# -*- coding: utf-8 -*-
"""
fit_response_curve.py
=====================
Uoc luong HAM PHAN UNG thi truong voi tin: f(S_t) -> target, bang 1 mang
neuron nhan tao (ANN) MUOT, roi doc dao ham bac 1 (f') va bac 2 (f'') de
tra ra NGUONG TOI HAN cua tin.

Y tuong:
  1. Data: banking_with_sentiment_vietstock.parquet (tu K4 refresh, da co
     S_t Gemini + Model va avg_return / log_volume / avg_range).
  2. ANN: MLP 2 lop an x 16 tanh (ham kich C-inf mooth) + HINH PHAT NANG
     LUONG CONG: loss = MSE + lam2 * mean(f''(x)^2) tren grid xac dinh
     -> ham tra ra mat, khong rung.
  3. Ensemble 6 seed -> f, f', f'' trung binh +- do lech chuan = dai tin cay.
  4. Dao ham bang torch.autograd (chinh xac, khong lai so).
  5. Doc "toi han":
       - x_peak   : |S_t| ma do nhay cam |f'| dat cuc dai
       - x_sat50  : |S_t| ma |f'| chi con 50% cuc dai -> bat dau bao hoa
       - x_sat20  : |S_t| ma |f'| chi con 20% -> vung toi han that su
       - x_onset  : |S_t| ma ham bat dau lac khoi tuyen tinh (phi tuyen)
       - diem uon : f'' doi dau (concave <-> convex)
       - bat doi xung : f(+m) vs f(-m) (tin tot khac tin xau?)
  6. Doi chieu: R^2 test (chia theo thoi gian) ANN vs OLS tuyen tinh.

Dau ra (data/processed/response_curves/):
  curve_<source>_<target>.csv   grid: S_t, f, f_lo/f_hi, f', f'_lo/hi, f'', ...
  threshold_summary.csv         bang nguong toi han cho moi (source, target)
  plot_<source>_<target>.png    3 panel: f (kem data + OLS), f', f''

Cach chay:
  python scripts/fit_response_curve.py                       # tat ca source x target
  python scripts/fit_response_curve.py --sources Gemini --targets avg_return
  python scripts/fit_response_curve.py --selftest            # kiem chung bang synthetic
  python scripts/fit_response_curve.py --skip-plot           # khong can matplotlib
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Tham so khung (z-space, du lieu da z-score)
GRID_LO, GRID_HI, GRID_N = -2.5, 2.5, 401
PEN_LO, PEN_HI, PEN_GRID_N = -2.0, 2.0, 128   # grid cho hinh phat f''
HIDDEN = (32, 32)         # 2 lop an, tanh
N_SEEDS = 6
MAX_EPOCHS = 2500
PATIENCE = 150
BATCH = 128
LR = 3e-3
WEIGHT_DECAY = 1e-4


# ----------------------------------------------------------------------------
# 1) MANG NEURON MUOT
# ----------------------------------------------------------------------------
class SmoothMLP(nn.Module):
    """MLP tanh: C-inf smooth, dao ham chinh xac moi cap bac."""

    def __init__(self, in_dim, hidden=HIDDEN):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def derivatives(model: nn.Module, x_z: torch.Tensor, others: torch.Tensor):
    """f(x_z) voi cac feature khac giu nguyen; tra f, f', f'' theo x_z.

    x_z   : (N,) requires_grad=True - cot S_t (z-space)
    others: (N, d-1) tensor cot giu nguyen (median = 0 trong z-space)
    """
    X = torch.cat([x_z.reshape(-1, 1), others], dim=1)
    f = model(X)
    g1 = torch.autograd.grad(f.sum(), x_z, create_graph=True)[0]
    g2 = torch.autograd.grad(g1.sum(), x_z, create_graph=True)[0]
    return f, g1, g2


def train_one(X, y, x_col, lam2, seed):
    """Train 1 net. Tra model. X,y: tensor float32; x_col: vi tri cot S_t."""
    torch.manual_seed(seed)
    n, d = X.shape
    # chia theo thoi gian: train / val (early stop) — test da cat truoc do
    n_tr = int(n * 0.95)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr:], y[n_tr:]

    model = SmoothMLP(d)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # grid penalty: x_z deu trong [PEN_LO, PEN_HI], feature khac = 0 (median)
    x_pen = torch.linspace(PEN_LO, PEN_HI, PEN_GRID_N, requires_grad=True)
    oth_pen = torch.zeros(PEN_GRID_N, d - 1)

    others_tr = torch.cat([X_tr[:, :x_col], X_tr[:, x_col + 1:]], dim=1)
    others_va = torch.cat([X_va[:, :x_col], X_va[:, x_col + 1:]], dim=1)
    x_tr_leaf = X_tr[:, x_col].clone().requires_grad_(True)
    x_va_leaf = X_va[:, x_col].clone().requires_grad_(True)

    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            xb = x_tr_leaf[idx].detach().clone().requires_grad_(True)
            ob = others_tr[idx]
            Xb = torch.cat([xb.reshape(-1, 1), ob], dim=1)
            pred = model(Xb)
            loss = nn.functional.mse_loss(pred, y_tr[idx])
            # hinh phat nang luong cong (dung grid xac dinh, moi epoch)
            _, _, g2p = derivatives(model, x_pen, oth_pen)
            loss = loss + lam2 * (g2p ** 2).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
        val_mse = float(nn.functional.mse_loss(val_pred, y_va))
        if val_mse < best_val - 1e-6:
            best_val, wait = val_mse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def eval_grid(model, x_col, d, n_grid=GRID_N):
    """f, f', f'' theo grid z-space (feature khac = 0 = median)."""
    x = torch.linspace(GRID_LO, GRID_HI, n_grid, requires_grad=True)
    oth = torch.zeros(n_grid, d - 1)
    f, g1, g2 = derivatives(model, x, oth)
    return (x.detach().numpy(), f.detach().numpy(),
            g1.detach().numpy(), g2.detach().numpy())


# ----------------------------------------------------------------------------
# 2) DOC NGUONG TOI HAN tu f', f''
# ----------------------------------------------------------------------------
def _nanmedian_safe(arr):
    """nanmedian khong warning khi toan bo la NaN."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def _nanstd_safe(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size <= 1 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanstd(arr))


def _sat_from_fp(x_z: np.ndarray, fp: np.ndarray, frac: float) -> float:
    """|x| ma |f'| lap giam xuong frac * |f'|max (du 20 diem lien tiep)."""
    i_pk = int(np.argmax(np.abs(fp)))
    peak_abs = float(np.abs(fp[i_pk]))
    inside = np.abs(x_z) <= 2.0
    x_in, fp_in = x_z[inside], np.abs(fp[inside])
    order = np.argsort(np.abs(x_in))
    run = 0
    for k in order:
        if fp_in[k] <= frac * peak_abs:
            run += 1
            if run >= 20:
                return float(abs(x_in[k]))
        else:
            run = 0
    return float("nan")


def detect_thresholds(x_z, f_m, fp_m, fpp_m, sigma_f, x_data_z, y_all, mu_y, sd_y):
    """Tra dict nguong. f_m/fp_m/fpp_m: trung binh ensemble (z-space y)."""
    out = {}
    # peak sensitivity
    i_pk = int(np.argmax(np.abs(fp_m)))
    out["x_peak_z"] = x_z[i_pk]
    out["fprime_peak_z"] = fp_m[i_pk]
    # bao hoa: diem |x| dau tien (quet tu 0 ra ngoai, du 20 diem sau)
    out["x_sat50_z"] = _sat_from_fp(x_z, fp_m, 0.50)
    out["x_sat20_z"] = _sat_from_fp(x_z, fp_m, 0.20)
    # diem uon: f'' doi dau
    sgn = np.sign(fpp_m)
    flips = np.where(np.diff(sgn) != 0)[0]
    flips = [k for k in flips if abs(x_z[k]) <= 2.0]
    out["n_inflections"] = len(flips)
    out["inflection_z"] = ";".join(f"{x_z[k]:+.2f}" for k in flips[:6])
    # bat doi xung: f(+m) vs f(-m), m = phan vi 95 cua |S_t| (z-space)
    m = float(np.percentile(np.abs(x_data_z), 95))
    m = min(m, 2.0)
    out["m_z"] = m
    out["f_at_plus_m"] = float(np.interp(m, x_z, f_m)) * sd_y + mu_y
    out["f_at_minus_m"] = float(np.interp(-m, x_z, f_m)) * sd_y + mu_y
    out["asym_f_plus_minus_f_minus"] = out["f_at_plus_m"] - out["f_at_minus_m"]
    out["x_m_raw"] = None  # dien sau (can mu_x, sd_x)
    out["_m_z"] = m
    out["_x_z"] = x_z
    out["_f_m"] = f_m
    out["_sigma_f"] = sigma_f
    out["_fp_m"] = fp_m
    out["_fpp_m"] = fpp_m
    out["_i_pk"] = i_pk
    return out


def onset_nonlinearity(x_z, f_m, f_lin_z, sigma_f):
    """|x| nho nhat ma |f - f_lin| > 1 sigma_ens, du lai 20 diem lien tiep."""
    dev = np.abs(f_m - f_lin_z)
    exceed = dev > np.maximum(sigma_f, 1e-9)
    for k in range(len(x_z)):
        if x_z[k] < 0 or not exceed[k] or k + 20 > len(x_z):
            continue
        if exceed[k:k + 20].all():
            return float(x_z[k])
    return float("nan")


# ----------------------------------------------------------------------------
# 3) CHAY 1 CAP (source, target)
# ----------------------------------------------------------------------------
FEATURES = ["S_t", "delta_S_t", "return_lag", "vol_lag"]


def prepare_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    d = df.sort_values("trade_date").copy()
    d["delta_S_t"] = d["S_t"].diff()
    d["return_lag"] = d["avg_return"].shift(1)
    d["vol_lag"] = d["log_volume"].shift(1)
    d = d.dropna(subset=FEATURES + [target]).reset_index(drop=True)
    return d


def run_pair(d: pd.DataFrame, source: str, target: str, outdir: Path,
             lam2: float, skip_plot: bool) -> dict:
    n = len(d)
    x_raw = d["S_t"].to_numpy(dtype=float)
    y_raw = d[target].to_numpy(dtype=float)
    mu_x, sd_x = float(np.mean(x_raw)), float(np.std(x_raw))
    mu_y, sd_y = float(np.mean(y_raw)), float(np.std(y_raw))

    X_z = (d[FEATURES].to_numpy(dtype=float) - d[FEATURES].to_numpy().mean(0)) \
        / d[FEATURES].to_numpy().std(0)
    X_z = X_z.astype(np.float32)
    y_z = ((y_raw - mu_y) / sd_y).astype(np.float32)
    x_col = FEATURES.index("S_t")

    # chia theo thoi gian: 80% dau (train+val) / 20% cuoi (test, khong tham gia train)
    n_test = int(n * 0.20)
    X_tr, y_tr = X_z[:n - n_test], y_z[:n - n_test]
    X_te, y_te = X_z[n - n_test:], y_z[n - n_test:]

    # ---- ensemble ----
    grids, fs, fps, fpps, models = [], [], [], [], []
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    for s in range(N_SEEDS):
        m = train_one(X_tr_t, y_tr_t, x_col, lam2, seed=1000 + s)
        models.append(m)
        xg, f, g1, g2 = eval_grid(m, x_col, X_z.shape[1])
        grids.append(xg)
        fs.append(f)
        fps.append(g1)
        fpps.append(g2)
    x_z = grids[0]
    f_m = np.mean(fs, 0)
    f_s = np.std(fs, 0)
    fp_m = np.mean(fps, 0)
    fp_s = np.std(fps, 0)
    fpp_m = np.mean(fpps, 0)
    fpp_s = np.std(fpps, 0)

    # NGUONG PER-SEED roi lay median (gan trung hon vi khong bi lam phang
    # boi trung binh cac duong f' khac nhau giua cac seed)
    per_pk = [x_z[int(np.argmax(np.abs(fp)))] for fp in fps]
    per_s50 = [_sat_from_fp(x_z, fp, 0.50) for fp in fps]
    per_s20 = [_sat_from_fp(x_z, fp, 0.20) for fp in fps]
    med_pk = _nanmedian_safe(per_pk)
    med_s50 = _nanmedian_safe(per_s50)
    med_s20 = _nanmedian_safe(per_s20)

    # ---- linear baseline (OLS cung features) ----
    try:
        from sklearn.linear_model import LinearRegression
        ols = LinearRegression().fit(X_tr, y_tr)
        r2_lin = float(ols.score(X_te, y_te))
        f_lin_z = ols.coef_[x_col] * x_z + ols.intercept_
    except Exception:
        r2_lin, f_lin_z = float("nan"), f_m * 0.0 + float(np.mean(y_tr))

    # R2 ANN tren test (dung day du features, khong chi slice) - tai su dung ensemble
    with torch.no_grad():
        preds = [m(X_te_t).numpy() for m in models]
        ann_pred = np.mean(preds, 0)
    ss_res = float(np.sum((y_te - ann_pred) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    r2_ann = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # ---- nguong toi han ----
    th = detect_thresholds(x_z, f_m, fp_m, fpp_m, f_s, X_z[:, x_col], y_z, mu_y, sd_y)
    # de median per-seed (chinh xac hon) cho 3 nguong chinh
    th["x_peak_z"], th["x_sat50_z"], th["x_sat20_z"] = med_pk, med_s50, med_s20
    th["_sat50_spread_z"] = _nanstd_safe(per_s50)
    th["_sat20_spread_z"] = _nanstd_safe(per_s20)
    x_onset = onset_nonlinearity(x_z, f_m, f_lin_z, f_s)

    # nhan hinh dang + f(0) + attention onset (cho response dang V / |S_t|)
    f0_z = float(np.interp(0.0, x_z, f_m))
    fm_z = float(np.interp(2.0, x_z, f_m))
    sig0 = float(np.interp(0.0, x_z, f_s))
    if f_m[-1] > f0_z + 2 * max(sig0, 1e-9) and f_m[0] > f0_z + 2 * max(sig0, 1e-9):
        shape = "V (|S_t|-shape): 2 phia cung tang"
    elif f_m[-1] > f0_z and f0_z > f_m[0]:
        shape = "tang dan (S-shape)"
    elif f_m[0] > f0_z and f0_z > f_m[-1]:
        shape = "giam dan"
    else:
        shape = "phang / phuc tap"
    # attention onset: |x| nho nhat ma f(|x|) > f(0) + 1 sigma_ens (du 20 diem)
    x_att = float("nan")
    for k in range(len(x_z) - 20):
        if x_z[k] < 0:
            continue
        if f_m[k] > f0_z + max(f_s[k], 1e-9) and np.all(f_m[k:k + 20] > f0_z):
            x_att = float(x_z[k])
            break

    # raw-space + ghi CSV curve
    x_raw_grid = mu_x + sd_x * x_z
    f_raw = mu_y + sd_y * f_m
    f_lo = mu_y + sd_y * (f_m - f_s)
    f_hi = mu_y + sd_y * (f_m + f_s)
    fp_raw = sd_y / sd_x * fp_m
    fp_lo = sd_y / sd_x * (fp_m - fp_s)
    fp_hi = sd_y / sd_x * (fp_m + fp_s)
    fpp_raw = sd_y / (sd_x ** 2) * fpp_m
    fpp_lo = sd_y / (sd_x ** 2) * (fpp_m - fpp_s)
    fpp_hi = sd_y / (sd_x ** 2) * (fpp_m + fpp_s)

    curve = pd.DataFrame({
        "S_t_raw": x_raw_grid, "S_t_z": x_z,
        "f_raw": f_raw, "f_lo": f_lo, "f_hi": f_hi, "f_ens_sd_raw": sd_y * f_s,
        "fprime_raw": fp_raw, "fprime_lo": fp_lo, "fprime_hi": fp_hi,
        "fpp_raw": fpp_raw, "fpp_lo": fpp_lo, "fpp_hi": fpp_hi,
        "f_lin_raw": mu_y + sd_y * f_lin_z,
    })
    outdir.mkdir(parents=True, exist_ok=True)
    curve_path = outdir / f"curve_{source}_{target}.csv"
    curve.to_csv(curve_path, index=False, encoding="utf-8-sig", float_format="%.6g")

    summary = {
        "source": source, "target": target, "n_days": n,
        "r2_ann_test": round(r2_ann, 4), "r2_ols_test": round(r2_lin, 4),
        "delta_r2": round(r2_ann - r2_lin, 4),
        "S_t_peak_raw": round(mu_x + sd_x * th["x_peak_z"], 4),
        "S_t_peak_z": round(th["x_peak_z"], 4),
        "fprime_peak_raw": round(sd_y / sd_x * th["fprime_peak_z"], 6),
        "S_t_sat50_raw": max(0.0, round(mu_x + sd_x * th["x_sat50_z"], 4)) if th["x_sat50_z"] == th["x_sat50_z"] else float("nan"),
        "S_t_sat20_raw": max(0.0, round(mu_x + sd_x * th["x_sat20_z"], 4)) if th["x_sat20_z"] == th["x_sat20_z"] else float("nan"),
        "sat50_spread_z": round(th["_sat50_spread_z"], 4),
        "sat20_spread_z": round(th["_sat20_spread_z"], 4),
        "S_t_onset_raw": round(mu_x + sd_x * x_onset, 4) if x_onset == x_onset else float("nan"),
        "shape": shape,
        "f_at_0_raw": round(mu_y + sd_y * f0_z, 6),
        "S_t_attention_raw": max(0.0, round(mu_x + sd_x * x_att, 4)) if x_att == x_att else float("nan"),
        "n_inflections": th["n_inflections"],
        "inflection_z": th["inflection_z"],
        "f_at_plus_m_raw": round(th["f_at_plus_m"], 6),
        "f_at_minus_m_raw": round(th["f_at_minus_m"], 6),
        "asymmetry_raw": round(th["asym_f_plus_minus_f_minus"], 6),
        "ens_sd_max_raw": round(sd_y * float(np.max(f_s)), 6),
    }

    # ---- ve 3 panel ----
    if not skip_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.family"] = "DejaVu Sans"
            plt.rcParams["axes.unicode_minus"] = False

            fig, axes = plt.subplots(3, 1, figsize=(9, 10.5), sharex=True,
                                     constrained_layout=True)
            ax = axes[0]
            sub = np.linspace(0, n - 1, min(500, n)).astype(int)
            ax.scatter(x_raw[sub], y_raw[sub], s=7, alpha=0.28,
                       color="#8899aa", label="dữ liệu ngày")
            ax.plot(x_raw_grid, f_raw, lw=2.4, color="#c0392b",
                    label="ANN f(S_t) (trung bình ensemble)")
            ax.fill_between(x_raw_grid, f_lo, f_hi, color="#c0392b", alpha=0.16,
                            label="dải tin cậy ±1σ")
            ax.plot(x_raw_grid, mu_y + sd_y * f_lin_z, ls="--", lw=1.4,
                    color="#2c3e50", label="OLS tuyến tính")
            ax.set_ylabel(target)
            ax.set_title(f"Hàm phản ứng f(S_t) — {source} → {target}  (n={n})")
            ax.legend(loc="best", fontsize=8)

            ax = axes[1]
            ax.plot(x_raw_grid, fp_raw, lw=2.2, color="#1a6f3c", label="f′ — hiệu ứng biên")
            ax.fill_between(x_raw_grid, fp_lo, fp_hi, color="#1a6f3c", alpha=0.16)
            for xv, c, lab in [
                (summary["S_t_peak_raw"], "#111111", "đỉnh nhạy cảm"),
                (summary.get("S_t_sat50_raw"), "#e67e22", "bão hòa 50%"),
                (summary.get("S_t_sat20_raw"), "#8e44ad", "tới hạn 20%"),
            ]:
                if isinstance(xv, (int, float)) and xv == xv:
                    ax.axvline(xv, color=c, ls=":", lw=1.3, label=lab)
            ax.set_ylabel(f"d{target}/dS_t")
            ax.legend(loc="best", fontsize=8)

            ax = axes[2]
            ax.plot(x_raw_grid, fpp_raw, lw=2.0, color="#2471a3", label="f″ — độ cong")
            ax.fill_between(x_raw_grid, fpp_lo, fpp_hi, color="#2471a3", alpha=0.16)
            ax.axhline(0, color="#555555", lw=1.0)
            ax.set_ylabel(f"d²{target}/dS_t²")
            ax.set_xlabel("S_t (chỉ số sentiment ngày, đơn vị gốc)")
            ax.legend(loc="best", fontsize=8)

            out_png = outdir / f"plot_{source}_{target}.png"
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            summary["_plot"] = str(out_png)
        except Exception as e:
            summary["_plot"] = f"skip ({e})"

    # ---- in ke_t qua ----
    print(f"\n--- {source} -> {target} (n={n}) ---")
    print(f"  R2 test : ANN {r2_ann:+.3f} | OLS {r2_lin:+.3f} "
          f"| delta(R2 phi tuyen) {r2_ann - r2_lin:+.3f}")
    print(f"  Dinh nhay cam  : S_t = {summary['S_t_peak_raw']:+.3f} "
          f"| f' max = {summary['fprime_peak_raw']:+.4g} / 1 don vi S_t")
    s50 = summary.get("S_t_sat50_raw")
    s20 = summary.get("S_t_sat20_raw")
    so  = summary.get("S_t_onset_raw")
    print(f"  Bao hoa 50%     : |S_t| = {s50:.3f}" if s50 == s50
          else "  Bao hoa 50%     : khong tim thay")
    print(f"  Toi han 20%     : |S_t| = {s20:.3f}" if s20 == s20
          else "  Toi han 20%     : khong tim thay")
    print(f"  Onset phi tuyen : S_t = {so:+.3f}" if so == so
          else "  Onset phi tuyen : tuyen tinh tren toan mien")
    att = summary.get("S_t_attention_raw")
    print(f"  Hinh dang       : {summary['shape']} | f(0)={summary['f_at_0_raw']:+.4g}")
    print(f"  Attention onset : |S_t| = {att:.3f}" if att == att
          else "  Attention onset : khong xac dinh")
    print(f"  Diem uon (z)    : {th['inflection_z'] or 'khong'} "
          f"(n={th['n_inflections']})")
    print(f"  Bat doi xung    : f(+m)={summary['f_at_plus_m_raw']:+.4g} "
          f"vs f(-m)={summary['f_at_minus_m_raw']:+.4g} "
          f"(D={summary['asymmetry_raw']:+.4g})")
    # chi tra ve cac key chinh (bo key noi bo dang '_' de ghi CSV sach)
    return {k: v for k, v in summary.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------
# 4) SELFTEST - synthetic co dap an biet truoc
# ----------------------------------------------------------------------------
def selftest(lam2: float, skip_plot: bool) -> int:
    """f_true(x_z) = 0.9*tanh(2.2*x_z) + 0.25*x_z (z-space).

    Dap an: |f'| max = 2.23 tai x=0; sat50 x=0.444; sat20 x=0.836;
    diem uon tai x=0 (tanh). Script phai dung lai gan dung."""
    rng = np.random.default_rng(7)
    n = 1200
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = np.clip(0.75 * x[i - 1] + 0.55 * rng.standard_normal(), -1.0, 1.0)
    x = x / np.std(x)  # de sau z-score std ~ 1
    eps = rng.standard_normal(n) * 0.25
    y_z = 0.9 * np.tanh(2.2 * x) + 0.25 * x + eps
    vol_lag = rng.standard_normal(n) * 0.5
    d = pd.DataFrame({
        "trade_date": pd.date_range("2022-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "S_t": x, "delta_S_t": np.concatenate([[0], np.diff(x)]),
        "avg_return": y_z, "log_volume": vol_lag, "avg_range": y_z * 0.8,
        "S_t_source": "Gemini",
    })
    print("=" * 68)
    print("SELFTEST - du lieu synthetic, dap an biet truoc (z-space)")
    print("  truth: f = 0.9*tanh(2.2*x) + 0.25*x  -> f'max=2.23 @x=0")
    print("         sat50 x=0.444 | sat20 x=0.836 | diem uon x=0")
    print("=" * 68)
    dd = prepare_frame(d, "avg_return")
    summary = run_pair(dd, "Gemini", "avg_return",
                       Path("/tmp/response_selftest"), lam2, skip_plot)

    def ok(got, want, tol):
        return got == got and abs(abs(got) - want) <= tol

    checks = [
        ("sat50 ~ 0.44 (+-0.20)", ok(summary.get("S_t_sat50_raw"), 0.444, 0.20)),
        ("sat20 ~ 0.84 (+-0.25)", ok(summary.get("S_t_sat20_raw"), 0.836, 0.25)),
        ("f'max ~ 2.23", ok(summary.get("fprime_peak_raw"), 2.23, 0.45)),
        ("R2 ANN > 0.45", summary["r2_ann_test"] > 0.45),
        ("delta R2 > 0.02", summary["delta_r2"] > 0.02),
    ]
    print("\nSELFTEST RESULT:")
    all_ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        all_ok &= passed
    print("  => " + ("MO HINH DOC DUNG HINH DANG + NGUONG." if all_ok
                     else "CAN HIEU CHINH (lam2 / epochs)."))
    return 0 if all_ok else 1


# ----------------------------------------------------------------------------
# 5) MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="data/processed/banking_with_sentiment_vietstock.parquet")
    ap.add_argument("--outdir", default="data/processed/response_curves")
    ap.add_argument("--sources", default="", help="VD: Gemini,Model (rong = tat ca)")
    ap.add_argument("--targets", default="avg_return,log_volume,avg_range")
    ap.add_argument("--lam2", type=float, default=0.0005,
                    help="trong so hinh phat nang luong cong mean(f''^2)")
    ap.add_argument("--skip-plot", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest(args.lam2, args.skip_plot))

    print("=" * 68)
    print("FIT RESPONSE CURVE - ANN muot + dao ham + nguong toi han")
    print("=" * 68)
    data = pd.read_parquet(args.data)
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.strftime("%Y-%m-%d")
    if "log_volume" not in data.columns and "total_volume" in data.columns:
        data["log_volume"] = np.log(data["total_volume"].clip(lower=1))
    sources = ([s.strip() for s in args.sources.split(",") if s.strip()]
               or sorted(data["S_t_source"].dropna().unique()))
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    rows = []
    outdir = Path(args.outdir)
    for src in sources:
        df_src = data[data["S_t_source"] == src]
        if len(df_src) < 120:
            print(f"[!] Bo qua {src}: chi {len(df_src)} ngay (<120)")
            continue
        for tg in targets:
            if tg not in df_src.columns:
                print(f"[!] Bo qua {src}->{tg}: khong co cot {tg}")
                continue
            dd = prepare_frame(df_src, tg)
            if len(dd) < 120:
                print(f"[!] Bo qua {src}->{tg}: sau dropna chi con {len(dd)}")
                continue
            rows.append(run_pair(dd, src, tg, outdir, args.lam2, args.skip_plot))

    if rows:
        outdir.mkdir(parents=True, exist_ok=True)
        sum_path = outdir / "threshold_summary.csv"
        pd.DataFrame(rows).drop(columns=[c for c in ["_plot", "_m_z", "_x_z", "_f_m",
                                                     "_sigma_f", "_fp_m", "_fpp_m",
                                                     "_i_pk"] if c in rows[0]]
                                  ).to_csv(sum_path, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 68)
        print("OK -> " + str(sum_path))
        print("    + curve_<source>_<target>.csv | plot_<source>_<target>.png")
        print("\nGiai nghia nhanh:")
        print("  - f'  > 0 va dat dinh gan S_t=0 : tin vua roi khoi trung tinh")
        print("    tac dong manh nhat; f' giam ve 0 hai ben = THI TRUONG HAP THU")
        print("    tin (tinh chat TOI HAN).")
        print("  - f'' doi dau (diem uon): chuyen vung tang toc -> giam toc")
        print("    cua phan ung; do la bien gioi giu 2 che do.")
        print("  - sat20 la 'ngung doc them tin' : qua do |S_t|, them tin nua")
        print("    khong lam target thay doi them dang ke.")
        print("  - delta_r2 > 0 : hinh phi tuyen mang them suc du bao thuc su")
        print("    so voi OLS tuyen tinh tren tap test cuoi ky.")


if __name__ == "__main__":
    main()

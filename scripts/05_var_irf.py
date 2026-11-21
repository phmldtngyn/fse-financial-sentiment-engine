# -*- coding: utf-8 -*-
"""
05_var_irf.py
=============
Ước lượng impulse response function qua Local Projections (Jordà 2005) cho
cú sốc sentiment lên return / volume / intraday_range.

Tại sao Local Projections thay vì VAR-Kalman?
- LP không cần chỉ định đúng VAR lag order.
- Mỗi horizon là 1 OLS riêng → dễ diễn giải, dễ tính CI.
- Ổn định hơn khi quan hệ phi tuyến (bài đã có response curve phi tuyến).

Cũng chạy VAR(p) chuẩn + IRF Cholesky để đối chiếu (qua statsmodels.tsa.api).

Input
-----
--input    : data/processed/banking_with_sentiment_vietstock.parquet
--s-col    : 'Sentiment_NLP'
--targets  : avg_return,intraday_range,log_volume
--horizon  : 14 (mặc định)
--lags     : 5 (mặc định)
--controls : vnindex_return,foreign_net_buy_log (tùy chọn, tách phẩy)

Output (data/processed/irf/)
- lp_irf_<target>.csv       (horizon, coef, std_err, ci_lo, ci_hi, p_value)
- lp_irf_<target>.png       (IRF plot với CI band)
- var_irf_<target>.csv      (đối chiếu VAR-Cholesky, nếu statsmodels OK)
- irf_summary.json          (cumulative IRF tại h=3, h=7, h=14)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.irf import local_projection_irf, lp_irf_table

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
except Exception:
    pass
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def fit_var_irf(y: np.ndarray, x: np.ndarray, horizon: int, lags: int) -> pd.DataFrame | None:
    """VAR(2) bivariate với Cholesky ordering (S, target) → IRF của target."""
    try:
        from statsmodels.tsa.api import VAR
    except ImportError:
        return None
    data = np.column_stack([x, y])
    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]
    if len(data) < 5 * lags:
        return None
    try:
        model = VAR(data)
        res = model.fit(maxlags=lags, ic="bic")
        irf = res.irf(horizon)
        # irf.irfs có shape (horizon+1, neqs, neqs); shock S (eq 0) → target (eq 1)
        coefs = irf.irfs[:, 1, 0]
        se = irf.stderr()[:, 1, 0] if hasattr(irf, "stderr") and irf.stderr() is not None else np.full_like(coefs, np.nan)
        return pd.DataFrame({
            "horizon": np.arange(len(coefs)),
            "var_coef": coefs,
            "var_se": se,
            "var_ci_lo": coefs - 1.96 * se,
            "var_ci_hi": coefs + 1.96 * se,
        })
    except Exception:
        return None


def plot_irf(lp_df: pd.DataFrame, var_df: pd.DataFrame | None, target: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    h = lp_df["horizon"].to_numpy()
    coef = lp_df["coef"].to_numpy()
    lo = lp_df["ci_lo"].to_numpy()
    hi = lp_df["ci_hi"].to_numpy()
    ax.plot(h, coef, color="#1f77b4", lw=2, label="Local Projection")
    ax.fill_between(h, lo, hi, color="#1f77b4", alpha=0.2, label="95% CI (LP)")
    if var_df is not None and "var_coef" in var_df.columns:
        ax.plot(var_df["horizon"], var_df["var_coef"], color="#d62728",
                lw=1.5, ls="--", label="VAR Cholesky")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Horizon (ngày)")
    ax.set_ylabel("Phản ứng của " + target + " (đơn vị σ sentiment)")
    ax.set_title(f"IRF: cú sốc sentiment (+1σ) → {target}")
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Projections + VAR IRF")
    parser.add_argument("--input", default="data/processed/banking_with_sentiment_vietstock.parquet")
    parser.add_argument("--s-col", default="Sentiment_NLP")
    parser.add_argument("--targets", default="avg_return,intraday_range,log_volume")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--lags", type=int, default=5)
    parser.add_argument("--controls", default="", help="cột macro tách phẩy, vd vnindex_return,foreign_net_buy_log")
    parser.add_argument("--out", default="data/processed/irf")
    parser.add_argument("--group-by", default="all")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[LỖI] Không tìm thấy input: {in_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    targets = [t.strip() for t in args.targets.split(",") if t.strip() in df.columns]
    controls = [c.strip() for c in args.controls.split(",") if c.strip() and c.strip() in df.columns]
    if args.s_col not in df.columns:
        print(f"[LỖI] Thiếu cột sentiment '{args.s_col}'.", file=sys.stderr)
        return 2

    group_col = None if args.group_by == "all" else args.group_by
    if group_col and group_col not in df.columns:
        print(f"[LỖI] Thiếu cột nhóm '{group_col}'.", file=sys.stderr)
        return 2

    if group_col is None:
        agg_map = {args.s_col: "mean"}
        for t in targets:
            agg_map[t] = "mean" if t in ("avg_return", "intraday_range", "log_volume") else "sum"
        for c in controls:
            agg_map[c] = "mean"
        work = df.groupby("date").agg(agg_map).reset_index()
        work["abs_return"] = work["avg_return"].abs() if "avg_return" in work.columns else np.nan
        groups_iter = [("ALL", work)]
    else:
        work = df.copy()
        groups_iter = [(name, sub) for name, sub in work.groupby(group_col, dropna=False)]

    print(f"[05] IRF | horizon={args.horizon} lags={args.lags} controls={controls or 'none'}")

    summaries = []
    for gname, sub in groups_iter:
        sub = sub.sort_values("date")
        for target in targets:
            ctrl_arrays = [sub[c].to_numpy() for c in controls] if controls else None
            try:
                lp = local_projection_irf(
                    sub[target].to_numpy(), sub[args.s_col].to_numpy(),
                    horizon=args.horizon, lags=args.lags, controls=ctrl_arrays,
                    alpha=0.05, standardize=True,
                )
            except Exception as e:
                print(f"  [!] LP fail {gname} {target}: {e}")
                continue
            lp.insert(0, "group", gname)
            lp.insert(1, "target", target)
            lp.insert(2, "sentiment", args.s_col)
            lp.to_csv(out_dir / f"lp_irf_{gname}_{target}.csv", index=False, encoding="utf-8-sig")

            # VAR đối chiếu
            var_df = fit_var_irf(
                sub[target].to_numpy(), sub[args.s_col].to_numpy(),
                horizon=args.horizon, lags=args.lags,
            )
            if var_df is not None:
                var_df.to_csv(out_dir / f"var_irf_{gname}_{target}.csv", index=False, encoding="utf-8-sig")

            # Plot
            plot_irf(lp, var_df, target, out_dir / f"irf_{gname}_{target}.png")

            # Cumulative IRF tại h=3, 7, 14
            cum = {}
            for h_check in [3, 7, args.horizon]:
                if h_check > args.horizon:
                    continue
                idx = lp["horizon"] <= h_check
                cum[f"cum_irf_h{h_check}"] = float(lp.loc[idx, "coef"].sum())
                cum[f"cum_irf_h{h_check}_p"] = float(
                    lp.loc[lp["horizon"] == h_check, "p_value"].iloc[0]
                    if h_check in lp["horizon"].values else np.nan
                )
            peak_row = lp.loc[lp["coef"].abs().idxmax()]
            summaries.append({
                "group": gname, "target": target,
                "peak_horizon": int(peak_row["horizon"]),
                "peak_coef": float(peak_row["coef"]),
                "peak_pvalue": float(peak_row["p_value"]),
                "irf_h0": float(lp.loc[lp["horizon"] == 0, "coef"].iloc[0]),
                "irf_h7": float(lp.loc[lp["horizon"] == 7, "coef"].iloc[0]
                                if 7 in lp["horizon"].values else np.nan),
                "irf_h14": float(lp.loc[lp["horizon"] == 14, "coef"].iloc[0]
                                 if 14 in lp["horizon"].values else np.nan),
                "n_obs": int(lp["n"].iloc[0]),
                **cum,
            })

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(out_dir / "irf_summary.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "local_projections",
        "horizon": args.horizon,
        "lags": args.lags,
        "controls": controls,
        "n_target_pairs": len(summaries),
        "interpretation": (
            "IRF cho thấy phản ứng của target sau cú sốc sentiment +1σ. "
            "Đỉnh IRF tại h>0 = hiệu ứng trễ. Nếu cum_irf_h14 khác 0 đáng kể "
            "→ tác động bền vững. Nếu CI bao gồm 0 ở mọi h → không có tác động đáng kể."
        ),
    }
    with open(out_dir / "irf_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Tóm tắt IRF ===")
    for s in summaries[:6]:
        print(f"  {s['group']:6s} {s['target']:18s} peak h={s['peak_horizon']:2d} coef={s['peak_coef']:+.4f} p={s['peak_pvalue']:.3f}")
    print(f"\nĐã ghi: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

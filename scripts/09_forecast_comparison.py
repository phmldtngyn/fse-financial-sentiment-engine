# -*- coding: utf-8 -*-
"""
09_forecast_comparison.py
=========================
So sánh 3 mô hình dự báo return ngày T+1 trên 500 ngày test cuối (walk-forward):

1. Baseline: AR(5) + day-of-week
2. + S_t (Gemini): AR(5) + DoW + S_t (Gemini)
3. + S_t (PhoBERT model): AR(5) + DoW + S_t (model)

Báo cáo:
- RMSE, MAE, MAPE cho mỗi mô hình
- Diebold–Mariano test (so sánh baseline vs mỗi mở rộng)
- Clark–West test (nested models, bias-corrected)
- Hit rate (dự báo đúng chiều tăng/giảm)

Input
-----
--input            : data/processed/banking_with_sentiment_vietstock.parquet
--s-col            : 'Sentiment_NLP'  (Gemini)
--s-model-col      : 's_model'        (PhoBERT, từ script 02)
--target           : 'avg_return'
--test-size        : 500 (mặc định, walk-forward)

Output (data/processed/forecast/)
- forecast_results.csv     (y_true, y_hat_baseline, y_hat_gemini, y_hat_model, date)
- metrics_summary.csv      (RMSE, MAE, MAPE, hit_rate cho 3 mô hình)
- dm_tests.csv             (Diebold–Mariano + Clark–West)
- forecast_summary.json
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
from scipy import stats


def make_features(series: np.ndarray, sentiment: np.ndarray | None,
                  lags: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """X = [const, lag_1..lag_lags, DoW dummies, sentiment_lag_1]."""
    n = len(series)
    if n <= lags + 2:
        raise ValueError(f"n={n} quá nhỏ cho lags={lags}")
    rows = []
    y = []
    for t in range(lags, n - 1):
        feat = [series[t - k] for k in range(1, lags + 1)]
        rows.append(feat)
        y.append(series[t + 1])
    X = np.array(rows, dtype=float)
    Y = np.array(y, dtype=float)
    if sentiment is not None:
        # S_t tại ngày t (dùng để dự báo t+1) → sentiment[t] ứng với row index = t - lags
        s_lag1 = sentiment[lags:n - 1]  # sentiment tại t (sẽ dự báo t+1)
        X = np.column_stack([X, s_lag1])
    return X, Y


def add_dow_features(X: np.ndarray, dates: pd.Series, lags: int) -> np.ndarray:
    """Thêm day-of-week dummies (loại Monday)."""
    # dates ứng với row i = ngày tại t = lags + i
    dow = pd.to_datetime(dates.iloc[lags:len(dates) - 1]).dt.dayofweek.to_numpy()
    # Mon=0, ..., Fri=4
    dummies = np.zeros((len(dow), 4))
    for d in range(1, 5):
        dummies[:, d - 1] = (dow == d).astype(float)
    return np.column_stack([np.ones(len(dow)), X, dummies])


def ols_fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return X_test @ beta


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold–Mariano test. H0: equal predictive accuracy.
    e1, e2: residual arrays. loss = squared error.
    """
    d = e1 ** 2 - e2 ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return float("nan"), 1.0
    dbar = float(np.mean(d))
    # HAC variance (Newey–West with h lags)
    gamma = np.var(d, ddof=1) if n > 1 else 0.0
    for k in range(1, h + 1):
        if n - k < 1:
            break
        g = np.cov(d[k:], d[:-k])[0, 1]
        gamma += 2 * g
    se = np.sqrt(gamma / n) if gamma > 0 else float("nan")
    if not np.isfinite(se) or se == 0:
        return float("nan"), 1.0
    stat = dbar / se
    p = float(stats.norm.sf(abs(stat)) * 2)
    return float(stat), p


def clark_west_test(e_restricted: np.ndarray, e_full: np.ndarray,
                    y_true: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Clark–West (2007) test cho nested models. H0: equal MSPE.
    Statistic = mean((e_restricted² - e_full² + (y_full_hat - y_restricted_hat)²)) / SE.
    Tuy nhiên ta không có y_full_hat và y_restricted_hat riêng → xấp xỉ:
    CW ≈ (MSPE_r - MSPE_f + var(diff_pred)) / SE.
    Ở đây xấp xỉ bằng hiệu MSPE không hiệu chỉnh (conservative).
    """
    er = e_restricted[np.isfinite(e_restricted)]
    ef = e_full[np.isfinite(e_full)]
    if len(er) != len(ef) or len(er) < 5:
        return float("nan"), 1.0
    n = len(er)
    d = er ** 2 - ef ** 2
    # Clark-West adjustment: thêm variance of forecast difference
    # Ước lượng var(y_hat_full - y_hat_restricted) ≈ var(e_restricted - e_full)
    adj = np.var(e_restricted - e_full, ddof=1) / n
    dbar = float(np.mean(d) + adj)
    se = float(np.sqrt(np.var(d, ddof=1) / n)) if n > 1 else float("nan")
    if not np.isfinite(se) or se == 0:
        return float("nan"), 1.0
    stat = dbar / se
    p = float(stats.norm.sf(abs(stat)) * 2)
    return float(stat), p


def main() -> int:
    parser = argparse.ArgumentParser(description="Out-of-sample forecast + DM/CW tests")
    parser.add_argument("--input", default="data/processed/banking_with_sentiment_vietstock.parquet")
    parser.add_argument("--s-col", default="Sentiment_NLP", help="S_t Gemini")
    parser.add_argument("--s-model-col", default="s_model", help="S_t PhoBERT (từ script 02)")
    parser.add_argument("--target", default="avg_return")
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--lags", type=int, default=5)
    parser.add_argument("--out", default="data/processed/forecast")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[LỖI] Không tìm thấy input: {in_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "ticker" in df.columns:
        # Basket aggregate
        df = df.groupby("date").agg({
            args.target: "mean",
            args.s_col: "mean",
            **({args.s_model_col: "mean"} if args.s_model_col in df.columns else {}),
        }).reset_index().sort_values("date")
    if args.s_col not in df.columns:
        print(f"[LỖI] Thiếu cột sentiment '{args.s_col}'.", file=sys.stderr)
        return 2
    if args.s_model_col not in df.columns:
        print(f"[!] Cột '{args.s_model_col}' không có → chạy 2 mô hình thay vì 3.")
        has_model = False
    else:
        has_model = True

    y = df[args.target].to_numpy()
    s_g = df[args.s_col].to_numpy()
    s_m = df[args.s_model_col].to_numpy() if has_model else None
    dates = df["date"]
    n = len(y)
    if n < args.test_size + args.lags * 2 + 50:
        print(f"[LỖI] n={n} quá nhỏ cho test_size={args.test_size} lags={args.lags}", file=sys.stderr)
        return 2

    test_start = n - args.test_size
    print(f"[09] Forecast | n={n} test={args.test_size} lags={args.lags}")

    # Walk-forward 1-step
    y_true, y_hat_b, y_hat_g, y_hat_m = [], [], [], []
    test_dates = []
    for i in range(test_start, n - 1):
        y_train = y[:i]
        s_g_train = s_g[:i]
        s_m_train = s_m[:i] if has_model else None
        dates_train = dates.iloc[:i]

        # Baseline: AR(lags) + DoW, không sentiment
        Xb_tr, yb_tr = make_features(y_train, None, lags=args.lags)
        Xb_tr = add_dow_features(Xb_tr, dates_train, lags=args.lags)
        # Gemini: AR + DoW + S_gemini
        Xg_tr, yg_tr = make_features(y_train, s_g_train, lags=args.lags)
        Xg_tr = add_dow_features(Xg_tr, dates_train, lags=args.lags)
        if has_model:
            Xm_tr, ym_tr = make_features(y_train, s_m_train, lags=args.lags)
            Xm_tr = add_dow_features(Xm_tr, dates_train, lags=args.lags)

        # Test row at t = i (dự báo t+1 = i+1)
        if i < args.lags:
            continue
        feat_b = np.concatenate([[1.0], y[i - args.lags:i][::-1], [0, 0, 0, 0]])
        feat_g = np.concatenate([[1.0], y[i - args.lags:i][::-1], [0, 0, 0, 0], [s_g[i]]])
        if has_model:
            feat_m = np.concatenate([[1.0], y[i - args.lags:i][::-1], [0, 0, 0, 0], [s_m[i]]])
        # DoW dummy cho test: tại index i, ngày = dates.iloc[i]
        dow_i = pd.to_datetime(dates.iloc[i]).dayofweek()
        for d in range(1, 5):
            feat_b[args.lags + 1 + d - 1] = 1.0 if dow_i == d else 0.0
            feat_g[args.lags + 1 + d - 1] = 1.0 if dow_i == d else 0.0
            if has_model:
                feat_m[args.lags + 1 + d - 1] = 1.0 if dow_i == d else 0.0

        # Fit trên train
        try:
            yb_pred = ols_fit_predict(Xb_tr, yb_tr, feat_b.reshape(1, -1))[0]
            yg_pred = ols_fit_predict(Xg_tr, yg_tr, feat_g.reshape(1, -1))[0]
            if has_model:
                ym_pred = ols_fit_predict(Xm_tr, ym_tr, feat_m.reshape(1, -1))[0]
            else:
                ym_pred = float("nan")
        except Exception:
            continue
        y_true.append(y[i + 1])
        y_hat_b.append(yb_pred)
        y_hat_g.append(yg_pred)
        y_hat_m.append(ym_pred)
        test_dates.append(dates.iloc[i + 1])

    y_true = np.array(y_true)
    y_hat_b = np.array(y_hat_b)
    y_hat_g = np.array(y_hat_g)
    y_hat_m = np.array(y_hat_m)

    res_df = pd.DataFrame({
        "date": test_dates, "y_true": y_true,
        "y_hat_baseline": y_hat_b,
        "y_hat_gemini": y_hat_g,
        "y_hat_model": y_hat_m if has_model else np.nan,
    })
    res_df.to_csv(out_dir / "forecast_results.csv", index=False, encoding="utf-8-sig")

    # Metrics
    def _metrics(y_true, y_hat):
        mask = np.isfinite(y_true) & np.isfinite(y_hat)
        yt, yh = y_true[mask], y_hat[mask]
        if len(yt) == 0:
            return {"rmse": float("nan"), "mae": float("nan"),
                    "mape": float("nan"), "hit_rate": float("nan"), "n": 0}
        rmse = float(np.sqrt(np.mean((yt - yh) ** 2)))
        mae = float(np.mean(np.abs(yt - yh)))
        mape = float(np.mean(np.abs((yt - yh) / (np.abs(yt) + 1e-9))))
        # Hit rate: dự báo đúng chiều (sign của yt - 0 so với yh - 0)
        hit = float(np.mean(np.sign(yt) == np.sign(yh)))
        return {"rmse": rmse, "mae": mae, "mape": mape, "hit_rate": hit, "n": int(len(yt))}

    mb = _metrics(y_true, y_hat_b)
    mg = _metrics(y_true, y_hat_g)
    mm = _metrics(y_true, y_hat_m) if has_model else {"rmse": float("nan"), "mae": float("nan"),
                                                      "mape": float("nan"), "hit_rate": float("nan"),
                                                      "n": 0}

    metrics_df = pd.DataFrame([
        {"model": "baseline", **mb},
        {"model": "+S_gemini", **mg},
        {"model": "+S_model", **mm},
    ])
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    # DM + CW tests: baseline vs +S_gemini, baseline vs +S_model
    e_b = y_true - y_hat_b
    e_g = y_true - y_hat_g
    e_m = y_true - y_hat_m if has_model else None
    tests = []
    dm_stat, dm_p = dm_test(e_b, e_g, h=1)
    cw_stat, cw_p = clark_west_test(e_b, e_g, y_true, h=1)
    tests.append({"comparison": "baseline vs +S_gemini",
                  "dm_stat": dm_stat, "dm_pvalue": dm_p,
                  "cw_stat": cw_stat, "cw_pvalue": cw_p,
                  "interpretation": "DM < 0 & p < 0.05 → baseline sai hơn (sentiment có giá trị)"})
    if has_model:
        dm_stat, dm_p = dm_test(e_b, e_m, h=1)
        cw_stat, cw_p = clark_west_test(e_b, e_m, y_true, h=1)
        tests.append({"comparison": "baseline vs +S_model",
                      "dm_stat": dm_stat, "dm_pvalue": dm_p,
                      "cw_stat": cw_stat, "cw_pvalue": cw_p,
                      "interpretation": "DM < 0 & p < 0.05 → baseline sai hơn"})
        dm_stat, dm_p = dm_test(e_g, e_m, h=1)
        tests.append({"comparison": "+S_gemini vs +S_model",
                      "dm_stat": dm_stat, "dm_pvalue": dm_p,
                      "cw_stat": float("nan"), "cw_pvalue": float("nan"),
                      "interpretation": "Non-nested: dùng DM (Clark–West không áp dụng)"})
    pd.DataFrame(tests).to_csv(out_dir / "dm_tests.csv", index=False, encoding="utf-8-sig")

    summary = {
        "test_size": int(args.test_size),
        "lags": args.lags,
        "metrics": {
            "baseline": mb, "gemini": mg, "model": mm,
        },
        "dm_baseline_vs_gemini_p": float(tests[0]["dm_pvalue"]),
        "dm_baseline_vs_model_p": float(tests[1]["dm_pvalue"]) if has_model else float("nan"),
        "interpretation": (
            "Nếu DM p < 0.05 (baseline vs +S) → sentiment có giá trị dự báo thực sự. "
            "Hit rate > 0.55 → sentiment có giá trị kinh tế (chiều tăng/giảm). "
            "Clark–West là test đúng cho nested models — ưu tiên báo cáo."
        ),
    }
    with open(out_dir / "forecast_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Tóm tắt forecast ===")
    print(f"Baseline    RMSE={mb['rmse']:.4f}  MAE={mb['mae']:.4f}  Hit={mb['hit_rate']:.2%}")
    print(f"+S_gemini   RMSE={mg['rmse']:.4f}  MAE={mg['mae']:.4f}  Hit={mg['hit_rate']:.2%}")
    if has_model:
        print(f"+S_model    RMSE={mm['rmse']:.4f}  MAE={mm['mae']:.4f}  Hit={mm['hit_rate']:.2%}")
    print(f"\nDM tests:")
    for t in tests:
        print(f"  {t['comparison']:30s}  DM p={t['dm_pvalue']:.4f}  CW p={t.get('cw_pvalue', float('nan')):.4f}")
    print(f"\nĐã ghi: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

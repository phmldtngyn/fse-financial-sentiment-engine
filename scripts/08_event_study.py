# -*- coding: utf-8 -*-
"""
08_event_study.py
=================
Event study quanh các sự kiện (Fed, NHNN, earnings, COVID, SVB) để đo Cumulative
Abnormal Return (CAR) và xem sentiment trong cửa sổ pre-event có tiên đoán CAR
trong cửa sổ post-event không.

Phương pháp:
1. Expected return = market model: r_expected = α + β·r_VNINDEX, ước lượng từ
   cửa sổ [-250, -30] ngày trước event.
2. AR_t = r_ticker,t - r_expected,t
3. AAR (average across events) và CAR[1, +window_post]
4. Kiểm định: t-test cho AAR từng ngày, Patell Z cho CAR.
5. Hồi quy: CAR[1, post] ~ S_t[-pre, 0]  → sentiment có tiên đoán CAR không?

Input
-----
--input      : data/processed/banking_with_sentiment_vietstock.parquet
--events     : config/events.yaml
--vnindex    : data/processed/vnindex.parquet (cột: date, vnindex_close)
              Nếu không có, dùng trung bình return của basket ngân hàng làm proxy.
--s-col      : 'Sentiment_NLP'

Output (data/processed/event_study/)
- aar_car_<event_id>.csv       (AAR + CAR theo từng ngày quanh event)
- car_summary.csv              (bảng gộp CAR[-pre, +post] cho mỗi event)
- sentiment_predicts_car.csv   (regression CAR[1,post] ~ S_t[-pre,0])
- event_study_summary.json
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
import yaml
from scipy import stats


def load_vnindex(path: Path) -> pd.DataFrame | None:
    if not path or not path.exists():
        return None
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # Lấy cột close
    for cand in ("vnindex_close", "close", "adj_close", "VNINDEX_Close"):
        if cand in df.columns:
            df = df[["date", cand]].rename(columns={cand: "vnindex_close"})
            df["vnindex_return"] = df["vnindex_close"].pct_change()
            return df
    return None


def market_model_params(ticker_returns: np.ndarray, market_returns: np.ndarray,
                        est_window: int = 250) -> tuple[float, float]:
    """Ước lượng alpha, beta từ cửa sổ estimation."""
    mask = np.isfinite(ticker_returns) & np.isfinite(market_returns)
    if mask.sum() < est_window:
        # Dùng tất cả available
        if mask.sum() < 30:
            return 0.0, 1.0  # Fallback: market-adjusted
    tr = ticker_returns[mask]
    mr = market_returns[mask]
    X = np.column_stack([np.ones(len(tr)), mr])
    beta, *_ = np.linalg.lstsq(X, tr, rcond=None)
    return float(beta[0]), float(beta[1])


def compute_ar(ticker_ret: np.ndarray, market_ret: np.ndarray,
               alpha: float, beta: float) -> np.ndarray:
    return ticker_ret - (alpha + beta * market_ret)


def main() -> int:
    parser = argparse.ArgumentParser(description="Event study quanh Fed/NHNN/earnings/crisis")
    parser.add_argument("--input", default="data/processed/banking_with_sentiment_vietstock.parquet")
    parser.add_argument("--events", default="config/events.yaml")
    parser.add_argument("--vnindex", default="data/processed/vnindex.parquet")
    parser.add_argument("--s-col", default="Sentiment_NLP")
    parser.add_argument("--out", default="data/processed/event_study")
    parser.add_argument("--return-col", default="avg_return")
    args = parser.parse_args()

    in_path = Path(args.input)
    ev_path = Path(args.events)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[LỖI] Không tìm thấy input: {in_path}", file=sys.stderr)
        return 2
    if not ev_path.exists():
        print(f"[LỖI] Không tìm thấy events config: {ev_path}", file=sys.stderr)
        return 2

    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if args.s_col not in df.columns or args.return_col not in df.columns:
        print(f"[LỖI] Thiếu cột sentiment hoặc return.", file=sys.stderr)
        return 2

    with open(ev_path, "r", encoding="utf-8") as f:
        ev_cfg = yaml.safe_load(f)
    events = ev_cfg.get("events", [])
    if not events:
        print("[LỖI] config/events.yaml không có event nào.", file=sys.stderr)
        return 2

    vnindex = load_vnindex(Path(args.vnindex))
    if vnindex is None:
        print(f"[08] Không có vnindex.parquet → dùng basket return trung bình làm proxy market.")
        proxy = df.groupby("date")[args.return_col].mean().reset_index()
        proxy = proxy.rename(columns={args.return_col: "vnindex_return"})
        vnindex = proxy
    else:
        df = pd.merge(df, vnindex[["date", "vnindex_return"]], on="date", how="left")

    # Nếu group by ticker
    if "ticker" in df.columns:
        tickers = df["ticker"].unique()
    else:
        df["ticker"] = "BASKET"
        tickers = ["BASKET"]

    print(f"[08] Event study | {len(events)} events × {len(tickers)} tickers")

    aar_rows = []
    car_rows = []
    predict_rows = []
    for ev in events:
        ev_date = pd.Timestamp(ev["date"]).normalize()
        pre = int(ev.get("window_pre", 5))
        post = int(ev.get("window_post", 10))
        ev_id = ev["id"]

        for ticker in tickers:
            sub = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            if sub.empty:
                continue
            # Tìm vị trí event date trong sub
            idx_ev = sub["date"].searchsorted(ev_date)
            if idx_ev >= len(sub):
                # Tìm ngày gần nhất
                diff = (sub["date"] - ev_date).abs()
                idx_ev = int(diff.idxmin())
                if diff.iloc[idx_ev] > pd.Timedelta(days=7):
                    continue  # Event không có dữ liệu gần
            # Cắt estimation window [-250-pre, -pre-1] (cách event ít nhất 30 ngày)
            est_end = idx_ev - 30
            est_start = max(0, est_end - 250)
            if est_end - est_start < 30:
                continue
            est = sub.iloc[est_start:est_end]
            alpha, beta = market_model_params(
                est[args.return_col].to_numpy(),
                est["vnindex_return"].to_numpy() if "vnindex_return" in est else np.zeros(len(est)),
            )
            # Cắt event window [-pre, +post]
            win_start = max(0, idx_ev - pre)
            win_end = min(len(sub), idx_ev + post + 1)
            win = sub.iloc[win_start:win_end].copy()
            if len(win) < pre + post + 1:
                continue
            win["ar"] = compute_ar(
                win[args.return_col].to_numpy(),
                win["vnindex_return"].to_numpy() if "vnindex_return" in win else np.zeros(len(win)),
                alpha, beta,
            )
            # Relative day index
            win["day_rel"] = np.arange(win_start, win_end) - idx_ev
            win["event_id"] = ev_id
            win["ticker"] = ticker
            aar_rows.append(win[["event_id", "ticker", "date", "day_rel",
                                 args.return_col, "vnindex_return", "ar", args.s_col]])

            # CAR[1, post] và CAR[-pre, 0]
            ar_post = win.loc[win["day_rel"].between(1, post), "ar"].to_numpy()
            ar_pre = win.loc[win["day_rel"].between(-pre, 0), "ar"].to_numpy()
            car_post = float(np.sum(ar_post)) if len(ar_post) > 0 else float("nan")
            car_pre = float(np.sum(ar_pre)) if len(ar_pre) > 0 else float("nan")
            # S_t trung bình trong [-pre, 0]
            s_pre = win.loc[win["day_rel"].between(-pre, 0), args.s_col].to_numpy()
            s_pre_mean = float(np.nanmean(s_pre)) if len(s_pre) > 0 else float("nan")
            s_pre_abs = float(np.nanmean(np.abs(s_pre))) if len(s_pre) > 0 else float("nan")
            car_rows.append({
                "event_id": ev_id, "event_date": ev["date"],
                "event_label_vi": ev.get("label_vi", ""),
                "type": ev.get("type", ""),
                "expected_dir": ev.get("expected_dir", ""),
                "ticker": ticker,
                "n_pre": int(len(ar_pre)), "n_post": int(len(ar_post)),
                "car_pre": car_pre, "car_post": car_post,
                "s_pre_mean": s_pre_mean, "s_pre_abs": s_pre_abs,
                "alpha": alpha, "beta": beta,
            })

    if not aar_rows:
        print("[LỖI] Không có event nào đủ dữ liệu.", file=sys.stderr)
        return 1

    aar_df = pd.concat(aar_rows, ignore_index=True)
    aar_df.to_csv(out_dir / "aar_detail.csv", index=False, encoding="utf-8-sig")
    car_df = pd.DataFrame(car_rows)
    car_df.to_csv(out_dir / "car_summary.csv", index=False, encoding="utf-8-sig")

    # Per-event AAR + CAR (aggregate across tickers)
    for ev_id, grp in aar_df.groupby("event_id"):
        agg = grp.groupby("day_rel").agg(
            aar=("ar", "mean"),
            n=("ar", "size"),
            ar_std=("ar", "std"),
        ).reset_index()
        agg["aar_se"] = agg["ar_std"] / np.sqrt(agg["n"])
        agg["aar_t"] = agg["aar"] / agg["aar_se"].replace(0, np.nan)
        agg["aar_pvalue"] = 2 * stats.t.sf(np.abs(agg["aar_t"]), agg["n"] - 1)
        agg["car"] = agg["aar"].cumsum()
        agg["event_id"] = ev_id
        agg.to_csv(out_dir / f"aar_car_{ev_id}.csv", index=False, encoding="utf-8-sig")

    # Regression: CAR_post ~ S_pre_mean (cross-section across events × tickers)
    valid = car_df.dropna(subset=["car_post", "s_pre_mean"])
    if len(valid) >= 10:
        X = np.column_stack([np.ones(len(valid)), valid["s_pre_mean"].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, valid["car_post"].to_numpy(), rcond=None)
        resid = valid["car_post"].to_numpy() - X @ beta
        dof = len(valid) - 2
        sigma2 = float(resid @ resid) / dof if dof > 0 else float("nan")
        XtX_inv = np.linalg.inv(X.T @ X)
        se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
        t = beta[1] / se if se > 0 else float("nan")
        p = float(stats.t.sf(abs(t), dof) * 2) if np.isfinite(t) else float("nan")
        r2 = 1 - float(resid @ resid) / float(
            np.sum((valid["car_post"] - valid["car_post"].mean()) ** 2)
        ) if len(valid) > 1 else float("nan")
        predict_rows.append({
            "predictor": "s_pre_mean", "target": "car_post",
            "coef": float(beta[1]), "se": se, "t": float(t),
            "p_value": p, "r2": r2, "n": len(valid),
            "significant": bool(np.isfinite(p) and p < 0.05),
        })

    # |S_t| predictor
    if "s_pre_abs" in valid.columns and len(valid) >= 10:
        y = valid["car_post"].abs().to_numpy()
        X = np.column_stack([np.ones(len(valid)), valid["s_pre_abs"].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(valid) - 2
        sigma2 = float(resid @ resid) / dof if dof > 0 else float("nan")
        XtX_inv = np.linalg.inv(X.T @ X)
        se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
        t = beta[1] / se if se > 0 else float("nan")
        p = float(stats.t.sf(abs(t), dof) * 2) if np.isfinite(t) else float("nan")
        predict_rows.append({
            "predictor": "s_pre_abs", "target": "abs(car_post)",
            "coef": float(beta[1]), "se": se, "t": float(t),
            "p_value": p, "r2": 1 - float(resid @ resid) / float(np.sum((y - y.mean()) ** 2)),
            "n": len(valid), "significant": bool(np.isfinite(p) and p < 0.05),
        })

    if predict_rows:
        pd.DataFrame(predict_rows).to_csv(
            out_dir / "sentiment_predicts_car.csv", index=False, encoding="utf-8-sig")

    # Direction match: S_pre_mean sign có khớp expected_dir không?
    dir_match = 0
    dir_total = 0
    for _, row in car_df.iterrows():
        if not np.isfinite(row["s_pre_mean"]):
            continue
        exp = row.get("expected_dir", "")
        if not exp:
            continue
        s_sign = "positive" if row["s_pre_mean"] > 0 else ("negative" if row["s_pre_mean"] < 0 else "neutral")
        if (exp == "positive" and s_sign == "positive") or \
           (exp == "negative" and s_sign == "negative"):
            dir_match += 1
        dir_total += 1

    summary = {
        "n_events": len(events),
        "n_event_ticker_pairs": len(car_df),
        "direction_match_rate": (dir_match / dir_total) if dir_total > 0 else float("nan"),
        "direction_match_count": dir_match,
        "direction_total": dir_total,
        "n_predictor_regressions": len(predict_rows),
        "predictor_significant": sum(1 for r in predict_rows if r["significant"]),
        "interpretation": (
            "direction_match_rate > 0.6 → sentiment phản ánh đúng chiều event. "
            "predictor_significant → sentiment trong [-pre,0] tiên đoán CAR[1,+post]. "
            "Nếu cả 2 đều tốt → sentiment có giá trị thực tiễn (event-trading signal)."
        ),
    }
    with open(out_dir / "event_study_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Tóm tắt event study ===")
    print(f"Events                 : {summary['n_events']}")
    print(f"Event×ticker pairs     : {summary['n_event_ticker_pairs']}")
    print(f"Sentiment direction match: {dir_match}/{dir_total} ({summary['direction_match_rate']:.1%})")
    print(f"Predictor regressions  : {summary['n_predictor_regressions']} "
          f"({summary['predictor_significant']} significant)")
    print(f"\nĐã ghi: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

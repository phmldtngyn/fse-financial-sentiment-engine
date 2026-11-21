# -*- coding: utf-8 -*-
"""
06_egarch_x.py
==============
Fit EGARCH-X và GJR-GARCH-X cho từng ticker (hoặc basket), với sentiment làm
biến ngoại sinh trong mean equation, leverage (γ) trong variance equation.

Câu hỏi nghiên cứu:
- Hệ số sentiment trong mean có significant không? (sentiment tiên đoán return)
- Hệ số leverage γ có significant không? (tin xấu tăng vol hơn tin tốt cùng độ lớn)
- So sánh AIC/BIC EGARCH-X vs EGARCH thường để xem sentiment có tăng fit không?

Input
-----
--input      : data/processed/banking_with_sentiment_vietstock.parquet
--s-col      : 'Sentiment_NLP'
--return-col : 'avg_return' (mặc định)
--ticker     : tất cả ticker trong cột 'ticker' (mặc định), hoặc chỉ định 1 ticker
--model      : egarch | gjr | both (mặc định)

Output (data/processed/garch_x/)
- egarch_x_results.csv      (mỗi dòng = 1 ticker, hệ số sentiment + γ + AIC/BIC)
- gjr_garch_x_results.csv
- per_ticker/<ticker>_fit.txt   (full summary text)
- garch_x_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.garch_x import fit_egarch_x, fit_gjr_garch_x, fit_panel_garch_x

# arch fit thường warning về convergence
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> int:
    parser = argparse.ArgumentParser(description="EGARCH-X / GJR-GARCH-X với sentiment")
    parser.add_argument("--input", default="data/processed/banking_with_sentiment_vietstock.parquet")
    parser.add_argument("--s-col", default="Sentiment_NLP")
    parser.add_argument("--return-col", default="avg_return")
    parser.add_argument("--model", choices=["egarch", "gjr", "both"], default="both")
    parser.add_argument("--ticker", default=None,
                        help="Chỉ fit 1 ticker (mặc định: tất cả trong cột 'ticker')")
    parser.add_argument("--basket", action="store_true",
                        help="Fit basket aggregate thay vì từng ticker")
    parser.add_argument("--out", default="data/processed/garch_x")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[LỖI] Không tìm thấy input: {in_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_ticker").mkdir(exist_ok=True)

    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if args.s_col not in df.columns or args.return_col not in df.columns:
        print(f"[LỖI] Thiếu cột sentiment hoặc return. Có: {list(df.columns)}", file=sys.stderr)
        return 2

    if args.basket:
        work = df.groupby("date").agg({
            args.s_col: "mean",
            args.return_col: "mean",
        }).reset_index().sort_values("date")
        work["ticker"] = "BASKET"
        groups = [("BASKET", work)]
    elif args.ticker:
        if "ticker" not in df.columns:
            print("[LỖI] Không có cột 'ticker'.", file=sys.stderr)
            return 2
        sub = df[df["ticker"] == args.ticker].sort_values("date")
        if sub.empty:
            print(f"[LỖI] Không tìm ticker '{args.ticker}'.", file=sys.stderr)
            return 2
        groups = [(args.ticker, sub)]
    else:
        if "ticker" not in df.columns:
            print("[LỖI] Không có cột 'ticker'. Dùng --basket để aggregate.", file=sys.stderr)
            return 2
        groups = [(name, sub.sort_values("date"))
                  for name, sub in df.groupby("ticker", dropna=False)]

    print(f"[06] GARCH-X | model={args.model} | {len(groups)} nhóm")

    egarch_rows = []
    gjr_rows = []
    for ticker, sub in groups:
        if len(sub) < 200:
            print(f"  [-] {ticker}: bỏ qua (n={len(sub)} < 200)")
            continue
        r = sub[args.return_col].to_numpy()
        s = sub[args.s_col].to_numpy()
        if args.model in ("egarch", "both"):
            try:
                res = fit_egarch_x(r, s, p=1, q=1, dist="normal")
                row = {"ticker": ticker}
                for k in ("model", "n_obs", "loglikelihood", "aic", "bic",
                          "sentiment_coef", "sentiment_se", "sentiment_t",
                          "sentiment_pvalue", "sentiment_significant",
                          "alpha", "beta", "gamma_leverage", "gamma_pvalue"):
                    row[k] = res.get(k, float("nan"))
                egarch_rows.append(row)
                # Full summary vào per_ticker/
                with open(out_dir / "per_ticker" / f"{ticker}_egarch_fit.txt", "w", encoding="utf-8") as f:
                    f.write(f"Ticker: {ticker}\n")
                    f.write(f"Model: EGARCH(1,1)-X\n")
                    f.write(f"n_obs: {res['n_obs']}\n")
                    f.write(f"LogL: {res['loglikelihood']:.2f}\n")
                    f.write(f"AIC: {res['aic']:.2f} | BIC: {res['bic']:.2f}\n")
                    f.write(f"Sentiment coef: {res['sentiment_coef']:.4f} "
                            f"(t={res['sentiment_t']:.2f}, p={res['sentiment_pvalue']:.4f})\n")
                    f.write(f"Leverage γ: {res['gamma_leverage']:.4f} (p={res['gamma_pvalue']:.4f})\n")
                    f.write(f"α: {res['alpha']:.4f} | β: {res['beta']:.4f}\n")
            except Exception as e:
                egarch_rows.append({"ticker": ticker, "error": str(e)[:200]})

        if args.model in ("gjr", "both"):
            try:
                res = fit_gjr_garch_x(r, s, p=1, q=1, dist="normal")
                row = {"ticker": ticker}
                for k in ("model", "n_obs", "loglikelihood", "aic", "bic",
                          "sentiment_coef", "sentiment_pvalue", "sentiment_significant",
                          "alpha", "beta", "gamma_leverage", "gamma_pvalue"):
                    row[k] = res.get(k, float("nan"))
                gjr_rows.append(row)
                with open(out_dir / "per_ticker" / f"{ticker}_gjr_fit.txt", "w", encoding="utf-8") as f:
                    f.write(f"Ticker: {ticker}\n")
                    f.write(f"Model: GJR-GARCH(1,1)-X\n")
                    f.write(f"n_obs: {res['n_obs']} | AIC: {res['aic']:.2f} | BIC: {res['bic']:.2f}\n")
                    f.write(f"Sentiment: {res['sentiment_coef']:.4f} (p={res['sentiment_pvalue']:.4f})\n")
                    f.write(f"Leverage γ: {res['gamma_leverage']:.4f} (p={res['gamma_pvalue']:.4f})\n")
            except Exception as e:
                gjr_rows.append({"ticker": ticker, "error": str(e)[:200]})

    if egarch_rows:
        edf = pd.DataFrame(egarch_rows)
        edf.to_csv(out_dir / "egarch_x_results.csv", index=False, encoding="utf-8-sig")
    if gjr_rows:
        gdf = pd.DataFrame(gjr_rows)
        gdf.to_csv(out_dir / "gjr_garch_x_results.csv", index=False, encoding="utf-8-sig")

    def _stats(dfx):
        if dfx is None or dfx.empty or "sentiment_pvalue" not in dfx.columns:
            return {"n": 0, "n_sig_sentiment": 0, "n_sig_leverage": 0}
        n_sig_sent = int((dfx["sentiment_pvalue"].fillna(1) < 0.05).sum())
        n_sig_lev = int((dfx.get("gamma_pvalue", pd.Series([1])).fillna(1) < 0.05).sum())
        return {"n": len(dfx), "n_sig_sentiment": n_sig_sent, "n_sig_leverage": n_sig_lev}

    summary = {
        "model": args.model,
        "n_groups": len(groups),
        "egarch": _stats(egarch_rows and pd.DataFrame(egarch_rows)),
        "gjr_garch": _stats(gjr_rows and pd.DataFrame(gjr_rows)),
        "interpretation": (
            "Hệ số sentiment significant → tin tức đi trước return. "
            "Hệ số leverage γ significant → tin xấu tăng vol nhiều hơn tin tốt cùng độ lớn "
            "(consistent với behavioral finance: loss aversion)."
        ),
    }
    with open(out_dir / "garch_x_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Tóm tắt GARCH-X ===")
    if egarch_rows:
        es = summary["egarch"]
        print(f"EGARCH-X  : {es['n_sig_sentiment']}/{es['n']} ticker có sentiment significant")
        print(f"            {es['n_sig_leverage']}/{es['n']} ticker có leverage significant")
    if gjr_rows:
        gs = summary["gjr_garch"]
        print(f"GJR-GARCH-X: {gs['n_sig_sentiment']}/{gs['n']} ticker có sentiment significant")
        print(f"            {gs['n_sig_leverage']}/{gs['n']} ticker có leverage significant")
    print(f"\nĐã ghi: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

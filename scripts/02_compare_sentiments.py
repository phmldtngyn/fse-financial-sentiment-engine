# -*- coding: utf-8 -*-
"""
02_compare_sentiments.py
========================
So sánh hai nguồn sentiment: S_t từ PhoBERT (do FSE tạo) vs S_t từ Gemini
(nhãn LLM aggregate của VN-Quant, đang được dùng trong correlation_analysis.py).

Mục đích: trả lời câu hỏi phản biện "PhoBERT model để làm gì nếu correlation
dùng nhãn Gemini?". Script này:
1. Sinh S_t_model = mean(prob_positive - prob_negative) per day từ output của PhoBERT.
2. Tính tương quan đồng thời (concurrent) giữa S_t_model và S_t_gemini.
3. Chạy lại 8 cặp tương quan (theo docs/correlation_report.md) với cả 2 nguồn.
4. So sánh độ lớn và significance, đưa ra bảng đối chiếu.

Input
-----
--news-with-probs   : data/processed/news_with_probs.parquet
                       (cột: date, ticker?, probs (np.array[N,3]), hoặc prob_pos, prob_neg, prob_neu)
--banking-daily     : data/processed/banking_with_sentiment_vietstock.parquet
                       (cột: date, ticker, Sentiment_NLP [Gemini], avg_return, log_volume, avg_range)
--out               : data/processed/sentiment_comparison/

Output
- daily_sentiment_compare.parquet : date, s_gemini, s_model, n_news
- correlation_two_sources.csv      : Pearson, Spearman, n giữa 2 S_t
- banking_corr_comparison.csv      : bảng 8 cặp (source=S_t) vs (source=S_t_model)
- plot_corr_comparison.png         : scatter S_gemini vs S_model
- agreement_summary.json
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


# 8 cặp tương quan chính (theo docs/correlation_report.md)
TARGETS = ["avg_return", "intraday_range", "log_volume", "abs_return", "volume"]


def compute_s_model(news_df: pd.DataFrame) -> pd.DataFrame:
    """Tính S_t_model = mean(prob_pos - prob_neg) trên news trong ngày."""
    if "probs" in news_df.columns:
        probs = np.stack(news_df["probs"].to_numpy())
        news_df = news_df.assign(
            prob_pos=probs[:, 2] if probs.shape[1] == 3 else probs[:, -1],
            prob_neg=probs[:, 0],
            prob_neu=probs[:, 1] if probs.shape[1] == 3 else probs[:, 1],
        )
    for c in ("prob_pos", "prob_neg"):
        if c not in news_df.columns:
            raise ValueError(f"Thiếu cột '{c}' trong news_df (hoặc cột 'probs').")
    news_df["s_score"] = news_df["prob_pos"] - news_df["prob_neg"]
    if "date" not in news_df.columns:
        # Tìm cột ngày (trade_date, published_date, ...)
        for cand in ("trade_date", "published_date", "publish_date", "date_only"):
            if cand in news_df.columns:
                news_df = news_df.rename(columns={cand: "date"})
                break
        else:
            raise ValueError("Không tìm thấy cột ngày (date/trade_date/...).")
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
    daily = (
        news_df.groupby("date")
        .agg(s_model=("s_score", "mean"),
             s_model_abs=("s_score", lambda s: np.mean(np.abs(s))),
             n_news=("s_score", "size"))
        .reset_index()
    )
    return daily


def correlation_pair(x, y) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 30:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan"), "spearman_p": float("nan"), "n": int(len(x))}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp), "n": int(len(x))}


def main() -> int:
    parser = argparse.ArgumentParser(description="So sánh sentiment PhoBERT vs Gemini")
    parser.add_argument("--news-with-probs", default="data/processed/news_with_probs.parquet")
    parser.add_argument("--banking-daily", default="data/processed/banking_with_sentiment_vietstock.parquet")
    parser.add_argument("--out", default="data/processed/sentiment_comparison")
    args = parser.parse_args()

    news_path = Path(args.news_with_probs)
    bank_path = Path(args.banking_daily)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not news_path.exists():
        print(f"[LỖI] Không tìm thấy news probs: {news_path}", file=sys.stderr)
        print("       Nếu chưa có, chạy pipeline FSE để dự đoán probs cho toàn bộ corpus lịch sử,")
        print("       sau đó aggregate theo ngày.", file=sys.stderr)
        return 2
    if not bank_path.exists():
        print(f"[LỖI] Không tìm thấy banking daily: {bank_path}", file=sys.stderr)
        return 2

    print(f"[02] Đọc news probs: {news_path}")
    news_df = pd.read_parquet(news_path)
    s_model_daily = compute_s_model(news_df)
    print(f"     Đã tính S_t_model cho {len(s_model_daily)} ngày, "
          f"trung bình {s_model_daily['n_news'].mean():.1f} news/ngày.")

    print(f"[02] Đọc banking daily: {bank_path}")
    bank = pd.read_parquet(bank_path)
    bank["date"] = pd.to_datetime(bank["date"]).dt.normalize()

    # Phát hiện cột Gemini sentiment
    gemini_col = None
    for cand in ("Sentiment_NLP", "sentiment_gemini", "s_gemini", "Sentiment_NLP_vn_quant"):
        if cand in bank.columns:
            gemini_col = cand
            break
    if gemini_col is None:
        print(f"[LỖI] Không tìm cột sentiment Gemini trong banking daily. "
              f"Có: {list(bank.columns)}", file=sys.stderr)
        return 2
    print(f"     Cột Gemini sentiment: '{gemini_col}'")

    # Tính S_t_gemini hàng ngày trung bình trên 26 mã (nếu có ticker)
    if "ticker" in bank.columns:
        gemini_daily = bank.groupby("date")[gemini_col].mean().reset_index()
        gemini_daily = gemini_daily.rename(columns={gemini_col: "s_gemini"})
    else:
        gemini_daily = bank[["date", gemini_col]].rename(columns={gemini_col: "s_gemini"})

    # Merge
    daily = pd.merge(s_model_daily, gemini_daily, on="date", how="inner")
    daily = daily.sort_values("date").reset_index(drop=True)
    daily.to_parquet(out_dir / "daily_sentiment_compare.parquet", index=False)
    print(f"     Merge được {len(daily)} ngày chung (cả 2 nguồn đều có).")

    # 1) Tương quan giữa hai S_t
    pair_stats = correlation_pair(daily["s_gemini"], daily["s_model"])
    print(f"     S_gemini ↔ S_model: Pearson r={pair_stats['pearson_r']:.4f} (p={pair_stats['pearson_p']:.2e}), "
          f"Spearman ρ={pair_stats['spearman_r']:.4f}")
    pd.DataFrame([pair_stats]).to_csv(out_dir / "correlation_two_sources.csv",
                                      index=False, encoding="utf-8-sig")

    # 2) Bảng đối chiếu 8 cặp tương quan (nhóm basket ngân hàng, daily)
    #    Trước tiên cần aggregate daily cho cả basket.
    # avg_return, intraday_range: mean trên 26 mã
    # log_volume, volume: sum hoặc mean — ta dùng mean (đã được log_volume mean)
    if "ticker" in bank.columns:
        basket = (
            bank.groupby("date")
            .agg(avg_return=("avg_return", "mean"),
                 intraday_range=("intraday_range", "mean"),
                 log_volume=("log_volume", "mean") if "log_volume" in bank.columns else ("total_volume", lambda s: np.log(s.sum())),
                 volume=("total_volume", "sum") if "total_volume" in bank.columns else ("volume", "sum"),
                 s_gemini=(gemini_col, "mean"))
            .reset_index()
        )
    else:
        basket = bank.copy()
        basket = basket.rename(columns={gemini_col: "s_gemini"})
    basket["abs_return"] = basket["avg_return"].abs()

    basket = pd.merge(basket, s_model_daily[["date", "s_model"]], on="date", how="inner")
    basket = basket.sort_values("date").reset_index(drop=True)

    rows = []
    for target in TARGETS:
        if target not in basket.columns:
            continue
        x_g = basket["s_gemini"].to_numpy()
        x_m = basket["s_model"].to_numpy()
        y = basket[target].to_numpy()
        g = correlation_pair(x_g, y)
        m = correlation_pair(x_m, y)
        rows.append({
            "target": target,
            "r_gemini": g["pearson_r"], "p_gemini": g["pearson_p"], "n_gemini": g["n"],
            "r_model": m["pearson_r"], "p_model": m["pearson_p"], "n_model": m["n"],
            "r_diff": m["pearson_r"] - g["pearson_r"],
            "same_sign": bool(np.sign(g["pearson_r"]) == np.sign(m["pearson_r"])),
            "both_significant": bool(g["pearson_p"] < 0.05 and m["pearson_p"] < 0.05),
        })
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_dir / "banking_corr_comparison.csv", index=False, encoding="utf-8-sig")

    # 3) Scatter plot
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(daily["s_gemini"], daily["s_model"], s=10, alpha=0.4, edgecolor="none")
    if pair_stats["pearson_r"] == pair_stats["pearson_r"]:  # not nan
        xs = np.array([daily["s_gemini"].min(), daily["s_gemini"].max()])
        # Linear fit
        valid = np.isfinite(daily["s_gemini"]) & np.isfinite(daily["s_model"])
        if valid.sum() > 2:
            b1, b0 = np.polyfit(daily["s_gemini"][valid], daily["s_model"][valid], 1)
            ax.plot(xs, b1 * xs + b0, color="red", lw=1.5, label=f"OLS fit (r={pair_stats['pearson_r']:.3f})")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("S_t (Gemini, VN-Quant)")
    ax.set_ylabel("S_t (PhoBERT model)")
    ax.set_title("So sánh hai nguồn sentiment (daily)")
    ax.legend(loc="best")
    fig.savefig(out_dir / "plot_corr_comparison.png", dpi=120)
    plt.close(fig)

    # 4) Agreement summary
    summary = {
        "n_days_overlap": int(len(daily)),
        "pearson_two_sources": pair_stats["pearson_r"],
        "spearman_two_sources": pair_stats["spearman_r"],
        "p_value_pearson": pair_stats["pearson_p"],
        "n_target_pairs_compared": int(len(comp_df)),
        "n_pairs_same_sign": int(comp_df["same_sign"].sum()),
        "n_pairs_both_significant": int(comp_df["both_significant"].sum()),
        "mean_r_diff": float(comp_df["r_diff"].mean()),
        "interpretation": (
            "Nếu |r| > 0.7 và cả 8 cặp cùng dấu cùng significant → hai nguồn tương đương, "
            "có thể dùng PhoBERT (nhanh, rẻ, tái lập) thay Gemini. "
            "Nếu |r| < 0.3 hoặc khác dấu → hai nguồn đang đo thứ khác nhau, cần điều tra."
        ),
    }
    with open(out_dir / "agreement_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Tóm tắt so sánh ===")
    print(f"Số ngày chung          : {summary['n_days_overlap']}")
    print(f"Pearson(Gemini, Model) : {summary['pearson_two_sources']:.4f} (p={summary['p_value_pearson']:.2e})")
    print(f"Spearman               : {summary['spearman_two_sources']:.4f}")
    print(f"Cặp tương quan so sánh : {summary['n_target_pairs_compared']}")
    print(f"  Cùng dấu             : {summary['n_pairs_same_sign']}/{summary['n_target_pairs_compared']}")
    print(f"  Cùng significant     : {summary['n_pairs_both_significant']}/{summary['n_target_pairs_compared']}")
    print(f"\nĐã ghi: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

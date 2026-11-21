# -*- coding: utf-8 -*-
"""
03_robustness_battery.py
========================
Pin ba phép thử robustness cho bảng 8 cặp tương quan:

1. Block bootstrap 95% CI cho từng cặp (block size L=5, n_boot=5000)
2. Subsample stability: chia 2016–2019 / 2020–2022 / 2023–2026, chạy lại từng
   khoảng và báo cáo dấu + significance
3. Bai–Perron structural break test cho mỗi cặp (max 3 break, min_seg=60)
4. CUSUM of squares (Brown–Durbin–Evans) cho mỗi cặp
5. Rolling correlation 120 ngày để visualize stability

Kết quả gộp vào 1 file CSV với cột: ticker, feature, r_full, ci_lo, ci_hi,
stable_subsample, n_breaks, cusum_significant, ...

Input
-----
--input         : data/processed/banking_with_sentiment_vietstock.parquet
                  (hoặc file đã join sentiment + price ở mức ngày + ticker)
--corr-results  : data/processed/correlation/correlation_results.parquet (tùy chọn,
                  để lấy danh sách 8 cặp đã significance)
--s-col         : tên cột sentiment (mặc định 'Sentiment_NLP')
--targets       : danh sách cột target cách nhau bởi dấu phẩy
                  (mặc định: avg_return,intraday_range,log_volume,abs_return,volume)
--out           : data/processed/robustness/

Output
- bootstrap_ci.csv          (block bootstrap CI cho mỗi cặp)
- subsample_stability.csv   (chia 3 giai đoạn)
- structural_breaks.csv     (Bai–Perron + CUSUM)
- rolling_corr.csv          (rolling 120d, để vẽ)
- robustness_summary.json
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

# Thêm repo root vào path để import src.analysis
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.bootstrap import block_bootstrap_corr
from src.analysis.breaks import bai_perron_breaks, cusum_squares, rolling_corr_stability


def split_subsamples(df: pd.DataFrame, date_col: str = "date") -> list[tuple[str, pd.DataFrame]]:
    """Chia 3 giai đoạn: 2016–2019, 2020–2022, 2023–2026."""
    if date_col not in df.columns:
        return [("full", df)]
    d = pd.to_datetime(df[date_col])
    cuts = [
        ("2016-2019", (d >= "2016-01-01") & (d < "2020-01-01")),
        ("2020-2022", (d >= "2020-01-01") & (d < "2023-01-01")),
        ("2023-2026", (d >= "2023-01-01")),
    ]
    out = []
    for name, mask in cuts:
        sub = df.loc[mask]
        if len(sub) > 60:
            out.append((name, sub))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Robustness battery cho 8 cặp tương quan")
    parser.add_argument("--input", default="data/processed/banking_with_sentiment_vietstock.parquet")
    parser.add_argument("--s-col", default="Sentiment_NLP")
    parser.add_argument("--targets", default="avg_return,intraday_range,log_volume,abs_return,volume")
    parser.add_argument("--abs-sentiment", action="store_true",
                        help="Cũng chạy cho |S_t| — theo README cặp |S_t| ↔ volume/range")
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--out", default="data/processed/robustness")
    parser.add_argument("--group-by", default="ticker",
                        help="Nhóm để chạy correlation (mặc định 'ticker'; dùng 'all' để basket)")
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
    if not targets:
        print(f"[LỖI] Không có target nào trong df. Có: {list(df.columns)}", file=sys.stderr)
        return 2
    if args.s_col not in df.columns:
        print(f"[LỖI] Thiếu cột sentiment '{args.s_col}'. Có: {list(df.columns)}", file=sys.stderr)
        return 2

    df["abs_sentiment"] = df[args.s_col].abs()
    sentiment_cols = [args.s_col]
    if args.abs_sentiment:
        sentiment_cols.append("abs_sentiment")

    group_col = args.group_by if args.group_by != "all" else None
    if group_col is not None and group_col not in df.columns:
        print(f"[LỖI] Thiếu cột nhóm '{group_col}'. Có: {list(df.columns)}", file=sys.stderr)
        return 2

    print(f"[03] Robustness trên {len(df)} dòng, targets={targets}, sentiment={sentiment_cols}")
    print(f"     Group: {group_col or 'basket (all)'} | Block L={args.block_size} | n_boot={args.n_boot}")

    # Tạo basket aggregate nếu group='all'
    if group_col is None:
        basket = (
            df.groupby("date")
            .agg({args.s_col: "mean", "abs_sentiment": "mean",
                  **{t: ("mean" if t in ("avg_return", "intraday_range", "log_volume") else "sum")
                     for t in targets}})
            .reset_index()
        )
        basket["abs_return"] = basket["avg_return"].abs() if "avg_return" in basket.columns else np.nan
        work = basket
        groups_iter = [("ALL", work)]
    else:
        work = df.copy()
        groups_iter = [(name, sub) for name, sub in work.groupby(group_col, dropna=False)]

    # 1) Block bootstrap CI
    boot_rows = []
    for s_col in sentiment_cols:
        for target in targets:
            for gname, sub in groups_iter:
                sub = sub.sort_values("date")
                try:
                    res = block_bootstrap_corr(
                        sub[s_col].to_numpy(), sub[target].to_numpy(),
                        L=args.block_size, n_boot=args.n_boot, alpha=0.05,
                        method="pearson", seed=42,
                    )
                    boot_rows.append({
                        "group": gname, "sentiment": s_col, "target": target,
                        "corr": res.point, "ci_lo": res.lo, "ci_hi": res.hi,
                        "n_boot": res.n_boot, "n_obs": res.n_used,
                        "ci_excludes_zero": bool(res.lo > 0 or res.hi < 0),
                    })
                except Exception as e:
                    boot_rows.append({
                        "group": gname, "sentiment": s_col, "target": target,
                        "corr": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                        "error": str(e)[:100], "n_obs": len(sub),
                    })
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out_dir / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    # 2) Subsample stability
    sub_rows = []
    for s_col in sentiment_cols:
        for target in targets:
            for gname, sub in groups_iter:
                for sub_name, sub_sub in split_subsamples(sub):
                    sub_sub = sub_sub.sort_values("date")
                    x = sub_sub[s_col].to_numpy()
                    y = sub_sub[target].to_numpy()
                    mask = np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 30:
                        continue
                    from scipy import stats as st
                    r, p = st.pearsonr(x[mask], y[mask])
                    sub_rows.append({
                        "group": gname, "sentiment": s_col, "target": target,
                        "subsample": sub_name, "n": int(mask.sum()),
                        "r": float(r), "p_value": float(p),
                        "significant": bool(p < 0.05),
                        "sign": int(np.sign(r)),
                    })
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(out_dir / "subsample_stability.csv", index=False, encoding="utf-8-sig")

    # 3) Structural breaks: Bai–Perron + CUSUM
    break_rows = []
    for s_col in sentiment_cols:
        for target in targets:
            for gname, sub in groups_iter:
                sub = sub.sort_values("date")
                x = sub[s_col].to_numpy()
                y = sub[target].to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 120:
                    continue
                try:
                    bp = bai_perron_breaks(y[mask], x[mask], max_breaks=3, min_seg=60)
                    cs = cusum_squares(y[mask], x[mask])
                    # Map break indices back to dates
                    dates = sub["date"].to_numpy()[mask]
                    break_dates = [pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
                                   for i in bp.break_indices] if bp.break_indices else []
                    break_rows.append({
                        "group": gname, "sentiment": s_col, "target": target,
                        "n_obs": bp.n_obs,
                        "n_breaks_bai_perron": bp.n_breaks,
                        "break_dates": ",".join(break_dates),
                        "supF": bp.sup_f, "bp_pvalue": bp.p_value,
                        "bp_significant": bp.significant,
                        "cusum_stat": cs.statistic, "cusum_pvalue": cs.p_value,
                        "cusum_significant": cs.significant,
                        "cusum_max_pos": cs.max_pos,
                    })
                except Exception as e:
                    break_rows.append({
                        "group": gname, "sentiment": s_col, "target": target,
                        "error": str(e)[:100],
                    })
    break_df = pd.DataFrame(break_rows)
    break_df.to_csv(out_dir / "structural_breaks.csv", index=False, encoding="utf-8-sig")

    # 4) Rolling correlation (chỉ cho basket)
    if group_col is None:
        rc = rolling_corr_stability(
            work, args.s_col, targets[0], group_cols=[], window=120,
        )
    else:
        rc = rolling_corr_stability(
            work, args.s_col, targets[0], group_cols=[group_col], window=120,
        )
    rc.to_csv(out_dir / "rolling_corr.csv", index=False, encoding="utf-8-sig")

    # 5) Summary
    n_total = len(boot_df)
    n_excludes_zero = int(boot_df.get("ci_excludes_zero", pd.Series(dtype=bool)).sum()) if "ci_excludes_zero" in boot_df else 0
    n_stable = 0
    if not sub_df.empty:
        # Cặp ổn định = cùng dấu trong cả 3 subsample
        for (g, s, t), grp in sub_df.groupby(["group", "sentiment", "target"]):
            if len(grp) >= 3 and (grp["sign"].nunique() == 1):
                n_stable += 1
    n_breaks = int((break_df.get("n_breaks_bai_perron", pd.Series(dtype=int)) > 0).sum()) if not break_df.empty else 0

    summary = {
        "n_pairs_tested": n_total,
        "n_ci_excludes_zero": n_excludes_zero,
        "n_stable_across_subsamples": n_stable,
        "n_pairs_with_breaks": n_breaks,
        "block_size": args.block_size,
        "n_boot": args.n_boot,
        "subsamples": ["2016-2019", "2020-2022", "2023-2026"],
        "interpretation": (
            "Bảng đủ robust nếu: (1) >80% cặp có CI excludes zero, "
            "(2) >80% cặp stable sign across 3 subsamples, "
            "(3) <30% cặp có structural break. Nếu không đạt, viết limitation vào báo cáo."
        ),
    }
    with open(out_dir / "robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Tóm tắt robustness ===")
    print(f"Cặp test                : {summary['n_pairs_tested']}")
    print(f"CI excludes zero        : {summary['n_ci_excludes_zero']}")
    print(f"Stable 3 subsamples     : {summary['n_stable_across_subsamples']}")
    print(f"Có structural break     : {summary['n_pairs_with_breaks']}")
    print(f"\nĐã ghi: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

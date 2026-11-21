# -*- coding: utf-8 -*-
"""
04_granger_causality.py
=======================
Kiểm định nhân quả Granger và Toda–Yamamoto (bidirectional) cho các cặp:
    S_t → target  và  target → S_t

Lý do chạy cả 2: kiểm tra chiều nhân quả thực sự. Nếu chỉ S→target significant
→ sentiment tiên đoán. Nếu cả 2 significant → feedback loop. Nếu chỉ target→S
→ sentiment chỉ phản ứng, không tiên đoán.

Lý do chạy cả Granger và Toda–Yamamoto:
- Granger cần chuỗi dừng. Chuỗi tài chính thường có unit root → Granger sai.
- Toda–Yamamoto robust với unit root/cointegration, thêm d lags dummy rồi Wald
  trên p lags đầu. Đây là phiên bản "an toàn" để báo cáo.

Input
-----
--input    : data/processed/banking_with_sentiment_vietstock.parquet
--s-col    : 'Sentiment_NLP'
--targets  : avg_return,intraday_range,log_volume,abs_return,volume
--max-lag  : 10 (mặc định)
--method   : granger | toda_yamamoto | both (mặc định)
--d        : bậc tích hợp tối đa cho Toda–Yamamoto (mặc định 1)

Output (data/processed/causality/)
- granger_results.csv      (mỗi dòng: group, direction, lag, F, p_value, significant)
- toda_yamamoto_results.csv
- best_lag_summary.csv     (chọn lag theo BIC nhỏ nhất + significant)
- causality_summary.json
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
from src.analysis.causality import bidirectional_causality


def main() -> int:
    parser = argparse.ArgumentParser(description="Granger + Toda–Yamamoto causality")
    parser.add_argument("--input", default="data/processed/banking_with_sentiment_vietstock.parquet")
    parser.add_argument("--s-col", default="Sentiment_NLP")
    parser.add_argument("--targets", default="avg_return,intraday_range,log_volume,abs_return,volume")
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--method", choices=["granger", "toda_yamamoto", "both"], default="both")
    parser.add_argument("--group-by", default="all")
    parser.add_argument("--out", default="data/processed/causality")
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
    if args.s_col not in df.columns:
        print(f"[LỖI] Thiếu cột sentiment '{args.s_col}'.", file=sys.stderr)
        return 2

    group_col = None if args.group_by == "all" else args.group_by
    if group_col and group_col not in df.columns:
        print(f"[LỖI] Thiếu cột nhóm '{group_col}'.", file=sys.stderr)
        return 2

    # Aggregate basket nếu group='all'
    if group_col is None:
        agg_map = {args.s_col: "mean"}
        for t in targets:
            agg_map[t] = "mean" if t in ("avg_return", "intraday_range", "log_volume") else "sum"
        work = df.groupby("date").agg(agg_map).reset_index()
        work["abs_return"] = work["avg_return"].abs() if "avg_return" in work.columns else np.nan
        groups_iter = [("ALL", work)]
    else:
        work = df.copy()
        groups_iter = [(name, sub) for name, sub in work.groupby(group_col, dropna=False)]

    print(f"[04] Granger/TY | group={group_col or 'basket'} | max_lag={args.max_lag} | method={args.method}")

    granger_parts = []
    ty_parts = []
    for s_col in [args.s_col, f"abs_{args.s_col}"]:
        if s_col == f"abs_{args.s_col}":
            work[s_col] = work[args.s_col].abs()
        for target in targets:
            if target not in work.columns or s_col not in work.columns:
                continue
            if args.method in ("granger", "both"):
                res = bidirectional_causality(
                    work if group_col is None else work,
                    s_col, target, max_lag=args.max_lag,
                    method="granger", group_cols=[group_col] if group_col else None,
                )
                if not res.empty:
                    res["sentiment_col"] = s_col
                    res["target_col"] = target
                    granger_parts.append(res)
            if args.method in ("toda_yamamoto", "both"):
                res = bidirectional_causality(
                    work if group_col is None else work,
                    s_col, target, max_lag=args.max_lag,
                    method="toda_yamamoto", d=args.d,
                    group_cols=[group_col] if group_col else None,
                )
                if not res.empty:
                    res["sentiment_col"] = s_col
                    res["target_col"] = target
                    ty_parts.append(res)

    if granger_parts:
        granger_df = pd.concat(granger_parts, ignore_index=True)
    else:
        granger_df = pd.DataFrame()
    if ty_parts:
        ty_df = pd.concat(ty_parts, ignore_index=True)
    else:
        ty_df = pd.DataFrame()

    if not granger_df.empty:
        granger_df.to_csv(out_dir / "granger_results.csv", index=False, encoding="utf-8-sig")
    if not ty_df.empty:
        ty_df.to_csv(out_dir / "toda_yamamoto_results.csv", index=False, encoding="utf-8-sig")

    # Best-lag summary (chọn smallest p_value per direction)
    best_rows = []
    for dfx, name in [(granger_df, "granger"), (ty_df, "toda_yamamoto")]:
        if dfx.empty:
            continue
        for keys, grp in dfx.groupby(["sentiment_col", "target_col", "direction"] +
                                     ([group_col] if group_col else [])):
            if not isinstance(keys, tuple):
                keys = (keys,)
            best_row = grp.loc[grp["p_value"].idxmin()]
            row = dict(zip(["sentiment_col", "target_col", "direction"] +
                           ([group_col] if group_col else []), keys))
            row.update({
                "method": name,
                "best_lag": int(best_row["lag"]),
                "F": float(best_row["F_statistic"]),
                "p_value": float(best_row["p_value"]),
                "significant": bool(best_row["significant"]),
            })
            best_rows.append(row)
    best_df = pd.DataFrame(best_rows)
    if not best_df.empty:
        best_df.to_csv(out_dir / "best_lag_summary.csv", index=False, encoding="utf-8-sig")

    # Summary
    def _count_sig(dfx):
        if dfx.empty:
            return 0, 0
        # Đếm số direction (unique) có ít nhất 1 lag significant
        cols = ["sentiment_col", "target_col", "direction"] + ([group_col] if group_col else [])
        sig_dirs = dfx[dfx["significant"]].groupby(cols).ngroups
        total_dirs = dfx.groupby(cols).ngroups
        return sig_dirs, total_dirs

    g_sig, g_tot = _count_sig(granger_df)
    ty_sig, ty_tot = _count_sig(ty_df)

    summary = {
        "method": args.method,
        "max_lag": args.max_lag,
        "d_toda_yamamoto": args.d,
        "group": group_col or "basket",
        "granger_significant_directions": g_sig,
        "granger_total_directions": g_tot,
        "toda_yamamoto_significant_directions": ty_sig,
        "toda_yamamoto_total_directions": ty_tot,
        "interpretation": (
            "So sánh 2 hướng: nếu chỉ S→target significant (không có target→S), "
            "sentiment có vai trò tiên đoán. Nếu cả 2 hướng significant → feedback loop "
            "(sentiment vừa phản ứng vừa tiên đoán). Toda–Yamamoto là phiên bản robust "
            "với unit root — báo cáo ưu tiên nếu kết quả khác Granger."
        ),
    }
    with open(out_dir / "causality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Tóm tắt nhân quả ===")
    if not granger_df.empty:
        print(f"Granger       : {g_sig}/{g_tot} hướng significant")
    if not ty_df.empty:
        print(f"Toda-Yamamoto : {ty_sig}/{ty_tot} hướng significant")
    print(f"\nĐã ghi: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

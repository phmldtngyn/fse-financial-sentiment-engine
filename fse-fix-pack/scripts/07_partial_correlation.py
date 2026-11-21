# -*- coding: utf-8 -*-
"""
07_partial_correlation.py
=========================
Tính tương quan RIÊNG giữa sentiment và target khi kiểm soát các biến macro:
    vnindex_return, foreign_net_buy_log, vix, dxy_return, usd_vnd_return

Câu hỏi nghiên cứu:
- Sau khi partial out vnindex_return (hiệu ứng thị trường chung), sentiment có
  còn significant không? Nếu CÓ → sentiment mang thông tin GIA TĂNG (incremental
  informativeness), không chỉ đại diện cho thị trường chung.
- F-test so sánh R² của reduced (target ~ macro) vs full (target ~ sentiment + macro):
  nếu F significant → sentiment tăng R² đáng kể.

Input
-----
--input      : data/processed/banking_with_sentiment_vietstock.parquet
--macro      : data/processed/macro.parquet (do scripts/fetch_macro.py tạo,
               hoặc tự điền). Phải có cột 'date' và các biến trong config/macro_controls.yaml
--config     : config/macro_controls.yaml
--s-col      : 'Sentiment_NLP'
--targets    : avg_return,intraday_range,log_volume,abs_return,volume

Output (data/processed/partial_corr/)
- partial_corr_results.csv   (mỗi dòng: group, target, partial_corr, p_value,
                              sentiment_coef, sentiment_pvalue, r2_reduced, r2_full, delta_r2, F)
- incremental_r2.csv         (F-test cho delta R²)
- partial_corr_summary.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.partial_corr import partial_corr_table


def _apply_transform(series: pd.Series, transform: str) -> pd.Series:
    if transform == "level":
        return series
    if transform == "log":
        return np.log(series.abs().clip(lower=1e-9))
    if transform == "diff_1":
        return series.diff()
    if transform == "pct_1":
        return series.pct_change()
    return series


def load_macro(macro_path: Path, config: dict) -> pd.DataFrame:
    """Đọc data/processed/macro.parquet và áp dụng transform theo config."""
    if not macro_path.exists():
        print(f"[!] Không tìm thấy macro file: {macro_path}", file=sys.stderr)
        print("    Tạo file với các cột: date + các biến trong config/macro_controls.yaml.", file=sys.stderr)
        print("    Có thể skip bước này bằng --macro '' (chỉ chạy partial corr không controls).", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_parquet(macro_path) if macro_path.suffix == ".parquet" else pd.read_csv(macro_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    out = df[["date"]].copy()
    for v in config.get("variables", []):
        col = v.get("column", v["name"])
        if col not in df.columns:
            print(f"  [!] Macro thiếu cột '{col}' cho biến '{v['name']}'", file=sys.stderr)
            continue
        out[v["name"]] = _apply_transform(df[col], v.get("transform", "level"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Partial correlation kiểm soát macro")
    parser.add_argument("--input", default="data/processed/banking_with_sentiment_vietstock.parquet")
    parser.add_argument("--macro", default="data/processed/macro.parquet")
    parser.add_argument("--config", default="config/macro_controls.yaml")
    parser.add_argument("--s-col", default="Sentiment_NLP")
    parser.add_argument("--targets", default="avg_return,intraday_range,log_volume,abs_return,volume")
    parser.add_argument("--group-by", default="all")
    parser.add_argument("--out", default="data/processed/partial_corr")
    args = parser.parse_args()

    in_path = Path(args.input)
    cfg_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[LỖI] Không tìm thấy input: {in_path}", file=sys.stderr)
        return 2

    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    targets = [t.strip() for t in args.targets.split(",") if t.strip() in df.columns]
    if args.s_col not in df.columns:
        print(f"[LỖI] Thiếu cột sentiment '{args.s_col}'.", file=sys.stderr)
        return 2

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load macro
    macro_df = load_macro(Path(args.macro), config) if args.macro else pd.DataFrame()
    control_cols = config.get("partial_out", [])
    if macro_df.empty or not control_cols:
        print(f"[07] Không có macro → chạy partial corr không controls (= Pearson thường)")
        control_cols = []
    else:
        df = pd.merge(df, macro_df, on="date", how="left")
        missing = [c for c in control_cols if c not in df.columns]
        if missing:
            print(f"[!] Bỏ controls thiếu: {missing}")
            control_cols = [c for c in control_cols if c in df.columns]
        print(f"[07] Controls: {control_cols}")

    group_col = None if args.group_by == "all" else args.group_by
    if group_col and group_col not in df.columns:
        print(f"[LỖI] Thiếu cột nhóm '{group_col}'.", file=sys.stderr)
        return 2

    if group_col is None:
        agg_map = {args.s_col: "mean"}
        for t in targets:
            agg_map[t] = "mean" if t in ("avg_return", "intraday_range", "log_volume") else "sum"
        for c in control_cols:
            agg_map[c] = "mean"
        work = df.groupby("date").agg(agg_map).reset_index().sort_values("date")
        work["abs_return"] = work["avg_return"].abs() if "avg_return" in work.columns else np.nan
        groups_iter = [("ALL", work)]
        group_cols_for_fn = []
    else:
        work = df.copy()
        groups_iter = [(name, sub) for name, sub in work.groupby(group_col, dropna=False)]
        group_cols_for_fn = [group_col]

    print(f"[07] Partial corr | group={group_col or 'basket'} | n_targets={len(targets)} | controls={control_cols}")

    all_rows = []
    for s_col in [args.s_col, f"abs_{args.s_col}"]:
        if s_col == f"abs_{args.s_col}":
            work[s_col] = work[args.s_col].abs()
        for target in targets:
            if target not in work.columns:
                continue
            # Per-group partial corr
            parts = []
            for gname, sub in groups_iter:
                sub = sub.sort_values("date")
                ctrl_arrays = [sub[c].to_numpy() for c in control_cols] if control_cols else None
                try:
                    from src.analysis.partial_corr import (
                        partial_correlation, incremental_regression, incremental_r2_test,
                    )
                    pc = partial_correlation(sub[s_col].to_numpy(), sub[target].to_numpy(), ctrl_arrays)
                    ir = incremental_regression(sub[target].to_numpy(), sub[s_col].to_numpy(), ctrl_arrays)
                    rt = incremental_r2_test(sub[target].to_numpy(), sub[s_col].to_numpy(), ctrl_arrays)
                    row = {"group": gname, "sentiment": s_col, "target": target}
                    row.update({
                        "partial_corr": pc["partial_corr"],
                        "pc_pvalue": pc["p_value"],
                        "pc_significant": pc["significant"],
                        "sentiment_coef": ir["sentiment_coef"],
                        "sentiment_se": ir["sentiment_se"],
                        "sentiment_pvalue": ir["sentiment_pvalue"],
                        "sentiment_significant": ir["sentiment_significant"],
                        "r2_reduced": rt["r2_reduced"],
                        "r2_full": rt["r2_full"],
                        "delta_r2": rt["delta_r2"],
                        "F_test": rt["F"],
                        "F_pvalue": rt["p_value"],
                        "F_significant": rt["significant"],
                        "n": ir["n"],
                        "n_controls": ir["n_controls"],
                    })
                    parts.append(row)
                except Exception as e:
                    parts.append({"group": gname, "sentiment": s_col, "target": target,
                                  "error": str(e)[:150]})
            if parts:
                all_rows.extend(parts)

    if not all_rows:
        print("[LỖI] Không có kết quả nào được tạo.", file=sys.stderr)
        return 1
    res_df = pd.DataFrame(all_rows)
    res_df.to_csv(out_dir / "partial_corr_results.csv", index=False, encoding="utf-8-sig")

    # Incremental R² table (chỉ những dòng không error)
    inc_df = res_df[res_df.get("F_pvalue", pd.Series(dtype=float)).notna() &
                    (res_df.get("F_pvalue", pd.Series(dtype=float)) != "")].copy()
    if not inc_df.empty:
        inc_df = inc_df[["group", "sentiment", "target", "r2_reduced", "r2_full",
                         "delta_r2", "F_test", "F_pvalue", "F_significant"]]
        inc_df.to_csv(out_dir / "incremental_r2.csv", index=False, encoding="utf-8-sig")

    # Summary
    n_pairs = len(res_df)
    n_inc_sig = int(res_df.get("F_significant", pd.Series(dtype=bool)).fillna(False).sum()) if "F_significant" in res_df else 0
    n_sent_sig = int(res_df.get("sentiment_significant", pd.Series(dtype=bool)).fillna(False).sum()) if "sentiment_significant" in res_df else 0
    n_pc_sig = int(res_df.get("pc_significant", pd.Series(dtype=bool)).fillna(False).sum()) if "pc_significant" in res_df else 0
    summary = {
        "n_pairs_tested": n_pairs,
        "n_partial_corr_significant": n_pc_sig,
        "n_sentiment_coef_significant": n_sent_sig,
        "n_incremental_r2_significant": n_inc_sig,
        "controls_used": control_cols,
        "interpretation": (
            "Nếu sentiment_significant = True sau khi partial out macro → sentiment có "
            "sức giải thích GIA TĂNG. F_significant = True → sentiment tăng R² đáng kể. "
            "Đây là phép thử quyết định để chuyển từ 'correlated' sang 'incrementally informative'."
        ),
    }
    with open(out_dir / "partial_corr_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Tóm tắt partial correlation ===")
    print(f"Cặp test             : {summary['n_pairs_tested']}")
    print(f"Partial corr sig     : {summary['n_partial_corr_significant']}")
    print(f"Sentiment coef sig   : {summary['n_sentiment_coef_significant']}")
    print(f"Incremental R² sig   : {summary['n_incremental_r2_significant']}")
    print(f"Controls used        : {summary['controls_used']}")
    print(f"\nĐã ghi: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

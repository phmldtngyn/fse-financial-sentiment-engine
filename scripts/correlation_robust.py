"""
Hiệu chỉnh thống kê cho bảng kết quả tương quan hiện có — vỏ CLI cho
src/analysis/correlation_stats.py.

Input: parquet/csv kết quả từ correlation_analysis.py hiện có, mỗi dòng = 1 test
(cột: ticker, feature, corr, p_value). Output:
- q-value Benjamini–Hochberg cho từng test → quan hệ "đạt ý nghĩa" SAU khi
  kiểm soát false discovery (cột significance: raw vs fdr)
- Cảnh báo số quan hệ mất ý nghĩa sau FDR

Chạy từ gốc repo:
    python -m scripts.correlation_robust \
        --input data/processed/correlation/correlation_results.parquet

Ngoài CLI, dùng trực tiếp cho sâu hơn:
    from src.analysis.correlation_stats import out_of_sample_corr, lag_correlations
"""

from __future__ import annotations

import argparse
import json
import sys

# --- Windows an toàn: khi stdout/stderr là pipe/redirect, Python dùng codepage
# locale (vd cp1252) và sẽ crash khi in dấu tiếng Việt -> ép UTF-8 bắt buộc.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import pandas as pd

from src.analysis.correlation_stats import benjamini_hochberg


def main() -> int:
    parser = argparse.ArgumentParser(description="FDR hiệu chỉnh bảng tương quan")
    parser.add_argument("--input", required=True)
    parser.add_argument("--p-col", default="p_value")
    parser.add_argument("--corr-col", default="corr")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default=None,
                        help="Mặc định: cạnh file input, đuôi _fdr.parquet + _fdr.json")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"[LỖI] Không tìm thấy bảng kết quả: {path}", file=sys.stderr)
        return 2
    df = pd.read_parquet(path)
    if args.p_col not in df.columns:
        print(f"[LỖI] Thiếu cột p-value '{args.p_col}' (có: {list(df.columns)}). "
              f"Chỉnh --p-col cho khớp output của correlation_analysis.py.", file=sys.stderr)
        return 2

    p = pd.to_numeric(df[args.p_col], errors="coerce")
    valid = p.notna()
    df = df[valid].copy()
    p = p[valid].to_numpy()
    if len(p) == 0:
        print("[LỖI] Không có p-value hợp lệ nào.", file=sys.stderr)
        return 2

    df["q_value_bh"] = benjamini_hochberg(p)
    df["sig_raw"] = p <= args.alpha
    df["sig_fdr"] = df["q_value_bh"] <= args.alpha

    summary = {
        "n_tests": int(len(df)),
        "alpha": args.alpha,
        "significant_raw": int(df["sig_raw"].sum()),
        "significant_after_fdr": int(df["sig_fdr"].sum()),
        "lost_after_fdr": int((df["sig_raw"] & ~df["sig_fdr"]).sum()),
        "false_discovery_ratio_raw": round(
            1 - int(df["sig_fdr"].sum()) / max(1, int(df["sig_raw"].sum())), 3)
        if df["sig_raw"].sum() else None,
        "note": "lost_after_fdr = số quan hệ 'đạt ý nghĩa' trước đây chỉ do "
                "trùng hợp ngẫu nhiên khi test nhiều lần.",
    }

    out_base = Path(args.out) if args.out else path.with_name(path.stem + "_fdr")
    df.to_parquet(out_base.with_suffix(".parquet"), index=False)
    out_base.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")

    print(f"Số test              : {summary['n_tests']}")
    print(f"Đạt ý nghĩa (thô)    : {summary['significant_raw']}")
    print(f"Đạt ý nghĩa (sau FDR): {summary['significant_after_fdr']}")
    print(f"Bị loại sau FDR      : {summary['lost_after_fdr']}  ← đây là phần 'spurious'")
    print(f"→ {out_base.with_suffix('.parquet')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

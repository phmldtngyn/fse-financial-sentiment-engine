"""
Chia train/val/test THEO THỜI GIAN — chặn leakage tin cũ/tin mới.

Khác biệt sống còn so với chia ngẫu nhiên:
- Chia theo DANH SÁCH NGÀY DUY NHẤT (không phải theo số dòng): một ngày
  giao dịch tuyệt đối không xuất hiện ở 2 split cùng lúc → hết leakage ranh giới.
- Sort theo cột date → train: 80% ngày đầu, val: 10% giữa, test: 10% cuối.
- Ghi ra cả file regression (text, score, date) lẫn 3-class (text, label, score, date)
  với ngưỡng ±0.3 — giữ nguyên quy ước hiện có của repo.

Chạy từ gốc repo (sau khi có file nguồn CÓ CỘT date):
    python -m scripts.prepare_splits \
        --source data/raw/vn_quant_fulltext.parquet \
        --score-col Sentiment_Score --date-col date

Yêu cầu file nguồn có cột date (ngày đăng/bạn tin). Nếu chưa có → script dừng
với hướng dẫn, KHÔNG tự bịa ngày.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def split_by_dates(dates: pd.Series, fractions=(0.8, 0.1, 0.1)) -> dict[str, set]:
    """Chia danh sách ngày đã sort thành 3 tập KHÔNG GIAO NHAU theo tỷ lệ ngày."""
    unique = sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique())
    n = len(unique)
    if n < 30:
        raise ValueError(f"Chỉ có {n} ngày duy nhất — quá ít để chia theo thời gian.")
    cut1 = int(n * fractions[0])
    cut2 = int(n * (fractions[0] + fractions[1]))
    return {
        "train": set(unique[:cut1]),
        "val": set(unique[cut1:cut2]),
        "test": set(unique[cut2:]),
    }


def to_3class(score: pd.Series, threshold: float = 0.3) -> pd.Series:
    """Quy đổi score [-1,1] → nhãn 3 lớp theo ngưỡng ±0.3 (quy ước repo)."""
    s = pd.to_numeric(score, errors="coerce")
    return np.where(s < -threshold, 0, np.where(s > threshold, 2, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Chia train/val/test theo thời gian")
    parser.add_argument("--source", required=True, help="Parquet nguồn, PHẢI có cột date")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--min-text-chars", type=int, default=10)
    parser.add_argument("--fractions", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"[LỖI] Không tìm thấy file nguồn: {src}", file=sys.stderr)
        return 2
    df = pd.read_parquet(src)

    for col in (args.text_col, args.score_col, args.date_col):
        if col not in df.columns:
            print(f"[LỖI] File nguồn thiếu cột '{col}' (có: {list(df.columns)}).\n"
                  f"      Nếu chưa có cột ngày: hãy build date trước (từ bảng VN-Quant "
                  f"hoặc published_at của crawler) — KHÔNG chia ngẫu nhiên.", file=sys.stderr)
            return 2

    # lọc + chuẩn hóa
    df = df[df[args.text_col].astype(str).str.len() >= args.min_text_chars].copy()
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.dropna(subset=[args.score_col, args.date_col]).sort_values(args.date_col)
    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce").clip(-1, 1)
    df = df.dropna(subset=[args.score_col]).reset_index(drop=True)
    if abs(sum(args.fractions) - 1.0) > 1e-6:
        print("[LỖI] --fractions phải cộng đúng 1.0", file=sys.stderr)
        return 2

    splits = split_by_dates(df[args.date_col], tuple(args.fractions))

    # kiểm tra không giao nhau
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b and splits[a] & splits[b]:
                print(f"[LỖI] Ngày giao nhau giữa {a} và {b} — bug logic chia.", file=sys.stderr)
                return 3

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, days in splits.items():
        part = df[df[args.date_col].dt.normalize().isin(days)].copy()
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
        out3 = pd.DataFrame({
            args.text_col: part[args.text_col].astype(str),
            "label": to_3class(part[args.score_col], args.threshold),
            "score": part[args.score_col],
            args.date_col: part[args.date_col],
        })
        out3.to_parquet(out_dir / f"{name}_3class.parquet", index=False)

        dist = out3["label"].value_counts().sort_index().to_dict()
        print(f"{name:5s}: {len(part):5d} câu | {len(days):4d} ngày | "
              f"{part[args.date_col].min():%Y-%m-%d} → {part[args.date_col].max():%Y-%m-%d} | "
              f"phân bố {dist}")

    print("\n✅ Đã ghi 6 file vào", out_dir)
    print("⚠️  Lưu ý: kiểm tra cảnh báo split bằng verify_splits khi train "
          "(train_fse_sft tự chạy nếu date_col được cấu hình trong train_config.yaml).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

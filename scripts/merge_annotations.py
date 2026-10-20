"""Gộp 20 file claude_batch_XX.json + all_sampled.csv → gold_full.parquet."""
import json
from pathlib import Path
import pandas as pd

GOLD = Path("data/gold")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

def main():
    # 1. Đọc tất cả JSON Claude trả về
    all_labels = []
    json_files = sorted(GOLD.glob("claude_batch_*.json"))
    if not json_files:
        raise FileNotFoundError("Chưa có file claude_batch_*.json nào trong data/gold/")

    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        all_labels.extend(data)
        print(f"  {jf.name}: {len(data)} câu")

    df_labels = pd.DataFrame(all_labels)
    print(f"\nTổng nhãn Claude: {len(df_labels):,}")

    # 2. Đọc text gốc (all_sampled.csv có đủ id + text)
    df_text = pd.read_csv(GOLD / "all_sampled.csv")
    print(f"Tổng text gốc: {len(df_text):,}")

    # 3. Merge
    merged = df_text.merge(df_labels, on="id", how="left")
    n_missing = merged["score"].isna().sum()
    if n_missing > 0:
        print(f"⚠️  {n_missing} câu thiếu nhãn Claude (Claude bỏ sót)")

    # 4. Xuất Excel để spot-check
    review_path = GOLD / "for_review.xlsx"
    merged.to_excel(review_path, index=False)
    print(f"\n✅ Xuất file spot-check: {review_path}")
    print(f"   → Mở Excel, đọc 400 câu ngẫu nhiên (20%), sửa nếu sai")
    print(f"   → Lưu lại cùng tên, rồi chạy: python -m scripts.finalize_gold")

if __name__ == "__main__":
    main()
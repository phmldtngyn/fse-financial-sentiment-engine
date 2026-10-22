"""Verify 3-class dataset files."""
import pandas as pd
from pathlib import Path

print("=" * 60)
print("FILE SIZES")
print("=" * 60)
for f in Path("data/processed").glob("*_3class.parquet"):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:30s} {size_kb:8.1f} KB")

print("\n" + "=" * 60)
print("ROW COUNTS + LABEL DISTRIBUTION")
print("=" * 60)
for split in ["train", "val", "test"]:
    p = Path(f"data/processed/{split}_3class.parquet")
    if not p.exists():
        print(f"  ⚠️  {p.name} MISSING")
        continue
    df = pd.read_parquet(p)
    labels = df["label"].value_counts().sort_index().to_dict()
    print(f"  {split:6s}: {len(df):,} rows | labels={labels}")

print("\n" + "=" * 60)
print("SAMPLE (first row)")
print("=" * 60)
df = pd.read_parquet("data/processed/train_3class.parquet")
row = df.iloc[0]
print(f"  Label: {row['label']}")
print(f"  Text:  {row['text'][:150]}...")
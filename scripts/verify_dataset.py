"""Verify dataset: distribution, length, samples."""
import pandas as pd

df = pd.read_parquet("data/processed/train.parquet")

print(f"Rows: {len(df):,}")
print(f"Cols: {list(df.columns)}")
print()

# Distribution
neg = (df["score"] < -0.3).sum()
neu = df["score"].between(-0.3, 0.3).sum()
pos = (df["score"] > 0.3).sum()
total = len(df)

print("=== Score distribution ===")
print(f"Negative (<-0.3): {neg:,} ({neg/total*100:.1f}%)")
print(f"Neutral  (-0.3..0.3): {neu:,} ({neu/total*100:.1f}%)")
print(f"Positive (>0.3):  {pos:,} ({pos/total*100:.1f}%)")
print()

# Text length
lengths = df["text"].str.len()
print("=== Text length ===")
print(f"Min:  {lengths.min():,}")
print(f"Mean: {lengths.mean():.0f}")
print(f"Max:  {lengths.max():,}")
print()

# Tier distribution
print("=== Tier distribution ===")
print(df["tier"].value_counts().sort_index())
print()

# Samples
print("=== 3 random samples ===")
for _, r in df.sample(3, random_state=42).iterrows():
    text_preview = r["text"][:100].replace("\n", " ")
    print(f"  Score {r['score']:+.2f} | Tier {r['tier']} | {text_preview}...")
"""Chẩn đoán label gốc — tìm scale thật của Sentiment_Score."""
import sqlite3
from pathlib import Path
import pandas as pd

DB = Path("data/raw/vn_quant/master_quant_database.db")
conn = sqlite3.connect(DB)

# 1. Distribution gốc
print("=" * 60)
print("ORIGINAL Sentiment_Score distribution")
print("=" * 60)
df = pd.read_sql_query("SELECT Sentiment_Score FROM sentiment_LLM_labeled", conn)
print(df["Sentiment_Score"].describe().round(3))
print(f"\nValue counts (top 15):")
print(df["Sentiment_Score"].value_counts().head(15).sort_index())

# 2. Bucket distribution
print("\n" + "=" * 60)
print("Buckets of original score")
print("=" * 60)
buckets = pd.cut(df["Sentiment_Score"],
                 bins=[-0.01, 0, 0.25, 0.5, 0.75, 1.0],
                 labels=["= 0", "0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"])
print(buckets.value_counts().sort_index())

# 3. Sample với cả Title và Score gốc
print("\n" + "=" * 60)
print("10 random samples (Title + original Score)")
print("=" * 60)
samples = pd.read_sql_query(
    "SELECT Title, Sentiment_Score FROM sentiment_LLM_labeled ORDER BY RANDOM() LIMIT 10",
    conn,
)
for _, r in samples.iterrows():
    title = str(r["Title"])[:100].replace("\n", " ")
    print(f"  Score {r['Sentiment_Score']:+.3f} | {title}")

# 4. Đếm neutral thật (score = 0.5 hoặc gần đó)
print("\n" + "=" * 60)
print("Neutral check")
print("=" * 60)
for val in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75]:
    n = (df["Sentiment_Score"] == val).sum()
    print(f"  Score exactly {val}: {n:,}")

conn.close()
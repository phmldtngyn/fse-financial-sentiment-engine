"""Inspect VN-Quant database schema."""
import sqlite3
from pathlib import Path

DB = Path("data/raw/vn_quant/master_quant_database.db")

conn = sqlite3.connect(DB)
cur = conn.cursor()

print("=" * 60)
print("TABLES")
print("=" * 60)
tables = [row[0] for row in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
)]
for t in tables:
    n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t:40s} ({n:,} rows)")

# Inspect bảng chính
MAIN = "UNIFIED_RESEARCH_MATRIX"
if MAIN in tables:
    print("\n" + "=" * 60)
    print(f"SCHEMA {MAIN}")
    print("=" * 60)
    for row in cur.execute(f"PRAGMA table_info({MAIN})"):
        col_id, col_name, col_type, notnull, default, pk = row
        print(f"  {col_name:30s} {col_type:15s} notnull={notnull}")

    print("\n" + "=" * 60)
    print("SAMPLE ROWS (Sentiment_NLP not null)")
    print("=" * 60)
    try:
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({MAIN})")]
        # Chọn cột sample — ưu tiên cột text nếu có
        text_like = [c for c in cols if any(
            k in c.lower() for k in ["text", "title", "headline", "content", "news"]
        )]
        sample_cols = ["date", "ticker"] + text_like + ["Sentiment_NLP"]
        sample_cols = [c for c in sample_cols if c in cols]
        
        q = f"SELECT {', '.join(sample_cols)} FROM {MAIN} WHERE Sentiment_NLP IS NOT NULL LIMIT 5"
        for row in cur.execute(q):
            print()
            for col, val in zip(sample_cols, row):
                if isinstance(val, str) and len(val) > 80:
                    val = val[:80] + "..."
                print(f"  {col:20s} = {val}")
    except Exception as e:
        print(f"  Error: {e}")
else:
    print(f"\n⚠️  Table {MAIN} not found. Available: {tables}")

conn.close()
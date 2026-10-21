"""Inspect 2 bảng có khả năng chứa text+label trong VN-Quant."""
import sqlite3
from pathlib import Path

DB = Path("data/raw/vn_quant/master_quant_database.db")
conn = sqlite3.connect(DB)
cur = conn.cursor()

for table in ["sentiment_LLM_labeled", "google_news_10years_FULLTEXT"]:
    print("\n" + "=" * 70)
    print(f"TABLE: {table}")
    print("=" * 70)

    # Schema
    cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
    print("\nSchema:")
    for row in cols:
        print(f"  {row[1]:30s} {row[2]}")

    # Count
    n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"\nRows: {n:,}")

    # Sample 3 dòng — truncate text nếu dài
    print("\nSample 3 dòng:")
    col_names = [c[1] for c in cols]
    q = f"SELECT * FROM {table} LIMIT 3"
    for i, row in enumerate(cur.execute(q), 1):
        print(f"\n  --- Row {i} ---")
        for name, val in zip(col_names, row):
            if isinstance(val, str) and len(val) > 150:
                val = val[:150] + "..."
            elif val is None:
                val = "(null)"
            print(f"    {name:25s} = {val}")

conn.close()
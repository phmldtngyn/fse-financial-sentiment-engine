"""Check Sentiment_NLP coverage trong VN-Quant."""
import sqlite3
from pathlib import Path
import pandas as pd

DB = Path("data/raw/vn_quant/master_quant_database.db")
conn = sqlite3.connect(DB)

print("=" * 60)
print("Sentiment_NLP coverage")
print("=" * 60)

# Tổng quát
total = conn.execute("SELECT COUNT(*) FROM UNIFIED_RESEARCH_MATRIX").fetchone()[0]
notnull = conn.execute("SELECT COUNT(*) FROM UNIFIED_RESEARCH_MATRIX WHERE Sentiment_NLP IS NOT NULL").fetchone()[0]
print(f"Total rows:     {total:,}")
print(f"Non-null NLP:   {notnull:,} ({notnull/total*100:.1f}%)")

# Distinct dates
dates = conn.execute(
    "SELECT COUNT(DISTINCT date) FROM UNIFIED_RESEARCH_MATRIX WHERE Sentiment_NLP IS NOT NULL"
).fetchone()[0]
print(f"Distinct dates: {dates:,}")

# Date range
rng = conn.execute(
    "SELECT MIN(date), MAX(date) FROM UNIFIED_RESEARCH_MATRIX WHERE Sentiment_NLP IS NOT NULL"
).fetchone()
print(f"Date range:     {rng[0]} → {rng[1]}")

# Daily aggregate (mean Sentiment_NLP per date)
daily = pd.read_sql_query(
    """SELECT date,
              AVG(Sentiment_NLP) as S_t,
              COUNT(*) as n_obs
       FROM UNIFIED_RESEARCH_MATRIX
       WHERE Sentiment_NLP IS NOT NULL
       GROUP BY date
       ORDER BY date""",
    conn,
)
conn.close()

print(f"\nAggregated: {len(daily)} days")
print(f"\nS_t distribution (avg by date):")
print(daily["S_t"].describe().round(3))

print(f"\nLatest 5 days:")
print(daily.tail(5).to_string(index=False))

# Save
out = Path("data/processed/vnquant_sentiment_daily.parquet")
daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
daily.to_parquet(out)
print(f"\n✅ Saved → {out}")
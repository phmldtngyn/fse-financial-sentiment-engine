# check_db.py -- chay: python check_db.py
import sqlite3

DB = r"data/raw/vn_quant/master_quant_database.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

print("=" * 70)
print("1) TOAN BO BANG TRONG DB")
print("=" * 70)
tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
for name in tables:
    n = cur.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
    print(f"{name:45s} {n:>10,} rows")

print()
print("=" * 70)
print("2) SCHEMA UNIFIED_RESEARCH_MATRIX")
print("=" * 70)
if "UNIFIED_RESEARCH_MATRIX" in tables:
    for row in cur.execute("PRAGMA table_info(UNIFIED_RESEARCH_MATRIX)"):
        print(f"  {row[1]:25s} {row[2]}")
    print()
    print("Mau gia VCB 5 ngay gan nhat:")
    for row in cur.execute(
        "SELECT ticker, date, open, high, low, close, volume, pct "
        "FROM UNIFIED_RESEARCH_MATRIX WHERE ticker='VCB' "
        "ORDER BY date DESC LIMIT 5"):
        print(" ", row)
else:
    print("Khong thay UNIFIED_RESEARCH_MATRIX trong DB!")

conn.close()
"""Extract daily aggregate cho nhóm cổ phiếu ngân hàng từ VN-Quant."""
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

DB = Path("data/raw/vn_quant/master_quant_database.db")
OUT = Path("data/processed/banking_daily.parquet")

BANKING_TICKERS = [
    "VCB", "BID", "CTG", "TCB", "VPB", "MBB", "ACB", "STB", "SHB", "HDB",
    "LPB", "TPB", "VIB", "MSB", "OCB", "SSB", "EIB", "NAB", "BVB", "SGB",
    "VBB", "KLB", "PGB", "ABB", "BAB", "PVCB",
]


def main():
    print("=" * 60)
    print("Extract banking daily data")
    print("=" * 60)
    print(f"Tickers: {len(BANKING_TICKERS)} mã")

    conn = sqlite3.connect(DB)
    placeholders = ",".join("?" * len(BANKING_TICKERS))
    query = f"""
        SELECT ticker, date, open, high, low, close, volume, pct
        FROM UNIFIED_RESEARCH_MATRIX
        WHERE ticker IN ({placeholders})
          AND volume IS NOT NULL
          AND volume > 0
          AND close IS NOT NULL
        ORDER BY date, ticker
    """
    df = pd.read_sql_query(query, conn, params=BANKING_TICKERS)
    conn.close()

    print(f"\nLoaded {len(df):,} rows from {len(df['ticker'].unique())} tickers")

    if len(df) == 0:
        raise ValueError("Không có data ngân hàng.")

    found = sorted(df["ticker"].unique().tolist())
    missing = set(BANKING_TICKERS) - set(found)
    print(f"Found: {len(found)}/{len(BANKING_TICKERS)}")
    if missing:
        print(f"⚠️  Missing: {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["intraday_range"] = (df["high"] - df["low"]) / df["close"]

    agg = df.groupby("date").agg(
        total_volume=("volume", "sum"),
        avg_return=("pct", "mean"),
        avg_range=("intraday_range", "mean"),
        avg_close=("close", "mean"),
        n_banks=("ticker", "nunique"),
    ).reset_index()

    agg = agg.rename(columns={"date": "trade_date"})
    agg = agg[agg["n_banks"] >= 5].reset_index(drop=True)

    print(f"\nDate range: {agg['trade_date'].min()} → {agg['trade_date'].max()}")
    print(f"\nStats:")
    print(agg[["total_volume", "avg_return", "avg_range", "n_banks"]].describe().round(4).to_string())
    print(f"\nLatest 5 days:")
    print(agg.tail(5).to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT)
    print(f"\n✅ Saved → {OUT}")


if __name__ == "__main__":
    main()
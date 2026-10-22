"""Build forecast dataset từ Sentiment_NLP (VN-Quant) + market aggregate.

Dùng Sentiment_NLP proxy cho S_t (vì S_t FSE chỉ có 3 ngày).
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

DB = Path("data/raw/vn_quant/master_quant_database.db")
OUT = Path("data/processed/market_with_sentiment.parquet")


def load_all():
    conn = sqlite3.connect(DB)

    # S_t proxy: avg(Sentiment_NLP) per date
    st = pd.read_sql_query(
        """SELECT date, AVG(Sentiment_NLP) as S_t, COUNT(*) as n_obs
           FROM UNIFIED_RESEARCH_MATRIX
           WHERE Sentiment_NLP IS NOT NULL
           GROUP BY date ORDER BY date""",
        conn,
    )

    # Market aggregate: sum(volume), avg(pct)
    market = pd.read_sql_query(
        """SELECT date,
                  SUM(volume) as vol,
                  AVG(pct) as return,
                  COUNT(DISTINCT ticker) as n_tickers
           FROM UNIFIED_RESEARCH_MATRIX
           WHERE volume IS NOT NULL AND pct IS NOT NULL
           GROUP BY date ORDER BY date""",
        conn,
    )

    conn.close()
    return st, market


def main():
    print("=" * 60)
    print("Build forecast dataset (Sentiment_NLP proxy)")
    print("=" * 60)

    st, market = load_all()
    print(f"S_t (NLP): {len(st)} days")
    print(f"Market:    {len(market)} days")

    # Normalize date
    for df in [st, market]:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Join
    df = st[["date", "S_t"]].merge(
        market[["date", "vol", "return", "n_tickers"]],
        on="date", how="inner",
    )
    print(f"Joined: {len(df)} days")

    if len(df) < 100:
        raise ValueError(f"Chỉ {len(df)} ngày — không đủ train. Kiểm tra lại DB.")

    # Features
    df["delta_S_t"] = df["S_t"].diff().fillna(0)
    df["vol_lag"] = df["vol"].shift(1)
    df["return_lag"] = df["return"].shift(1)

    # Dummy v_ctx (sẽ thay bằng RAG thật sau)
    np.random.seed(42)
    for i in range(8):
        df[f"v_ctx_{i}"] = np.random.randn(len(df)) * 0.1

    df = df.dropna(subset=["vol_lag", "return_lag"]).reset_index(drop=True)

    print(f"\nAfter feature build: {len(df)} days")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"\nLatest 5 days:")
    print(df[["date", "S_t", "vol", "return"]].tail(5).to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(f"\n✅ Saved → {OUT}")


if __name__ == "__main__":
    main()
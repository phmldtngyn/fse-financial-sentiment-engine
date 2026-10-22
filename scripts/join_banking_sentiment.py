"""Join Sentiment_NLP + banking_daily → dataset cho correlation analysis."""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

DB = Path("data/raw/vn_quant/master_quant_database.db")
BANKING_PATH = Path("data/processed/banking_daily.parquet")
OUT = Path("data/processed/banking_with_sentiment.parquet")


def load_sentiment_daily():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        """SELECT date,
                  AVG(Sentiment_NLP) as S_t,
                  COUNT(*) as n_obs
           FROM UNIFIED_RESEARCH_MATRIX
           WHERE Sentiment_NLP IS NOT NULL
           GROUP BY date ORDER BY date""",
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.rename(columns={"date": "trade_date"})


def main():
    print("=" * 60)
    print("Join S_t + banking daily")
    print("=" * 60)

    if not BANKING_PATH.exists():
        raise FileNotFoundError(f"Thiếu {BANKING_PATH}. Chạy extract_banking_daily.py trước.")

    banking = pd.read_parquet(BANKING_PATH)
    print(f"Banking: {len(banking)} days ({banking['trade_date'].min()} → {banking['trade_date'].max()})")

    sentiment = load_sentiment_daily()
    print(f"Sentiment: {len(sentiment)} days")

    df = sentiment[["trade_date", "S_t"]].merge(
        banking[["trade_date", "total_volume", "avg_return", "avg_range", "n_banks"]],
        on="trade_date", how="inner",
    )
    print(f"Joined: {len(df)} days")

    if len(df) < 100:
        raise ValueError(f"Chỉ {len(df)} ngày — không đủ.")

    df["abs_S_t"] = df["S_t"].abs()
    df["log_volume"] = np.log(df["total_volume"].clip(lower=1))
    df["abs_return"] = df["avg_return"].abs()
    df = df.sort_values("trade_date").reset_index(drop=True)

    print(f"\nDate range: {df['trade_date'].min()} → {df['trade_date'].max()}")
    print(f"S_t non-zero: {(df['S_t'] != 0).sum()} / {len(df)} ({(df['S_t'] != 0).mean()*100:.1f}%)")

    print(f"\nStats:")
    print(df[["S_t", "total_volume", "avg_return", "avg_range"]].describe().round(4).to_string())

    print(f"\nLatest 5 days:")
    print(df.tail(5)[["trade_date", "S_t", "total_volume", "avg_return", "avg_range"]].to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(f"\n✅ Saved → {OUT}")


if __name__ == "__main__":
    main()
"""Join sentiment_LLM_labeled + google_news_10years_FULLTEXT → SFT dataset.

Bảng 1: Title + Sentiment_Score (0-1)
Bảng 2: Title + Content (full text)
→ Join qua Title → dataset có Content + Score

Normalize score: [0, 1] → [-1, 1] qua công thức 2*s - 1.

Usage:
    python -m scripts.prepare_vn_quant_dataset                # chỉ report
    python -m scripts.prepare_vn_quant_dataset --write        # ghi parquet
"""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DB = Path("data/raw/vn_quant/master_quant_database.db")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tables():
    """Load 2 bảng qua pandas read_sql."""
    conn = sqlite3.connect(DB)

    sentiment = pd.read_sql_query(
        "SELECT Date, Title, Sentiment_Score, Magnitude, Reason FROM sentiment_LLM_labeled",
        conn,
    )
    news = pd.read_sql_query(
        "SELECT Date, Title, Content FROM google_news_10years_FULLTEXT",
        conn,
    )
    conn.close()

    print(f"sentiment_LLM_labeled: {len(sentiment):,} rows")
    print(f"google_news_FULLTEXT:  {len(news):,} rows")
    return sentiment, news


def join_by_title(sentiment: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Join qua Title (exact match)."""
    # Chuẩn hóa Title để tăng match rate
    for df in [sentiment, news]:
        df["title_key"] = df["Title"].astype(str).str.strip().str.lower()

    merged = sentiment.merge(
        news[["title_key", "Content"]],
        on="title_key",
        how="inner",
    )
    print(f"\nJoin by exact title: {len(merged):,} rows")

    # Fallback: nếu ít, thử fuzzy join qua 100 ký tự đầu
    if len(merged) < len(sentiment) * 0.5:
        print("⚠️  Match rate thấp, thử join qua title prefix (100 ký tự)...")
        for df in [sentiment, news]:
            df["title_prefix"] = df["Title"].astype(str).str.strip().str.lower().str[:100]

        merged2 = sentiment.merge(
            news[["title_prefix", "Content"]],
            on="title_prefix",
            how="inner",
        )
        print(f"Join by title prefix: {len(merged2):,} rows")

        if len(merged2) > len(merged):
            merged = merged2

    return merged.drop_duplicates(subset=["title_key"] if "title_key" in merged.columns else ["title_prefix"])


def normalize_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa score, lọc câu lỗi."""
    # Chuẩn hóa score: [0, 1] → [-1, 1]
    df["score"] = (df["Sentiment_Score"].astype(float) * 2 - 1).clip(-1.0, 1.0)

    # Lọc câu quá ngắn hoặc Content null
    before = len(df)
    df = df.dropna(subset=["Content", "score"])
    df = df[df["Content"].astype(str).str.len() >= 50]
    df = df[df["score"].between(-1.0, 1.0)]
    print(f"\nAfter cleaning: {len(df):,} / {before:,} rows")

    # Tier: default 3 (tin tài chính chứng khoán → liên quan trực tiếp)
    # Có thể refine sau: nếu Title/Content chứa mã cổ phiếu → tier=3, còn không → tier=2
    df["tier"] = 3

    return df[["Title", "Content", "score", "tier"]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Ghi parquet files (mặc định chỉ report)")
    args = parser.parse_args()

    # 1. Load
    sentiment, news = load_tables()

    # 2. Join
    merged = join_by_title(sentiment, news)

    if len(merged) == 0:
        print("\n❌ Không join được row nào. Kiểm tra Title/Date trong 2 bảng.")
        return

    # 3. Clean
    df = normalize_and_clean(merged)

    # 4. Report
    print("\n" + "=" * 60)
    print("DATASET STATS")
    print("=" * 60)
    print(f"Total samples: {len(df):,}")
    print(f"\nScore distribution (sau normalize [-1,1]):")
    print(df["score"].describe().round(3))
    print(f"\nScore buckets:")
    print(f"  Rất tiêu cực [-1.0, -0.5]: {(df['score'] <= -0.5).sum():,}")
    print(f"  Tiêu cực     (-0.5, 0.0): {((df['score'] > -0.5) & (df['score'] < 0)).sum():,}")
    print(f"  Trung tính   [ 0.0, 0.0]: {(df['score'] == 0).sum():,}")
    print(f"  Tích cực     ( 0.0, 0.5): {((df['score'] > 0) & (df['score'] < 0.5)).sum():,}")
    print(f"  Rất tích cực [ 0.5, 1.0]: {(df['score'] >= 0.5).sum():,}")

    print(f"\nContent length (chars):")
    print(f"  Mean: {df['Content'].str.len().mean():.0f}")
    print(f"  Max:  {df['Content'].str.len().max():,}")

    if args.write:
        # 5. Split 80/10/10
        train, rest = train_test_split(df, test_size=0.2, random_state=42)
        val, test = train_test_split(rest, test_size=0.5, random_state=42)

        train.to_parquet(OUT_DIR / "train.parquet")
        val.to_parquet(OUT_DIR / "val.parquet")
        test.to_parquet(OUT_DIR / "test.parquet")

        print(f"\n✅ Saved:")
        print(f"   train={len(train):,} val={len(val):,} test={len(test):,}")
        print(f"   → {OUT_DIR}/")
    else:
        print(f"\n💡 Chạy lại với --write để ghi parquet files.")


if __name__ == "__main__":
    main()
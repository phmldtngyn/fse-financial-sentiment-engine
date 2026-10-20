"""Đọc for_review.xlsx đã spot-check → chia train/val/test → commit."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

GOLD = Path("data/gold")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_excel(GOLD / "for_review.xlsx")

    # Lọc câu có nhãn đầy đủ
    df = df.dropna(subset=["score", "tier"])
    df["score"] = df["score"].astype(float).clip(-1.0, 1.0)
    df["tier"] = df["tier"].astype(int).clip(1, 3)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 30]

    print(f"Gold standard: {len(df):,} câu")
    print(f"\nPhân bố score:\n{df['score'].value_counts().sort_index()}")
    print(f"\nPhân bố tier:\n{df['tier'].value_counts().sort_index()}")

    # Lưu full (để commit, làm reference cố định)
    df[["text", "score", "tier"]].to_parquet(PROC / "gold_full.parquet")
    print(f"\n✅ gold_full.parquet (commit file này lên Git)")

    # Chia 80/10/10
    train, rest = train_test_split(df[["text", "score", "tier"]],
                                    test_size=0.2, random_state=42)
    val, test = train_test_split(rest, test_size=0.5, random_state=42)

    train.to_parquet(PROC / "train.parquet")
    val.to_parquet(PROC / "val.parquet")
    test.to_parquet(PROC / "test.parquet")

    print(f"\n✅ train={len(train):,} | val={len(val):,} | test={len(test):,}")

if __name__ == "__main__":
    main()
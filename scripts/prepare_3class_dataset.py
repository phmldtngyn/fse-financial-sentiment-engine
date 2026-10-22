"""Convert regression dataset → 3-class classification.

Classes:
  0 = negative (score < -0.3)
  1 = neutral  (-0.3 ≤ score ≤ 0.3)
  2 = positive (score > 0.3)

Giữ cột `score` gốc để backup + eval MAE nếu cần.
"""
import pandas as pd
from pathlib import Path
from collections import Counter

THRESHOLD = 0.3

def score_to_class(score):
    if score < -THRESHOLD:
        return 0
    elif score > THRESHOLD:
        return 2
    return 1


def main():
    for split in ["train", "val", "test"]:
        src = Path(f"data/processed/{split}.parquet")
        if not src.exists():
            print(f"⚠️  {src} missing, skip")
            continue

        df = pd.read_parquet(src)
        df["label"] = df["score"].apply(score_to_class)

        # Reorder columns: text, label, score
        df = df[["text", "label", "score"]]
        dst = Path(f"data/processed/{split}_3class.parquet")
        df.to_parquet(dst)

        counts = Counter(df["label"])
        total = len(df)
        print(f"\n✅ {split}: {total:,} rows → {dst.name}")
        for cls, name in [(0, "negative"), (1, "neutral"), (2, "positive")]:
            n = counts.get(cls, 0)
            print(f"   {name:9s}: {n:5,} ({n/total*100:5.1f}%)")

        # Class weights cho train set
        if split == "train":
            n_classes = 3
            weights = [total / (counts.get(c, 1) * n_classes) for c in range(n_classes)]
            print(f"\n   Class weights (balanced): {[f'{w:.3f}' for w in weights]}")
            print(f"   → Dùng weights này trong loss")


if __name__ == "__main__":
    main()
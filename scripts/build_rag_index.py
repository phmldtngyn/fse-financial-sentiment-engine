"""Build RAG rolling index từ ViFiD 200k (hoặc corpus khác).

Test không cần Claude, không cần gold standard.

Usage:
    python -m scripts.build_rag_index              # build 10k câu (test nhanh)
    python -m scripts.build_rag_index --n 50000    # build 50k câu
"""
import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.rag.embedder import Embedder
from src.rag.indexer import RollingIndex

VIFID = Path(r"C:\Users\ADMIN\.cache\kagglehub\datasets\daddychillonkaggle\vifid-segraw-200k\versions\3\ViFiD-200K.csv")
CONFIG = Path("config/rag.yaml")


def load_corpus(n_samples: int) -> pd.DataFrame:
    """Load ViFiD, sample n câu, gán published_at giả trong 30 ngày qua."""
    df = pd.read_csv(VIFID)[["label"]].rename(columns={"label": "text"})
    df = df[df["text"].astype(str).str.len().between(40, 800)]
    df = df.drop_duplicates("text").reset_index(drop=True)
    print(f"Corpus available: {len(df):,}")

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    print(f"Using: {len(df):,} câu")

    # Gán published_at phân bố đều trong 30 ngày qua
    now = datetime.now(timezone.utc)
    random.seed(42)
    df["published_at"] = [
        now - timedelta(days=random.uniform(0, 30))
        for _ in range(len(df))
    ]
    df["category"] = "news"  # placeholder
    df["id"] = range(1, len(df) + 1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000, help="Số câu build (mặc định 10k)")
    args = parser.parse_args()

    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Loading corpus (n={args.n})...")
    df = load_corpus(args.n)

    print(f"\nLoading embedder: {cfg['embedder']['model_name']}")
    embedder = Embedder(
        model_name=cfg["embedder"]["model_name"],
        max_seq_length=cfg["embedder"]["max_seq_length"],
    )
    print(f"  dim = {embedder.dim}")

    print(f"\nEncoding {len(df):,} câu...")
    vectors = embedder.encode(
        df["text"].tolist(),
        batch_size=cfg["embedder"]["batch_size"],
    )

    print(f"\nBuilding FAISS index...")
    index = RollingIndex(dim=embedder.dim, window_days=cfg["index"]["window_days"])
    metadata = [
        {
            "text": r["text"],
            "published_at": r["published_at"],
            "category": r["category"],
        }
        for _, r in df.iterrows()
    ]
    index.add(df["id"].tolist(), vectors, metadata)
    print(f"  Index size: {len(index):,}")

    out_dir = Path(cfg["index"]["dir"])
    index.save(out_dir)
    print(f"\n✅ Saved index → {out_dir}/")
    print(f"   index.faiss, metadata.json, version.txt")


if __name__ == "__main__":
    main()
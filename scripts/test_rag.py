"""Test RAG retriever. Usage: python -m scripts.test_rag"""
import yaml
from pathlib import Path

from src.rag.embedder import Embedder
from src.rag.indexer import RollingIndex
from src.rag.retriever import Retriever


def main():
    with open("config/rag.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("Loading index...")
    index = RollingIndex.load(
        cfg["index"]["dir"],
        dim=cfg["embedder"]["dim"],
        window_days=cfg["index"]["window_days"],
    )
    print(f"  {len(index):,} vectors")

    print("Loading embedder...")
    embedder = Embedder(
        model_name=cfg["embedder"]["model_name"],
        max_seq_length=cfg["embedder"]["max_seq_length"],
    )

    retriever = Retriever(
        index=index,
        embedder=embedder,
        alpha=cfg["retriever"]["alpha"],
        lambda_decay=cfg["retriever"]["lambda_decay"],
    )

    queries = [
        "cổ phiếu ngân hàng tăng trần",
        "thị trường chứng khoán lao dốc",
        "lãi suất điều hành giảm",
        "khối ngoại mua ròng",
    ]

    for q in queries:
        print(f"\n{'='*70}")
        print(f"Query: {q}")
        print('='*70)
        results = retriever.query(
            q,
            k=cfg["retriever"]["top_k"],
            top_m=cfg["retriever"]["top_m"],
            window_hours=cfg["retriever"]["window_hours"],
        )
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r['score']:.3f} (sim={r['sim']:.3f}, "
                  f"age={r['age_hours']:.0f}h)")
            print(f"    {r['text'][:100]}...")


if __name__ == "__main__":
    main()
"""FAISS rolling index — thêm, xóa theo thời gian, lưu/load disk."""

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import faiss
import numpy as np


class RollingIndex:
    """FAISS IndexFlatIP (cosine similarity trên vector normalized).

    Metadata (text, published_at, category) lưu riêng dạng dict.
    """

    def __init__(self, dim=768, window_days=30):
        self.dim = dim
        self.window_days = window_days
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.metadata = {}  # id -> {"text", "published_at", "category"}

    def add(self, ids, vectors, metadata_list):
        """Thêm batch vectors + metadata."""
        if len(ids) != len(vectors) or len(ids) != len(metadata_list):
            raise ValueError("ids, vectors, metadata_list phải cùng độ dài")
        ids_np = np.array(ids, dtype="int64")
        vectors = vectors.astype("float32")
        self.index.add_with_ids(vectors, ids_np)
        for i, meta in zip(ids, metadata_list):
            self.metadata[int(i)] = meta

    def remove_older_than(self, cutoff: datetime):
        """Xóa các vector có published_at < cutoff. Trả về số đã xóa."""
        to_remove = [
            i for i, m in self.metadata.items()
            if m["published_at"] < cutoff
        ]
        if not to_remove:
            return 0
        self.index.remove_ids(np.array(to_remove, dtype="int64"))
        for i in to_remove:
            del self.metadata[i]
        return len(to_remove)

    def search(self, query_vectors, k=20):
        """Trả về (scores, ids) shape (N, k)."""
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        scores, ids = self.index.search(query_vectors.astype("float32"), k)
        return scores, ids

    def save(self, dir_path):
        """Lưu index + metadata ra disk."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(dir_path / "index.faiss"))

        meta_serializable = {
            str(k): {
                "text": v["text"],
                "published_at": v["published_at"].isoformat(),
                "category": v.get("category", "unknown"),
            }
            for k, v in self.metadata.items()
        }
        with open(dir_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_serializable, f, ensure_ascii=False)

        (dir_path / "version.txt").write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8"
        )

    @classmethod
    def load(cls, dir_path, dim=768, window_days=30):
        """Load index từ disk."""
        dir_path = Path(dir_path)
        obj = cls(dim=dim, window_days=window_days)
        obj.index = faiss.read_index(str(dir_path / "index.faiss"))

        with open(dir_path / "metadata.json", encoding="utf-8") as f:
            raw = json.load(f)
        obj.metadata = {
            int(k): {
                "text": v["text"],
                "published_at": datetime.fromisoformat(v["published_at"]),
                "category": v.get("category", "unknown"),
            }
            for k, v in raw.items()
        }
        return obj

    def __len__(self):
        return self.index.ntotal
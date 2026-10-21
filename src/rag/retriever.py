"""Retriever — query FAISS top-K + time-weighted rerank."""

from datetime import datetime, timedelta, timezone

import numpy as np


class Retriever:
    """Time-weighted retrieval từ RollingIndex.

    Score cuối = alpha * cosine_sim + (1 - alpha) * exp(-lambda * age_hours)
    """

    def __init__(self, index, embedder, alpha=0.6, lambda_decay=0.05):
        self.index = index
        self.embedder = embedder
        self.alpha = alpha
        self.lambda_decay = lambda_decay

    def query(self, text, k=20, top_m=5, window_hours=168,
              now: datetime = None):
        """Query → list of {id, text, published_at, score, sim, age_hours}."""
        now = now or datetime.now(timezone.utc)
        qvec = self.embedder.encode(text)
        scores, ids = self.index.search(qvec, k)
        scores = scores[0]
        ids = ids[0]

        cutoff = now - timedelta(hours=window_hours)
        results = []
        for sim, idx in zip(scores, ids):
            if idx == -1:
                continue
            meta = self.index.metadata.get(int(idx))
            if meta is None:
                continue

            pub = meta["published_at"]
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)

            if pub < cutoff:
                continue

            age_hours = (now - pub).total_seconds() / 3600
            recency = np.exp(-self.lambda_decay * age_hours)
            final = self.alpha * float(sim) + (1 - self.alpha) * float(recency)

            results.append({
                "id": int(idx),
                "text": meta["text"],
                "published_at": pub.isoformat(),
                "category": meta.get("category", "unknown"),
                "sim": round(float(sim), 4),
                "age_hours": round(age_hours, 1),
                "recency": round(float(recency), 4),
                "score": round(float(final), 4),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_m]

    def aggregate(self, text, dim=768, **kwargs):
        """Trả về v_context = mean của top-M vectors (đã rerank)."""
        results = self.query(text, **kwargs)
        if not results:
            return np.zeros(dim, dtype="float32")

        vecs = []
        for r in results:
            v = np.zeros(dim, dtype="float32")
            # Lấy lại vector từ FAISS: reconstruct
            idx = np.array([r["id"]], dtype="int64")
            try:
                v = self.index.index.reconstruct(int(r["id"]))
            except Exception:
                continue
            vecs.append(v)

        if not vecs:
            return np.zeros(dim, dtype="float32")

        return np.mean(np.stack(vecs, axis=0), axis=0).astype("float32")
"""Embedder cho RAG — dùng SentenceTransformer với model Vietnamese SBERT."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Encode text → vector normalized (L2) cho FAISS IndexFlatIP (cosine)."""

    def __init__(self, model_name="keepitreal/vietnamese-sbert", device=None,
                 max_seq_length=256):
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=32, normalize=True):
        """Encode 1 hoặc nhiều câu → np.ndarray (N, dim) float32."""
        if isinstance(texts, str):
            texts = [texts]
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 200,
        )
        return emb.astype("float32")
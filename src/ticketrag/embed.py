"""Thin wrapper over sentence-transformers so the model is swappable (ablations) and the
query/document asymmetry (BGE query instruction) lives in one place."""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": _BGE_QUERY_PREFIX,
    "BAAI/bge-base-en-v1.5": _BGE_QUERY_PREFIX,
    "BAAI/bge-large-en-v1.5": _BGE_QUERY_PREFIX,
    # bge-m3 and most other models need no prefix
}


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.query_prefix = QUERY_PREFIX.get(model_name, "")

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # cosine == inner product
            show_progress_bar=len(texts) > 200,
        )
        return np.ascontiguousarray(vecs, dtype="float32")

    def encode_docs(self, texts, batch_size: int = 64) -> np.ndarray:
        return self._encode(list(texts), batch_size)

    def encode_queries(self, texts, batch_size: int = 64) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts], batch_size)

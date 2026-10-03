"""Vector store: exact (flat) inner-product FAISS index with stable int64 ids.

Kept behind this small interface so it can be swapped for pgvector / Qdrant / Azure AI Search
in production without touching retrieval code. Upsert and delete support incremental ingestion
of new tickets and new ticket classes without a full rebuild."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

_INDEX_FILE, _META_FILE = "vectors.faiss", "meta.json"


class VectorStore:
    def __init__(self, dim: int, meta: dict | None = None):
        self.dim = dim
        self.meta = meta or {}
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def __len__(self) -> int:
        return int(self.index.ntotal)

    def remove(self, ids: Sequence[int]) -> None:
        self.index.remove_ids(np.asarray(ids, dtype="int64"))

    def upsert(self, ids: Sequence[int], vecs: np.ndarray) -> None:
        ids_arr = np.asarray(ids, dtype="int64")
        if vecs.shape != (len(ids_arr), self.dim):
            raise ValueError(f"expected vecs {(len(ids_arr), self.dim)}, got {vecs.shape}")
        self.index.remove_ids(ids_arr)  # no-op for unseen ids
        self.index.add_with_ids(vecs, ids_arr)

    def search(self, queries: np.ndarray, k: int = 5) -> list[list[tuple[int, float]]]:
        k = min(k, len(self)) or 1
        scores, ids = self.index.search(queries, k)
        return [[(int(i), float(s)) for i, s in zip(r_ids, r_scores) if i != -1]
                for r_ids, r_scores in zip(ids, scores)]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / _INDEX_FILE))
        (d / _META_FILE).write_text(json.dumps({**self.meta, "dim": self.dim, "n": len(self)}, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        d = Path(directory)
        meta = json.loads((d / _META_FILE).read_text())
        store = cls(meta["dim"], meta)
        store.index = faiss.read_index(str(d / _INDEX_FILE))
        return store

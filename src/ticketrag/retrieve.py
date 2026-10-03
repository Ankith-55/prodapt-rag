"""Pattern-level retrieval: raw complaint -> nearest historical ticket patterns, each with its
aggregated resolution distribution (the evidence the generator will cite).

Each pattern is indexed as several vectors (its label + synthetic example complaints). A query is
scored against every vector and collapsed to the best-matching vector per pattern (max-pooling)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ticketrag.embed import Embedder
from ticketrag.store import VectorStore

SLOT_BITS = 4
SLOTS = 1 << SLOT_BITS  # vector id = pattern_int << 4 | slot ; slot 0 = label, 1..15 = example complaints


def _tidy(s: str) -> str:
    # HPD labels are ALL CAPS ("WATER LEAK"); normalise so the embedder sees the same style everywhere.
    return s.title() if s.isupper() else s


def pattern_text(complaint_type: str, descriptor: str, descriptor_2: str = "") -> str:
    parts = [_tidy(p) for p in (complaint_type, descriptor, descriptor_2) if p]
    return ". ".join(parts)


def pid_to_int(pattern_id: str) -> int:
    """pattern_id is a 10-hex-char hash (40 bits); shifted by SLOT_BITS it still fits an int64 id."""
    return int(pattern_id, 16)


def vec_id(pattern_id: str, slot: int) -> int:
    return (pid_to_int(pattern_id) << SLOT_BITS) | slot


@dataclass
class Hit:
    pattern_id: str
    score: float  # raw cosine similarity of the best-matching vector (used for abstention)
    complaint_type: str
    descriptor: str
    descriptor_2: str
    tier: str
    total_cases: int
    top1_share: float
    median_close_hours: float
    p90_close_hours: float = 0.0
    resolutions: list[dict] = field(default_factory=list)  # [{resolution_id, share, count, text}]


class PatternRetriever:
    def __init__(self, index_dir: str | Path = "data/processed/index",
                 processed_dir: str | Path = "data/processed", embedder: Embedder | None = None,
                 support_weight: float = 0.0, min_cases: int = 0):
        self.index_dir, self.processed_dir = Path(index_dir), Path(processed_dir)
        self.support_weight, self.min_cases = support_weight, min_cases
        store, patterns, resolution_text = self._load()
        model_name = store.meta["model_name"]
        if embedder is not None and embedder.model_name != model_name:
            raise ValueError(f"index built with {model_name}, embedder is {embedder.model_name}")
        self.embedder = embedder or Embedder(model_name)
        self.store, self.patterns, self.resolution_text = store, patterns, resolution_text

    def _load(self):
        store = VectorStore.load(self.index_dir)
        patterns = pd.read_parquet(self.processed_dir / "patterns.parquet")
        patterns = {int_id: row for int_id, row in
                    zip(patterns["pattern_id"].map(pid_to_int), patterns.to_dict("records"))}
        res = pd.read_parquet(self.processed_dir / "resolutions.parquet")
        return store, patterns, dict(zip(res["resolution_id"], res["resolution_description"]))

    def reload(self) -> None:
        """Pick up an ingested batch without restarting. New objects are built first, then swapped in,
        so a concurrent search sees either the old or the new state, never a mix."""
        self.store, self.patterns, self.resolution_text = self._load()

    def rank(self, qvecs: np.ndarray, k: int = 5, support_weight: float | None = None,
             min_cases: int | None = None) -> list[list[tuple[int, float]]]:
        """For each query vector: top-k [(pattern_int, raw_cosine)] ordered by
        cosine + support_weight*log10(1+n_cases). The support prior damps one-off tail patterns."""
        alpha = self.support_weight if support_weight is None else support_weight
        floor = self.min_cases if min_cases is None else min_cases
        out = []
        for row in self.store.search(qvecs, k * SLOTS):
            best: dict[int, float] = {}
            for vid, s in row:
                pid = vid >> SLOT_BITS
                if pid in self.patterns and s > best.get(pid, -1.0):
                    best[pid] = s
            scored = [(pid, s) for pid, s in best.items() if self.patterns[pid]["total_cases"] >= floor]
            scored.sort(key=lambda t: t[1] + alpha * math.log10(1 + self.patterns[t[0]]["total_cases"]),
                        reverse=True)
            out.append(scored[:k])
        return out

    def _hit(self, int_id: int, score: float, top_resolutions: int) -> Hit:
        r = self.patterns[int_id]
        res = [{**d, "text": self.resolution_text[d["resolution_id"]]}
               for d in list(r["resolution_dist"])[:top_resolutions]]
        return Hit(r["pattern_id"], score, r["complaint_type"], r["descriptor"], r["descriptor_2"],
                   r["tier"], int(r["total_cases"]), float(r["top1_share"]),
                   float(r["median_close_hours"]), float(r["p90_close_hours"]), res)

    def search(self, query: str, k: int = 5, top_resolutions: int = 3) -> list[Hit]:
        qv = self.embedder.encode_queries([query])
        return [self._hit(i, s, top_resolutions) for i, s in self.rank(qv, k)[0]]

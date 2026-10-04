"""Test doubles and a tiny synthetic corpus: no API key, no model download, runs in seconds."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

NOISE_A = "Police responded and observed no criminal violation upon arrival. Please contact 311 if it persists."
NOISE_B = "Police responded and issued a summons for the loud music violation."
NOISE_C = "Police responded but the caller could not be reached and the case was closed."
HEAT_A = "The housing agency inspected and found heat was restored. No violation was issued."
HEAT_B = "The housing agency could not gain access to the apartment to inspect."
POTHOLE = "The transportation department inspected the pothole and repaired the problem."

SPEC = [  # (labels), [(weight, resolution text)], n tickets
    (("Noise - Residential", "Loud Music/Party", ""), [(0.40, NOISE_A), (0.35, NOISE_B), (0.25, NOISE_C)], 60),  # rag_core
    (("Heat/Hot Water", "No Heat", ""), [(0.60, HEAT_A), (0.40, HEAT_B)], 40),  # rag_light
    (("Street Condition", "Pothole", ""), [(1.0, POTHOLE)], 40),  # fast_lookup
]


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors: shared words give high cosine, disjoint words give ~0."""
    model_name = "fake-bow"
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts, batch_size: int = 64) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]).astype("float32")

    encode_queries = encode_docs


class StubLLM:
    """Stands in for ticketrag.llm.LLM. Records calls; answers are canned and valid unless overridden."""
    model = "stub"

    def __init__(self) -> None:
        self.usage = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.calls: list[str] = []
        self.answers: list[dict] = []  # queued answers for the "answer" call (to test retry paths)
        self.fail = False
        self.parse_pick = 1
        self.severity = 3

    def json(self, system, user, schema, name="result", temperature=0.0, max_tokens=1000, retries=6):
        if self.fail:
            raise RuntimeError("llm down")
        self.calls.append(name)
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += 100
        self.usage["completion_tokens"] += 20
        if name == "parse":
            return {"reason": "stub", "selected_candidate": self.parse_pick, "confidence": "high",
                    "severity": self.severity, "sentiment": "frustrated"}
        if name == "answer":
            return self.answers.pop(0) if self.answers else good_answer()
        if name == "complaints":
            label = re.search(r"resident\): (.*)", user).group(1)
            return {"complaints": [f"I have a problem with {label} case {i}" for i in range(4)]}
        raise AssertionError(f"unexpected llm call {name}")


def good_answer() -> dict:
    return {
        "opening": "I am sorry you are dealing with this.",
        "summary": {"text": "In past tickets the outcomes were mixed.", "citations": ["P1"]},
        "steps": [{"text": "Tell the customer that past tickets recorded several different outcomes.", "citations": ["P1"]},
                  {"text": "Tell the customer what the agency recorded in similar cases.", "citations": ["R1"]}],
    }


def make_raw(spec=SPEC, prefix: str = "T", start: datetime = datetime(2026, 3, 1)) -> pd.DataFrame:
    rows, k = [], 0
    for (ctype, desc, desc2), outcomes, n in spec:
        texts: list[str] = []
        for weight, text in outcomes:
            texts += [text] * round(weight * n)
        texts = (texts + [outcomes[-1][1]] * n)[:n]
        for t in texts:
            created = start + timedelta(minutes=10 * k)
            rows.append({"unique_key": f"{prefix}{k}", "created_date": created.isoformat(),
                         "closed_date": (created + timedelta(hours=2 * (k % 5 + 1))).isoformat(),
                         "complaint_type": ctype, "descriptor": desc, "descriptor_2": desc2,
                         "resolution_description": t, "status": "Closed"})
            k += 1
    return pd.DataFrame(rows)

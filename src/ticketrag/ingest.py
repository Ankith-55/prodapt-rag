"""Incremental ingestion of new closed tickets.

State directory layout (everything the service needs):
  tickets.parquet, patterns.parquet, resolutions.parquet, examples.jsonl, index/{vectors.faiss,meta.json},
  ingest_log.jsonl (one audit record per batch)

A batch (1) dedupes by unique_key, (2) appends the new tickets, (3) recomputes pattern statistics and tiers,
(4) for patterns never seen before ("new ticket classes") asks the LLM for example complaints and upserts their
vectors. Existing patterns keep their vectors (their label did not change); only their statistics refresh.
Re-ingesting the same batch is a no-op. At much larger scale step (3) becomes incremental aggregation and the
embed step runs as an async worker (see the production notes)."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ticketrag.embed import Embedder
from ticketrag.examples import generate_complaints
from ticketrag.llm import LLM
from ticketrag.patterns import build_pattern_table, clean_tickets
from ticketrag.retrieve import SLOTS, pattern_text, vec_id
from ticketrag.store import VectorStore


def _index_patterns(store: VectorStore, embedder: Embedder, pat: pd.DataFrame, examples: dict) -> int:
    ids, texts = [], []
    for r in pat.itertuples():
        ids.append(vec_id(r.pattern_id, 0))
        texts.append(pattern_text(r.complaint_type, r.descriptor, r.descriptor_2))
        for slot, complaint in enumerate(examples.get(r.pattern_id, [])[: SLOTS - 1], start=1):
            ids.append(vec_id(r.pattern_id, slot))
            texts.append(complaint)
    if ids:
        store.upsert(ids, embedder.encode_docs(texts))
    return len(ids)


def init_state(clean: pd.DataFrame, state_dir: str | Path, embedder: Embedder,
               examples_path: str | Path | None = None) -> dict:
    """Build a fresh state from cleaned tickets, reusing already generated example complaints."""
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    pat, res = build_pattern_table(clean)
    examples = {}
    if examples_path and Path(examples_path).exists():
        keep = set(pat["pattern_id"])
        for line in Path(examples_path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["pattern_id"] in keep:
                examples[row["pattern_id"]] = row["complaints"]
    store = VectorStore(embedder.dim, meta={"model_name": embedder.model_name,
                                            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                            "doc_format": "label + synthetic example complaints"})
    n_vec = _index_patterns(store, embedder, pat, examples)
    store.meta.update({"patterns": len(pat), "example_vectors": n_vec - len(pat)})
    store.save(state / "index")
    clean.to_parquet(state / "tickets.parquet", index=False)
    pat.to_parquet(state / "patterns.parquet", index=False)
    res.to_parquet(state / "resolutions.parquet", index=False)
    with open(state / "examples.jsonl", "w", encoding="utf-8") as f:
        for pid, cs in examples.items():
            f.write(json.dumps({"pattern_id": pid, "complaints": cs}, ensure_ascii=False) + "\n")
    return {"tickets": len(clean), "patterns": len(pat), "vectors": n_vec}


def apply_batch(state_dir: str | Path, raw: pd.DataFrame, embedder: Embedder, llm: LLM | None = None,
                n_examples: int = 4, workers: int = 8) -> dict:
    t0 = time.perf_counter()
    state = Path(state_dir)
    old_t = pd.read_parquet(state / "tickets.parquet")
    old_p = pd.read_parquet(state / "patterns.parquet").set_index("pattern_id")
    old_r = set(pd.read_parquet(state / "resolutions.parquet")["resolution_id"])

    new_clean, filt = clean_tickets(raw)
    fresh = new_clean[~new_clean["unique_key"].isin(set(old_t["unique_key"]))]
    report = {"received": len(raw), "usable": len(new_clean), "duplicates_skipped": len(new_clean) - len(fresh),
              "ingested": len(fresh), "filter": filt}
    if fresh.empty:
        report |= {"new_patterns": [], "updated_patterns": 0, "tier_changes": [], "new_resolutions": 0,
                   "examples_generated": 0, "seconds": round(time.perf_counter() - t0, 2)}
        return report

    tickets = pd.concat([old_t, fresh], ignore_index=True)
    pat, res = build_pattern_table(tickets)
    new_p = pat.set_index("pattern_id")
    new_ids = new_p.index.difference(old_p.index)
    common = new_p.index.intersection(old_p.index)
    updated = common[(new_p.loc[common, "total_cases"] != old_p.loc[common, "total_cases"]).values]
    tier_changes = [{"pattern": pattern_text(*new_p.loc[p, ["complaint_type", "descriptor", "descriptor_2"]]),
                     "from": old_p.loc[p, "tier"], "to": new_p.loc[p, "tier"]}
                    for p in common if old_p.loc[p, "tier"] != new_p.loc[p, "tier"]]

    # New ticket classes: LLM-written example complaints, then upsert their vectors.
    new_rows = pat[pat["pattern_id"].isin(new_ids)]
    examples: dict = {}
    if llm is not None and len(new_rows):
        def gen(r):
            return r.pattern_id, generate_complaints(llm, pattern_text(r.complaint_type, r.descriptor, r.descriptor_2),
                                                     "index", n_examples)
        with ThreadPoolExecutor(workers) as ex:
            examples = dict(ex.map(gen, list(new_rows.itertuples())))

    store = VectorStore.load(state / "index")
    if store.meta["model_name"] != embedder.model_name:
        raise ValueError(f"index built with {store.meta['model_name']}, embedder is {embedder.model_name}")
    _index_patterns(store, embedder, new_rows, examples)
    store.meta.update({"patterns": len(pat), "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    store.save(state / "index")

    tickets.to_parquet(state / "tickets.parquet", index=False)
    pat.to_parquet(state / "patterns.parquet", index=False)
    res.to_parquet(state / "resolutions.parquet", index=False)
    with open(state / "examples.jsonl", "a", encoding="utf-8") as f:
        for pid, cs in examples.items():
            f.write(json.dumps({"pattern_id": pid, "complaints": cs}, ensure_ascii=False) + "\n")

    report |= {
        "new_patterns": [pattern_text(r.complaint_type, r.descriptor, r.descriptor_2) + f" (n={r.total_cases})"
                         for r in new_rows.itertuples()],
        "updated_patterns": len(updated), "tier_changes": tier_changes,
        "new_resolutions": len(set(res["resolution_id"]) - old_r),
        "examples_generated": sum(len(v) for v in examples.values()),
        "index_vectors": len(store), "seconds": round(time.perf_counter() - t0, 2),
    }
    with open(state / "ingest_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **report},
                           ensure_ascii=False, default=str) + "\n")
    return report

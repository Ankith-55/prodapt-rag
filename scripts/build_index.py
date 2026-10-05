"""Embed every pattern (label + synthetic example complaints) and write the FAISS index.

  python scripts/build_index.py                      # bge-small, label + examples -> data/processed/index
  python scripts/build_index.py --no-examples        # label-only baseline
  python scripts/build_index.py --model BAAI/bge-m3 --out data/processed/index_bge-m3
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ticketrag import runlog
from ticketrag.embed import DEFAULT_MODEL, Embedder
from ticketrag.paths import default_state_dir, examples_file
from ticketrag.retrieve import SLOTS, pattern_text, vec_id
from ticketrag.store import VectorStore

ap = argparse.ArgumentParser()
ap.add_argument("--patterns", default=str(default_state_dir() / "patterns.parquet"))
ap.add_argument("--examples", default=str(examples_file("index")))
ap.add_argument("--no-examples", action="store_true")
ap.add_argument("--out", default="data/processed/index")
ap.add_argument("--model", default=DEFAULT_MODEL)
a = ap.parse_args()
runlog.start("build_index_" + Path(a.out).name)

pat = pd.read_parquet(a.patterns)
ids, texts = [], []
for r in pat.itertuples():
    ids.append(vec_id(r.pattern_id, 0))
    texts.append(pattern_text(r.complaint_type, r.descriptor, r.descriptor_2))

n_examples = 0
if not a.no_examples:
    ex_path = Path(a.examples)
    if not ex_path.exists():
        raise SystemExit(f"{ex_path} not found: run expand_patterns.py --role index, or pass --no-examples")
    for line in ex_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for slot, complaint in enumerate(row["complaints"][: SLOTS - 1], start=1):
            ids.append(vec_id(row["pattern_id"], slot))
            texts.append(complaint)
            n_examples += 1
assert len(set(ids)) == len(ids), "vector id collision"

t0 = time.time()
emb = Embedder(a.model)
vecs = emb.encode_docs(texts)
store = VectorStore(emb.dim, meta={
    "model_name": a.model,
    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "doc_format": "label + synthetic example complaints" if n_examples else "label only",
    "patterns": len(pat),
    "example_vectors": n_examples,
})
store.upsert(ids, vecs)
store.save(a.out)
print(f"indexed {len(store)} vectors ({len(pat)} labels + {n_examples} examples) with {a.model} "
      f"(dim={emb.dim}) in {time.time() - t0:.1f}s -> {a.out}")
print("sample docs:", *texts[:2], texts[len(pat)] if n_examples else "", sep="\n  ")

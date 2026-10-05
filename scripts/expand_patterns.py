"""Generate synthetic raw customer complaints for every pattern (the corpus has labels, not prose).

  --role index : neutral, clear complaints -> embedded alongside the label to improve retrieval
  --role eval  : messy, noisy complaints from an independent prompt -> held-out retrieval eval set
Resumable: re-running skips patterns already in the output file. Responses are also disk-cached."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ticketrag import runlog
from ticketrag.examples import STYLES, generate_complaints
from ticketrag.llm import LLM
from ticketrag.paths import default_state_dir
from ticketrag.retrieve import pattern_text

ap = argparse.ArgumentParser()
ap.add_argument("--role", choices=STYLES, required=True)
ap.add_argument("--n", type=int, default=4, help="complaints per pattern")
ap.add_argument("--limit", type=int, default=0, help="only the first N patterns (0 = all)")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--patterns", default=str(default_state_dir() / "patterns.parquet"))
ap.add_argument("--suffix", default="", help="output name suffix, e.g. _n8 for an ablation set")
a = ap.parse_args()

runlog.start(f"expand_patterns_{a.role}{a.suffix}")
out_path = Path("data/processed") / f"pattern_examples_{a.role}{a.suffix}.jsonl"
pat = pd.read_parquet(a.patterns)
if a.limit:
    pat = pat.head(a.limit)
done = set()
if out_path.exists():
    done = {json.loads(line)["pattern_id"] for line in out_path.read_text(encoding="utf-8").splitlines()}
todo = [r for r in pat.itertuples() if r.pattern_id not in done]
print(f"{len(pat)} patterns, {len(done)} already done, {len(todo)} to generate (role={a.role})")

llm = LLM()


def gen(r):
    label = pattern_text(r.complaint_type, r.descriptor, r.descriptor_2)
    try:
        complaints = generate_complaints(llm, label, a.role, a.n)
    except Exception as e:  # one bad pattern must not kill the batch; a rerun retries it
        return {"failed": r.pattern_id, "error": f"{type(e).__name__}: {e}"}
    return {"pattern_id": r.pattern_id, "label": label, "complaints": complaints}


failed = []
with ThreadPoolExecutor(a.workers) as ex, out_path.open("a", encoding="utf-8") as f:
    for i, row in enumerate(ex.map(gen, todo), 1):
        if "failed" in row:
            failed.append(row)
        else:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        if i % 100 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  failed={len(failed)}  usage={llm.usage}")

print("done ->", out_path)
if failed:
    print(f"{len(failed)} patterns failed (rerun this command to retry them):", *failed[:5], sep="\n  ")

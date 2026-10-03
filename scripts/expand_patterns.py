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
from ticketrag.llm import LLM
from ticketrag.retrieve import pattern_text

STYLES = {
    "index": ("Write clear, natural first-person complaints a resident would phone in. Vary length "
              "(1-3 sentences), tone and the details mentioned (time, duration, impact)."),
    "eval": ("Write MESSY, realistic complaints as people actually type or say them: typos, slang, run-ons, "
             "emotion, irrelevant side details, sometimes vague. Vary length from one line to a short rant."),
}
SCHEMA = {
    "type": "object",
    "properties": {"complaints": {"type": "array", "items": {"type": "string"}}},
    "required": ["complaints"],
    "additionalProperties": False,
}

ap = argparse.ArgumentParser()
ap.add_argument("--role", choices=STYLES, required=True)
ap.add_argument("--n", type=int, default=4, help="complaints per pattern")
ap.add_argument("--limit", type=int, default=0, help="only the first N patterns (0 = all)")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--patterns", default="data/processed/patterns.parquet")
a = ap.parse_args()

runlog.start(f"expand_patterns_{a.role}")
out_path = Path("data/processed") / f"pattern_examples_{a.role}.jsonl"
pat = pd.read_parquet(a.patterns)
if a.limit:
    pat = pat.head(a.limit)
done = set()
if out_path.exists():
    done = {json.loads(line)["pattern_id"] for line in out_path.read_text(encoding="utf-8").splitlines()}
todo = [r for r in pat.itertuples() if r.pattern_id not in done]
print(f"{len(pat)} patterns, {len(done)} already done, {len(todo)} to generate (role={a.role})")

llm = LLM()
SYSTEM = ("You write realistic complaints that residents of New York City submit to the 311 hotline. "
          f"{STYLES[a.role]} Never mention '311', ticket categories, or agency names unless a real person "
          "naturally would. Do not copy the category label verbatim; describe the situation in your own "
          "words. Each complaint must be about the described problem only.")


def gen(r):
    label = pattern_text(r.complaint_type, r.descriptor, r.descriptor_2)
    user = f"Problem category (hidden from the resident): {label}\nWrite {a.n} different complaints."
    try:
        res = llm.json(SYSTEM, user, SCHEMA, name="complaints", temperature=0.9, max_tokens=1500)
    except Exception as e:  # one bad pattern must not kill the batch; a rerun retries it
        return {"failed": r.pattern_id, "error": f"{type(e).__name__}: {e}"}
    return {"pattern_id": r.pattern_id, "label": label, "complaints": res["complaints"][: a.n]}


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
for line in out_path.read_text(encoding="utf-8").splitlines()[:3]:
    row = json.loads(line)
    print("\n", row["label"])
    for c in row["complaints"]:
        print("   -", c)

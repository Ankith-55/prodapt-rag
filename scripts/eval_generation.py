"""End-to-end system-health eval: routing, abstention, grounding/faithfulness, latency, cost.

  python scripts/eval_generation.py --n 150 --tag gen_v1
Writes eval/reports/generation/<tag>.json, output/eval_<tag>.txt, output/<tag>_samples.jsonl."""
import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ticketrag import runlog
from ticketrag.judge import judge_answer
from ticketrag.llm import LLM
from ticketrag.paths import examples_file
from ticketrag.pipeline import Assistant
from ticketrag.retrieve import PatternRetriever, pid_to_int

ap = argparse.ArgumentParser()
ap.add_argument("--index", default=None, help="index dir (default: <state>/index)")
ap.add_argument("--eval-file", default=str(examples_file("eval")))
ap.add_argument("--ood-outside", default="eval/ood_outside.txt")
ap.add_argument("--ood-borderline", default="eval/ood_borderline.txt")
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--gate", type=float, default=0.74)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--siblings", type=int, default=0, help="sibling patterns added to rag_core sources (0 = off)")
ap.add_argument("--judge-model", default="gpt-4.1-mini")
ap.add_argument("--tag", default="gen_v1")
a = ap.parse_args()
runlog.start(f"eval_{a.tag}")

pool = []
for line in Path(a.eval_file).read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    pool += [(c, row["pattern_id"]) for c in row["complaints"]]
random.Random(a.seed).shuffle(pool)
sample = pool[: a.n]

llm, judge = LLM(), LLM(model=a.judge_model)
retriever = PatternRetriever(a.index)
assistant = Assistant(retriever, llm, gate=a.gate, k=a.k, siblings=a.siblings)
print(f"generator={llm.model} judge={judge.model} n={len(sample)} k={a.k} gate={a.gate}")


def run_one(item):
    q, gold_pid = item
    r = assistant.ask(q)
    verdicts = judge_answer(judge, r.source_block, r.answer) if not r.abstained else []
    return r, verdicts


results = []
with ThreadPoolExecutor(a.workers) as ex:
    for i, res in enumerate(ex.map(run_one, sample), 1):
        results.append(res)
        if i % 10 == 0 or i == len(sample):
            print(f"  {i}/{len(sample)} done  generator={llm.usage}", flush=True)
n = len(results)
answered = [(r, v, g) for (r, v), (_, g) in zip(results, sample) if not r.abstained]
gold = {pid: retriever.patterns[pid_to_int(pid)] for _, pid in sample}


def pct(x, d):
    return round(100 * x / d, 1) if d else None


items = [it for _, v, _ in answered for it in v]
verdict_counts = Counter(it["verdict"] for it in items)
full_ok = sum(all(it["verdict"] == "supported" for it in v) for _, v, _ in answered)
lat = np.array([r.timing_ms["total"] for r, _, _ in answered]) if answered else np.array([0])
by_method = {}
for m in ("llm", "template"):
    sel = [it for r, v, _ in answered if r.method == m for it in v]
    by_method[m] = {"answers": sum(r.method == m for r, _, _ in answered), "items": len(sel),
                    "supported_pct": pct(sum(i["verdict"] == "supported" for i in sel), len(sel))}
llm_answers = [r for r, _, _ in answered if r.method == "llm" or r.grounding.get("attempts")]

rep = {
    "tag": a.tag, "generator": llm.model, "judge": judge.model, "n": n, "k": a.k, "gate": a.gate,
    "answered_pct": pct(len(answered), n),
    "abstain_reasons": dict(Counter((r.abstain_reason or "").split(" ")[0] for r, _ in results if r.abstained)),
    "tier_mix_answered": dict(Counter(r.tier for r, _, _ in answered)),
    "end_to_end_category_acc_pct": pct(sum(r.category is not None and r.category.lower() == gold[g]["complaint_type"].lower()
                                           for r, _, g in answered), n),
    "category_acc_when_answered_pct": pct(sum(r.category.lower() == gold[g]["complaint_type"].lower()
                                              for r, _, g in answered), len(answered)),
    "validator_first_attempt_violation_pct": pct(sum(r.grounding.get("first_attempt_violations", 0) > 0
                                                     for r in llm_answers), len(llm_answers)),
    "validator_remaining_violation_pct": pct(sum(bool(r.grounding.get("remaining_violations")) for r in llm_answers),
                                             len(llm_answers)),
    "template_fallback_from_llm_tier": sum(r.method == "template" and r.tier != "fast_lookup" for r, _, _ in answered),
    "faithfulness_item_pct": {k: pct(v, len(items)) for k, v in verdict_counts.items()},
    "faithfulness_answer_fully_supported_pct": pct(full_ok, len(answered)),
    "faithfulness_by_method": by_method,
    "latency_ms": {"p50": int(np.percentile(lat, 50)), "p95": int(np.percentile(lat, 95)),
                   "mean_stage": {s: int(np.mean([r.timing_ms.get(s, 0) for r, _, _ in answered]))
                                  for s in ("retrieve", "parse", "generate")} if answered else {}},
    "usage_generator": llm.usage, "usage_judge": judge.usage,
}

# Out-of-domain abstention (two sets)
for name, path in (("ood_outside", a.ood_outside), ("ood_borderline", a.ood_borderline)):
    qs = [q for q in Path(path).read_text(encoding="utf-8").splitlines() if q.strip()]
    with ThreadPoolExecutor(a.workers) as ex:
        rs = list(ex.map(assistant.ask, qs))
    rep[name] = {"n": len(qs), "abstained_pct": pct(sum(r.abstained for r in rs), len(qs)),
                 "reasons": dict(Counter((r.abstain_reason or "answered").split(" ")[0] for r in rs)),
                 "passed_through_examples": [f"{r.category} / {r.product} <- {r.complaint[:80]}"
                                             for r in rs if not r.abstained][:8]}

print(json.dumps(rep, indent=2, default=str))
Path("eval/reports/generation").mkdir(parents=True, exist_ok=True)
Path(f"eval/reports/generation/{a.tag}.json").write_text(json.dumps(rep, indent=2, default=str))
with open(f"output/{a.tag}_samples.jsonl", "w", encoding="utf-8") as f:
    for (r, v), (_, g) in zip(results, sample):
        d = asdict(r)
        d["gold"] = {k: gold[g][k] for k in ("complaint_type", "descriptor", "descriptor_2")}
        d["verdicts"] = v
        f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
print(f"\nreport -> eval/reports/generation/{a.tag}.json ; samples -> output/{a.tag}_samples.jsonl")

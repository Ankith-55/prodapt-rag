"""Evaluate LLM candidate selection (ablation A5) against retrieval top-1 on a sample of eval queries.

  python scripts/eval_parse.py --n 1000 --tag parse_small_top5
Writes eval/reports/parse/<tag>.json, output/eval_<tag>.txt and output/<tag>_samples.jsonl."""
import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ticketrag import runlog
from ticketrag.llm import LLM
from ticketrag.parse import candidate_label, parse_complaint
from ticketrag.retrieve import PatternRetriever, pid_to_int

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="data/processed/index")
ap.add_argument("--eval-file", default="data/processed/pattern_examples_eval.jsonl")
ap.add_argument("--ood-file", default="eval/ood_queries.txt")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--k", type=int, default=5, help="candidates shown to the LLM")
ap.add_argument("--gate", type=float, default=0.74, help="cosine abstention threshold")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=4, help="keep low: the account limit is 200k tokens/min")
ap.add_argument("--tag", default="parse_small_top5")
a = ap.parse_args()
runlog.start(f"eval_{a.tag}")

pool = []
for line in Path(a.eval_file).read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    pool += [(c, pid_to_int(row["pattern_id"])) for c in row["complaints"]]
random.Random(a.seed).shuffle(pool)
sample = pool[: a.n]
ood = [q for q in Path(a.ood_file).read_text(encoding="utf-8").splitlines() if q.strip()]

retriever = PatternRetriever(a.index)
llm = LLM()
print(f"model={llm.model} n={len(sample)} k={a.k} gate={a.gate}")


def run(queries):
    qv = retriever.embedder.encode_queries(queries)
    ranked = retriever.rank(qv, k=a.k)
    hit_lists = [[retriever._hit(p, s, 1) for p, s in r] for r in ranked]
    with ThreadPoolExecutor(a.workers) as ex:
        parsed = list(ex.map(lambda t: parse_complaint(llm, t[0], t[1]), zip(queries, hit_lists)))
    return ranked, hit_lists, parsed


ranked, hit_lists, parsed = run([q for q, _ in sample])
gold = [g for _, g in sample]
n = len(sample)
gcat = [retriever.patterns[g]["complaint_type"] for g in gold]

in_topk = [g in [p for p, _ in r] for r, g in zip(ranked, gold)]
ret_cat = sum(hl[0].complaint_type == c for hl, c in zip(hit_lists, gcat)) / n
ret_pat = sum(r[0][0] == g for r, g in zip(ranked, gold)) / n
llm_cat = sum(p.category == c for p, c in zip(parsed, gcat)) / n
llm_pat = sum(p.hit is not None and pid_to_int(p.hit.pattern_id) == g for p, g in zip(parsed, gold)) / n
given = [(p.hit is not None and pid_to_int(p.hit.pattern_id) == g) for p, g, ok in zip(parsed, gold, in_topk) if ok]
answered = sum(p.hit is not None and h[0].score >= a.gate for p, h in zip(parsed, hit_lists)) / n
llm_none_in = sum(p.hit is None for p in parsed) / n

_, ood_hits, ood_parsed = run(ood)
rej_cos = sum(h[0].score < a.gate for h in ood_hits) / len(ood)
rej_llm = sum(p.hit is None for p in ood_parsed) / len(ood)
rej_any = sum(h[0].score < a.gate or p.hit is None for h, p in zip(ood_hits, ood_parsed)) / len(ood)

rep = {
    "tag": a.tag, "model": llm.model, "n": n, "k": a.k, "gate": a.gate,
    "ceiling_gold_in_topk": sum(in_topk) / n,
    "retrieval_top1_category": ret_cat, "retrieval_top1_pattern": ret_pat,
    "llm_category": llm_cat, "llm_pattern": llm_pat,
    "llm_pick_gold_given_in_topk": sum(given) / max(len(given), 1),
    "in_domain_answered_both_gates": answered, "in_domain_llm_none": llm_none_in,
    "ood_n": len(ood), "ood_reject_cosine": rej_cos, "ood_reject_llm": rej_llm, "ood_reject_either": rej_any,
    "severity_dist": dict(sorted(Counter(p.severity for p in parsed).items())),
    "sentiment_dist": dict(Counter(p.sentiment for p in parsed)),
    "confidence_dist": dict(Counter(p.confidence for p in parsed)),
    "usage": llm.usage,
}
print(json.dumps(rep, indent=2))

print("\nOOD queries the gates let through:")
for q, h, p in zip(ood, ood_hits, ood_parsed):
    if h[0].score >= a.gate and p.hit is not None:
        print(f"  cos={h[0].score:.3f} -> {candidate_label(p.hit)} | {q[:90]}")

Path("eval/reports/parse").mkdir(parents=True, exist_ok=True)
Path(f"eval/reports/parse/{a.tag}.json").write_text(json.dumps(rep, indent=2))
with open(f"output/{a.tag}_samples.jsonl", "w", encoding="utf-8") as f:
    for (q, g), hl, p in zip(sample, hit_lists, parsed):
        f.write(json.dumps({
            "query": q, "gold": candidate_label(retriever._hit(g, 1.0, 1)),
            "retrieval_top1": candidate_label(hl[0]), "llm_pick": candidate_label(p.hit) if p.hit else None,
            "severity": p.severity, "sentiment": p.sentiment, "confidence": p.confidence, "reason": p.reason,
        }, ensure_ascii=False) + "\n")
print("\nreport -> eval/reports/parse/%s.json ; samples -> output/%s_samples.jsonl" % (a.tag, a.tag))

"""Retrieval eval on held-out messy synthetic complaints (+ out-of-domain queries for abstention).

  python scripts/run_eval.py --index data/processed/index --tag small_examples
Writes eval/reports/<tag>.json and output/eval_<tag>.txt."""
import argparse
import json
from pathlib import Path

import numpy as np

from ticketrag import runlog
from ticketrag.retrieve import PatternRetriever, pid_to_int

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="data/processed/index")
ap.add_argument("--eval-file", default="data/processed/pattern_examples_eval.jsonl")
ap.add_argument("--ood-file", default="eval/ood_queries.txt")
ap.add_argument("--tag", default="default")
ap.add_argument("--alphas", default="0,0.01,0.02,0.04", help="support-prior weights to compare")
ap.add_argument("--ks", default="1,3,5,10")
a = ap.parse_args()
runlog.start(f"eval_{a.tag}")

ks = [int(x) for x in a.ks.split(",")]
alphas = [float(x) for x in a.alphas.split(",")]
retriever = PatternRetriever(a.index)
print("index meta:", {k: retriever.store.meta.get(k) for k in ("model_name", "doc_format", "patterns", "example_vectors")})

queries, gold = [], []
for line in Path(a.eval_file).read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    for c in row["complaints"]:
        queries.append(c)
        gold.append(pid_to_int(row["pattern_id"]))
print(f"{len(queries)} eval queries over {len(set(gold))} patterns")

qv = retriever.embedder.encode_queries(queries)
cases = np.array([retriever.patterns[g]["total_cases"] for g in gold])
gold_cat = [retriever.patterns[g]["complaint_type"] for g in gold]
buckets = {"n>=100": cases >= 100, "30<=n<100": (cases >= 30) & (cases < 100),
           "5<=n<30": (cases >= 5) & (cases < 30), "n<5": cases < 5}

dists = {pid: {d["resolution_id"]: d["share"] for d in row["resolution_dist"]}
         for pid, row in retriever.patterns.items()}
total = sum(row["total_cases"] for row in retriever.patterns.values())
prior: dict[str, float] = {}
for pid, row in retriever.patterns.items():
    for k, v in dists[pid].items():
        prior[k] = prior.get(k, 0.0) + v * row["total_cases"] / total
prior_tv = float(np.mean([0.5 * sum(abs(dists[g].get(k, 0) - prior.get(k, 0)) for k in set(dists[g]) | set(prior))
                          for g in gold]))
print(f"no-retrieval baseline (global resolution prior): resolution_tv={prior_tv:.3f}")

report = {"index": retriever.store.meta, "n_queries": len(queries), "no_retrieval_tv": prior_tv, "by_alpha": {}}
for alpha in alphas:
    ranked = retriever.rank(qv, k=max(ks), support_weight=alpha)
    pos = np.array([next((i for i, (p, _) in enumerate(r) if p == g), 10**6) for r, g in zip(ranked, gold)])
    m = {f"recall@{k}": float((pos < k).mean()) for k in ks}
    m["mrr@10"] = float(np.where(pos < 10, 1.0 / (pos + 1), 0.0).mean())
    m["category@1"] = float(np.mean([retriever.patterns[r[0][0]]["complaint_type"] == c
                                     for r, c in zip(ranked, gold_cat)]))
    m["volume_weighted_recall@1"] = float(((pos < 1) * cases).sum() / cases.sum())
    # Resolution-level usefulness: does the retrieved evidence carry the right resolutions, even if the exact
    # pattern is a sibling? tv = total-variation distance between gold and top-1 resolution distributions
    # (0 = identical, 1 = disjoint). coverage@3 = share of gold's resolution mass present in the top-3 patterns.
    tv, cov3 = [], []
    for r, g in zip(ranked, gold):
        gd = dists[g]
        top1 = dists[r[0][0]]
        tv.append(0.5 * sum(abs(gd.get(k, 0) - top1.get(k, 0)) for k in set(gd) | set(top1)))
        seen = {k for p, _ in r[:3] for k, v in dists[p].items() if v >= 0.05}
        cov3.append(sum(v for k, v in gd.items() if k in seen))
    m["resolution_tv@1"] = float(np.mean(tv))
    m["resolution_coverage@3"] = float(np.mean(cov3))
    m["recall@1_by_support"] = {b: float((pos[mask] < 1).mean()) for b, mask in buckets.items() if mask.any()}
    report["by_alpha"][alpha] = m
    print(f"\nalpha={alpha}")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if not isinstance(v, dict)))
    print("  recall@1 by support:", {b: round(v, 3) for b, v in m["recall@1_by_support"].items()},
          {b: int(mask.sum()) for b, mask in buckets.items()})

# Abstention calibration: top-1 raw cosine for in-domain vs out-of-domain queries.
ood = [q for q in Path(a.ood_file).read_text(encoding="utf-8").splitlines() if q.strip()]
in_top = np.array([r[0][1] for r in retriever.rank(qv, k=1)])
ood_top = np.array([r[0][1] for r in retriever.rank(retriever.embedder.encode_queries(ood), k=1)])
auc = float((in_top[:, None] > ood_top[None, :]).mean() + 0.5 * (in_top[:, None] == ood_top[None, :]).mean())
print(f"\nAbstention: {len(ood)} OOD queries. top-1 cosine  in-domain median={np.median(in_top):.3f} "
      f"p5={np.percentile(in_top, 5):.3f} | OOD median={np.median(ood_top):.3f} max={ood_top.max():.3f} | AUROC={auc:.3f}")
rows = []
for t in np.arange(0.50, 0.85, 0.02):
    rows.append({"threshold": round(float(t), 2), "in_domain_kept": float((in_top >= t).mean()),
                 "ood_rejected": float((ood_top < t).mean())})
    print(f"  thr={t:.2f}  keep in-domain {rows[-1]['in_domain_kept']:.1%}   reject OOD {rows[-1]['ood_rejected']:.1%}")
report["abstention"] = {"auroc": auc, "curve": rows}

out = Path("eval/reports")
out.mkdir(parents=True, exist_ok=True)
(out / f"{a.tag}.json").write_text(json.dumps(report, indent=2, default=str))
print("\nreport ->", out / f"{a.tag}.json")

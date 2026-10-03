"""Regenerate the numeric tables in docs/ablations.md from eval/reports/*.json.

Run after every eval:  python scripts/make_ablation_table.py
Only the block between the ABLATION_TABLES markers is rewritten; the decision log is hand-written."""
import json
import re
from pathlib import Path

DOC = Path("docs/ablations.md")
START, END = "<!-- ABLATION_TABLES:START -->", "<!-- ABLATION_TABLES:END -->"

reports = sorted(Path("eval/reports").glob("*.json"), key=lambda p: p.stat().st_mtime)
loaded = [(p.stem, json.loads(p.read_text())) for p in reports]


def pct(x):
    return f"{x * 100:.1f}%"


lines = ["### Retrieval ablations (support prior α=0; held-out messy synthetic queries)", "",
         "| Run | Embedder | Index docs | Vectors | R@1 | R@5 | R@10 | Category@1 | Res. TV@1 ↓ | Res. cov@3 | Abstain AUROC |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for tag, r in loaded:
    m, meta = r["by_alpha"]["0.0"], r["index"]
    lines.append(f"| {tag} | {meta['model_name'].split('/')[-1]} | {meta.get('doc_format', '?')} | {meta.get('n', '?')} "
                 f"| {pct(m['recall@1'])} | {pct(m['recall@5'])} | {pct(m['recall@10'])} | {pct(m['category@1'])} "
                 f"| {m['resolution_tv@1']:.3f} | {pct(m['resolution_coverage@3'])} | {r['abstention']['auroc']:.3f} |")
if loaded:
    lines += ["", f"No-retrieval baseline (global resolution prior): TV = {loaded[0][1].get('no_retrieval_tv', float('nan')):.3f}"]

lines += ["", "### Support-prior sweep (score = cosine + α·log10(1+n_cases))", "",
          "| Run | α | R@1 | R@5 | Category@1 | Volume-weighted R@1 | Res. TV@1 ↓ |", "|---|---|---|---|---|---|---|"]
for tag, r in loaded:
    for alpha, m in r["by_alpha"].items():
        lines.append(f"| {tag} | {alpha} | {pct(m['recall@1'])} | {pct(m['recall@5'])} | {pct(m['category@1'])} "
                     f"| {pct(m['volume_weighted_recall@1'])} | {m['resolution_tv@1']:.3f} |")

parse_reports = sorted(Path("eval/reports/parse").glob("*.json"), key=lambda p: p.stat().st_mtime)
if parse_reports:
    lines += ["", "### Parser: LLM candidate selection vs retrieval top-1 (sampled eval queries)", "",
              "| Run | LLM | n | k | Gold in top-k | Category: retrieval → LLM | Pattern: retrieval → LLM | "
              "LLM picks gold when present | In-domain answered | OOD rejected: cosine / LLM / either |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for p in parse_reports:
        r = json.loads(p.read_text())
        lines.append(f"| {r['tag']} | {r['model']} | {r['n']} | {r['k']} | {pct(r['ceiling_gold_in_topk'])} "
                     f"| {pct(r['retrieval_top1_category'])} → {pct(r['llm_category'])} "
                     f"| {pct(r['retrieval_top1_pattern'])} → {pct(r['llm_pattern'])} "
                     f"| {pct(r['llm_pick_gold_given_in_topk'])} | {pct(r['in_domain_answered_both_gates'])} "
                     f"| {pct(r['ood_reject_cosine'])} / {pct(r['ood_reject_llm'])} / {pct(r['ood_reject_either'])} |")

gen_reports = sorted(Path("eval/reports/generation").glob("*.json"), key=lambda p: p.stat().st_mtime)
if gen_reports:
    lines += ["", "### End-to-end generation / system health (LLM-judged faithfulness, separate judge model)", "",
              "| Run | Generator / judge | n | Answered | Category acc (e2e) | Validator viol. (1st attempt) | "
              "Claims supported / partial / unsupported | Answers fully supported | Latency p50 / p95 (s) | "
              "OOD outside abstained | OOD borderline abstained |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in gen_reports:
        r = json.loads(p.read_text())
        f = r["faithfulness_item_pct"]
        lines.append(f"| {r['tag']} | {r['generator']} / {r['judge']} | {r['n']} | {r['answered_pct']}% "
                     f"| {r['end_to_end_category_acc_pct']}% | {r['validator_first_attempt_violation_pct']}% "
                     f"| {f.get('supported', 0)}% / {f.get('partial', 0)}% / {f.get('unsupported', 0)}% "
                     f"| {r['faithfulness_answer_fully_supported_pct']}% "
                     f"| {r['latency_ms']['p50'] / 1000:.1f} / {r['latency_ms']['p95'] / 1000:.1f} "
                     f"| {r['ood_outside']['abstained_pct']}% | {r['ood_borderline']['abstained_pct']}% |")

block = "\n".join([START, *lines, END])
text = DOC.read_text(encoding="utf-8")
if START not in text:
    raise SystemExit(f"markers missing in {DOC}")
DOC.write_text(re.sub(f"{re.escape(START)}.*?{re.escape(END)}", lambda _: block, text, flags=re.S), encoding="utf-8")
print(f"updated {DOC} from {len(loaded)} reports")

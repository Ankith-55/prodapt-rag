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

block = "\n".join([START, *lines, END])
text = DOC.read_text(encoding="utf-8")
if START not in text:
    raise SystemExit(f"markers missing in {DOC}")
DOC.write_text(re.sub(f"{re.escape(START)}.*?{re.escape(END)}", lambda _: block, text, flags=re.S), encoding="utf-8")
print(f"updated {DOC} from {len(loaded)} reports")

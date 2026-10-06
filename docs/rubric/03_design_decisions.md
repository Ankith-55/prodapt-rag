# 3. Design decisions

Every decision below was measured, and the full setup, numbers and reasoning are in [`docs/ablations.md`](../ablations.md) (entries A1 to A9; raw metrics in [`eval/reports/`](../../eval/reports/)).

| # | Decision | Evidence | Outcome |
|---|---|---|---|
| A1 | Index patterns, enriched with LLM-written example complaints | Category accuracy 67.6% to 76.0%, recall@5 65.8% to 73.2% | Adopted |
| A2 | bge-small over bge-m3 | Same accuracy at 60x lower build cost and a far smaller image | bge-small |
| A3 | No popularity prior in ranking | Every setting lowered accuracy | Rank by similarity alone |
| A4 | Abstain below cosine 0.74 plus an LLM "none" gate | 97.6% of valid queries kept, 76% of off-topic rejected | Two gates |
| A5 | LLM picks among the top-5 | Category 75.7% to 83.1%, exact pattern 42.8% to 57.2% | Adopted |
| A6 | 8 example complaints instead of 4 | About 1.5 points more recall at double the index size | Not adopted |
| A7 | 8 candidates instead of 5 | Higher ceiling but a lower pick rate; confounded with a prompt change | Provisional, kept at 5 |
| A8 | Facts from code, prose from the LLM; separate uncited opening | Fully supported answers 28.1% to 72.6%, latency 7.1 s to 4.2 s | Adopted |
| A9 | Hard validator including "no predictions", no sibling patterns, outcomes sum to 100% | Unsupported claims 2.4% to 1.7%, fully supported 72.6% to 76.7% | Adopted |
| Tiers | Route by ticket history | Real later tickets: `fast_lookup` realised 93.5% vs 96.4% predicted | Adopted; tiers need hysteresis |

## Negative results that shaped the design
bge-m3 gave nothing for its cost. The popularity prior hurt. Sibling patterns brought unrelated resolution texts into answers. The first generator prompt conflicted with its own validator (55% of answers had violations): the fix was a prompt change, not a better model.

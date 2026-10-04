# Evolving data: feasibility check on real later tickets

Requirement 3: *handle evolving data and ticket classes.* We tested this on **real** NYC 311 tickets created
2026-03-10 to 03-12 (`dataset/next_days.csv`), i.e. after the 2026-03-01..03-09 corpus the system was built on.
Source: `EDA/EDA_next_days_ingestion_check.ipynb` (executed copy in `output/`). The check feeds the batch through the
pipeline's own functions (`clean_tickets`, `build_pattern_table`, `apply_batch`) on a scratch copy of the state;
`data/processed` is never modified. Date of check: 2026-10-03.

**Verdict: PASS (8/8 checks). The batch can be ingested by the current pipeline without code changes.**

## 1. Cleaning funnel

| Stage | Rows | Share of raw |
|---|---|---|
| Raw closed tickets in file | 34,573 | 100% |
| With a real resolution | 34,270 | 99.1% |
| Usable (labels present) | 33,773 | 97.7% |
| Rows with double-encoded text, repaired by the cleaner | 1,108 | 3.2% |
| Duplicate `unique_key` inside the batch | 0 | n/a |
| Overlap with the training state | 0 | n/a |

## 2. Coverage: known patterns vs new ticket classes

| Measure | Value |
|---|---|
| Training state | 94,674 tickets, 1,266 patterns, 434 resolution templates |
| New tickets in an already-known pattern | 98.8% |
| New patterns | 132 (399 tickets, 1.2%) |
| Complaint types never seen in training | Hazardous Material, Radioactive Material |
| Tickets with a never-seen resolution template | 63 (28 distinct new templates, 0.2%) |
| Patterns / templates after ingestion | 1,398 / 462 |

Largest new patterns (a whole class family emerging in 3 days):

| New pattern | Tickets |
|---|---|
| Missed Collection / Bulky Trash | 55 |
| Missed Collection / Recycling - Paper/Metal/Glass/Rigid Plastic | 53 |
| Missed Collection / Recycling - Paper | 43 |
| Missed Collection / Bulky Recycling | 41 |
| Missed Collection / Recycling - Metal/Glass/Rigid Plastic | 35 |
| Missed Collection / Trash and Bulky Trash | 11 |

## 3. Drift of resolution distributions (15 biggest known patterns)

Total-variation distance between the training distribution and the new batch (0 = identical, 1 = disjoint).

| Pattern | Train n | New n | TV distance |
|---|---|---|---|
| Street Condition / Pothole | 5,107 | 2,599 | 0.103 |
| Illegal Parking / Posted Parking Sign Violation | 3,918 | 1,413 | 0.027 |
| Illegal Parking / Blocked Hydrant | 4,328 | 1,356 | 0.061 |
| Noise - Residential / Banging/Pounding | 3,739 | 1,155 | 0.070 |
| Blocked Driveway / No Access | 3,754 | 1,084 | 0.044 |
| Noise - Residential / Loud Music/Party | 5,560 | 961 | 0.126 |
| Abandoned Vehicle / With License Plate | 2,099 | 919 | 0.073 |
| Noise - Street/Sidewalk / Loud Music/Party | 2,380 | 872 | 0.092 |
| Heat/Hot Water / Entire Building / No Heat | 2,737 | 628 | 0.116 |
| Illegal Parking / Blocked Sidewalk | 1,973 | 544 | 0.078 |
| Dirty Condition / Trash / Littering | 1,404 | 527 | 0.065 |
| Derelict Vehicles / Derelict Vehicles | 1,114 | 463 | 0.086 |
| Illegal Dumping / Removal Request | 828 | 440 | 0.119 |
| Heat/Hot Water / Entire Building / No Hot Water | 900 | 417 | 0.140 |
| Blocked Driveway / Partial Access | 1,184 | 416 | 0.086 |

Range 0.03 to 0.14: refreshing statistics matters, but drift over 3 days is modest.

## 4. Tier transitions after ingesting 3 more days (rows = before, columns = after)

| Before \ After | fast_lookup | rag_light | rag_core | Total before |
|---|---|---|---|---|
| fast_lookup | 17 | 2 | 0 | 19 |
| rag_light | 11 | 860 | 44 | 915 |
| rag_core | 0 | 24 | 308 | 332 |
| absent (new pattern) | 0 | 131 | 1 | 132 |

82 of 1,266 existing patterns (6.5%) changed tier after only 3 more days. Tiers computed from small samples are
noisy. **Design consequence:** tiers are a derived property recomputed on every ingest; production should add
hysteresis (change tier only when a pattern clears the threshold by a margin) before routing on it.

## 5. Ingestion dry-run (scratch copy, no LLM)

| Check | Result |
|---|---|
| Required columns present | PASS |
| Rows survive the cleaner | PASS |
| No `unique_key` overlap with training | PASS |
| Ingestion ran | PASS: 33,773 ingested |
| `ingested == usable - duplicates` | PASS |
| Re-ingesting the same batch is a no-op | PASS: 0 ingested, 33,773 duplicates skipped |
| Pattern count grew by exactly the number of new patterns | PASS: 1,266 → 1,398 |
| New patterns retrievable right after ingestion | PASS: 10/10 by their own label |

## Caveats

- The dry-run used `llm=None`: new classes were indexed by **label only**. In production the LLM writes 4 example
  complaints per new class first (132 calls, about $0.03), which is what improves natural-language retrieval.
- This shows **implementability**, not retrieval quality on new classes.
- `data/processed` stays the reproducible 9-day baseline the eval is built on; ingestion is exercised in a separate
  state (`scripts/demo_evolution.py`, `scripts/ingest.py`) and in the API.

## 6. Temporal backtest on real tickets (`scripts/backtest_temporal.py`, report `eval/reports/backtest/backtest_mar10_12.json`)

Train state: 94,674 tickets (03-01..03-09). Test: 33,773 real closed tickets (03-10..03-12). No LLM or embeddings:
each test ticket's pattern is looked up and we measure the probability the pattern's *historical* resolution
distribution gave to the ticket's *real* resolution. Baselines: complaint-type distribution, global distribution.

| Slice | Test tickets | Mean p(real) pattern | Mean p(real) type | Mean p(real) global | Top-1 realised | Top-1 predicted | Top-3 coverage |
|---|---|---|---|---|---|---|---|
| All seen patterns (98.8% of tickets) | 33,374 | **0.315** | 0.271 | 0.040 | 40.6% | 43.4% | 78.5% |
| fast_lookup | 1,064 | 0.905 | 0.846 | 0.021 | 93.5% | 96.4% | 97.5% |
| rag_light | 7,995 | 0.487 | 0.380 | 0.021 | 58.1% | 65.6% | 90.7% |
| rag_core | 24,315 | 0.233 | 0.210 | 0.047 | 32.5% | 33.8% | 73.7% |
| Train support n>=100 | 27,541 | 0.295 | 0.260 | 0.044 | 38.4% | 40.7% | 78.3% |
| Train support 30..99 | 3,392 | 0.348 | 0.277 | 0.020 | 46.7% | 46.6% | 81.3% |
| Train support 5..29 | 1,725 | 0.500 | 0.387 | 0.015 | 59.7% | 63.3% | 84.9% |
| Train support n<5 | 716 | 0.476 | 0.411 | 0.007 | 48.3% | 87.0% | 55.7% |

**Conclusions**
- Pattern-level history predicts real outcomes about 8x better than a global guess and about 16% better than the complaint-type distribution (value of the descriptor / descriptor_2 labels).
- The `fast_lookup` threshold (top-1 share >= 0.9 and >= 30 cases) is well calibrated on real future data (93.5% realised vs 96.4% predicted).
- Patterns with fewer than 5 training tickets are badly over-confident (48% realised vs 87% predicted). This justifies the "low evidence" note and motivates shrinking small-sample shares toward the complaint-type distribution.
- Even the best case is about 40% top-1, so the system presents likely outcomes with their shares instead of predicting one.

## 7. Live evolution demo (`scripts/demo_evolution.py`, output in `output/demo_evolution.txt`)

| Step | Result |
|---|---|
| Initial state, days 1-7 only | 70,967 tickets, 1,153 patterns |
| 1. Telecom complaint before the class exists | Abstained (top cosine 0.713 < 0.74); nearest shown for a human |
| 2. Ingest real days 8-9 | 23,707 tickets ingested, many new real classes, 720 patterns refreshed, tier changes reported |
| 3. Ingest SYNTHETIC "Broadband Service" class (120 fabricated tickets, labelled as demo data) | 2 new patterns, 8 example complaints generated, 7 new resolution templates, 7.4 s |
| Idempotency | Re-ingesting the same batch: 0 ingested, 120 duplicates skipped |
| 4. Same telecom complaint after ingestion | Answered as Broadband Service / Intermittent Connection, citing the new telecom resolutions |

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

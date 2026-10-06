# 1. Problem background understanding

The question this project answers: *can semantic retrieval of past resolved tickets give a support agent useful, cited guidance for a new complaint?* Before building, we tested whether the data supports retrieval at all, and changed the dataset when it did not.

## What the EDA found

| Finding | Evidence | Consequence |
|---|---|---|
| The provided sample is too thin | 25,921 rows, 7 complaint types (Noise - Residential alone is 52%), 2 to 3 templated resolutions per descriptor | A lookup table reproduces most of it, so it cannot demonstrate retrieval |
| The official export is richer | NYC 311 full export: 100,000 rows pulled, 44 columns, 161 complaint types, 677 descriptors; 94,674 usable after keeping closed tickets with a real resolution | Used as the corpus |
| Complaints collapse into 1,266 patterns | `complaint_type + descriptor + descriptor_2` | The retrieval unit is a pattern with a resolution distribution, not a single ticket |
| Outcomes are mixed | 91% of ticket volume sits in patterns whose top resolution is under 75% | Retrieval is needed; a fixed lookup would be wrong most of the time |
| Few patterns are predictable | 576 patterns have a top share of 90% or more, but only 19 also have 30 or more tickets (4.3% of volume) | Only these take the cheap template path |
| `descriptor_2` adds detail | Present on 47.8% of usable tickets; 40% more distinct input-to-resolution representations when used | Kept as optional |
| Resolutions are templated | 434 distinct resolution texts across 94,674 tickets | They behave like a knowledge base of standard responses |
| Data quality issues | 3.2% double-encoded text; about 3% of resolutions cut off at 500 characters in the source | Repaired or flagged in the answer |

## Why real data, and what is synthetic
Choosing real data was deliberate. The ticket history and resolutions are real. LLM-generated data is cleaner and less noisy, with neatly separated classes that even a simple ML classifier can usually tell apart, so it would not test retrieval; real tickets carry natural noise: mixed outcomes and overlapping sibling categories. The only synthetic text is customer-style complaints, because the corpus has category labels but no customer prose. They are used as a proxy for queries and checked against real later tickets (the temporal backtest).

## Why this is not a telecom corpus
The problem statement frames a telecom desk. The provided data was NYC 311 as well, so we extended it with the same source. The engine does not depend on the domain; the evolution demo onboards a synthetic telecom class to show that.

## Where to look
- Dataset selection: [`docs/EDA and Dataset Selection — NYC 311 RAG Project.md`](<../EDA and Dataset Selection — NYC 311 RAG Project.md>)
- Dataset EDA summary: [`docs/prodapt_rag_dataset_eda_summary.md`](../prodapt_rag_dataset_eda_summary.md)
- Notebooks: [`EDA/`](../../EDA/)
- Plain-language walkthrough: [`docs/how_it_works.md`](../how_it_works.md)

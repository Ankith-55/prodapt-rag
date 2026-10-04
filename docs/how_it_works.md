# How it works, in plain language

Written so one person can explain the whole system in ten minutes. Diagrams are in `docs/architecture.md`; every number
below is traceable to `docs/ablations.md` or `docs/evolving_data.md`.

## 1. The problem and the idea

A support agent pastes a raw complaint. Keyword search misses past tickets that describe the same problem in other
words. We find the most similar *types* of past resolved ticket by meaning, show how those tickets were really resolved,
and draft guidance for the agent that **cites the history it used**.

**Data.** 94,674 closed NYC 311 tickets (2026-03-01 to 03-09). Each has `complaint_type`, `descriptor`, `descriptor_2` and
a recorded `resolution_description`. It is city-services data, not telecom; the engine does not depend on the domain
(the evolution demo onboards a telecom class in seconds).

**Why retrieval-augmented generation and not a classifier or a lookup?** Grouping tickets by their three labels gives
about 1,266 patterns. For 91% of ticket volume the *same* pattern ends in several different resolutions (no outcome above
75%), so a fixed lookup would be wrong most of the time. The honest answer is a distribution of outcomes, with sources.
Only 4% of volume sits in patterns with a consistent outcome (90% or more over at least 30 tickets), so those take a cheap
template path.

## 2. The three ideas that make it work

1. **Index patterns, not tickets.** Embedding 94k near-duplicate tickets would return five copies of the same thing.
   Each pattern becomes a handful of vectors (its label plus 4 LLM-written example complaints, because the corpus has
   labels but no customer prose). A query matches by its best vector per pattern. Example complaints lifted category
   accuracy from 67.6% to 76% (ablation A1).
2. **Facts come from code, language from the LLM.** Outcome shares, ticket counts and close times are computed from the
   table and rendered by code. The LLM writes only a one-line empathy opening, a summary and 3 to 4 steps, and may only
   use numbered sources. This cut latency by half and removed a whole class of errors (A8).
3. **Say "I don't know" in several ways.** Below a similarity threshold, or if the LLM judges that none of the top-5
   candidates fits, the system abstains and routes to a human. Borderline matches and thin evidence carry visible warnings.

## 3. One request, step by step

1. **Retrieve.** Embed the complaint (bge-small, runs on CPU), search the FAISS index, collapse to the best vector per
   pattern, take the top 5 patterns with their resolution distributions.
2. **Similarity gate.** If the best cosine is below 0.74, stop: no LLM call, return the nearest types for a human.
3. **Parse (LLM).** Given the complaint and the 5 candidates (each with a snippet of its agency text), pick the one that
   fits or answer "none"; also rate severity (1 to 5) and sentiment. Category = `complaint_type`; product =
   `descriptor` + `descriptor_2`, both taken from the chosen pattern, so they are grounded. LLM picking lifted category
   accuracy from 75.7% to 83.1% and exact-pattern accuracy from 42.8% to 57.2% (A5).
4. **Route by tier.** `fast_lookup` patterns get a template answer with no generation call. Others go to the LLM.
5. **Generate (LLM).** The prompt contains numbered sources: `[P1]` pattern statistics, `[R1]..` recorded resolution texts.
   It returns an opening, a summary and steps, each with citations.
6. **Validate.** Every summary/step must cite known sources, use only numbers present in the sources, and not predict
   ("will", "going to"). Violations go back to the model once as feedback; items that still fail are dropped. If nothing
   survives, a deterministic template is used.
7. **Assemble.** Add the outcome shares (rendered by code, with an "other" line so they sum to 100%), policy notes
   (severity 4+ prioritise, severity 5 adds a fixed 911 line, fewer than 30 past tickets warns of low evidence, borderline
   similarity warns to verify the category), timings and token usage.

If the LLM or network fails at any step, `ask_safe` returns a statistics-only answer (`template_degraded`) instead of an error.

## 4. Evolving data and new ticket classes

New closed tickets arrive through `POST /ingest` or `scripts/ingest.py`. The batch is cleaned, deduplicated by ticket key
(re-ingesting is a no-op), appended, and the pattern table and tiers are recomputed. A pattern never seen before gets
4 LLM-written example complaints and its vectors are added to the index with no rebuild; known patterns just refresh
their statistics. The running service swaps in the new state atomically (`reload`).

Tested on **real** later tickets (`docs/evolving_data.md`): of 33,773 new tickets, 98.8% fell in known patterns and
132 new patterns appeared, including a whole "Missed Collection" family. A temporal backtest showed pattern history gives
the real outcome 8x the probability of a global guess, and the `fast_lookup` threshold is calibrated on real future data
(93.5% realised vs 96.4% predicted).

## 5. Guardrails and why each exists

| Guardrail | Failure it prevents |
|---|---|
| Similarity gate and LLM "none" | Confident answers about problems the history does not cover |
| Citation + number validator | Invented facts, phone numbers or timeframes |
| "No predictions" check | Promises like "the agency will fix it" |
| Outcomes rendered by code | Wrong shares or percentages |
| Separate judge model (evals only) | Self-grading bias when measuring faithfulness |
| Low-evidence and borderline notes | Over-trusting thin or marginal matches (small patterns promised 87% accuracy and delivered 48%) |
| Prompt-injection rule | Following instructions hidden in a complaint |
| Degraded mode | An LLM outage becoming an outage of the whole desk tool |
| No complaint text in logs | Leaking customer PII (only a hash and length are logged) |

## 6. How we know it works (and what it does not prove)

- Retrieval, parsing and generation each have evals; every design change has an ablation with the decision and the reason,
  including negative results (bge-m3 gave nothing for 60x the build cost; a popularity prior hurt).
- Faithfulness is judged by a different model than the generator: claims supported 94.2%, fully supported answers 76.7% (A9).
- The temporal backtest uses real tickets and real resolutions.
- **Not proven:** most quality numbers use synthetic complaints (no real customer text exists in this corpus); severity and
  sentiment are unvalidated; about 30% of clearly out-of-scope private-company complaints still match a 311 consumer
  category. These are stated limitations, not hidden ones.

## 7. Monitoring signals (`GET /metrics`)

Abstain rate and reasons, tier mix, answer method (llm / template / degraded), validator retries, latency per stage,
token cost, severity mix, and the distribution of top-1 cosine. A rising abstain rate or a drifting cosine distribution is
the earliest sign that incoming complaints no longer look like the history.

## 8. Scaling path (what changes at 100x)

| Today | At scale |
|---|---|
| In-process flat FAISS index, about 6k vectors | pgvector / Qdrant / Azure AI Search, HNSW or IVF, sharded by agency; metadata filters |
| Each replica loads its own index | Shared vector database; ingestion writes once, all replicas see it |
| Pattern table recomputed from all tickets on ingest | Incremental aggregation in a database; embedding as an asynchronous worker |
| One service for serving and ingestion | Separate ingestion worker and scheduled feed |
| Per-call LLM with on-disk cache | Shared semantic cache, token-budget rate limiter, smaller model for parsing, streaming |
| Tiers recomputed on each ingest | Add hysteresis so a pattern changes tier only when it clears the threshold by a margin (6.5% of patterns flipped in 3 days) |
| API key header | Real authentication, PII redaction before the LLM, audit trail |

## 9. Honest limitations

- City-services corpus; telecom is demonstrated by a synthetic class only.
- No separate knowledge base: agency resolution templates play that role.
- Resolution texts record what an agency did, so steps are guidance for the agent, not technical repair procedures.
- Some resolution texts are cut off at 500 characters in the source data; they are flagged in the answer.
- Exact-pattern accuracy tops out around 57% because sibling patterns are hard to separate and outcomes are inherently mixed;
  the system presents a distribution, it does not predict one outcome.

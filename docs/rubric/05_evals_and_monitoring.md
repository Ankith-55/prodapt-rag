# 5. Checkpoints, evals and monitoring

## Evals
| Eval | What it measures | Report |
|---|---|---|
| Retrieval (`run_eval.py`) | Recall, category accuracy, resolution-distribution error, abstention curve | `eval/reports/*.json` |
| Parsing (`eval_parse.py`) | LLM selection vs nearest neighbour, false abstentions | `eval/reports/parse/` |
| Generation (`eval_generation.py`) | Groundedness by an independent judge model, validator violations, latency, out-of-domain abstention | `eval/reports/generation/` |
| Temporal backtest (`backtest_temporal.py`) | Do historical resolution distributions predict **real** later outcomes (33,773 tickets) | `eval/reports/backtest/` |
| Ingestion check | 8 checks on real later tickets, idempotency, new classes searchable | [`docs/evolving_data.md`](../evolving_data.md) |

Key results: 94.2% of claims supported, 1.7% unsupported, 76.7% of answers fully supported; the pattern history gives the true outcome probability 0.315 vs 0.040 for a global guess; `fast_lookup` calibrated at 93.5% realised vs 96.4% predicted.

## Checkpoints (states you can resume from or roll back to)
Index metadata (model, build time, counts), a per-batch ingestion audit log, resumable generation jobs with an LLM response cache, committed eval reports, an immutable versioned state image (`kubectl rollout undo`), and retained ReplicaSets. Not yet built: an index version stamp on `/ready`.

## Monitoring
`GET /metrics` exposes abstain rate and reasons, tier mix, answer method, validator retries, latency per stage, token cost, severity mix and the distribution of top-1 similarity (the main drift signal). Proposed alert rules (high abstain rate, similarity drift, degraded mode, rising retries, slow answers, errors) are in [`docs/production_scale.md`](../production_scale.md). Logs are structured with a request ID and never contain the complaint text.

## Honest gaps
Most accuracy numbers use LLM-written complaints (the backtest is the real-data check); severity and sentiment are not validated against human labels; the alert rules are designed but not deployed to a live Prometheus.

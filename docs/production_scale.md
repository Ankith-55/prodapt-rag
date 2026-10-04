# Production-scale considerations

What exists today, what was verified, and what would change at 10x and 100x. Evidence files are in `docs/evidence/`.
Honest status first, design after.

## 1. Status: built and verified vs designed only

| Capability | Status | Evidence |
|---|---|---|
| Stateless API (`/ask`, `/ingest`, `/reload`, `/health`, `/ready`, `/metrics`) | Built, 48 tests | `tests/` |
| Container image (CPU-only torch, model baked in, non-root, health check) | Built, ran locally with Docker Compose | `Dockerfile`, `docker-compose.yml` |
| Kubernetes deployment on AKS (probes, limits, HPA, secret, private registry) | **Deployed and verified**: pod 1/1 Running, probes 200, HPA active, request authenticated and answered | `docs/evidence/aks_proof.txt`, `aks_logs.txt` |
| CI/CD (lint, tests, retrieval gate, container smoke test, publish image) | Written, **not yet run on GitHub** | `.github/workflows/ci.yml` |
| Metrics and structured logs | Built, `/metrics` live | `src/ticketrag/api.py` |
| Alerting and dashboards | Rules designed below, **not deployed** | section 6 |
| Shared vector store, async ingestion, semantic cache, real auth, PII redaction | **Designed only** | sections 3 to 5 |

## 2. Deployed topology (verified on AKS, indiasouthcentral)

- One node (`Standard_B2ms`: 2 vCPU, 8 GB), one pod, ClusterIP service. The AKS control plane is on the free tier; image in a private Azure Container Registry (Basic).
- **State is baked into the image** (`ticketrag-demo`): code + pre-built tickets/patterns/index. Every replica is identical and rollback is `kubectl rollout undo`. The cost: live `/ingest` would diverge across replicas, so the cluster runs with `TICKETRAG_IMMUTABLE=1` and `/ingest` returns 403. Ingestion is an offline step that produces a new state image.
- Security posture: non-root user, read-only root filesystem (`/tmp` is an emptyDir), all capabilities dropped, seccomp default, secrets in a Kubernetes Secret, access key required on write and answer endpoints, no public endpoint by default.
- Rollouts: rolling update with `maxUnavailable: 0` so the serving replica is never removed before its replacement is ready; startup, readiness (`/ready`: index loaded) and liveness (`/health`) probes are separate because the model takes a while to load.
- Cost of the demo: roughly $2 per day (one small node plus registry). `az group delete` removes everything.

## 3. Where the current design stops scaling, and the fix

| Constraint today | Breaks at | Fix |
|---|---|---|
| **In-process flat FAISS index**, about 6k vectors, loaded per pod | Many replicas needing the same fresh data; millions of vectors (RAM, linear scan) | Shared vector database (pgvector, Qdrant or Azure AI Search), HNSW or IVF indexes, sharded by agency, metadata filters. The `VectorStore` interface (`store.py`) is the seam. |
| **Ingestion in the serving process**, recomputing the pattern table from all tickets | Tens of millions of tickets; any multi-replica deployment | Separate ingestion worker on a schedule; incremental aggregation (counts and resolution shares as running totals in a database); embedding of new classes as an asynchronous job; serving replicas read from the shared store |
| **State baked into the image** | Frequent data updates | State in object storage with a versioned manifest; pods pull on start (init container) and on a version change; index version shown on `/ready` |
| **One LLM provider, one key, exact-match response cache on local disk** | Burst traffic, provider rate limits (a 429 was hit in testing at 200k tokens/min) | Client-side token-budget rate limiter; shared semantic cache (embedding similarity on the complaint) in Redis; smaller model for parsing; queue with backpressure; second provider as fallback |
| **Pattern-level tiers recomputed on each ingest** | Tier flapping on small samples (6.5% of patterns changed tier after 3 days of new data) | Hysteresis: change tier only when the threshold is cleared by a margin; confidence intervals on shares |
| **Single node, HPA only adds pods** | Load beyond one node | AKS cluster autoscaler (`--enable-cluster-autoscaler --min-count 1 --max-count 3`); the student quota allows about 2 nodes of this size |
| **API-key header authentication** | Multi-team or external use | Entra ID / OIDC at an ingress, per-tenant quotas and rate limits, audit trail |

## 4. Capacity and cost model (measured)

| Quantity | Measured |
|---|---|
| Latency per answer | about 4 to 5 s (retrieve 0.05 to 0.25 s, parse 1 to 2 s, generate about 3 s); abstentions about 0.05 to 0.25 s with no LLM call |
| Tokens per answered request | about 1,400 to 2,300 |
| Cost per answered request (gpt-4o-mini list price) | about $0.0004 to $0.0005; `fast_lookup` answers skip the generation call |
| Memory per pod | about 1 to 1.5 GB working set (torch + bge-small + index); request 1 GiB, limit 3 GiB |
| Throughput per pod | bounded by the LLM calls, not CPU: with a 2 vCPU pod and about 4.2 s per request, roughly 0.5 to 1 request per second per pod before the provider limit matters |

Back-of-envelope: 10,000 requests per day is about $5 per day in LLM cost and fits one or two pods; 1,000,000 per day is about $500 per day and needs the shared cache, a token limiter and a smaller parsing model first. These are estimates from per-request measurements, not load tests; a load test is still to do.

## 5. Evolving data in production

Tested: 33,773 real later tickets ingested, 8 of 8 checks, idempotent, new classes searchable without a rebuild (`docs/evolving_data.md`). Production design:
1. Scheduled job pulls newly closed tickets (a data contract on columns and the closed-with-resolution filter).
2. Clean, dedupe by key, aggregate incrementally, recompute tiers with hysteresis.
3. New classes get LLM-written example complaints and are embedded and upserted.
4. Publish a new versioned state; replicas pick it up on version change; keep the previous version for rollback.
5. Monitor the share of tickets landing in new patterns as an early signal that the taxonomy is changing.

## 6. Monitoring and alerts

`GET /metrics` exposes the signals below. Proposed alert rules (Prometheus syntax; thresholds to be tuned on real traffic):

```yaml
groups:
  - name: ticketrag
    rules:
      - alert: HighAbstainRate            # incoming complaints no longer look like the history
        expr: sum(rate(ticketrag_requests_total{outcome="abstained"}[30m])) / sum(rate(ticketrag_requests_total[30m])) > 0.25
        for: 30m
      - alert: TopSimilarityDrift         # median best-match score drifting down
        expr: histogram_quantile(0.5, sum(rate(ticketrag_top_cosine_bucket[1h])) by (le)) < 0.72
        for: 1h
      - alert: LlmDegradedMode            # answers are falling back to statistics only
        expr: sum(rate(ticketrag_generation_total{method="template_degraded"}[10m])) > 0
        for: 10m
      - alert: ValidatorRetriesRising     # the model is breaking grounding rules more often
        expr: sum(rate(ticketrag_validator_first_attempt_violations_total[1h])) / sum(rate(ticketrag_requests_total{outcome="answered"}[1h])) > 0.25
        for: 1h
      - alert: SlowAnswers
        expr: histogram_quantile(0.95, sum(rate(ticketrag_stage_latency_seconds_bucket{stage="total"}[10m])) by (le)) > 10
        for: 15m
      - alert: ErrorsPresent
        expr: sum(rate(ticketrag_errors_total[5m])) > 0
        for: 5m
```

Logs are one JSON line per request with a request ID, a SHA-1 prefix and length of the complaint (never the text), tier, method, severity, similarity and latency, so a trace can be followed without storing personal data. Quality monitoring beyond these counters (faithfulness over time) would sample answers into the judge offline.

## 7. Delivery pipeline (CI/CD)

`.github/workflows/ci.yml`: lint and tests on every push and PR (stub LLM, no secrets); a retrieval gate with the real embedder on a tiny fixture; a container smoke test that runs the built image with a dummy LLM key and asserts the degraded path works; publish of the image to GitHub Container Registry on `main`. Cluster deployment stays a manual script (`deploy/deploy_aks.ps1`) because the demo image bundles local state; in production a deploy job would roll an image tag and a state version. Rollback: `kubectl rollout undo` for the image; previous state version for data.

## 8. Failure modes and handling

| Failure | Behaviour |
|---|---|
| LLM outage, key problem or rate limit after retries | Degraded statistics-only answer, flagged in the response and UI; `template_degraded` metric |
| Nothing in the history is close | Abstain, no LLM cost, route to a human |
| Model breaks grounding rules | Retry once with feedback, drop failing items, fall back to a template if nothing remains |
| Bad or missing source text (cut off at 500 characters in the data) | Flagged in the answer and the UI |
| Network blocked between an operator and the cluster API (happened during this deployment) | `az aks command invoke` runs kubectl through Azure's management plane |
| Pod restart | Readiness gate keeps traffic away until the index is loaded; image contains the model so no download at start |

## 9. Security checklist: done and not done

Done: non-root, read-only filesystem, dropped capabilities, secrets outside the image and repo, key required on `/ask`, strict CSP on the UI, no complaint text in logs, no public endpoint by default.
Not done (production needs them): TLS and authenticated ingress, per-user authentication and quotas, PII redaction before text reaches the LLM, key rotation through a key vault, network policies, image scanning and signing in CI, dependency pinning with a lock file.

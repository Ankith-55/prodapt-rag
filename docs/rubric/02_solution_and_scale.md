# 2. Solution depth and production scale

## What was built
A stateless FastAPI service (`/ask`, `/ingest`, `/reload`, `/health`, `/ready`, `/metrics`), a hand-written UI served by the same container, and the pipeline behind it: retrieval, an abstention gate, LLM parsing, routing by ticket history, grounded generation, validation, and a deterministic fallback.

## Verified in production-like conditions
| Capability | Evidence |
|---|---|
| Container, non-root, health check, model baked in | `Dockerfile`, ran with Docker Compose |
| Kubernetes on AKS: probes, resource limits, autoscaler, private registry, secret, authenticated request answered | [`docs/evidence/aks_proof.txt`](../evidence/aks_proof.txt), [`aks_logs.txt`](../evidence/aks_logs.txt) |
| Five-stage CI/CD | `.github/workflows/ci.yml` (quality, security, test, build and verify, publish) |
| Graceful degradation | If the LLM fails the service answers from the retrieved statistics and flags it; tested |
| Evolving data | 33,773 real tickets ingested, idempotent, hot reload, new classes searchable ([`docs/evolving_data.md`](../evolving_data.md)) |

## Measured cost and speed
About 4 to 5 s and 1,400 to 2,300 tokens (about $0.0005) per answered request; abstentions return in under 0.3 s with no LLM call; per-pod working set about 1 to 1.5 GB.

## What breaks first at scale, and the fix
In-process flat FAISS index (shared vector database), ingestion inside the serving process (separate worker with incremental aggregation), state baked into the image (versioned object storage), one provider and a local cache (token limiter, shared semantic cache, fallback provider), tiers flipping on small samples (hysteresis), single node (cluster autoscaler), API-key authentication (OIDC with per-user quotas).

## Where to look
- Full document with the cost model, alert rules and a security checklist: [`docs/production_scale.md`](../production_scale.md)
- Diagrams: [`docs/architecture.md`](../architecture.md)
- Manifests and deploy script: [`deploy/`](../../deploy/)

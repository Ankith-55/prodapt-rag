# 4. Code

## Structure
| Module | Responsibility |
|---|---|
| `patterns.py` | Clean tickets, repair double-encoded text, build the pattern table and tiers |
| `retrieve.py`, `store.py`, `embed.py` | Multi-vector retrieval over a FAISS store behind an interface (upsert and remove by id) |
| `parse.py` | LLM chooses among the retrieved candidates or says none; severity and sentiment |
| `generate.py` | Numbered-source prompt, citation and number validator, retry, deterministic template |
| `pipeline.py` | The flow, tier routing, policy notes, degraded mode |
| `ingest.py` | Dedupe, refresh statistics, new classes, hot reload |
| `api.py`, `static/` | FastAPI service, metrics, structured logs, the UI |
| `llm.py`, `judge.py` | Cached, retrying LLM wrapper; independent faithfulness judge |

## Quality
- **48 tests, 84% coverage**, using a stub LLM and a fake embedder (no key, no downloads, about 6 s). They cover the validator, routing, ingestion, abstention, degraded mode, auth and the API contract. Not covered: the LLM wrapper's network paths and the eval-only judge.
- **CI gates:** ruff with a pinned rule set, hadolint, kubeconform, gitleaks, bandit, coverage floor of 80%, a retrieval gate with the real embedder, a Trivy scan, a container smoke test.
- **Reproducible:** `artifacts/` holds a ready-to-run state; `scripts/fetch_data.py` rebuilds from the official source; every eval writes a JSON report.

## Where to look
- Source: [`src/ticketrag/`](../../src/ticketrag/), tests: [`tests/`](../../tests/), workflow: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
- How to run and read the output: [`docs/testing_guide.md`](../testing_guide.md)

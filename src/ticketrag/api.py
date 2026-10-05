"""FastAPI service.

  uvicorn ticketrag.api:app --host 0.0.0.0 --port 8000

Config (environment): TICKETRAG_STATE (default data/processed if built, else artifacts/state), TICKETRAG_INDEX (default <state>/index),
TICKETRAG_GATE (0.74), TICKETRAG_K (5), TICKETRAG_API_KEY (if set, /ask /ingest /reload require X-API-Key),
OPENAI_API_KEY, OPENAI_MODEL.

Design notes
- Endpoints are synchronous (`def`), so FastAPI runs them in a thread pool; the LLM calls never block the event loop.
- /ask degrades to a retrieval-only answer if the LLM path fails (ask_safe) instead of returning a 500.
- Raw complaint text is never logged (it can contain PII); only a hash prefix and length.
- /metrics exposes Prometheus counters/histograms for abstain rate, tier mix, latency per stage, token cost and
  top-cosine distribution (the main drift signals)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from ticketrag.ingest import apply_batch
from ticketrag.paths import default_state_dir
from ticketrag.pipeline import Answer, Assistant

log = logging.getLogger("ticketrag")
REQUIRED_TICKET_COLUMNS = ["unique_key", "created_date", "closed_date", "complaint_type", "descriptor",
                           "descriptor_2", "resolution_description"]


class AskRequest(BaseModel):
    complaint: str = Field(min_length=5, max_length=2000, description="raw customer complaint")


class AskResponse(BaseModel):
    request_id: str
    abstained: bool
    abstain_reason: str | None = None
    top_score: float = 0.0
    category: str | None = None
    product: str | None = None
    severity: int | None = None
    sentiment: str | None = None
    tier: str | None = None
    method: str | None = None
    answer: dict[str, Any] | None = None
    sources: dict[str, Any] = {}
    policy_notes: list[str] = []
    candidates: list[Any] = []
    grounding: dict[str, Any] = {}
    timing_ms: dict[str, Any] = {}
    llm_usage: dict[str, Any] = {}


class IngestRequest(BaseModel):
    tickets: list[dict[str, Any]] = Field(min_length=1, max_length=50000)


class Metrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.requests = Counter("ticketrag_requests_total", "Requests by outcome", ["outcome", "reason"], registry=r)
        self.tiers = Counter("ticketrag_tier_total", "Answered requests by routing tier", ["tier"], registry=r)
        self.methods = Counter("ticketrag_generation_total", "Answer method (llm/template/template_degraded)",
                               ["method"], registry=r)
        self.violations = Counter("ticketrag_validator_first_attempt_violations_total",
                                  "LLM answers that needed a retry or lost items to the validator", registry=r)
        self.latency = Histogram("ticketrag_stage_latency_seconds", "Latency by stage", ["stage"], registry=r,
                                 buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21))
        self.tokens = Counter("ticketrag_llm_tokens_total", "LLM tokens (best effort under concurrency)", ["kind"],
                              registry=r)
        self.top_cosine = Histogram("ticketrag_top_cosine", "Top-1 cosine similarity of incoming complaints",
                                    registry=r, buckets=(0.5, 0.55, 0.6, 0.65, 0.7, 0.74, 0.8, 0.85, 0.9, 0.95))
        self.severity = Counter("ticketrag_severity_total", "Answered requests by severity", ["level"], registry=r)
        self.errors = Counter("ticketrag_errors_total", "Unhandled errors", ["where"], registry=r)
        self.ingested = Counter("ticketrag_ingested_tickets_total", "Tickets ingested", registry=r)
        self.new_patterns = Counter("ticketrag_new_patterns_total", "New ticket classes created by ingestion", registry=r)
        self.index_vectors = Gauge("ticketrag_index_vectors", "Vectors in the index", registry=r)
        self.patterns = Gauge("ticketrag_patterns", "Patterns in the pattern table", registry=r)

    def record(self, a: Answer) -> None:
        reason = (a.abstain_reason or "").split(" ")[0] or "none"
        self.requests.labels("abstained" if a.abstained else "answered", reason if a.abstained else "none").inc()
        self.top_cosine.observe(a.top_score)
        for stage, ms in a.timing_ms.items():
            self.latency.labels(stage).observe(ms / 1000)
        for kind in ("prompt_tokens", "completion_tokens"):
            self.tokens.labels(kind).inc(a.llm_usage.get(kind, 0))
        if not a.abstained:
            self.tiers.labels(a.tier or "unknown").inc()
            self.methods.labels(a.method or "unknown").inc()
            if a.severity is not None:
                self.severity.labels(str(a.severity)).inc()
            if a.grounding.get("first_attempt_violations", 0) > 0:
                self.violations.inc()


def _build_assistant() -> Assistant:
    from ticketrag.llm import LLM
    from ticketrag.retrieve import PatternRetriever

    state = str(default_state_dir())
    index = os.getenv("TICKETRAG_INDEX", f"{state}/index")
    retriever = PatternRetriever(index, state)
    return Assistant(retriever, LLM(), gate=float(os.getenv("TICKETRAG_GATE", "0.74")),
                     k=int(os.getenv("TICKETRAG_K", "5")), siblings=int(os.getenv("TICKETRAG_SIBLINGS", "0")))


def create_app(assistant: Assistant | None = None) -> FastAPI:
    """`assistant` can be injected (tests use a stub LLM); otherwise it is built at startup from the environment."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    metrics = Metrics()
    holder: dict[str, Any] = {"assistant": assistant}
    ingest_lock = threading.Lock()  # one ingestion at a time; searches keep serving the old state meanwhile
    api_key = os.getenv("TICKETRAG_API_KEY")
    immutable = os.getenv("TICKETRAG_IMMUTABLE") == "1"  # baked-in state: live ingestion would diverge across replicas

    def update_gauges() -> None:
        a = holder["assistant"]
        if a is not None:
            metrics.index_vectors.set(len(a.retriever.store))
            metrics.patterns.set(len(a.retriever.patterns))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if holder["assistant"] is None:
            holder["assistant"] = _build_assistant()
        update_gauges()
        yield

    app = FastAPI(title="Support ticket resolution assistant", version="0.1.0", lifespan=lifespan)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key and x_api_key != api_key:
            raise HTTPException(401, "invalid or missing X-API-Key")

    def get_assistant() -> Assistant:
        if holder["assistant"] is None:
            raise HTTPException(503, "service is starting")
        return holder["assistant"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        a = get_assistant()
        return {"status": "ready", "patterns": len(a.retriever.patterns), "vectors": len(a.retriever.store),
                "embedder": a.retriever.embedder.model_name, "llm": a.llm.model, "gate": a.gate}

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
    def ask(req: AskRequest, response: Response, x_request_id: str | None = Header(default=None)) -> AskResponse:
        a = get_assistant()
        rid = (x_request_id or uuid.uuid4().hex)[:32]
        complaint = " ".join(req.complaint.split())
        t0 = time.perf_counter()
        try:
            result = a.ask_safe(complaint)
        except Exception:  # noqa: BLE001
            metrics.errors.labels("ask").inc()
            log.exception(json.dumps({"event": "ask_error", "request_id": rid}))
            raise HTTPException(500, "internal error") from None
        metrics.record(result)
        log.info(json.dumps({
            "event": "ask", "request_id": rid, "complaint_sha1": hashlib.sha1(complaint.encode()).hexdigest()[:10],
            "complaint_chars": len(complaint), "abstained": result.abstained, "reason": result.abstain_reason,
            "tier": result.tier, "method": result.method, "severity": result.severity, "top_score": round(result.top_score, 3),
            "latency_ms": round((time.perf_counter() - t0) * 1000), "tokens": result.llm_usage.get("prompt_tokens", 0)
            + result.llm_usage.get("completion_tokens", 0)}))
        response.headers["X-Request-ID"] = rid
        body = asdict(result)
        body.pop("complaint")  # never echo raw customer text
        return AskResponse(request_id=rid, **body)

    @app.post("/ingest", dependencies=[Depends(require_key)])
    def ingest(req: IngestRequest) -> dict:
        if immutable:
            raise HTTPException(403, "state is immutable in this deployment: ingest offline and roll out a new state image")
        a = get_assistant()
        df = pd.DataFrame(req.tickets).astype(str)
        missing = [c for c in REQUIRED_TICKET_COLUMNS if c not in df.columns]
        if missing:
            raise HTTPException(422, f"missing columns: {missing}")
        with ingest_lock:
            report = apply_batch(a.retriever.processed_dir, df, a.retriever.embedder, a.llm)
            a.retriever.reload()  # atomic swap; in-flight searches are not disturbed
        update_gauges()
        metrics.ingested.inc(report["ingested"])
        metrics.new_patterns.inc(len(report["new_patterns"]))
        report.pop("filter", None)
        return report

    @app.post("/reload", dependencies=[Depends(require_key)])
    def reload() -> dict:
        a = get_assistant()
        with ingest_lock:
            a.retriever.reload()
        update_gauges()
        return {"patterns": len(a.retriever.patterns), "vectors": len(a.retriever.store)}

    @app.get("/metrics")
    def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    ui_paths = {"/", "/index.html", "/app.js", "/style.css"}

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if request.url.path in ui_paths:  # strict CSP for the UI only; /docs (Swagger) loads assets from a CDN
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        return response

    # Registered last so API routes win; serves the hand-written UI (no build step) at "/".
    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
    return app


app = create_app()

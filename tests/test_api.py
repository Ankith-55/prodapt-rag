import pytest
from fastapi.testclient import TestClient
from helpers import make_raw

from ticketrag.api import create_app


@pytest.fixture
def client(assistant):
    return TestClient(create_app(assistant))


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/ready").json()
    assert body["status"] == "ready" and body["patterns"] == 3 and body["llm"] == "stub"


def test_ask_returns_answer_and_never_echoes_the_complaint(client):
    r = client.post("/ask", json={"complaint": "loud music party noise next door"}, headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200 and r.headers["X-Request-ID"] == "abc123"
    body = r.json()
    assert body["request_id"] == "abc123" and not body["abstained"] and body["category"] == "Noise - Residential"
    assert "complaint" not in body and body["answer"]["steps"]


def test_ask_abstains_on_unrelated_text(client):
    body = client.post("/ask", json={"complaint": "quantum espresso invoice"}).json()
    assert body["abstained"] and body["answer"] is None and body["candidates"]


def test_ask_validates_input(client):
    assert client.post("/ask", json={"complaint": "hi"}).status_code == 422
    assert client.post("/ask", json={"complaint": "x" * 2001}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_api_key_is_enforced_when_configured(assistant, monkeypatch):
    monkeypatch.setenv("TICKETRAG_API_KEY", "secret")
    c = TestClient(create_app(assistant))
    payload = {"complaint": "loud music party noise next door"}
    assert c.post("/ask", json=payload).status_code == 401
    assert c.post("/ask", json=payload, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/ask", json=payload, headers={"X-API-Key": "secret"}).status_code == 200
    assert c.get("/health").status_code == 200  # probes stay open


def test_llm_outage_returns_degraded_answer_not_500(client, stub):
    stub.fail = True
    r = client.post("/ask", json={"complaint": "loud music party noise next door"})
    assert r.status_code == 200 and r.json()["method"] == "template_degraded"


def test_metrics_reflect_traffic(client):
    client.post("/ask", json={"complaint": "loud music party noise next door"})
    client.post("/ask", json={"complaint": "quantum espresso invoice"})
    text = client.get("/metrics").text
    assert 'ticketrag_requests_total{outcome="answered",reason="none"} 1.0' in text
    assert 'ticketrag_requests_total{outcome="abstained",reason="low_similarity"} 1.0' in text
    assert 'ticketrag_tier_total{tier="rag_core"} 1.0' in text
    assert "ticketrag_index_vectors" in text and "ticketrag_top_cosine_bucket" in text


def test_ingest_adds_a_class_and_serves_it_immediately(client):
    spec = [(("Broadband Service", "Outage", ""), [(1.0, "Provider restored the line remotely.")], 35)]
    tickets = make_raw(spec, prefix="N").drop(columns=["status"]).to_dict("records")
    r = client.post("/ingest", json={"tickets": tickets})
    assert r.status_code == 200 and len(r.json()["new_patterns"]) == 1
    assert client.get("/ready").json()["patterns"] == 4
    body = client.post("/ask", json={"complaint": "broadband service outage"}).json()
    assert body["category"] == "Broadband Service"


def test_ingest_rejects_missing_columns(client):
    assert client.post("/ingest", json={"tickets": [{"unique_key": "1"}]}).status_code == 422


def test_ui_is_served_with_strict_csp_and_api_routes_still_win(client):
    page = client.get("/")
    assert page.status_code == 200 and "Resolution Assistant" in page.text
    assert "script-src" not in page.headers["content-security-policy"] and "default-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/app.js").status_code == 200 and client.get("/style.css").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}  # mounting "/" did not shadow API routes
    assert "content-security-policy" not in client.get("/health").headers

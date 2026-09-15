import uuid

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    with TestClient(main.app) as c:  # context manager runs the startup hook
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["cache_backend"] == "sqlite"


def test_analyze_invalid_body_returns_422(client):
    assert client.post("/analyze", json={"body": "no title"}).status_code == 422
    assert client.post("/analyze", json={"title": ""}).status_code == 422


def test_analyze_second_call_is_cache_hit(client, fake_analyzer):
    req = {"title": "Headline", "body": "text", "url": f"https://example.com/{uuid.uuid4()}"}

    first = client.post("/analyze", json=req)
    assert first.status_code == 200
    assert first.headers["X-Cache"] == "MISS"
    assert first.json()["cached"] is False

    second = client.post("/analyze", json=req)
    assert second.headers["X-Cache"] == "HIT"
    assert second.json()["cached"] is True
    assert second.json()["clickbait_pct"] == first.json()["clickbait_pct"]
    assert fake_analyzer.calls == 1


def test_metrics_keys_and_increment(client):
    before = client.get("/metrics").json()
    assert set(before) == {"requests", "latency_ms_24h", "predictions_24h"}
    assert {"total", "cache_hits", "cache_misses", "cache_hit_rate"} <= set(before["requests"])

    client.get("/health")
    after = client.get("/metrics").json()
    assert after["requests"]["total"] > before["requests"]["total"]


def test_related_with_mocked_fetcher(client, monkeypatch):
    articles = [
        {"title": "Left take.", "url": "https://l.example", "source": "L", "political_score": -0.6,
         "sentiment_score": -0.3, "snippet": "Left snippet."},
        {"title": "Right take.", "url": "https://r.example", "source": "R", "political_score": -0.5,
         "sentiment_score": -0.4, "snippet": "Right snippet."},
    ]
    calls = []
    monkeypatch.setattr(main, "fetch_related", lambda **kw: calls.append(kw) or articles)

    topic = f"topic {uuid.uuid4().hex[:8]}"
    r = client.get("/related", params={"topic": topic, "political_score": -0.5, "sentiment_score": -0.3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["articles"]) == 2
    assert body["dominant_leaning"] == "left"
    assert body["dominant_pct"] == 100.0
    assert body["case"] == "echo_chamber"
    assert calls[0]["topic"] == topic

    # second call served from cache, fetcher not called again
    assert client.get("/related", params={"topic": topic, "political_score": -0.5,
                                          "sentiment_score": -0.3}).headers["X-Cache"] == "HIT"
    assert len(calls) == 1

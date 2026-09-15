import hashlib
import time

import cache as cache_mod
from cache import ArticleCache, _hash, get_cache

DATA = {"clickbait_pct": 73.5, "political_score": -0.4, "sentiment_score": -0.6, "case": "echo_chamber"}


def test_hash_is_stable_sha256():
    url = "https://example.com/a"
    assert _hash(url) == _hash(url) == hashlib.sha256(url.encode()).hexdigest()
    assert _hash(url) != _hash(url + "b")


def test_set_then_get_returns_value(tmp_path):
    c = ArticleCache(str(tmp_path / "c.db"))
    c.set_analysis("https://example.com/1", DATA)
    assert c.get_analysis("https://example.com/1") == {**DATA, "cached": True}
    assert c.get_analysis("https://example.com/missing") is None

    c.set_related("topic", {"summary": "s"})
    assert c.get_related("topic") == {"summary": "s"}


def test_expired_ttl_is_a_miss(tmp_path, monkeypatch):
    c = ArticleCache(str(tmp_path / "c.db"))
    c.set_analysis("https://example.com/old", DATA)
    c.set_related("old topic", {"summary": "s"})

    future = time.time() + cache_mod.TTL_SECONDS + 1
    monkeypatch.setattr(cache_mod.time, "time", lambda: future)
    assert c.get_analysis("https://example.com/old") is None
    assert c.get_related("old topic") is None


def test_get_cache_falls_back_to_local_without_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert type(get_cache()) is ArticleCache

"""
cache.py — SQLite 24-hour Caching Layer
========================================

Two tables:
    article_analyses  — caches /analyze results by URL hash
    related_articles  — caches /related results by topic hash

Entries expire after 24 hours. Stale entries are purged on startup.
"""

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

import redis

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DB = os.getenv(
    "CACHE_DB",
    os.path.join(_BACKEND_DIR, "..", "cache", "unblur.db"),
)

TTL_SECONDS = 24 * 60 * 60  # 24 hours


# ═══════════════════════════════════════════════════════════════
#  DB HELPERS
# ═══════════════════════════════════════════════════════════════

@contextmanager
def _get_conn(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════
#  ArticleCache
# ═══════════════════════════════════════════════════════════════

class ArticleCache:
    """
    SQLite-backed cache for analyzed articles and related-article batches.

    Usage
    -----
    cache = ArticleCache()           # uses DEFAULT_CACHE_DB
    cache = ArticleCache("/tmp/t.db")

    # Store / retrieve analysis
    cache.set_analysis(url, data)
    result = cache.get_analysis(url)   # None if missing/expired

    # Store / retrieve related articles
    cache.set_related(topic, data)
    result = cache.get_related(topic)  # None if missing/expired
    """

    def __init__(self, db_path: str = DEFAULT_CACHE_DB):
        self._db = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self._db), exist_ok=True)
        self._init_db()
        self._purge_expired()

    # ── Schema ───────────────────────────────────────────────────
    def _init_db(self) -> None:
        with _get_conn(self._db) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS article_analyses (
                    url_hash        TEXT PRIMARY KEY,
                    url             TEXT,
                    title           TEXT,
                    clickbait_pct   REAL,
                    political_score REAL,
                    sentiment_score REAL,
                    case_label      TEXT,
                    created_at      REAL
                );

                CREATE TABLE IF NOT EXISTS related_articles (
                    topic_hash  TEXT PRIMARY KEY,
                    topic       TEXT,
                    payload     TEXT,   -- JSON blob
                    created_at  REAL
                );
            """)

    # ── Expiry cleanup ───────────────────────────────────────────
    def _purge_expired(self) -> None:
        cutoff = time.time() - TTL_SECONDS
        with _get_conn(self._db) as conn:
            conn.execute(
                "DELETE FROM article_analyses WHERE created_at < ?", (cutoff,)
            )
            conn.execute(
                "DELETE FROM related_articles WHERE created_at < ?", (cutoff,)
            )

    # ── Article analysis cache ───────────────────────────────────
    def get_analysis(self, url: str) -> Optional[dict]:
        """Return cached analysis for url, or None if absent/expired."""
        key = _hash(url)
        cutoff = time.time() - TTL_SECONDS
        with _get_conn(self._db) as conn:
            row = conn.execute(
                "SELECT * FROM article_analyses WHERE url_hash = ? AND created_at >= ?",
                (key, cutoff),
            ).fetchone()
        if row is None:
            return None
        return {
            "clickbait_pct":    row["clickbait_pct"],
            "political_score":  row["political_score"],
            "sentiment_score":  row["sentiment_score"],
            "case":             row["case_label"],
            "cached":           True,
        }

    def set_analysis(self, url: str, data: dict) -> None:
        """Store analysis result for url."""
        key = _hash(url)
        with _get_conn(self._db) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO article_analyses
                    (url_hash, url, title, clickbait_pct, political_score,
                     sentiment_score, case_label, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    url,
                    data.get("title", ""),
                    data.get("clickbait_pct"),
                    data.get("political_score"),
                    data.get("sentiment_score"),
                    data.get("case", ""),
                    time.time(),
                ),
            )

    # ── Related articles cache ───────────────────────────────────
    def get_related(self, topic: str) -> Optional[dict]:
        """Return cached related-articles payload, or None if absent/expired."""
        key = _hash(topic)
        cutoff = time.time() - TTL_SECONDS
        with _get_conn(self._db) as conn:
            row = conn.execute(
                "SELECT payload FROM related_articles WHERE topic_hash = ? AND created_at >= ?",
                (key, cutoff),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload"])

    def set_related(self, topic: str, data: dict) -> None:
        """Store related-articles payload for topic."""
        key = _hash(topic)
        with _get_conn(self._db) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO related_articles
                    (topic_hash, topic, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, topic, json.dumps(data), time.time()),
            )


# ═══════════════════════════════════════════════════════════════
#  RedisArticleCache
# ═══════════════════════════════════════════════════════════════

class RedisArticleCache:
    """
    Redis-backed cache for analyzed articles and related-article batches.

    Same interface as ArticleCache (SQLite): get_analysis/set_analysis/
    get_related/set_related, same 24h TTL, same key scheme (sha256 of
    url/topic). Redis's native EX/SETEX handles expiry — no manual purge.

    Usage
    -----
    cache = RedisArticleCache("redis://localhost:6379/0")
    cache = RedisArticleCache("rediss://default:pw@host:6379")   # Upstash TLS
    """

    def __init__(self, redis_url: str):
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    # ── Article analysis cache ───────────────────────────────────
    def get_analysis(self, url: str) -> Optional[dict]:
        """Return cached analysis for url, or None if absent/expired."""
        raw = self._client.get(f"analysis:{_hash(url)}")
        if raw is None:
            return None
        data = json.loads(raw)
        return {
            "clickbait_pct":   data.get("clickbait_pct"),
            "political_score": data.get("political_score"),
            "sentiment_score": data.get("sentiment_score"),
            "case":            data.get("case", ""),
            "cached":          True,
        }

    def set_analysis(self, url: str, data: dict) -> None:
        """Store analysis result for url."""
        payload = {
            "clickbait_pct":   data.get("clickbait_pct"),
            "political_score": data.get("political_score"),
            "sentiment_score": data.get("sentiment_score"),
            "case":            data.get("case", ""),
        }
        self._client.set(f"analysis:{_hash(url)}", json.dumps(payload), ex=TTL_SECONDS)

    # ── Related articles cache ───────────────────────────────────
    def get_related(self, topic: str) -> Optional[dict]:
        """Return cached related-articles payload, or None if absent/expired."""
        raw = self._client.get(f"related:{_hash(topic)}")
        if raw is None:
            return None
        return json.loads(raw)

    def set_related(self, topic: str, data: dict) -> None:
        """Store related-articles payload for topic."""
        self._client.set(f"related:{_hash(topic)}", json.dumps(data), ex=TTL_SECONDS)


# ═══════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = ArticleCache(db_path)

        # Analysis round-trip
        url  = "https://example.com/article/123"
        data = {"clickbait_pct": 73.5, "political_score": -0.4,
                "sentiment_score": -0.6, "case": "echo_chamber", "title": "Test"}
        cache.set_analysis(url, data)
        result = cache.get_analysis(url)
        assert result is not None
        assert result["clickbait_pct"] == 73.5
        assert result["cached"] is True
        print("✓ article_analyses round-trip passed")

        # Related round-trip
        topic   = "immigration policy"
        related = {"articles": [{"title": "Test"}], "summary": "s", "dominant_leaning": "left", "dominant_pct": 60.0}
        cache.set_related(topic, related)
        result2 = cache.get_related(topic)
        assert result2 is not None
        assert result2["dominant_leaning"] == "left"
        print("✓ related_articles round-trip passed")

        # Miss
        assert cache.get_analysis("https://notcached.com") is None
        print("✓ cache miss returns None")

    print("\n✓ All ArticleCache (SQLite) tests passed.")

    # ── RedisArticleCache smoke test (requires REDIS_URL) ────────
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        print("\nREDIS_URL not set — skipping RedisArticleCache smoke test.")
        sys.exit(0)

    rcache = RedisArticleCache(redis_url)

    r_url  = "https://example.com/article/redis-123"
    r_data = {"clickbait_pct": 42.0, "political_score": 0.2,
              "sentiment_score": 0.1, "case": "balanced", "title": "Redis Test"}
    rcache.set_analysis(r_url, r_data)
    r_result = rcache.get_analysis(r_url)
    assert r_result is not None
    assert r_result["clickbait_pct"] == 42.0
    assert r_result["cached"] is True
    print("✓ [Redis] article_analyses round-trip passed")

    r_topic   = "immigration policy"
    r_related = {"articles": [{"title": "Test"}], "summary": "s", "dominant_leaning": "right", "dominant_pct": 55.0}
    rcache.set_related(r_topic, r_related)
    r_result2 = rcache.get_related(r_topic)
    assert r_result2 is not None
    assert r_result2["dominant_leaning"] == "right"
    print("✓ [Redis] related_articles round-trip passed")

    assert rcache.get_analysis("https://notcached-redis.com") is None
    print("✓ [Redis] cache miss returns None")

    print("\n✓ All RedisArticleCache tests passed.")

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
#  Smoke test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
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

    print("\n✓ All cache tests passed.")

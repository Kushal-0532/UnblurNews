"""
metrics.py — Lightweight SQLite-backed Request & Prediction Metrics
====================================================================

Tracks per-endpoint latency, cache hit/miss rate, request counts, and
prediction score distributions without any external dependencies.

Design choice: SQLite (stdlib) over Prometheus/StatsD
  - Zero extra dependencies; ships in the same DB as the cache
  - Sufficient for a single-instance service; upgrade to Prometheus
    when horizontal scaling is needed (the /metrics JSON schema is
    intentionally Prometheus-compatible for a future migration)

Usage
-----
metrics = MetricsStore()
metrics.record_request("/analyze", latency_ms=142.3, cache_hit=False)
metrics.record_prediction(clickbait_pct=72.1, political_score=-0.4, sentiment_score=-0.3)
stats = metrics.get_stats()
"""

import os
import sqlite3
import statistics
import time
from contextlib import contextmanager
from typing import Optional

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METRICS_DB = os.getenv(
    "METRICS_DB",
    os.path.join(_BACKEND_DIR, "..", "cache", "unblur_metrics.db"),
)

# Keep 7 days of raw request rows, then summarise
RAW_TTL_SECONDS = 7 * 24 * 60 * 60


@contextmanager
def _conn(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


class MetricsStore:
    """
    Thread-safe (WAL mode) metrics store.

    Tables
    ------
    requests    — one row per API call (endpoint, latency_ms, cache_hit, ts)
    predictions — one row per model inference (scores, ts)
    """

    def __init__(self, db_path: str = DEFAULT_METRICS_DB):
        self._db = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self._db), exist_ok=True)
        self._init()
        self._purge_old()

    def _init(self) -> None:
        with _conn(self._db) as c:
            c.executescript("""
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS requests (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint    TEXT    NOT NULL,
                    latency_ms  REAL    NOT NULL,
                    cache_hit   INTEGER NOT NULL DEFAULT 0,
                    status_code INTEGER NOT NULL DEFAULT 200,
                    ts          REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    clickbait_pct   REAL NOT NULL,
                    political_score REAL NOT NULL,
                    sentiment_score REAL NOT NULL,
                    ts              REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_req_endpoint ON requests(endpoint);
                CREATE INDEX IF NOT EXISTS idx_req_ts       ON requests(ts);
                CREATE INDEX IF NOT EXISTS idx_pred_ts      ON predictions(ts);
            """)

    def _purge_old(self) -> None:
        cutoff = time.time() - RAW_TTL_SECONDS
        with _conn(self._db) as c:
            c.execute("DELETE FROM requests    WHERE ts < ?", (cutoff,))
            c.execute("DELETE FROM predictions WHERE ts < ?", (cutoff,))

    # ── Write ────────────────────────────────────────────────────

    def record_request(
        self,
        endpoint: str,
        latency_ms: float,
        cache_hit: bool = False,
        status_code: int = 200,
    ) -> None:
        with _conn(self._db) as c:
            c.execute(
                "INSERT INTO requests (endpoint, latency_ms, cache_hit, status_code, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (endpoint, latency_ms, int(cache_hit), status_code, time.time()),
            )

    def record_prediction(
        self,
        clickbait_pct: float,
        political_score: float,
        sentiment_score: float,
    ) -> None:
        with _conn(self._db) as c:
            c.execute(
                "INSERT INTO predictions (clickbait_pct, political_score, sentiment_score, ts) "
                "VALUES (?, ?, ?, ?)",
                (clickbait_pct, political_score, sentiment_score, time.time()),
            )

    # ── Read / aggregation ───────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Return a snapshot of current metrics.

        Percentiles are computed over the last 24 h of data.
        Histograms use fixed buckets to allow trend monitoring
        (sudden bucket shift = potential distribution drift).
        """
        now = time.time()
        window_24h = now - 86_400

        with _conn(self._db) as c:
            # ── Request counts & cache hit rate ──────────────────
            total = c.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            hits  = c.execute(
                "SELECT COUNT(*) FROM requests WHERE cache_hit = 1"
            ).fetchone()[0]

            # ── Per-endpoint latency (last 24 h) ─────────────────
            endpoints = [
                r["endpoint"]
                for r in c.execute(
                    "SELECT DISTINCT endpoint FROM requests"
                ).fetchall()
            ]
            latency_stats: dict = {}
            for ep in endpoints:
                rows = c.execute(
                    "SELECT latency_ms FROM requests WHERE endpoint = ? AND ts >= ?",
                    (ep, window_24h),
                ).fetchall()
                vals = sorted(r["latency_ms"] for r in rows)
                if vals:
                    latency_stats[ep] = _percentiles(vals)

            # ── Error rate ────────────────────────────────────────
            errors = c.execute(
                "SELECT COUNT(*) FROM requests WHERE status_code >= 500"
            ).fetchone()[0]

            # ── Prediction distribution (last 24 h) ───────────────
            pred_rows = c.execute(
                "SELECT clickbait_pct, political_score, sentiment_score "
                "FROM predictions WHERE ts >= ?",
                (window_24h,),
            ).fetchall()

        pred_count = len(pred_rows)
        cb_dist = pol_dist = sent_dist = {}
        if pred_count:
            cb_vals   = [r["clickbait_pct"]   for r in pred_rows]
            pol_vals  = [r["political_score"]  for r in pred_rows]
            sent_vals = [r["sentiment_score"]  for r in pred_rows]

            cb_dist   = _score_histogram(cb_vals,  buckets=[(0,33),(33,67),(67,100)])
            pol_dist  = _score_histogram(pol_vals, buckets=[(-1,-0.3),(-0.3,0.3),(0.3,1)])
            sent_dist = _score_histogram(sent_vals,buckets=[(-1,-0.2),(-0.2,0.2),(0.2,1)])

        return {
            "requests": {
                "total":         total,
                "cache_hits":    hits,
                "cache_misses":  total - hits,
                "cache_hit_rate": round(hits / total * 100, 1) if total else 0.0,
                "errors_5xx":    errors,
                "error_rate_pct": round(errors / total * 100, 1) if total else 0.0,
            },
            "latency_ms_24h": latency_stats,
            "predictions_24h": {
                "count": pred_count,
                "clickbait_distribution":  cb_dist,
                "political_distribution":  pol_dist,
                "sentiment_distribution":  sent_dist,
            },
        }


# ── Helpers ──────────────────────────────────────────────────────

def _percentiles(sorted_vals: list[float]) -> dict:
    n = len(sorted_vals)
    return {
        "count": n,
        "p50":   round(_pct(sorted_vals, 50), 1),
        "p95":   round(_pct(sorted_vals, 95), 1),
        "p99":   round(_pct(sorted_vals, 99), 1),
        "mean":  round(statistics.mean(sorted_vals), 1),
        "min":   round(sorted_vals[0], 1),
        "max":   round(sorted_vals[-1], 1),
    }


def _pct(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile (same as numpy default)."""
    if not sorted_vals:
        return 0.0
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _score_histogram(
    vals: list[float],
    buckets: list[tuple[float, float]],
) -> dict:
    """Count values falling into each labelled bucket."""
    total = len(vals)
    result = {}
    for lo, hi in buckets:
        label = f"{lo}_to_{hi}"
        count = sum(1 for v in vals if lo <= v < hi)
        result[label] = {"count": count, "pct": round(count / total * 100, 1)}
    return result


# ── Smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import json, tempfile

    with tempfile.TemporaryDirectory() as tmp:
        m = MetricsStore(os.path.join(tmp, "test_metrics.db"))

        for i in range(10):
            m.record_request("/analyze",  latency_ms=100 + i*10, cache_hit=(i % 3 == 0))
            m.record_request("/related",  latency_ms=200 + i*5,  cache_hit=(i % 4 == 0))
            m.record_prediction(
                clickbait_pct=20 + i*5,
                political_score=-0.5 + i*0.1,
                sentiment_score=-0.3 + i*0.06,
            )

        stats = m.get_stats()
        print(json.dumps(stats, indent=2))
        assert stats["requests"]["total"] == 20
        assert stats["requests"]["cache_hits"] > 0
        assert stats["predictions_24h"]["count"] == 10
        print("\n✓ MetricsStore smoke test passed.")

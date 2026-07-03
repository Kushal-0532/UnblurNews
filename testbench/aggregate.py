"""Turns raw.csv into summary stats: percentiles, hit rate, error rate."""
from pathlib import Path

import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _percentiles(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"p50": None, "p95": None, "p99": None, "count": 0, "error_rate_pct": None}
    q = df["latency_ms"].quantile([.5, .95, .99])
    return {
        "p50": q[.5],
        "p95": q[.95],
        "p99": q[.99],
        "count": len(df),
        "error_rate_pct": (~df["success"]).mean() * 100,
    }


def summarize(df: pd.DataFrame) -> dict:
    overall = _percentiles(df)

    by_cache_status = {}
    if not df.empty:
        for status, group in df.dropna(subset=["cache_status"]).groupby("cache_status"):
            by_cache_status[status] = _percentiles(group)

    by_topic_hit_rate = {}
    if not df.empty:
        cached = df.dropna(subset=["cache_status"])
        for topic, group in cached.groupby("topic"):
            by_topic_hit_rate[topic] = (group["cache_status"] == "HIT").mean()

    by_concurrency = {}
    if not df.empty:
        for concurrency, group in df.groupby("concurrency"):
            by_concurrency[concurrency] = {
                "p95": group["latency_ms"].quantile(.95),
                "error_rate_pct": (~group["success"]).mean() * 100,
            }

    return {
        "overall": overall,
        "by_cache_status": by_cache_status,
        "by_topic_hit_rate": by_topic_hit_rate,
        "by_concurrency": by_concurrency,
    }

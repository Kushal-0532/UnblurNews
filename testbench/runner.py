"""Wires simulated users + backend client + recorder into a concurrent load run.

CLI: python testbench/runner.py --scenario mixed --users 50 --concurrency 50
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from testbench.aggregate import summarize
from testbench.charts import generate_all_charts
from testbench.client import BackendClient
from testbench.corpus.loader import group_by_topic, load_corpus
from testbench.recorder import RequestRecord, ResultRecorder
from testbench.users import TOPIC_WEIGHTS, make_users


async def run_session(session: list[dict], client: BackendClient, recorder: ResultRecorder,
                       semaphore: asyncio.Semaphore, concurrency_label: int):
    for article in session:
        async with semaphore:
            analyze_result = await client.analyze(article)
        recorder.add(RequestRecord(
            timestamp=time.time(),
            endpoint=analyze_result.endpoint,
            topic=article["topic"],
            cache_status=analyze_result.cache_status,
            latency_ms=analyze_result.latency_ms,
            success=analyze_result.success,
            status_code=analyze_result.status_code,
            error=analyze_result.error,
            concurrency=concurrency_label,
        ))

        scores = analyze_result.data or {}
        political_score = scores.get("political_score", 0.0)
        sentiment_score = scores.get("sentiment_score", 0.0)

        async with semaphore:
            related_result = await client.related(article["topic"], political_score, sentiment_score)
        recorder.add(RequestRecord(
            timestamp=time.time(),
            endpoint=related_result.endpoint,
            topic=article["topic"],
            cache_status=related_result.cache_status,
            latency_ms=related_result.latency_ms,
            success=related_result.success,
            status_code=related_result.status_code,
            error=related_result.error,
            concurrency=concurrency_label,
        ))


async def run_load_test(sessions: list[list[dict]], base_url: str, concurrency: int, recorder: ResultRecorder = None) -> ResultRecorder:
    recorder = recorder if recorder is not None else ResultRecorder()
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as hc:
        client = BackendClient(hc)
        await asyncio.gather(*[
            run_session(session, client, recorder, semaphore, concurrency)
            for session in sessions
        ])
    return recorder


def _print_summary(summary: dict):
    o = summary["overall"]
    print(f"\nOverall: n={o['count']} p50={o['p50']:.1f}ms p95={o['p95']:.1f}ms "
          f"p99={o['p99']:.1f}ms error_rate={o['error_rate_pct']:.1f}%")
    for status, stats in summary["by_cache_status"].items():
        print(f"  {status}: n={stats['count']} p50={stats['p50']:.1f}ms p95={stats['p95']:.1f}ms")
    print("Cache hit rate by topic:")
    for topic, rate in summary["by_topic_hit_rate"].items():
        print(f"  {topic}: {rate * 100:.1f}%")
    if summary["by_concurrency"]:
        print("By concurrency level:")
        for level, stats in sorted(summary["by_concurrency"].items()):
            print(f"  {level}: p95={stats['p95']:.1f}ms error_rate={stats['error_rate_pct']:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(TOPIC_WEIGHTS.keys()), default="mixed")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--concurrency-levels", default=None,
                         help="comma-separated concurrency levels, e.g. 1,10,50; overrides --concurrency")
    parser.add_argument("--session-min", type=int, default=3)
    parser.add_argument("--session-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--p95-threshold-ms", type=float, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("testbench/results") / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")

    grouped = group_by_topic(load_corpus())
    users = make_users(args.users, args.scenario, args.seed, (args.session_min, args.session_max))
    sessions = [u.generate_session(grouped) for u in users]

    levels = [int(x) for x in args.concurrency_levels.split(",")] if args.concurrency_levels else [args.concurrency]

    recorder = ResultRecorder()
    for level in levels:
        asyncio.run(run_load_test(sessions, args.base_url, level, recorder))

    df = recorder.to_dataframe()
    recorder.write_csv(out_dir / "raw.csv")

    summary = summarize(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    generate_all_charts(df, summary, out_dir)

    _print_summary(summary)
    print(f"\nResults written to {out_dir}")

    p95 = summary["overall"]["p95"]
    if args.p95_threshold_ms is not None and p95 is not None and p95 > args.p95_threshold_ms:
        print(f"THRESHOLD BREACH: p95={p95:.1f}ms > {args.p95_threshold_ms}ms")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

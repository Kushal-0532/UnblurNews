"""Renders Phase 07's summary + raw dataframe into PNG charts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_latency_histogram(df, out_path):
    fig, ax = plt.subplots()
    ax.hist(df["latency_ms"].dropna(), bins=30)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Latency distribution")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_latency_by_cache(df, out_path):
    fig, ax = plt.subplots()
    cached = df.dropna(subset=["cache_status"])
    groups = [g["latency_ms"].values for _, g in cached.groupby("cache_status")]
    labels = list(cached.groupby("cache_status").groups.keys())
    if groups:
        ax.boxplot(groups, tick_labels=labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency by cache status")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_hit_rate_by_topic(summary, out_path):
    fig, ax = plt.subplots()
    rates = summary["by_topic_hit_rate"]
    ax.bar(list(rates.keys()), list(rates.values()))
    ax.set_ylabel("Hit rate")
    ax.set_title("Cache hit rate by topic")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_latency_vs_concurrency(summary, out_path):
    fig, ax = plt.subplots()
    by_conc = dict(sorted(summary["by_concurrency"].items()))
    ax.plot(list(by_conc.keys()), [v["p95"] for v in by_conc.values()], marker="o")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_title("p95 latency vs concurrency")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_error_rate_vs_concurrency(summary, out_path):
    fig, ax = plt.subplots()
    by_conc = dict(sorted(summary["by_concurrency"].items()))
    ax.plot(list(by_conc.keys()), [v["error_rate_pct"] for v in by_conc.values()], marker="o")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Error rate vs concurrency")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_all_charts(df, summary, out_dir) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        ("latency_histogram.png", plot_latency_histogram, (df,)),
        ("latency_by_cache.png", plot_latency_by_cache, (df,)),
        ("hit_rate_by_topic.png", plot_hit_rate_by_topic, (summary,)),
        ("latency_vs_concurrency.png", plot_latency_vs_concurrency, (summary,)),
        ("error_rate_vs_concurrency.png", plot_error_rate_vs_concurrency, (summary,)),
    ]
    paths = []
    for filename, fn, args in charts:
        out_path = str(out_dir / filename)
        fn(*args, out_path)
        paths.append(out_path)
    return paths

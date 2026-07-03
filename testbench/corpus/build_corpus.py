"""One-time script: pull AG News via HF `datasets`, label by topic, write corpus.json.

Run: python -m testbench.corpus.build_corpus
Requires: pip install datasets (not a runtime dependency of the rest of testbench)
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

TOPIC_MAP = {0: "world", 1: "sports", 2: "business", 3: "sci_tech"}

OUT_PATH = Path(__file__).parent / "corpus.json"


def build(seed: int, per_topic_count: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split="train")

    by_topic = defaultdict(list)
    for row in ds:
        by_topic[TOPIC_MAP[row["label"]]].append(row["text"])

    rng = random.Random(seed)
    records = []
    next_id = 0
    for topic, texts in sorted(by_topic.items()):
        sample = rng.sample(texts, min(per_topic_count, len(texts)))
        for text in sample:
            text = text.strip()
            records.append({
                "id": next_id,
                "topic": topic,
                "title": text[:80],
                "body": text,
                "url": f"https://corpus.local/article/{next_id}",
            })
            next_id += 1
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-topic-count", type=int, default=250)
    args = parser.parse_args()

    records = build(args.seed, args.per_topic_count)
    OUT_PATH.write_text(json.dumps(records, indent=2))

    counts = defaultdict(int)
    for r in records:
        counts[r["topic"]] += 1
    print(f"Wrote {len(records)} records to {OUT_PATH}")
    for topic, count in sorted(counts.items()):
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()

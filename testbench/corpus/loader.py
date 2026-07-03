"""Loads corpus.json into memory and groups it by topic."""
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "corpus.json"


def load_corpus(path: Path | None = None) -> list[dict]:
    path = path or DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m testbench.corpus.build_corpus` first."
        )
    return json.loads(path.read_text())


def group_by_topic(articles: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for article in articles:
        grouped[article["topic"]].append(article)
    return dict(grouped)

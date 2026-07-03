"""Collects per-request results in memory and writes them to raw.csv."""
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd


@dataclass
class RequestRecord:
    timestamp: float
    endpoint: str
    topic: str
    cache_status: str | None
    latency_ms: float
    success: bool
    status_code: int | None
    error: str | None
    concurrency: int


class ResultRecorder:
    def __init__(self):
        self.records: list[RequestRecord] = []

    def add(self, record: RequestRecord):
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        columns = [f.name for f in fields(RequestRecord)]
        return pd.DataFrame([r.__dict__ for r in self.records], columns=columns)

    def write_csv(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

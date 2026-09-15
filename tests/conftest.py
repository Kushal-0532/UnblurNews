"""Shared test setup: temp DB paths and a fake analyzer so ModernBERT/torch never load."""

import os
import sys
import tempfile
import types

_tmp = tempfile.mkdtemp(prefix="unblur-tests-")
# cache.py / metrics.py read these at import time
os.environ["CACHE_DB"] = os.path.join(_tmp, "cache.db")
os.environ["METRICS_DB"] = os.path.join(_tmp, "metrics.db")
os.environ.pop("REDIS_URL", None)
os.environ.pop("OPENAI_API_KEY", None)


class FakeAnalyzer:
    model_loaded = True
    _load_error = None

    def __init__(self):
        self.calls = 0

    def analyze(self, title, body=""):
        self.calls += 1
        return {"clickbait_pct": 12.5, "political_score": -0.4, "sentiment_score": 0.1}


_fake = FakeAnalyzer()
_stub = types.ModuleType("analyzer")
_stub.UnBlurAnalyzer = type("UnBlurAnalyzer", (), {"get_instance": staticmethod(lambda: _fake)})
sys.modules["analyzer"] = _stub

import pytest


@pytest.fixture
def fake_analyzer():
    _fake.calls = 0
    return _fake

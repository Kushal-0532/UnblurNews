"""Thin async wrapper around httpx.AsyncClient for /analyze and /related."""
import time
from dataclasses import dataclass

import httpx


@dataclass
class RequestResult:
    endpoint: str
    topic: str | None
    cache_status: str | None
    latency_ms: float
    success: bool
    status_code: int | None
    error: str | None
    data: dict | None = None


class BackendClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def _request(self, endpoint: str, topic: str | None, coro) -> RequestResult:
        start = time.perf_counter()
        try:
            response = await coro
            latency_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            return RequestResult(
                endpoint=endpoint,
                topic=topic,
                cache_status=response.headers.get("X-Cache"),
                latency_ms=latency_ms,
                success=True,
                status_code=response.status_code,
                error=None,
                data=response.json(),
            )
        except httpx.HTTPStatusError as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestResult(
                endpoint=endpoint, topic=topic, cache_status=None, latency_ms=latency_ms,
                success=False, status_code=e.response.status_code, error=str(e),
            )
        except httpx.HTTPError as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestResult(
                endpoint=endpoint, topic=topic, cache_status=None, latency_ms=latency_ms,
                success=False, status_code=None, error=str(e),
            )

    async def analyze(self, article: dict) -> RequestResult:
        body = {"title": article["title"], "body": article["body"], "url": article["url"]}
        return await self._request(
            "analyze", article.get("topic"), self.client.post("/analyze", json=body)
        )

    async def related(self, topic: str, political_score: float, sentiment_score: float) -> RequestResult:
        params = {"topic": topic, "political_score": political_score, "sentiment_score": sentiment_score}
        return await self._request("related", topic, self.client.get("/related", params=params))

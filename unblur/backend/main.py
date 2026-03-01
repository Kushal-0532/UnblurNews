"""
main.py — UnBlur FastAPI Backend
==================================

Endpoints:
    POST /analyze   → clickbait_pct, political_score, sentiment_score, case
    GET  /related   → related articles + summary + dominant leaning
    GET  /metrics   → request latency, cache hit rate, prediction distributions
    GET  /health    → service health + model status + uptime

Run:
    uvicorn backend.main:app --reload --port 8000
    # or from the UnBlur/ directory:
    uvicorn backend.main:app --reload
"""

import os
import sys
import time

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Local imports ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer     import UnBlurAnalyzer
from cache        import ArticleCache
from case_logic   import determine_case, dominant_leaning
from metrics      import MetricsStore
from news_fetcher import fetch_related, extract_keywords
from summarizer   import generate_summary

# ── App setup ────────────────────────────────────────────────────
app = FastAPI(
    title="UnBlur API",
    description="Detect echo chambers and media bias in real time.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (initialised at startup) ──────────────────────────
analyzer: UnBlurAnalyzer = None
cache:    ArticleCache   = None
metrics:  MetricsStore   = None
_start_time: float       = 0.0


@app.on_event("startup")
async def startup():
    global analyzer, cache, metrics, _start_time
    _start_time = time.time()
    analyzer = UnBlurAnalyzer.get_instance()
    cache    = ArticleCache()
    metrics  = MetricsStore()
    print(f"[startup] model_loaded={analyzer.model_loaded}")


# ── Timing middleware ────────────────────────────────────────────
# Records every request's latency and HTTP status to the metrics store.
# Design choice: middleware (not per-route decorator) so we get
# uniform coverage including 4xx/5xx errors without touching handler code.

@app.middleware("http")
async def record_latency(request: Request, call_next) -> Response:
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000

    if metrics is not None:
        metrics.record_request(
            endpoint=request.url.path,
            latency_ms=latency_ms,
            cache_hit=response.headers.get("X-Cache") == "HIT",
            status_code=response.status_code,
        )

    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.1f}"
    return response


# ═══════════════════════════════════════════════════════════════
#  Request / Response models
# ═══════════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    body:  str = Field("", max_length=5000)
    url:   str = Field("", max_length=2000)


class AnalyzeResponse(BaseModel):
    clickbait_pct:   float
    political_score: float
    sentiment_score: float
    case:            str
    cached:          bool


class RelatedArticle(BaseModel):
    title:           str
    url:             str
    source:          str
    political_score: float
    sentiment_score: float
    snippet:         str


class RelatedResponse(BaseModel):
    articles:         list[RelatedArticle]
    summary:          str
    dominant_leaning: str
    dominant_pct:     float
    case:             str = "balanced"


# ═══════════════════════════════════════════════════════════════
#  POST /analyze
# ═══════════════════════════════════════════════════════════════

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_article(req: AnalyzeRequest, response: Response):
    """
    Analyze a news article for clickbait, political leaning, and sentiment.
    Results are cached by URL for 24 hours.
    """
    # Try cache first
    if req.url:
        cached = cache.get_analysis(req.url)
        if cached:
            response.headers["X-Cache"] = "HIT"
            return AnalyzeResponse(**cached)

    response.headers["X-Cache"] = "MISS"

    if not analyzer.model_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {analyzer._load_error}",
        )

    scores = analyzer.analyze(title=req.title, body=req.body)

    # Record prediction distribution for drift monitoring
    if metrics is not None:
        metrics.record_prediction(
            clickbait_pct=scores["clickbait_pct"],
            political_score=scores["political_score"],
            sentiment_score=scores["sentiment_score"],
        )

    case = "balanced"  # refined case computed in /related with full coverage context

    result = {
        **scores,
        "case":   case,
        "cached": False,
        "title":  req.title,
    }

    if req.url:
        cache.set_analysis(req.url, result)

    return AnalyzeResponse(
        clickbait_pct=scores["clickbait_pct"],
        political_score=scores["political_score"],
        sentiment_score=scores["sentiment_score"],
        case=case,
        cached=False,
    )


# ═══════════════════════════════════════════════════════════════
#  GET /related
# ═══════════════════════════════════════════════════════════════

@app.get("/related", response_model=RelatedResponse)
async def get_related(
    response: Response,
    topic:           str   = Query(..., min_length=1, max_length=200),
    political_score: float = Query(0.0, ge=-1.0, le=1.0),
    sentiment_score: float = Query(0.0, ge=-1.0, le=1.0),
):
    """
    Fetch related articles on the same topic, scored and sorted by
    distance from the current article's position.
    """
    cache_key = f"{topic}|{round(political_score, 1)}|{round(sentiment_score, 1)}"
    cached = cache.get_related(cache_key)
    if cached:
        response.headers["X-Cache"] = "HIT"
        return RelatedResponse(**cached)

    response.headers["X-Cache"] = "MISS"

    articles = fetch_related(
        topic=topic,
        current_political=political_score,
        current_sentiment=sentiment_score,
        analyzer=analyzer if analyzer.model_loaded else None,
        max_results=10,
    )

    case = determine_case(
        articles,
        current_political=political_score,
        current_sentiment=sentiment_score,
    )

    dom_lean, dom_pct = dominant_leaning(articles)

    keywords = extract_keywords(topic)
    summary  = generate_summary(topic=" ".join(keywords) or topic, articles=articles)

    shaped = [
        RelatedArticle(
            title=a.get("title", ""),
            url=a.get("url", ""),
            source=a.get("source", ""),
            political_score=a.get("political_score", 0.0),
            sentiment_score=a.get("sentiment_score", 0.0),
            snippet=a.get("snippet", ""),
        )
        for a in articles
    ]

    payload = {
        "articles":         [a.model_dump() for a in shaped],
        "summary":          summary,
        "dominant_leaning": dom_lean,
        "dominant_pct":     dom_pct,
        "case":             case,
    }

    cache.set_related(cache_key, payload)

    return RelatedResponse(
        articles=shaped,
        summary=summary,
        dominant_leaning=dom_lean,
        dominant_pct=dom_pct,
        case=case,
    )


# ═══════════════════════════════════════════════════════════════
#  GET /metrics
# ═══════════════════════════════════════════════════════════════

@app.get("/metrics")
async def get_metrics():
    """
    Return request latency percentiles, cache hit rates, and prediction
    score distributions for the last 24 hours.

    These metrics help with:
      - SLA monitoring (p95 latency)
      - Cache efficiency (hit rate should increase over time as URLs repeat)
      - Distribution drift detection (sudden shift in political_distribution
        may indicate a dataset or source quality change)
    """
    if metrics is None:
        raise HTTPException(status_code=503, detail="Metrics store not initialised")
    return metrics.get_stats()


# ═══════════════════════════════════════════════════════════════
#  GET /health
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """
    Liveness + readiness probe.
    Returns 200 even if the model is not loaded (so the container stays up
    and the error is surfaced via model_loaded=false rather than a crash).
    """
    uptime_s = round(time.time() - _start_time, 1) if _start_time else 0

    # Quick cache size estimate
    cache_size = 0
    try:
        db_path = cache._db if cache else None
        if db_path and os.path.exists(db_path):
            cache_size = os.path.getsize(db_path)
    except Exception:
        pass

    return {
        "status":       "ok",
        "model_loaded": analyzer.model_loaded if analyzer else False,
        "model_error":  analyzer._load_error  if analyzer and not analyzer.model_loaded else None,
        "uptime_s":     uptime_s,
        "cache_db_bytes": cache_size,
    }


# ═══════════════════════════════════════════════════════════════
#  Dev entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

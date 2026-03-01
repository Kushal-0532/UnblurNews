"""
news_fetcher.py — Related Article Fetcher
==========================================

Priority order:
  1. NewsAPI          (if NEWSAPI_KEY set) — best quality, 100 req/day free
  2. Google News RSS  (no key needed)      — topic-specific search, always relevant
  3. Empty list       (both failed)

After fetching, articles are:
  - Filtered to remove clearly off-topic results
  - Scored by MediaAnalyzer
  - Sorted by 2D distance from current article (most different perspective first)
"""

import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request

try:
    from newsapi import NewsApiClient
    _NEWSAPI_AVAILABLE = True
except ImportError:
    _NEWSAPI_AVAILABLE = False

_NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
MAX_ARTICLES = 10


# ═══════════════════════════════════════════════════════════════
#  Keyword extraction
# ═══════════════════════════════════════════════════════════════

_STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "this","that","it","its","as","up","out","not","no","new","what","why",
    "how","who","after","says","said","will","can","do","did","more","over",
    "about","into","than","their","they","them","his","her","he","she","we",
    "you","i","just","also","some","such","when","then","there","here",
    "would","could","should","may","might","must","now","one","two","three",
}

def extract_keywords(title: str, max_keywords: int = 6) -> list[str]:
    """Return the most meaningful words from a headline for use as a search query."""
    words = re.findall(r"[a-zA-Z0-9]+", title)
    seen, out = set(), []
    for w in words:
        wl = w.lower()
        if wl not in _STOP_WORDS and len(wl) > 2 and wl not in seen:
            seen.add(wl)
            out.append(w)   # preserve original casing for proper nouns
    return out[:max_keywords]


def _relevance_keywords(topic: str) -> set[str]:
    """Set of lowercase keywords that a relevant article should contain at least one of."""
    return {w.lower() for w in re.findall(r"[a-zA-Z0-9]+", topic)
            if w.lower() not in _STOP_WORDS and len(w) > 2}


def _is_relevant(article: dict, must_contain: set[str]) -> bool:
    """Return True if the article title/snippet shares at least one keyword with the query."""
    if not must_contain:
        return True
    haystack = (article.get("title", "") + " " + article.get("snippet", "")).lower()
    return any(kw in haystack for kw in must_contain)


# ═══════════════════════════════════════════════════════════════
#  1. NewsAPI
# ═══════════════════════════════════════════════════════════════

def _fetch_newsapi(query: str, page_size: int = MAX_ARTICLES) -> list[dict]:
    if not _NEWSAPI_AVAILABLE or not _NEWSAPI_KEY:
        return []
    try:
        client   = NewsApiClient(api_key=_NEWSAPI_KEY)
        response = client.get_everything(
            q=query,
            page_size=page_size,
            language="en",
            sort_by="relevancy",   # relevancy > publishedAt for our use case
        )
        results = []
        for art in response.get("articles", []):
            title  = art.get("title")  or ""
            url    = art.get("url")    or ""
            source = (art.get("source") or {}).get("name") or ""
            desc   = art.get("description") or ""
            if title and url and "[Removed]" not in title:
                results.append({"title": title, "snippet": desc[:300],
                                 "url": url, "source": source})
        print(f"[news_fetcher] NewsAPI: {len(results)} articles for '{query[:60]}'")
        return results
    except Exception as exc:
        print(f"[news_fetcher] NewsAPI error: {exc}")
        return []


# ═══════════════════════════════════════════════════════════════
#  2. Google News RSS search  (no API key needed)
# ═══════════════════════════════════════════════════════════════

def _fetch_gnews_rss(query: str, max_items: int = MAX_ARTICLES) -> list[dict]:
    """
    Search Google News via its public RSS endpoint.
    Returns relevant articles for any topic — no API key required.
    """
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; UnBlur/1.0)"})
        with urlopen(req, timeout=8) as resp:
            raw = resp.read()

        root  = ET.fromstring(raw)
        items = root.findall(".//item")
        results = []
        for item in items[:max_items]:
            title_el  = item.find("title")
            link_el   = item.find("link")
            desc_el   = item.find("description")
            source_el = item.find("source")

            title  = (title_el.text  or "").strip()
            link   = (link_el.text   or "").strip()
            desc   = (desc_el.text   or "") if desc_el is not None else ""
            source = (source_el.text or "") if source_el is not None else ""

            # Google News descriptions contain HTML — strip it
            desc = re.sub(r"<[^>]+>", " ", desc).strip()
            # Google News titles include the source at the end: "Title - Source Name"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0].strip()
                if not source:
                    source = parts[1].strip()

            if title and link:
                results.append({"title": title, "snippet": desc[:300],
                                 "url": link, "source": source})

        print(f"[news_fetcher] Google News RSS: {len(results)} articles for '{query[:60]}'")
        return results
    except Exception as exc:
        print(f"[news_fetcher] Google News RSS error: {exc}")
        return []


# ═══════════════════════════════════════════════════════════════
#  Scoring
# ═══════════════════════════════════════════════════════════════

def _score_articles(raw: list[dict], analyzer) -> list[dict]:
    scored = []
    for art in raw:
        try:
            scores = analyzer.analyze(title=art["title"], body=art.get("snippet", ""))
            scored.append({**art, **scores})
        except Exception as exc:
            print(f"[news_fetcher] Scoring error: {exc}")
    return scored


def _distance(article: dict, pol: float, sent: float) -> float:
    dp = article["political_score"] - pol
    ds = article["sentiment_score"]  - sent
    return (dp**2 + ds**2) ** 0.5


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

def fetch_related(
    topic: str,
    current_political: float = 0.0,
    current_sentiment: float  = 0.0,
    analyzer=None,
    max_results: int = 10,
) -> list[dict]:
    """
    Fetch, filter, score, and sort related articles by distance from current position.

    Parameters
    ----------
    topic             : str   Full title or keyword string to search for
    current_political : float Political score of the current article
    current_sentiment : float Sentiment score of the current article
    analyzer          : MediaAnalyzer instance (or None)
    max_results       : int   Max articles to return
    """
    must_contain = _relevance_keywords(topic)

    # Build a focused search query: use at most 6 keywords so the query isn't too broad
    keywords = extract_keywords(topic, max_keywords=6)
    query    = " ".join(keywords) if keywords else topic

    # 1. Try NewsAPI
    raw = _fetch_newsapi(query, page_size=max_results + 5)

    # 2. Fall back to Google News RSS (topic-specific, not generic top-news)
    if not raw:
        raw = _fetch_gnews_rss(query, max_items=max_results + 5)

    if not raw:
        print("[news_fetcher] No articles found from any source.")
        return []

    # 3. Filter to relevance — drop articles with zero keyword overlap
    relevant = [a for a in raw if _is_relevant(a, must_contain)]
    if not relevant:
        print("[news_fetcher] Relevance filter removed all articles — using unfiltered set.")
        relevant = raw   # don't return empty; better something than nothing

    # 4. Score with model
    if analyzer is not None:
        scored = _score_articles(relevant, analyzer)
    else:
        scored = [{**a, "political_score": 0.0, "sentiment_score": 0.0, "clickbait_pct": 0.0}
                  for a in relevant]

    # 5. Sort by 2D distance from current article (most different perspective first)
    scored.sort(key=lambda a: _distance(a, current_political, current_sentiment), reverse=True)

    return scored[:max_results]

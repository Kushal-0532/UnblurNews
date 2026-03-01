"""
summarizer.py — Article Summary Generator
==========================================

Generates a 1-2 sentence summary of opposing views across related articles.

Primary:  OpenAI GPT-3.5-turbo (if OPENAI_API_KEY is set)
Fallback: Extractive summary — first sentence from top-3 most-different articles

Output is a plain string, max ~150 chars.
"""

import os
import re
from typing import Optional

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Optional OpenAI client ──────────────────────────────────────
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

SUMMARY_MAX_CHARS = 200

_PROMPT_TEMPLATE = """\
Given these {n} news article headlines about "{topic}", \
summarize the main arguments being made in 2 sentences. \
Focus on what perspective they represent. \
Headlines:
{headlines}"""


# ═══════════════════════════════════════════════════════════════
#  OpenAI primary
# ═══════════════════════════════════════════════════════════════

def _summarize_openai(topic: str, headlines: list[str]) -> Optional[str]:
    """Call GPT-3.5-turbo. Returns None on any error."""
    if not _OPENAI_AVAILABLE or not _OPENAI_API_KEY:
        return None
    try:
        client = OpenAI(api_key=_OPENAI_API_KEY)
        prompt = _PROMPT_TEMPLATE.format(
            n=len(headlines),
            topic=topic,
            headlines="\n".join(f"- {h}" for h in headlines),
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        return text[:SUMMARY_MAX_CHARS]
    except Exception as exc:
        print(f"[summarizer] OpenAI error: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════
#  Extractive fallback
# ═══════════════════════════════════════════════════════════════

def _first_sentence(text: str) -> str:
    """Extract the first sentence from a snippet or headline."""
    # Try splitting on '. ' or '! ' or '? '
    match = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)
    return match[0].rstrip(".!?") if match else text.strip()


def _summarize_extractive(articles: list[dict]) -> str:
    """
    Take the first sentence from each of the top-3 most distant articles
    and join them into a short summary.
    """
    if not articles:
        return "No related articles found."

    # Pick up to 3 articles
    selected = articles[:3]
    sentences = []
    for art in selected:
        source = art.get("source") or art.get("url", "")
        title  = art.get("title", "")
        snippet = art.get("snippet", "")
        text = snippet if snippet else title
        sent = _first_sentence(text)
        if sent:
            sentences.append(sent)

    summary = " ".join(sentences)
    return summary[:SUMMARY_MAX_CHARS]


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

def generate_summary(
    topic: str,
    articles: list[dict],
    use_openai: bool = True,
) -> str:
    """
    Generate a short summary of the main arguments across related articles.

    Parameters
    ----------
    topic     : str          The topic / search query used
    articles  : list[dict]   Scored related articles (must have 'title' key)
    use_openai: bool         Whether to try OpenAI first (default True)

    Returns
    -------
    str — plain-text summary, ≤ SUMMARY_MAX_CHARS characters
    """
    if not articles:
        return "No related coverage found."

    headlines = [a["title"] for a in articles if a.get("title")][:8]

    if use_openai:
        result = _summarize_openai(topic, headlines)
        if result:
            return result

    return _summarize_extractive(articles)


# ═══════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    MOCK_ARTICLES = [
        {
            "title": "GOP tax cuts threaten middle-class families",
            "snippet": "Economists warn that the Republican budget proposals "
                       "will widen inequality and hurt working Americans.",
            "source": "MSNBC",
            "political_score": -0.6,
            "sentiment_score": -0.5,
        },
        {
            "title": "Tax reform brings economic growth and opportunity",
            "snippet": "Supporters argue lower corporate taxes spur investment "
                       "and create jobs across all income levels.",
            "source": "Fox News",
            "political_score": 0.7,
            "sentiment_score": 0.4,
        },
        {
            "title": "Budget analysts split on tax cut impact",
            "snippet": "Independent analysts present mixed evidence on the "
                       "long-term effects of the proposed tax changes.",
            "source": "Reuters",
            "political_score": 0.0,
            "sentiment_score": 0.0,
        },
    ]

    print("=" * 60)
    print("  summarizer — Smoke Test")
    print("=" * 60)

    # Extractive (no API key)
    summary = generate_summary("tax cuts", MOCK_ARTICLES, use_openai=False)
    print(f"\nExtractive summary:\n  {summary}")
    assert len(summary) > 0

    # OpenAI (only if key set)
    if _OPENAI_API_KEY:
        summary_ai = generate_summary("tax cuts", MOCK_ARTICLES, use_openai=True)
        print(f"\nOpenAI summary:\n  {summary_ai}")
    else:
        print("\nSkipping OpenAI test (OPENAI_API_KEY not set)")

    print("\n✓ Smoke test complete.")

"""
pipeline.py — End-to-End: Fetch → Classify → Determine Case
============================================================

This script ties everything together:

    1. Fetch news articles  (using retrieve_news.py)
    2. Run each article through the multi-head model
    3. Apply deterministic case logic
    4. Print a table of results

Run
---
    python model/pipeline.py
    python model/pipeline.py --query "climate change"
    python model/pipeline.py --query "immigration" --count 5
"""

import os
import sys
import argparse

# ── Make project root importable ────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from inference import classify_article, determine_case


# ═══════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(query: str = "breaking news", count: int = 5):
    """
    Fetch articles for a query, classify each one, and print results.

    Parameters
    ----------
    query : str     Search query for the news API.
    count : int     Number of articles to fetch.
    """
    print("=" * 70)
    print("  UnblurNews — End-to-End Pipeline")
    print("=" * 70)

    # ── Step 1: Fetch articles ──────────────────────────────
    print(f"\n  Fetching up to {count} articles for: \"{query}\" ...")

    try:
        from retrieve_news import find_similar_articles
        articles = find_similar_articles(query, page_size=count)
    except ImportError:
        print("  ⚠ Could not import retrieve_news.py.")
        print("    Make sure you have a .env file with API_KEY set.")
        articles = []
    except Exception as e:
        print(f"  ⚠ Error fetching articles: {e}")
        articles = []

    if not articles:
        print("  No articles found. Using built-in examples instead.\n")
        articles = _get_example_articles()

    print(f"  ✓ Got {len(articles)} articles.\n")

    # ── Step 2: Classify each article ───────────────────────
    results = []

    for i, article in enumerate(articles, 1):
        # Combine title + description + content for analysis
        title   = article.get("title", "")
        desc    = article.get("description", "")
        content = article.get("content", "")
        source  = article.get("source", "Unknown")

        # Use as much text as available
        full_text = f"{title}. {desc}. {content}".strip()

        # Get predictions
        scores = classify_article(full_text)
        case   = determine_case(
            scores["clickbait_score"],
            scores["leaning_score"],
            scores["sentiment_score"],
        )

        results.append({
            "index":     i,
            "title":     title,
            "source":    source,
            "scores":    scores,
            "case":      case,
        })

    # ── Step 3: Print results table ─────────────────────────
    print("─" * 70)
    print(f"  {'#':<3} {'Source':<18} {'Clickbait':>9} {'Leaning':>9} {'Sentiment':>10}  Case")
    print("─" * 70)

    for r in results:
        s = r["scores"]
        title_short = r["title"][:50] if r["title"] else "(no title)"
        print(
            f"  {r['index']:<3} {r['source']:<18} "
            f"{s['clickbait_score']:>8.2f} "
            f"{s['leaning_score']:>+8.2f} "
            f"{s['sentiment_score']:>+9.2f}  "
            f"{r['case']}"
        )
        print(f"      └─ {title_short}")

    print("─" * 70)

    # ── Summary ─────────────────────────────────────────────
    n_neutral   = sum(1 for r in results if r["case"] == "NEUTRAL")
    n_clickbait = sum(1 for r in results if "CLICKBAIT" in r["case"])
    n_biased    = sum(1 for r in results if "BIASED"    in r["case"])
    n_emotional = sum(1 for r in results if "EMOTIONAL" in r["case"])

    print(f"\n  Summary:  {n_neutral} neutral  |  {n_clickbait} clickbait  |  "
          f"{n_biased} biased  |  {n_emotional} emotional")
    print("=" * 70)

    return results


# ═══════════════════════════════════════════════════════════════
#  FALLBACK  EXAMPLE  ARTICLES
# ═══════════════════════════════════════════════════════════════

def _get_example_articles():
    """Return a few hard-coded example articles for demo purposes."""
    return [
        {
            "title":       "Federal Reserve Holds Rates Steady Amid Economic Uncertainty",
            "description": "The Fed decided to maintain current interest rates, "
                           "citing moderate growth and stable inflation.",
            "content":     "",
            "source":      "Reuters",
        },
        {
            "title":       "SHOCKING: What This Celebrity Did Will Leave You Speechless!",
            "description": "Fans are STUNNED after the star's latest move went viral.",
            "content":     "",
            "source":      "BuzzFeed",
        },
        {
            "title":       "Biden's Radical Agenda Threatens American Values",
            "description": "Critics warn that the administration's progressive policies "
                           "are pushing the country too far left.",
            "content":     "",
            "source":      "Fox News",
        },
        {
            "title":       "GOP Budget Cuts Would Devastate Working Families",
            "description": "Analysis shows Republican spending proposals disproportionately "
                           "affect low-income households and social programs.",
            "content":     "",
            "source":      "MSNBC",
        },
        {
            "title":       "Global Markets Post Strong Gains Following Trade Agreement",
            "description": "Stocks rose sharply across major exchanges after the two "
                           "nations signed a comprehensive trade deal.",
            "content":     "",
            "source":      "Associated Press",
        },
    ]


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UnblurNews end-to-end pipeline")
    parser.add_argument("--query", type=str, default="breaking news",
                        help="News search query (default: 'breaking news')")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of articles to fetch (default: 5)")
    args = parser.parse_args()

    run_pipeline(query=args.query, count=args.count)

"""
case_logic.py — Deterministic Echo-Chamber Case Logic
======================================================

Takes a list of scored articles and classifies the coverage pattern:

    echo_chamber    > 70% share same leaning + sentiment polarity
    contradiction   Both political extremes present (std of scores > 0.5)
    internal_split  Same leaning side, but high sentiment variance (std > 0.4)
    balanced        Default — roughly even distribution

Constants
---------
LEANING_THRESHOLD  = 0.3  abs(political_score) > this → article is "leaning"
SENTIMENT_THRESHOLD = 0.2  abs(sentiment_score) > this → article is "charged"
"""

import statistics
from typing import Optional

LEANING_THRESHOLD  = 0.3
SENTIMENT_THRESHOLD = 0.2
ECHO_PCT_THRESHOLD = 0.70   # fraction of articles needed for echo chamber
CONTRADICTION_STD  = 0.5    # political std above this → contradiction
SPLIT_STD          = 0.4    # sentiment std above this (same side) → split


def _leaning_bucket(political_score: float) -> str:
    """Map a political score to 'left' | 'center' | 'right'."""
    if political_score < -LEANING_THRESHOLD:
        return "left"
    if political_score > LEANING_THRESHOLD:
        return "right"
    return "center"


def _sentiment_polarity(sentiment_score: float) -> str:
    """Map a sentiment score to 'negative' | 'neutral' | 'positive'."""
    if sentiment_score < -SENTIMENT_THRESHOLD:
        return "negative"
    if sentiment_score > SENTIMENT_THRESHOLD:
        return "positive"
    return "neutral"


def determine_case(
    articles: list[dict],
    current_political: Optional[float] = None,
    current_sentiment: Optional[float] = None,
) -> str:
    """
    Classify the coverage pattern of a set of related articles.

    Parameters
    ----------
    articles : list[dict]
        Each article must have "political_score" and "sentiment_score" keys.
    current_political : float, optional
        Political score of the article being analyzed (for echo-chamber check).
    current_sentiment : float, optional
        Sentiment score of the article being analyzed (for echo-chamber check).

    Returns
    -------
    str — one of: "echo_chamber" | "contradiction" | "internal_split" | "balanced"
    """
    if not articles:
        return "balanced"

    pol_scores  = [a["political_score"]  for a in articles]
    sent_scores = [a["sentiment_score"] for a in articles]
    n = len(pol_scores)

    # ── CONTRADICTION: wide political spread ─────────────────────
    if n >= 2:
        pol_std = statistics.stdev(pol_scores)
        if pol_std > CONTRADICTION_STD:
            return "contradiction"

    # ── ECHO CHAMBER: > 70 % cluster around same leaning & sentiment
    # Include the current article in the comparison if provided.
    ref_pol  = current_political  if current_political  is not None else (sum(pol_scores)  / n)
    ref_sent = current_sentiment if current_sentiment is not None else (sum(sent_scores) / n)

    ref_leaning   = _leaning_bucket(ref_pol)
    ref_sentiment = _sentiment_polarity(ref_sent)

    matching = sum(
        1 for a in articles
        if _leaning_bucket(a["political_score"]) == ref_leaning
        and _sentiment_polarity(a["sentiment_score"]) == ref_sentiment
    )
    if matching / n > ECHO_PCT_THRESHOLD:
        return "echo_chamber"

    # ── INTERNAL SPLIT: same leaning side, divergent sentiment ───
    # "Same side" = majority share the reference leaning
    same_side = [a for a in articles if _leaning_bucket(a["political_score"]) == ref_leaning]
    if len(same_side) >= 2:
        same_side_sentiments = [a["sentiment_score"] for a in same_side]
        side_sent_std = statistics.stdev(same_side_sentiments)
        if side_sent_std > SPLIT_STD:
            return "internal_split"

    return "balanced"


def dominant_leaning(articles: list[dict]) -> tuple[str, float]:
    """
    Return the dominant political leaning and its percentage.

    Returns
    -------
    (leaning_label, pct)  e.g. ("left", 67.3)
    """
    if not articles:
        return ("center", 0.0)

    counts = {"left": 0, "center": 0, "right": 0}
    for a in articles:
        counts[_leaning_bucket(a["political_score"])] += 1

    dominant = max(counts, key=counts.__getitem__)
    pct = round(counts[dominant] / len(articles) * 100, 1)
    return dominant, pct


# ═══════════════════════════════════════════════════════════════
#  Unit tests (run directly: python backend/case_logic.py)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def _a(pol, sent):
        return {"political_score": pol, "sentiment_score": sent}

    # Echo chamber — all left-negative
    articles_echo = [_a(-0.6, -0.5)] * 8 + [_a(0.1, 0.1)]  # 8/9 = 89%
    case = determine_case(articles_echo, current_political=-0.6, current_sentiment=-0.5)
    assert case == "echo_chamber", f"Expected echo_chamber, got {case}"
    print(f"✓ echo_chamber: {case}")

    # Contradiction — wide political spread
    articles_contra = [_a(-0.8, -0.4), _a(-0.7, -0.3), _a(0.8, 0.4), _a(0.7, 0.3)]
    case = determine_case(articles_contra)
    assert case == "contradiction", f"Expected contradiction, got {case}"
    print(f"✓ contradiction: {case}")

    # Internal split — same side (left), divergent sentiment
    articles_split = [_a(-0.5, -0.8), _a(-0.6, -0.7), _a(-0.4, 0.8), _a(-0.5, 0.7)]
    case = determine_case(articles_split, current_political=-0.5)
    assert case == "internal_split", f"Expected internal_split, got {case}"
    print(f"✓ internal_split: {case}")

    # Balanced — even distribution
    articles_balanced = [_a(-0.5, 0.0), _a(0.0, 0.0), _a(0.5, 0.0), _a(-0.2, 0.1)]
    case = determine_case(articles_balanced)
    assert case == "balanced", f"Expected balanced, got {case}"
    print(f"✓ balanced: {case}")

    # Dominant leaning test
    articles_dom = [_a(-0.6, -0.3)] * 6 + [_a(0.5, 0.2)] * 2 + [_a(0.0, 0.0)] * 2
    label, pct = dominant_leaning(articles_dom)
    assert label == "left", f"Expected left, got {label}"
    assert pct == 60.0, f"Expected 60.0, got {pct}"
    print(f"✓ dominant_leaning: {label} at {pct}%")

    # Edge case — empty list
    assert determine_case([]) == "balanced"
    print("✓ empty list → balanced")

    print("\n✓ All case_logic tests passed.")

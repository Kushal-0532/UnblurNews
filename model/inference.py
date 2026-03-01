"""
inference.py — Prediction Functions + Deterministic Case Logic
==============================================================

This file provides four ready-to-use functions:

    predict_clickbait(headline)  → float   (0 = safe, 1 = clickbait)
    predict_leaning(text)        → float   (-1 = left, 0 = center, +1 = right)
    predict_sentiment(text)      → float   (-1 = negative, 0 = neutral, +1 = positive)

    classify_article(article)    → dict    (all three scores at once)

And a deterministic routing function:

    determine_case(scores)       → str     ("NEUTRAL", "CLICKBAIT", "BIASED_LEFT", etc.)

Run this file directly to test on example headlines:

    python model/inference.py
"""

import os
import sys
import torch
from transformers import AutoTokenizer

# ── Imports from this package ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from multi_head_model import MultiHeadModernBERT
from train_clickbait import load_model


# ═══════════════════════════════════════════════════════════════
#  GLOBAL MODEL  &  TOKENIZER  (loaded once, reused everywhere)
# ═══════════════════════════════════════════════════════════════

_model = None
_tokenizer = None
_device = None


def _load_model_if_needed():
    """
    Lazy-load the model and tokenizer the first time any predict
    function is called.  After that, the same objects are reused.
    """
    global _model, _tokenizer, _device

    if _model is not None:
        return  # already loaded

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try loading the best available checkpoint
    for path in [config.FULL_MODEL_PATH, config.POLITICAL_MODEL_PATH, config.CLICKBAIT_MODEL_PATH]:
        if os.path.exists(path):
            print(f"Loading model from: {path}")
            _model = load_model(path, _device)
            break

    if _model is None:
        raise FileNotFoundError(
            "No trained model found!  Run a training script first:\n"
            "  python model/train_clickbait.py\n"
            "  python model/train_political.py\n"
            "  python model/train_sentiment.py"
        )

    _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    _model.eval()
    print(f"✓ Model loaded on {_device}.\n")


def _tokenize(text, max_length=None):
    """Tokenize a single string and return tensors on the correct device."""
    if max_length is None:
        max_length = config.MAX_LENGTH

    tokens = _tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    # Move every tensor to the model's device
    return {key: val.to(_device) for key, val in tokens.items()}


# ═══════════════════════════════════════════════════════════════
#  INDIVIDUAL  PREDICTION  FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def predict_clickbait(headline: str) -> float:
    """
    Predict whether a headline is clickbait.

    Parameters
    ----------
    headline : str
        The headline text to analyze.

    Returns
    -------
    float   0.0 → definitely NOT clickbait
            1.0 → definitely clickbait

    Example
    -------
    >>> predict_clickbait("You Won't Believe What Happened Next!")
    0.92
    """
    _load_model_if_needed()
    tokens = _tokenize(headline)

    with torch.no_grad():
        logits = _model(tokens["input_ids"], tokens["attention_mask"], task="clickbait")
        probs  = torch.softmax(logits, dim=1)  # shape (1, 2)

    # Return P(clickbait) — the probability of class 1
    clickbait_prob = probs[0, 1].item()
    return round(clickbait_prob, 4)


def predict_leaning(text: str) -> float:
    """
    Predict the political leaning of an article or headline.

    Parameters
    ----------
    text : str
        Headline, first few paragraphs, or full article text.

    Returns
    -------
    float   -1.0 → strong Left
             0.0 → Center
            +1.0 → strong Right

    How it works
    ------------
    The model outputs 3-class probabilities:  P(Left), P(Center), P(Right).
    We compute a weighted sum:
        score = -1 × P(Left) + 0 × P(Center) + 1 × P(Right)
    This gives a smooth continuous score.

    Example
    -------
    >>> predict_leaning("GOP tax cuts threaten social safety net programs")
    -0.45
    """
    _load_model_if_needed()
    tokens = _tokenize(text, max_length=config.MAX_LENGTH_ARTICLE)

    with torch.no_grad():
        logits = _model(tokens["input_ids"], tokens["attention_mask"], task="leaning")
        probs  = torch.softmax(logits, dim=1)  # shape (1, 3)

    # Weighted sum:  -1 × P(Left)  +  0 × P(Center)  +  1 × P(Right)
    p_left   = probs[0, 0].item()
    p_center = probs[0, 1].item()
    p_right  = probs[0, 2].item()

    score = (-1.0 * p_left) + (0.0 * p_center) + (1.0 * p_right)
    return round(score, 4)


def predict_sentiment(text: str) -> float:
    """
    Predict the emotional tone / sentiment of text.

    Parameters
    ----------
    text : str
        Any news text (headline or body).

    Returns
    -------
    float   -1.0 → very negative / alarming
             0.0 → neutral / factual
            +1.0 → very positive / uplifting

    How it works
    ------------
    Same weighted-sum approach as leaning:
        score = -1 × P(Negative) + 0 × P(Neutral) + 1 × P(Positive)

    Example
    -------
    >>> predict_sentiment("Markets crash amid fears of global recession")
    -0.72
    """
    _load_model_if_needed()
    tokens = _tokenize(text, max_length=config.MAX_LENGTH_ARTICLE)

    with torch.no_grad():
        logits = _model(tokens["input_ids"], tokens["attention_mask"], task="sentiment")
        probs  = torch.softmax(logits, dim=1)

    p_neg     = probs[0, 0].item()
    p_neutral = probs[0, 1].item()
    p_pos     = probs[0, 2].item()

    score = (-1.0 * p_neg) + (0.0 * p_neutral) + (1.0 * p_pos)
    return round(score, 4)


# ═══════════════════════════════════════════════════════════════
#  UNIFIED  CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def classify_article(article_text: str) -> dict:
    """
    Run ALL three heads on a piece of text in one call.

    Parameters
    ----------
    article_text : str
        Full article text (headline + body), or just a headline.

    Returns
    -------
    dict with keys:
        clickbait_score  : float  (0 to 1)
        leaning_score    : float  (-1 to +1)
        sentiment_score  : float  (-1 to +1)

    Example
    -------
    >>> classify_article("Breaking: Scientists shocked by new discovery!")
    {
        'clickbait_score': 0.87,
        'leaning_score': -0.05,
        'sentiment_score': 0.12,
    }
    """
    _load_model_if_needed()
    tokens = _tokenize(article_text, max_length=config.MAX_LENGTH_ARTICLE)

    with torch.no_grad():
        # Run ALL heads at once  (one backbone pass → three outputs)
        outputs = _model(tokens["input_ids"], tokens["attention_mask"], task="all")

    # --- Clickbait: P(clickbait) ---
    cb_probs = torch.softmax(outputs["clickbait"], dim=1)
    clickbait_score = cb_probs[0, 1].item()

    # --- Leaning: weighted sum ---
    lean_probs = torch.softmax(outputs["leaning"], dim=1)
    leaning_score = (
        -1.0 * lean_probs[0, 0].item()
        + 0.0 * lean_probs[0, 1].item()
        + 1.0 * lean_probs[0, 2].item()
    )

    # --- Sentiment: weighted sum ---
    sent_probs = torch.softmax(outputs["sentiment"], dim=1)
    sentiment_score = (
        -1.0 * sent_probs[0, 0].item()
        + 0.0 * sent_probs[0, 1].item()
        + 1.0 * sent_probs[0, 2].item()
    )

    return {
        "clickbait_score":  round(clickbait_score, 4),
        "leaning_score":    round(leaning_score, 4),
        "sentiment_score":  round(sentiment_score, 4),
    }


# ═══════════════════════════════════════════════════════════════
#  DETERMINISTIC  CASE  LOGIC
# ═══════════════════════════════════════════════════════════════

def determine_case(
    clickbait_score: float,
    leaning_score: float,
    sentiment_score: float,
    clickbait_threshold: float = None,
    leaning_threshold: float = None,
    sentiment_threshold: float = None,
) -> str:
    """
    Apply deterministic thresholds to decide what "case" an article falls into.

    This function is pure logic — no model inference happens here.
    It takes the three scores (from classify_article) and returns a
    human-readable label describing the article's primary concern(s).

    Parameters
    ----------
    clickbait_score  : float  (0 to 1)
    leaning_score    : float  (-1 to +1)
    sentiment_score  : float  (-1 to +1)
    *_threshold      : float  (optional overrides for config thresholds)

    Returns
    -------
    str — one of:
        "NEUTRAL"           No flags triggered
        "CLICKBAIT"         Only clickbait flag
        "BIASED_LEFT"       Only left-leaning flag
        "BIASED_RIGHT"      Only right-leaning flag
        "EMOTIONAL_NEG"     Only strong negative sentiment
        "EMOTIONAL_POS"     Only strong positive sentiment
        "MIXED: X, Y, ..."  Multiple flags triggered

    Examples
    --------
    >>> determine_case(0.9, 0.1, 0.0)
    'CLICKBAIT'

    >>> determine_case(0.2, -0.6, -0.5)
    'MIXED: BIASED_LEFT, EMOTIONAL_NEG'

    >>> determine_case(0.1, 0.05, 0.1)
    'NEUTRAL'
    """
    # Use provided thresholds or fall back to config defaults
    ct = clickbait_threshold  if clickbait_threshold  is not None else config.CLICKBAIT_THRESHOLD
    lt = leaning_threshold    if leaning_threshold    is not None else config.LEANING_THRESHOLD
    st = sentiment_threshold  if sentiment_threshold  is not None else config.SENTIMENT_THRESHOLD

    # Collect all flags that are triggered
    flags = []

    # --- Clickbait check ---
    if clickbait_score > ct:
        flags.append("CLICKBAIT")

    # --- Political-leaning check ---
    if leaning_score < -lt:         # negative score = left-leaning
        flags.append("BIASED_LEFT")
    elif leaning_score > lt:        # positive score = right-leaning
        flags.append("BIASED_RIGHT")

    # --- Sentiment check ---
    if sentiment_score < -st:       # negative sentiment
        flags.append("EMOTIONAL_NEG")
    elif sentiment_score > st:      # positive sentiment
        flags.append("EMOTIONAL_POS")

    # --- Determine final case ---
    if len(flags) == 0:
        return "NEUTRAL"
    elif len(flags) == 1:
        return flags[0]
    else:
        return "MIXED: " + ", ".join(flags)


def classify_and_determine(article_text: str) -> dict:
    """
    Combined convenience function:  classify + determine case.

    Returns
    -------
    dict with keys:
        clickbait_score, leaning_score, sentiment_score, case
    """
    scores = classify_article(article_text)
    case   = determine_case(
        scores["clickbait_score"],
        scores["leaning_score"],
        scores["sentiment_score"],
    )
    return {**scores, "case": case}


# ═══════════════════════════════════════════════════════════════
#  DEMO:  Test on 5 sample articles
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  UnblurNews — Inference Demo")
    print("=" * 70)

    # Five sample articles with known biases for testing
    SAMPLES = [
        {
            "text": "You Won't BELIEVE What This Politician Said — "
                    "The Internet Is LOSING IT!",
            "expected": "Clickbait, probably neutral leaning",
        },
        {
            "text": "Republican tax cuts for the wealthy are destroying "
                    "the middle class and widening inequality. Workers "
                    "continue to suffer under trickle-down economics.",
            "expected": "Left-leaning, negative sentiment",
        },
        {
            "text": "The Biden administration's open-border policies have "
                    "led to an unprecedented crisis, with illegal crossings "
                    "skyrocketing and communities overwhelmed.",
            "expected": "Right-leaning, negative sentiment",
        },
        {
            "text": "Reuters — The Federal Reserve held interest rates steady "
                    "on Wednesday, citing stable inflation and moderate "
                    "economic growth. Markets reacted calmly.",
            "expected": "Neutral, center, factual",
        },
        {
            "text": "A heartwarming story: neighbors in a small town came "
                    "together to rebuild a family's home after a devastating "
                    "fire. The community raised over $50,000 in just two days.",
            "expected": "Neutral leaning, positive sentiment",
        },
    ]

    print("\n  Testing model predictions on 5 sample articles ...\n")

    for i, sample in enumerate(SAMPLES, 1):
        print(f"  ── Sample {i} ──────────────────────────────────")
        print(f"  Text:     {sample['text'][:75]}...")
        print(f"  Expected: {sample['expected']}")

        try:
            result = classify_and_determine(sample["text"])
            print(f"  Clickbait:  {result['clickbait_score']:.2f}")
            print(f"  Leaning:    {result['leaning_score']:+.2f}  "
                  f"(-1=Left, +1=Right)")
            print(f"  Sentiment:  {result['sentiment_score']:+.2f}  "
                  f"(-1=Neg, +1=Pos)")
            print(f"  Case:       {result['case']}")
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")
            break

        print()

    print("=" * 70)
    print("  Demo complete.")
    print("=" * 70)

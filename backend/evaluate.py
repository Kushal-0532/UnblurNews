"""
evaluate.py — Offline Model Evaluation Harness
================================================

Runs the UnBlur model against a curated, hand-labelled test set and
reports per-task accuracy and macro F1.  Run this after any model
update to catch regressions before deploying.

Usage
-----
    cd UnBlur/
    source venv/bin/activate
    python backend/evaluate.py                        # uses default model path
    python backend/evaluate.py --model ./backend/models/v2

Output
------
    Prints a per-task report to stdout.
    Saves full results to evaluation_results.json (overwritten each run).

Design choice: hardcoded test set vs. external dataset
    We keep 30 labelled examples directly in this file so the evaluation
    can run without network access or extra data pipeline setup.  The
    samples were chosen to cover corner cases (ambiguous leanings,
    mixed sentiment, edge-case clickbait patterns).  In production you
    would extend this with a larger held-out set stored in a file.

Scoring
-------
    Accuracy  = correct / total  (per task)
    Macro F1  = mean of per-class F1 scores (treats all classes equally,
                robust to class imbalance in the test set)
    Confusion matrix printed for qualitative error inspection.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────
#  Labelled test set
#  Fields:
#    title           str   — article headline
#    body            str   — short body excerpt (or "")
#    clickbait       int   — 0 = not clickbait, 1 = clickbait
#    leaning         str   — "left" | "center" | "right"
#    sentiment       str   — "negative" | "neutral" | "positive"
# ─────────────────────────────────────────────────────────────────

TEST_SET = [
    # ── Clickbait / sensationalist ────────────────────────────────
    {
        "title": "You WON'T BELIEVE What This Senator Just Did",
        "body":  "Fans reacted with shock after the viral clip spread across social media.",
        "clickbait": 1, "leaning": "center", "sentiment": "neutral",
    },
    {
        "title": "SHOCKING: Celebrity Reveals Explosive Secret",
        "body":  "Sources close to the star say the revelation has left everyone speechless.",
        "clickbait": 1, "leaning": "center", "sentiment": "neutral",
    },
    {
        "title": "10 Things Doctors Don't Want You To Know",
        "body":  "Medical experts have long hidden this simple trick from the public.",
        "clickbait": 1, "leaning": "center", "sentiment": "negative",
    },
    # ── Factual / non-clickbait ───────────────────────────────────
    {
        "title": "Federal Reserve Holds Interest Rates Steady for Third Consecutive Month",
        "body":  "The Federal Reserve kept its benchmark rate unchanged citing stable inflation.",
        "clickbait": 0, "leaning": "center", "sentiment": "neutral",
    },
    {
        "title": "Senate Passes Bipartisan Infrastructure Bill 69–30",
        "body":  "The $1.2 trillion package funds roads, bridges, broadband, and rail.",
        "clickbait": 0, "leaning": "center", "sentiment": "positive",
    },
    {
        "title": "WHO Releases Updated Guidance on COVID-19 Booster Vaccines",
        "body":  "The World Health Organisation recommends additional doses for high-risk groups.",
        "clickbait": 0, "leaning": "center", "sentiment": "neutral",
    },
    # ── Left-leaning ──────────────────────────────────────────────
    {
        "title": "GOP Tax Cuts Devastate Working Families as Billionaires Pocket Billions",
        "body":  "New analysis shows Republican tax legislation disproportionately benefits the ultra-wealthy.",
        "clickbait": 0, "leaning": "left", "sentiment": "negative",
    },
    {
        "title": "Climate Activists Demand Immediate Action as Carbon Emissions Break Records",
        "body":  "Protesters gathered outside Congress calling for an end to fossil fuel subsidies.",
        "clickbait": 0, "leaning": "left", "sentiment": "negative",
    },
    {
        "title": "Study Finds Universal Healthcare Would Save Millions of American Lives",
        "body":  "Harvard researchers conclude a single-payer system reduces preventable deaths by 40 percent.",
        "clickbait": 0, "leaning": "left", "sentiment": "positive",
    },
    {
        "title": "Workers Strike as Corporate Profits Soar to Record Highs",
        "body":  "Union members demand higher wages as CEO pay packages reached historic levels.",
        "clickbait": 0, "leaning": "left", "sentiment": "negative",
    },
    # ── Right-leaning ─────────────────────────────────────────────
    {
        "title": "Biden's Open-Border Policy Fuels Historic Surge in Illegal Immigration",
        "body":  "Border agents report record crossings as administration ends Title 42 enforcement.",
        "clickbait": 0, "leaning": "right", "sentiment": "negative",
    },
    {
        "title": "Second Amendment Victory: Supreme Court Strikes Down Gun Control Law",
        "body":  "The 6–3 ruling reaffirms constitutional carry rights for law-abiding citizens.",
        "clickbait": 0, "leaning": "right", "sentiment": "positive",
    },
    {
        "title": "Tax Cuts Drive Record Job Growth and Economic Expansion",
        "body":  "The Republican tax reform led to historic unemployment lows and wage increases.",
        "clickbait": 0, "leaning": "right", "sentiment": "positive",
    },
    {
        "title": "China Threat Grows as Biden Administration Weakens Military Spending",
        "body":  "Defense hawks warn America's military readiness is declining under current leadership.",
        "clickbait": 0, "leaning": "right", "sentiment": "negative",
    },
    # ── Center / neutral ──────────────────────────────────────────
    {
        "title": "Congress Reaches Deal on Budget After Weeks of Negotiations",
        "body":  "Lawmakers from both parties agreed on a continuing resolution to avoid shutdown.",
        "clickbait": 0, "leaning": "center", "sentiment": "neutral",
    },
    {
        "title": "Apple Reports Quarterly Revenue of $90 Billion, Beats Estimates",
        "body":  "The tech giant posted solid results driven by iPhone sales and services growth.",
        "clickbait": 0, "leaning": "center", "sentiment": "positive",
    },
    {
        "title": "NASA Confirms Water Ice Deposits on the Moon's South Pole",
        "body":  "Scientists say the discovery could support future lunar missions and habitation.",
        "clickbait": 0, "leaning": "center", "sentiment": "positive",
    },
    # ── Positive sentiment ────────────────────────────────────────
    {
        "title": "New Cancer Treatment Shows 90 Percent Remission Rate in Clinical Trial",
        "body":  "Researchers at Johns Hopkins published breakthrough results for an mRNA therapy.",
        "clickbait": 0, "leaning": "center", "sentiment": "positive",
    },
    {
        "title": "Renewable Energy Hits Record Milestone: 30 Percent of US Electricity",
        "body":  "Wind and solar combined to surpass coal for the first time in history.",
        "clickbait": 0, "leaning": "center", "sentiment": "positive",
    },
    # ── Negative sentiment ────────────────────────────────────────
    {
        "title": "Wildfires Destroy Thousands of Homes Across California and Oregon",
        "body":  "Emergency crews battle blazes as evacuations expand to 200,000 residents.",
        "clickbait": 0, "leaning": "center", "sentiment": "negative",
    },
    {
        "title": "Unemployment Spikes to 8 Percent as Tech Sector Announces Mass Layoffs",
        "body":  "Major firms cut more than 50,000 jobs, rattling financial markets.",
        "clickbait": 0, "leaning": "center", "sentiment": "negative",
    },
    # ── Ambiguous / edge cases ────────────────────────────────────
    {
        "title": "Is America Heading for a Recession? Economists Disagree",
        "body":  "Some analysts point to inflation and rate hikes; others see resilient consumer spending.",
        "clickbait": 0, "leaning": "center", "sentiment": "neutral",
    },
    {
        "title": "Poll: 52% of Americans Support Stricter Gun Laws",
        "body":  "A Gallup survey shows slim majority favouring background check expansion.",
        "clickbait": 0, "leaning": "center", "sentiment": "neutral",
    },
    {
        "title": "Democrats Slam Republican Healthcare Plan as 'Dangerous'",
        "body":  "Progressive lawmakers warned the bill would strip coverage from millions.",
        "clickbait": 0, "leaning": "left", "sentiment": "negative",
    },
    {
        "title": "Republicans Accuse Democrats of Weaponising DOJ Against Political Opponents",
        "body":  "GOP senators held press conference calling for special counsel investigations.",
        "clickbait": 0, "leaning": "right", "sentiment": "negative",
    },
    {
        "title": "Scientists Warn of Accelerating Arctic Melting, Dire Consequences Ahead",
        "body":  "A new Nature study found Arctic sea ice shrinking 30 percent faster than modelled.",
        "clickbait": 0, "leaning": "left", "sentiment": "negative",
    },
    {
        "title": "President Signs Executive Order Boosting Domestic Energy Production",
        "body":  "The White House says the move will lower gas prices and reduce foreign dependence.",
        "clickbait": 0, "leaning": "right", "sentiment": "positive",
    },
    {
        "title": "How One Small Town Is Rebuilding After the Tornado That Levelled It",
        "body":  "Residents of Clearbend, Oklahoma are rallying around each other six months after disaster.",
        "clickbait": 0, "leaning": "center", "sentiment": "positive",
    },
    {
        "title": "Doctors Are Stunned By This Simple Morning Habit That Burns Fat",
        "body":  "This revolutionary trick has big pharma terrified — and it works overnight!",
        "clickbait": 1, "leaning": "center", "sentiment": "positive",
    },
    {
        "title": "Stock Market Closes Mixed as Investors Weigh Inflation Data",
        "body":  "The S&P 500 edged up 0.2 percent while the Nasdaq fell on tech earnings disappointment.",
        "clickbait": 0, "leaning": "center", "sentiment": "neutral",
    },
]

# Map label strings to model output indices
LEANING_MAP  = {"left": 0, "center": 1, "right": 2}
SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}


# ─────────────────────────────────────────────────────────────────
#  Evaluation helpers
# ─────────────────────────────────────────────────────────────────

def _predicted_clickbait(pct: float) -> int:
    """Map continuous clickbait_pct → {0, 1}. Threshold at 50 %."""
    return 1 if pct >= 50.0 else 0


def _predicted_leaning(score: float) -> int:
    """Map continuous political_score → {0=left, 1=center, 2=right}."""
    if score < -0.3:
        return 0
    if score >  0.3:
        return 2
    return 1


def _predicted_sentiment(score: float) -> int:
    """Map continuous sentiment_score → {0=neg, 1=neu, 2=pos}."""
    if score < -0.2:
        return 0
    if score >  0.2:
        return 2
    return 1


def _macro_f1(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    """Compute macro-averaged F1 over num_classes classes."""
    f1s = []
    for cls in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def _confusion_matrix(y_true: list[int], y_pred: list[int], labels: list[str]) -> str:
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    col_w = max(len(l) for l in labels) + 2
    header = " " * (col_w + 2) + "  ".join(f"{l:>{col_w}}" for l in labels)
    lines = [header]
    for i, row in enumerate(matrix):
        line = f"{labels[i]:>{col_w}}  " + "  ".join(f"{v:>{col_w}}" for v in row)
        lines.append(line)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
#  Main evaluation loop
# ─────────────────────────────────────────────────────────────────

def evaluate(model_path: str | None = None) -> dict:
    # Import here so the file can be inspected without torch installed
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyzer import UnBlurAnalyzer

    print("=" * 60)
    print("  UnBlur Model Evaluation")
    print("=" * 60)

    kwargs = {} if model_path is None else {"model_path": model_path}
    analyzer = UnBlurAnalyzer(**kwargs)

    if not analyzer.model_loaded:
        print(f"\n[ERROR] Model not loaded: {analyzer._load_error}")
        sys.exit(1)

    print(f"\nModel path : {analyzer._model_path}")
    print(f"Test set   : {len(TEST_SET)} examples\n")

    # Accumulators
    cb_true, cb_pred     = [], []
    lean_true, lean_pred = [], []
    sent_true, sent_pred = [], []
    latencies: list[float] = []
    errors: list[dict] = []

    for idx, sample in enumerate(TEST_SET):
        t0 = time.perf_counter()
        result = analyzer.analyze(title=sample["title"], body=sample["body"])
        latencies.append((time.perf_counter() - t0) * 1000)

        pred_cb   = _predicted_clickbait(result["clickbait_pct"])
        pred_lean = _predicted_leaning(result["political_score"])
        pred_sent = _predicted_sentiment(result["sentiment_score"])

        true_cb   = sample["clickbait"]
        true_lean = LEANING_MAP[sample["leaning"]]
        true_sent = SENTIMENT_MAP[sample["sentiment"]]

        cb_true.append(true_cb);   cb_pred.append(pred_cb)
        lean_true.append(true_lean); lean_pred.append(pred_lean)
        sent_true.append(true_sent); sent_pred.append(pred_sent)

        # Collect errors for qualitative review
        wrong = []
        if pred_cb   != true_cb:   wrong.append("clickbait")
        if pred_lean != true_lean: wrong.append("leaning")
        if pred_sent != true_sent: wrong.append("sentiment")
        if wrong:
            errors.append({
                "idx": idx,
                "title": sample["title"][:60],
                "wrong": wrong,
                "scores": result,
                "true": {
                    "clickbait": true_cb,
                    "leaning": sample["leaning"],
                    "sentiment": sample["sentiment"],
                },
            })

    n = len(TEST_SET)

    # ── Per-task metrics ─────────────────────────────────────────
    cb_acc   = sum(t == p for t, p in zip(cb_true,   cb_pred))   / n
    lean_acc = sum(t == p for t, p in zip(lean_true, lean_pred)) / n
    sent_acc = sum(t == p for t, p in zip(sent_true, sent_pred)) / n

    cb_f1   = _macro_f1(cb_true,   cb_pred,   2)
    lean_f1 = _macro_f1(lean_true, lean_pred, 3)
    sent_f1 = _macro_f1(sent_true, sent_pred, 3)

    # ── Latency ──────────────────────────────────────────────────
    latencies.sort()
    avg_lat = sum(latencies) / len(latencies)
    p95_lat = latencies[int(0.95 * len(latencies))]

    # ── Report ───────────────────────────────────────────────────
    print("  Task Results")
    print("  " + "-" * 46)
    print(f"  {'Task':<18}  {'Accuracy':>8}  {'Macro F1':>8}")
    print("  " + "-" * 46)
    print(f"  {'Clickbait':<18}  {cb_acc:>7.1%}  {cb_f1:>8.3f}")
    print(f"  {'Political Leaning':<18}  {lean_acc:>7.1%}  {lean_f1:>8.3f}")
    print(f"  {'Sentiment':<18}  {sent_acc:>7.1%}  {sent_f1:>8.3f}")
    print()
    print(f"  Inference latency  avg={avg_lat:.1f}ms  p95={p95_lat:.1f}ms")
    print()

    print("  Confusion matrix — Clickbait")
    print(_confusion_matrix(cb_true, cb_pred, ["not-cb", "clickbait"]))
    print()
    print("  Confusion matrix — Political Leaning")
    print(_confusion_matrix(lean_true, lean_pred, ["left", "center", "right"]))
    print()
    print("  Confusion matrix — Sentiment")
    print(_confusion_matrix(sent_true, sent_pred, ["neg", "neutral", "pos"]))
    print()

    if errors:
        print(f"  Misclassified samples ({len(errors)}/{n}):")
        for e in errors:
            print(f"    [{e['idx']:02d}] {e['title']}")
            print(f"          wrong={e['wrong']}  scores={e['scores']}")
            print(f"          true ={e['true']}")
    else:
        print("  Perfect score — no misclassifications!")

    # ── Save to JSON ─────────────────────────────────────────────
    results = {
        "model_path": analyzer._model_path,
        "test_set_size": n,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": {
            "clickbait":          {"accuracy": round(cb_acc,   4), "macro_f1": round(cb_f1,   4)},
            "political_leaning":  {"accuracy": round(lean_acc, 4), "macro_f1": round(lean_f1, 4)},
            "sentiment":          {"accuracy": round(sent_acc, 4), "macro_f1": round(sent_f1, 4)},
        },
        "latency_ms": {"mean": round(avg_lat, 1), "p95": round(p95_lat, 1)},
        "errors": errors,
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {os.path.abspath(out_path)}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the UnBlur model.")
    parser.add_argument(
        "--model", default=None,
        help="Path to model directory (default: backend/models/)",
    )
    args = parser.parse_args()
    evaluate(model_path=args.model)

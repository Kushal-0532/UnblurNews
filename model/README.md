# UnblurNews — Multi-Head ModernBERT Model

A single **ModernBERT** backbone with three classification heads for:

| Head | Task | Output |
|------|------|--------|
| 1 | **Clickbait Detection** | 0 (safe) → 1 (clickbait) |
| 2 | **Political Leaning** | -1 (Left) → 0 (Center) → +1 (Right) |
| 3 | **Sentiment Analysis** | -1 (Negative) → 0 (Neutral) → +1 (Positive) |

```
                ┌──────────────┐
                │  Input Text  │
                └──────┬───────┘
                       │
              ┌────────▼────────┐
              │  ModernBERT     │   ← shared backbone
              │  (encoder)      │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
   ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
   │ Clickbait │ │ Leaning │ │ Sentiment │
   │   Head    │ │  Head   │ │   Head    │
   └───────────┘ └─────────┘ └───────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r model/requirements.txt
```

### 2. Train the model (step by step)

```bash
# Day 3–4: Train clickbait head
python model/train_clickbait.py

# Day 5–6: Train political leaning head (Option A or B)
python model/train_political.py                    # Option B: joint fine-tune (default)
python model/train_political.py --freeze-backbone  # Option A: freeze backbone

# Day 7: Train sentiment head
python model/train_sentiment.py

# Day 7: (Optional) Unified multi-task fine-tuning
python model/train_multitask.py
```

### 3. Run inference

```python
from model.inference import predict_clickbait, predict_leaning, predict_sentiment, classify_article

# Individual predictions
print(predict_clickbait("You Won't Believe What Happened Next!"))  # → 0.92
print(predict_leaning("GOP tax cuts benefit wealthy donors"))       # → 0.45
print(predict_sentiment("Markets crash amid recession fears"))      # → -0.72

# All-in-one
result = classify_article("Breaking: Scientists discover shocking truth!")
print(result)
# {'clickbait_score': 0.87, 'leaning_score': -0.05, 'sentiment_score': 0.12}
```

### 4. End-to-end pipeline

```bash
python model/pipeline.py --query "climate change" --count 5
```

---

## Datasets

### Clickbait

**Recommended:** The `christophsonntag/clickbait` dataset on HuggingFace (auto-downloaded).

**Alternative from Kaggle:**
1. Search Kaggle for "clickbait detection dataset"
2. Download and save as `model/data/clickbait.csv` with columns: `text`, `label` (0 or 1)

### Political Leaning

**Recommended approach — AllSides mapping:**
1. Collect articles from various news sources (using `retrieve_news.py` or any news corpus)
2. The code automatically labels articles by their source using AllSides ratings (see `config.py → ALLSIDES_RATINGS`)
3. Use `datasets_loader.create_political_dataset_from_articles()` to generate labeled data

**Or provide your own CSV:**
Save as `model/data/political_leaning.csv` with columns: `text`, `label` (0=Left, 1=Center, 2=Right)

**Kaggle options:**
- Search for "AllSides media bias" or "political news bias" datasets
- Download and reformat to the CSV format above

### Sentiment

**Recommended:** The `tweet_eval` sentiment subset from HuggingFace (auto-downloaded).

**Alternative — Knowledge distillation:**
```python
from model.datasets_loader import distill_sentiment_labels

texts = ["your news text here", ...]
labels = distill_sentiment_labels(texts)  # Uses cardiffnlp teacher model
```

**Or provide your own CSV:**
Save as `model/data/sentiment.csv` with columns: `text`, `label` (0=Negative, 1=Neutral, 2=Positive)

---

## File Structure

```
model/
├── __init__.py             # Package initialization
├── config.py               # All hyperparameters, thresholds, paths
├── datasets_loader.py      # Data loading for all 3 tasks
├── multi_head_model.py     # Multi-head ModernBERT architecture
├── train_clickbait.py      # Day 3–4: Train clickbait head
├── train_political.py      # Day 5–6: Train political leaning head
├── train_sentiment.py      # Day 7: Train sentiment head
├── train_multitask.py      # Day 7: Unified multi-task training
├── inference.py            # predict_* functions + determine_case()
├── pipeline.py             # End-to-end: fetch → classify → route
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Training Strategies

### Preventing Catastrophic Forgetting

When training new heads, earlier heads can "forget" what they learned. Two strategies:

| Strategy | Command | Pros | Cons |
|----------|---------|------|------|
| **Option A:** Freeze backbone | `--freeze-backbone` | Safe; no forgetting | New head has less capacity |
| **Option B:** Joint fine-tune | (default) | Better overall accuracy | Slightly more complex |

Option B uses a combined loss at each step:
```
total_loss = task_loss + replay_loss_from_earlier_tasks
```

### Multi-Task Loss Weights

In `config.py`, adjust:
```python
W_CLICKBAIT  = 1.0   # weight for clickbait loss
W_LEANING    = 1.0   # weight for leaning loss
W_SENTIMENT  = 1.0   # weight for sentiment loss
```

If one task dominates, increase the weight of underperforming tasks.

---

## Deterministic Case Logic

The `determine_case()` function applies thresholds to classify articles:

```python
from model.inference import determine_case

case = determine_case(
    clickbait_score=0.9,   # high clickbait
    leaning_score=0.1,     # center
    sentiment_score=0.0,   # neutral
)
# → "CLICKBAIT"
```

**Possible cases:**
- `NEUTRAL` — no flags triggered
- `CLICKBAIT` — high clickbait score
- `BIASED_LEFT` / `BIASED_RIGHT` — strong political leaning
- `EMOTIONAL_NEG` / `EMOTIONAL_POS` — strong sentiment
- `MIXED: X, Y` — multiple flags

**Tune thresholds** in `config.py`:
```python
CLICKBAIT_THRESHOLD  = 0.5   # P(clickbait) above this
LEANING_THRESHOLD    = 0.3   # |leaning_score| above this
SENTIMENT_THRESHOLD  = 0.3   # |sentiment_score| above this
```

---

## Requirements

- Python 3.9+
- `torch >= 2.0`
- `transformers >= 4.48` (for ModernBERT support)
- `datasets` (for HuggingFace dataset loading)
- GPU recommended but not required (CPU works, just slower)

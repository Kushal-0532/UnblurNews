# UnBlur — Real-Time Media Bias & Echo Chamber Detector

UnBlur is a browser extension backed by a fine-tuned NLP model that analyzes any news article for clickbait, political leaning, and sentiment — then shows you how the same story is covered across the political spectrum, so you can spot echo chambers before they form.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture Overview](#architecture-overview)
3. [ML Pipeline](#ml-pipeline)
4. [MLOps & Evaluation](#mlops--evaluation)
5. [Infrastructure & Optimization Decisions](#infrastructure--optimization-decisions)
6. [Project Structure](#project-structure)
7. [Setup](#setup)
8. [API Reference](#api-reference)
9. [Extension Usage](#extension-usage)
10. [Environment Variables](#environment-variables)
11. [Tech Stack](#tech-stack)

---

## What It Does

When you click the UnBlur icon on any news article, the extension:

1. Extracts the article title and body from the DOM
2. Sends it to the local FastAPI backend
3. Runs the article through a fine-tuned **ModernBERT** model with three task-specific heads
4. Fetches 10 related articles from **Google News RSS** (or NewsAPI) and scores them all
5. Classifies the media landscape as one of four cases (Echo Chamber / Contradiction / Internal Split / Balanced)
6. Renders an interactive sidebar with:
   - **Clickbait score** — 0–100% bar (green → red gradient)
   - **Bias map** — 2D scatter chart (political × sentiment) showing where every article sits
   - **Case diagnosis** — icon + description of the media landscape
   - **Summary** — dominant leaning stats and links to the most opposing perspectives

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser                                                        │
│                                                                 │
│  ┌──────────────┐   TOGGLE_SIDEBAR   ┌─────────────────────┐   │
│  │ background.js│ ──────────────────►│    content.js       │   │
│  │ (service     │                    │  • DOM extraction   │   │
│  │  worker)     │                    │  • iframe injection │   │
│  └──────────────┘                    └────────┬────────────┘   │
│                                               │ postMessage    │
│                                      ┌────────▼────────────┐   │
│                                      │    sidebar.html/js  │   │
│                                      │  • Chart.js scatter │   │
│                                      │  • Clickbait bar    │   │
│                                      │  • Case diagnosis   │   │
│                                      └────────┬────────────┘   │
└───────────────────────────────────────────────┼────────────────┘
                                                │ HTTP (localhost)
                         ┌──────────────────────▼────────────────┐
                         │  FastAPI Backend (uvicorn)             │
                         │                                        │
                         │  POST /analyze                         │
                         │    ┌──────────────┐                    │
                         │    │ ArticleCache │◄── SQLite 24h TTL  │
                         │    └──────┬───────┘                    │
                         │           │ MISS                       │
                         │    ┌──────▼───────┐                    │
                         │    │UnBlurAnalyzer│  ModernBERT +      │
                         │    │  (singleton) │  3 heads (CPU)     │
                         │    └──────────────┘                    │
                         │                                        │
                         │  GET /related                          │
                         │    ┌──────────────┐                    │
                         │    │ news_fetcher │  Google News RSS   │
                         │    │              │  → NewsAPI fallback │
                         │    └──────┬───────┘                    │
                         │           │ scored articles            │
                         │    ┌──────▼───────┐                    │
                         │    │  case_logic  │  Euclidean dist    │
                         │    │  summarizer  │  + GPT-3.5 / ext.  │
                         │    └──────────────┘                    │
                         │                                        │
                         │  GET /metrics   GET /health            │
                         │    ┌──────────────┐                    │
                         │    │ MetricsStore │  SQLite WAL        │
                         │    └──────────────┘                    │
                         └────────────────────────────────────────┘
```

### Request flow — analyze

```
Extension icon click
  → content.js extracts {title, body, url}
  → sidebar POST /analyze
      ├─ cache HIT  → return cached scores (< 1 ms)
      └─ cache MISS → UnBlurAnalyzer.analyze()
            → tokenize (title [SEP] body, max 512 tokens)
            → ModernBERT forward pass (CPU, ~80–200 ms)
            → softmax per head → scalar scores
            → store in SQLite, set X-Cache: MISS header
            → record_prediction() for drift monitoring
  → sidebar GET /related (parallel to rendering clickbait)
      ├─ cache HIT  → return cached articles
      └─ cache MISS → fetch_related()
            → Google News RSS search (topic keywords)
            → _is_relevant() filter (keyword overlap)
            → score each article with UnBlurAnalyzer
            → sort by 2D Euclidean distance from current article
            → determine_case() on the scored set
            → generate_summary() (GPT-3.5 or extractive fallback)
  → sidebar renders Chart.js scatter + summary cards
```

---

## ML Pipeline

### Model: MultiHeadModernBERT

```
Input text  ──►  ModernBERT-base backbone  ──►  [CLS] token embedding
                 (answerdotai/ModernBERT-base)       (768-dim)
                                                        │
                         ┌──────────────────────────────┤
                         │              │               │
                   clickbait_head  leaning_head  sentiment_head
                   (768→256→2)    (768→256→3)   (768→256→3)
                         │              │               │
                    softmax(2)    softmax(3)      softmax(3)
                         │              │               │
                  P(clickbait)   P(left/ctr/right) P(neg/neu/pos)
```

**Why ModernBERT?**
ModernBERT-base uses alternating local/global attention, RoPE positional embeddings, and a 512-token efficient context — giving BERT-level text understanding with 2–3× faster inference than standard BERT on CPU. It was released by Answer.AI in late 2024 and outperforms DeBERTa-v3-base on most classification benchmarks while being lighter.

**Why multi-task?**
Training all three heads jointly over a shared backbone forces the encoder to learn representations that capture both political framing *and* sensationalism simultaneously. This improves generalization vs. three separate models and reduces memory footprint at inference time (one backbone, not three).

**Output → score mapping:**
| Head | Raw output | Mapped score |
|------|------------|--------------|
| clickbait | P(class=1) | × 100 → 0–100% |
| leaning | weighted avg | −1×P(left) + 0×P(center) + 1×P(right) → [−1, +1] |
| sentiment | weighted avg | −1×P(neg) + 0×P(neu) + 1×P(pos) → [−1, +1] |

The weighted average for leaning/sentiment preserves ordinality (left < center < right) and produces a continuous score rather than a hard category, enabling the 2D scatter chart.

### Training

The Colab notebook at `model/UnblurNews_Training.ipynb` handles the full training loop:

1. **Clickbait** — `christophsonntag/clickbait` (~32k headlines)
2. **Political leaning** — `cajcodes/political-news-dataset` (with 4-source fallback chain)
3. **Sentiment** — `cardiffnlp/tweet_eval` sentiment subset (~45k)

Training strategy:
- Phase 1: each head trained independently for 3 epochs (prevents early interference between tasks)
- Phase 2: all heads fine-tuned jointly for 2 epochs (multi-task alignment)
- Mixed-precision (fp16) throughout for ~40% memory reduction on T4 GPU
- AdamW with linear warmup + cosine decay schedule

Exports: HuggingFace backbone dir + `task_heads.pt` + `model_full.pt` + tokenizer → zipped for download.

### Model Loading (backend)

`analyzer.py` supports two formats transparently:

```
backend/models/
├── config.json           ← HF backbone config
├── model.safetensors     ← backbone weights (safetensors, not pickle)
├── tokenizer.json        ← tokenizer
├── task_heads.pt         ← head weights only (state dict)
└── model_full.pt         ← full checkpoint (backbone + heads, from training)
```

The loader tries Format 1 (HF + heads) first, falls back to Format 2 (full .pt) if no `config.json`. This lets you use the training checkpoint directly without a conversion step.

**Why safetensors?**
`.safetensors` cannot execute arbitrary Python code during deserialization — unlike `.pt` (pickle-based). For any model weights you distribute or receive, this is a meaningful security improvement.

---

## MLOps & Evaluation

### Offline Evaluation (`backend/evaluate.py`)

Run after every model update to catch regressions before deploying:

```bash
python backend/evaluate.py
# or with a specific checkpoint:
python backend/evaluate.py --model ./backend/models/v2
```

The harness runs 30 hand-labelled test articles through all three heads and reports:

| Metric | What it tells you |
|--------|------------------|
| **Accuracy** | Fraction of examples classified correctly |
| **Macro F1** | Per-class F1 averaged equally — robust to class imbalance |
| **Confusion matrix** | Which specific errors the model makes (e.g. left↔center confusion) |
| **Inference latency** | avg + p95 ms per example (catches regressions from model size changes) |

Results are saved to `evaluation_results.json` for version tracking. Check this file into git alongside model checkpoints to maintain a paper trail of model performance over time.

**Why Macro F1 over accuracy?**
The test set intentionally has more center-leaning and negative-sentiment examples (reflecting real-world news distribution). Macro F1 gives equal weight to rare classes (e.g. "clickbait") so a model that predicts "not clickbait" for everything cannot score well.

### Runtime Metrics (`backend/metrics.py` + `GET /metrics`)

Every API request is timed by the FastAPI middleware and stored in a local SQLite database (`cache/unblur_metrics.db`). Hit `/metrics` to get:

```json
{
  "requests": {
    "total": 1420,
    "cache_hit_rate": 67.3,
    "error_rate_pct": 0.1
  },
  "latency_ms_24h": {
    "/analyze": { "p50": 94, "p95": 210, "p99": 380, "mean": 108 },
    "/related":  { "p50": 320, "p95": 750, "p99": 900, "mean": 350 }
  },
  "predictions_24h": {
    "count": 470,
    "clickbait_distribution": {
      "0_to_33": { "count": 310, "pct": 66.0 },
      "33_to_67": { "count": 95, "pct": 20.2 },
      "67_to_100": { "count": 65, "pct": 13.8 }
    },
    "political_distribution": { ... },
    "sentiment_distribution": { ... }
  }
}
```

**Why track prediction distributions?**
Distribution shift is one of the earliest signals of model drift. If the `political_distribution` bucket for "right" suddenly spikes from 15% to 45%, it likely means the news source mix in Google News changed, not that the world became more conservative overnight. Tracking this lets you trigger a re-evaluation before users notice degraded results.

**Why SQLite over Prometheus?**
For a single-instance service, SQLite is zero-dependency, zero-config, and the data lives in the same place as the cache. The `/metrics` JSON schema mirrors what Prometheus expects (counters, histograms), so migrating to a Prometheus exporter when you scale horizontally is a ~1-day task.

---

## Infrastructure & Optimization Decisions

### 1. SQLite 24-hour Cache

**What:** Every `/analyze` result is stored keyed by URL hash (SHA-256). Every `/related` result is stored keyed by `{topic}|{political_score}|{sentiment_score}`.

**Why:** The most expensive operation is running `ModernBERT` inference for 10+ articles in `/related`. For popular articles (e.g. a breaking news story), the first user pays ~2 seconds; every subsequent user pays ~5 ms. At typical single-user extension usage this hits a 60–70% cache hit rate within a few hours.

**Trade-off:** 24h TTL means stale coverage data for very long-lived stories. Tunable via `TTL_SECONDS` in `cache.py`.

### 2. Singleton Model (no per-request loading)

**What:** `UnBlurAnalyzer.get_instance()` loads the model exactly once at FastAPI startup and holds it in memory for the process lifetime.

**Why:** Loading ModernBERT from disk takes ~2–4 seconds. Per-request loading would make every cache miss unbearably slow. The singleton pattern is safe here because PyTorch's `torch.no_grad()` inference is read-only — the weights are never mutated.

### 3. Relevance Filter Before Scoring

**What:** `news_fetcher._is_relevant()` drops articles whose title+snippet share no keywords with the search query, before the expensive model scoring step.

**Why:** Google News RSS and NewsAPI occasionally return tangentially related results (e.g. a story about "OpenAI funding" returning articles about "funding" in general). Filtering early means we score fewer articles and the downstream 2D-distance sort is more meaningful.

### 4. 2D Euclidean Distance Ranking

**What:** Related articles are sorted by `sqrt((Δpolitical)² + (Δsentiment)²)` from the current article's position — most different perspective shown first.

**Why:** This directly operationalizes the concept of "opposing viewpoint". Sorting by relevancy alone (NewsAPI default) would show you the most similar coverage, defeating the purpose of the extension.

### 5. Mixed-Precision Training (fp16)

**What:** The Colab training notebook uses `torch.cuda.amp.autocast()` + `GradScaler`.

**Why:** On a T4 GPU, fp16 training cuts memory usage by ~40% and training time by ~30% with no meaningful accuracy loss (the scaler handles underflow automatically). This lets you fit a full batch on a free-tier Colab GPU.

### 6. Safetensors for Weight Persistence

**What:** The HuggingFace backbone is saved with `model.save_pretrained()` which writes `model.safetensors` instead of a pickle `.pt` file.

**Why:** Pickle files can execute arbitrary code during deserialization — a known supply-chain attack vector for ML models. `safetensors` is a zero-copy, memory-mapped format with no code execution path.

### 7. Chunked Tokenization (512-token max)

**What:** `analyzer.py` encodes `"title [SEP] body"` with `truncation=True, max_length=512`.

**Why:** ModernBERT supports up to 8192 tokens, but news articles rarely need more than 512 for bias/sentiment classification. Capping at 512 keeps inference at ~80–200 ms on CPU; using 8192 would push it to several seconds. The `[SEP]` delimiter helps the model distinguish title framing (higher signal for clickbait/leaning) from body text.

### 8. Graceful Degradation

**What:** If the model fails to load (missing files, OOM, corrupted weights), the API returns `503` with a human-readable error instead of crashing. `model_loaded=False` is visible in `/health`.

**Why:** The extension is still useful for news discovery even without bias scoring. A crash would give the user a blank sidebar with no explanation.

### 9. Non-Root Docker User

**What:** The Dockerfile creates a `unblur` user (UID 1001) and drops privileges before starting uvicorn.

**Why:** If an attacker exploited a deserialization bug in a model checkpoint or an RCE in a dependency, running as root inside the container would give them full control of the host (via mounted volumes). Running as an unprivileged user limits the blast radius to the `/app` directory.

### 10. WAL Mode for SQLite

**What:** `metrics.py` sets `PRAGMA journal_mode=WAL` when initializing the metrics database.

**Why:** WAL (Write-Ahead Logging) allows concurrent readers and a single writer without blocking each other. The default rollback-journal mode would cause read requests to block while the middleware writes a latency record — adding measurable overhead on every request.

---

## Project Structure

```
UnBlur/
├── backend/
│   ├── main.py           FastAPI app — endpoints, timing middleware
│   ├── analyzer.py       UnBlurAnalyzer singleton (ModernBERT multi-task)
│   ├── cache.py          SQLite 24-hour response cache
│   ├── metrics.py        SQLite request/prediction metrics store
│   ├── evaluate.py       Offline model evaluation harness (30 labelled examples)
│   ├── case_logic.py     Deterministic echo-chamber case classifier
│   ├── news_fetcher.py   Google News RSS + NewsAPI fallback + relevance filter
│   ├── summarizer.py     GPT-3.5 summary + extractive fallback
│   ├── requirements.txt
│   └── models/           ← place fine-tuned model files here
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── task_heads.pt
│
├── extension/
│   ├── manifest.json     Manifest V3 (Chrome + Firefox)
│   ├── background.js     Service worker — icon click → TOGGLE_SIDEBAR
│   ├── content.js        Article extraction + iframe sidebar injection
│   ├── options.html      Backend URL settings page
│   └── sidebar/
│       ├── sidebar.html  Sidebar UI (420px iframe)
│       ├── sidebar.js    API calls + Chart.js rendering
│       ├── sidebar.css   Dark theme
│       └── chart.umd.min.js  Bundled Chart.js (MV3 CSP requires local)
│
├── cache/                SQLite databases (created automatically)
│   ├── unblur.db         Response cache (24h TTL)
│   └── unblur_metrics.db Request metrics (7-day raw, then aggregated)
│
├── Dockerfile            Production container (python:3.11-slim, non-root)
├── docker-compose.yml    Compose with volume mounts + healthcheck
├── .env.example          Environment variable template
└── evaluation_results.json  Last evaluation run output (git-tracked)
```

Model training lives one level up at `../model/`:
```
model/
├── UnblurNews_Training.ipynb   Google Colab training notebook
├── export_for_backend.py       Converts training checkpoint → backend format
├── multi_head_model.py         Model architecture definition
├── inference.py                Inference utilities
└── saved/                      Training checkpoints
```

---

## Setup

### Option A — Local (venv)

```bash
cd UnBlur

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set NEWSAPI_KEY (free at https://newsapi.org)
#              optionally set OPENAI_API_KEY for GPT-3.5 summaries

# Place model files in backend/models/ (see "Getting Model Files" below)

# Start the server
uvicorn backend.main:app --reload --port 8000
```

### Option B — Docker

```bash
cd UnBlur
cp .env.example .env   # fill in keys

docker compose up --build
# API available at http://localhost:8000
```

### Getting Model Files

**Option 1 — Train your own (recommended for best accuracy)**

Open `model/UnblurNews_Training.ipynb` in Google Colab (T4 GPU, ~2–4 hours):
1. Run all cells top-to-bottom
2. The final cell downloads `UnBlur_model.zip`
3. Unzip into `UnBlur/backend/models/`

**Option 2 — Use the pre-trained checkpoint**

If `backend/models/` already contains `model.safetensors`, `task_heads.pt`, and `tokenizer.json`, you're ready to go.

**Option 3 — Export from existing training checkpoint**

```bash
# From project root (UnblurNews/)
python model/export_for_backend.py
# Writes to UnBlur/backend/models/ automatically
```

### Load the Extension

**Chrome / Edge:**
1. `chrome://extensions/` → Enable **Developer mode**
2. **Load unpacked** → select `UnBlur/extension/`

**Firefox:**
1. `about:debugging` → **This Firefox** → **Load Temporary Add-on**
2. Select `UnBlur/extension/manifest.json`

---

## API Reference

### `POST /analyze`

Analyze a news article for clickbait, political leaning, and sentiment.

**Request body:**
```json
{
  "title": "GOP Tax Cuts Threaten Families",
  "body":  "Economists warn the plan disproportionately benefits...",
  "url":   "https://example.com/article"
}
```

**Response:**
```json
{
  "clickbait_pct":   12.4,
  "political_score": -0.62,
  "sentiment_score": -0.41,
  "case":            "balanced",
  "cached":          false
}
```

Response header `X-Cache: HIT|MISS` indicates cache status.
Response header `X-Response-Time-Ms: 142.3` provides server-side latency.

---

### `GET /related`

Fetch related articles sorted by distance from the current article's bias position.

**Query params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `topic` | string | required | Search query (article title / keywords) |
| `political_score` | float [-1,1] | 0.0 | Current article's political score |
| `sentiment_score` | float [-1,1] | 0.0 | Current article's sentiment score |

**Response:**
```json
{
  "articles": [
    {
      "title": "Tax Reform Boosts Job Growth, Study Finds",
      "url": "https://...",
      "source": "Fox News",
      "political_score": 0.71,
      "sentiment_score": 0.55,
      "snippet": "..."
    }
  ],
  "summary": "Coverage is predominantly right-leaning...",
  "dominant_leaning": "right",
  "dominant_pct": 60.0,
  "case": "contradiction"
}
```

---

### `GET /metrics`

Returns request latency percentiles, cache statistics, and prediction distributions for the last 24 hours.

```bash
curl http://localhost:8000/metrics | python3 -m json.tool
```

---

### `GET /health`

Liveness + readiness probe.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_error": null,
  "uptime_s": 3612.4,
  "cache_db_bytes": 245760
}
```

---

## Extension Usage

1. Navigate to any news article
2. Click the **UnBlur** icon in the browser toolbar
3. The sidebar slides in from the right (article content shifts left)
4. Results appear in ~1–3 seconds (< 100 ms on cache hit)

**Settings:** Right-click the icon → **Options** to set a custom backend URL (useful when running on a remote server or different port).

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEWSAPI_KEY` | No* | — | NewsAPI key (newsapi.org). Falls back to Google News RSS if unset. |
| `OPENAI_API_KEY` | No | — | Enables GPT-3.5 summaries. Falls back to extractive summary. |
| `MODEL_PATH` | No | `./backend/models` | Path to model directory |
| `CACHE_DB` | No | `./cache/unblur.db` | SQLite cache database path |
| `METRICS_DB` | No | `./cache/unblur_metrics.db` | SQLite metrics database path |
| `PORT` | No | `8000` | Server port |

*NewsAPI is optional — Google News RSS works without any key.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Model backbone | ModernBERT-base | Faster CPU inference than BERT/RoBERTa, efficient attention |
| Multi-task heads | PyTorch `nn.Sequential` | Minimal, auditable; no framework lock-in |
| API server | FastAPI + uvicorn | Async, auto-generated OpenAPI docs, Pydantic validation |
| Caching | SQLite (stdlib) | Zero dependencies; sufficient for single-instance |
| Metrics | SQLite WAL mode | Concurrent reads without blocking writes |
| Containerisation | Docker + Compose | Reproducible environment; model weights mounted as volumes |
| Extension | Vanilla JS + Manifest V3 | No build step; Chart.js bundled locally (MV3 CSP compliance) |
| Charts | Chart.js (scatter) | Lightweight, no React/Vue dependency |
| Summaries | OpenAI GPT-3.5 | Best quality; extractive fallback for offline use |
| News search | Google News RSS | Free, no API key, topic-specific (not generic top-news) |
| Weight format | safetensors | No-code-execution deserialization; memory-mapped loading |

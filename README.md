---
title: UnBlur API
sdk: docker
app_port: 8000
pinned: false
---

# UnBlur: Real-Time Media Bias and Echo Chamber Detector

UnBlur is a browser extension backed by a fine-tuned NLP model. It reads a news article, scores it for clickbait, political leaning, and sentiment, then pulls related coverage from across the political spectrum and shows you where each version of the story sits. The goal is to make an echo chamber visible while you're still in it.

**Live API:** https://kushal0532-unblur.hf.space ([health](https://kushal0532-unblur.hf.space/health)). Hosted on Hugging Face Spaces (Docker SDK), with an Upstash Redis cache and the model loaded from a private HF Hub repo. The extension ships pointed at this URL. See [Environment Variables](#environment-variables) if you want to run it against a local backend.

---

## What It Does

When you click the UnBlur icon on a news article, the extension:

1. Extracts the article title and body from the DOM
2. Sends it to the FastAPI backend
3. Runs the article through a fine-tuned **ModernBERT** model with three task-specific heads
4. Fetches 10 related articles from **Google News RSS** (or NewsAPI) and scores them the same way
5. Classifies the media landscape as one of four cases: Echo Chamber, Contradiction, Internal Split, or Balanced
6. Renders a sidebar with:
   - **Clickbait score**, a 0-100% bar (green to red gradient)
   - **Bias map**, a 2D scatter chart (political x sentiment) plotting every article
   - **Case diagnosis**, an icon and description of the landscape
   - **Summary**, the dominant leaning stats and links to the most opposing perspectives

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
                                                │ HTTPS (HF Space,
                                                │  or local :8000)
                         ┌──────────────────────▼────────────────┐
                         │  FastAPI Backend (uvicorn, on Spaces)  │
                         │                                        │
                         │  POST /analyze                         │
                         │    ┌──────────────┐                    │
                         │    │ ArticleCache │◄── Redis 24h TTL,  │
                         │    │  (get_cache) │    SQLite fallback │
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

### Request flow: analyze

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
ModernBERT-base uses alternating local/global attention, RoPE positional embeddings, and a 512-token efficient context. That gives BERT-level text understanding at roughly 2-3x the CPU inference speed of standard BERT. Answer.AI released it in late 2024; it beats DeBERTa-v3-base on most classification benchmarks and is lighter.

**Why multi-task?**
Training all three heads over a shared backbone forces the encoder to represent political framing and sensationalism at the same time. In practice this generalizes better than three separate models and uses less memory at inference, since there is one backbone instead of three.

**Output to score mapping:**
| Head | Raw output | Mapped score |
|------|------------|--------------|
| clickbait | P(class=1) | × 100 → 0–100% |
| leaning | weighted avg | −1×P(left) + 0×P(center) + 1×P(right) → [−1, +1] |
| sentiment | weighted avg | −1×P(neg) + 0×P(neu) + 1×P(pos) → [−1, +1] |

The weighted average for leaning and sentiment keeps the ordering (left < center < right) and produces a continuous score instead of a hard category, which is what the 2D scatter chart needs.

### Training

The Colab notebook at `model/UnblurNews_Training.ipynb` runs the full training loop:

1. **Clickbait**: `christophsonntag/clickbait` (~32k headlines)
2. **Political leaning**: `cajcodes/political-news-dataset` (with a 4-source fallback chain)
3. **Sentiment**: `cardiffnlp/tweet_eval` sentiment subset (~45k)

Training strategy:
- Phase 1: each head trained independently for 3 epochs, so the tasks don't interfere early on
- Phase 2: all heads fine-tuned jointly for 2 epochs for multi-task alignment
- Mixed-precision (fp16) throughout, roughly 40% less memory on a T4 GPU
- AdamW with linear warmup and cosine decay

Exports: HuggingFace backbone dir, `task_heads.pt`, `model_full.pt`, and the tokenizer, zipped for download.

### Model Loading (backend)

`analyzer.py` handles two formats:

```
backend/models/
├── config.json           ← HF backbone config
├── model.safetensors     ← backbone weights (safetensors, not pickle)
├── tokenizer.json        ← tokenizer
├── task_heads.pt         ← head weights only (state dict)
└── model_full.pt         ← full checkpoint (backbone + heads, from training)
```

The loader tries Format 1 (HF + heads) first and falls back to Format 2 (full .pt) when there's no `config.json`. That means the training checkpoint works directly, with no conversion step.

**Why safetensors?**
A `.safetensors` file cannot execute Python code while it deserializes, which a pickle-based `.pt` file can. For weights you distribute or download, that removes a real attack path.

---

## MLOps and Evaluation

### Offline Evaluation (`backend/evaluate.py`)

Run this after every model update to catch regressions before they ship:

```bash
python backend/evaluate.py
# or with a specific checkpoint:
python backend/evaluate.py --model ./backend/models/v2
```

The harness runs 30 hand-labelled test articles through all three heads and reports:

| Metric | What it tells you |
|--------|------------------|
| **Accuracy** | Fraction of examples classified correctly |
| **Macro F1** | Per-class F1 averaged equally, robust to class imbalance |
| **Confusion matrix** | Which errors the model makes (e.g. left/center confusion) |
| **Inference latency** | avg and p95 ms per example, catches regressions from model size changes |

Results go to `evaluation_results.json` for version tracking. Check that file into git alongside model checkpoints to keep a record of model performance over time.

**Why Macro F1 instead of accuracy?**
The test set has more center-leaning and negative-sentiment examples on purpose, since that reflects the real news distribution. Macro F1 weights rare classes (like "clickbait") equally, so a model that always predicts "not clickbait" still scores badly.

### Runtime Metrics (`backend/metrics.py` and `GET /metrics`)

FastAPI middleware times every request and stores it in a local SQLite database (`cache/unblur_metrics.db`). Hit `/metrics` for:

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
Distribution shift is one of the earliest signs of model drift. If the "right" bucket in `political_distribution` jumps from 15% to 45%, the likely cause is a change in the Google News source mix, not the world turning conservative overnight. Watching this lets you re-evaluate before users notice worse results.

**Why SQLite instead of Prometheus?**
For a single-instance service, SQLite is zero-dependency, zero-config, and the data sits next to the cache. The `/metrics` JSON matches what Prometheus expects (counters, histograms), so swapping in a Prometheus exporter once you scale horizontally is about a day of work.

---

## Infrastructure and Optimization Decisions

### 1. Redis cache (SQLite fallback), 24-hour TTL

**What:** Every `/analyze` result is stored keyed by URL hash (SHA-256). Every `/related` result is keyed by `{topic}|{political_score}|{sentiment_score}`. `cache.get_cache()` returns `RedisArticleCache` when `REDIS_URL` is set (production, Upstash), otherwise `ArticleCache` (SQLite, local dev). Same interface, same TTL, same key scheme.

**Why:** The expensive step is running ModernBERT inference for 10+ articles in `/related`. For a popular story the first user waits about 2 seconds; everyone after that waits about 5 ms. Redis gives a shared cache across HF Space restarts and replicas; SQLite keeps local dev dependency-free.

**Trade-off:** A 24h TTL means stale coverage data for very long-lived stories. Tune it with `TTL_SECONDS` in `cache.py`.

### 2. Singleton model (no per-request loading)

**What:** `UnBlurAnalyzer.get_instance()` loads the model once at FastAPI startup and keeps it in memory for the process lifetime.

**Why:** Loading ModernBERT from disk takes 2-4 seconds. Loading per request would make every cache miss painfully slow. The singleton is safe here because `torch.no_grad()` inference is read-only and never mutates the weights.

### 3. Relevance filter before scoring

**What:** `news_fetcher._is_relevant()` drops articles whose title and snippet share no keywords with the search query, before the model scores them.

**Why:** Google News RSS and NewsAPI sometimes return loosely related results (a search for "OpenAI funding" returning articles about "funding" generally). Filtering early means fewer articles to score and a more meaningful 2D-distance sort afterward.

### 4. 2D Euclidean distance ranking

**What:** Related articles are sorted by `sqrt((Δpolitical)² + (Δsentiment)²)` from the current article's position, most different perspective first.

**Why:** That is a concrete definition of "opposing viewpoint". Sorting by relevance alone (the NewsAPI default) surfaces the most similar coverage, which is the opposite of what the extension is for.

### 5. Mixed-precision training (fp16)

**What:** The Colab training notebook uses `torch.cuda.amp.autocast()` and `GradScaler`.

**Why:** On a T4 GPU, fp16 training cuts memory use by about 40% and training time by about 30%, with no meaningful accuracy loss (the scaler handles underflow). That lets a full batch fit on a free-tier Colab GPU.

### 6. Safetensors for weight persistence

**What:** The HuggingFace backbone is saved with `model.save_pretrained()`, which writes `model.safetensors` rather than a pickle `.pt`.

**Why:** Pickle files can execute code during deserialization, a known supply-chain vector for ML models. `safetensors` is a zero-copy, memory-mapped format with no code execution path.

### 7. Chunked tokenization (512-token max)

**What:** `analyzer.py` encodes `"title [SEP] body"` with `truncation=True, max_length=512`.

**Why:** ModernBERT supports up to 8192 tokens, but news articles rarely need more than 512 for bias and sentiment classification. Capping at 512 keeps inference at 80-200 ms on CPU; 8192 would push it to several seconds. The `[SEP]` delimiter helps the model separate title framing (a strong signal for clickbait and leaning) from body text.

### 8. Graceful degradation

**What:** If the model fails to load (missing files, OOM, corrupted weights), the API returns `503` with a readable error instead of crashing. `/health` reports `model_loaded=False`.

**Why:** The extension is still useful for news discovery without bias scoring. A crash would leave the user with a blank sidebar and no explanation.

### 9. Non-root Docker user

**What:** The Dockerfile creates a `unblur` user (UID 1001) and drops privileges before starting uvicorn.

**Why:** If an attacker exploited a deserialization bug in a checkpoint or an RCE in a dependency, running as root in the container would hand them the host through mounted volumes. An unprivileged user limits the damage to `/app`.

### 10. WAL mode for SQLite

**What:** `metrics.py` sets `PRAGMA journal_mode=WAL` when it initializes the metrics database.

**Why:** WAL lets concurrent readers and a single writer proceed without blocking each other. The default rollback-journal mode would block read requests while the middleware writes a latency record, adding overhead to every request.

---

## Project Structure

The repo root doubles as the Hugging Face Space: this README's YAML frontmatter is the Space card, and the root `Dockerfile` is what Spaces builds.

```
.
├── backend/
│   ├── main.py           FastAPI app — endpoints, timing middleware
│   ├── analyzer.py       UnBlurAnalyzer singleton (ModernBERT multi-task);
│   │                       loads from MODEL_REPO_ID (HF Hub) or local MODEL_PATH
│   ├── cache.py          get_cache() → RedisArticleCache (if REDIS_URL set)
│   │                       or ArticleCache (SQLite fallback), 24h TTL
│   ├── metrics.py        SQLite request/prediction metrics store
│   ├── evaluate.py       Offline model evaluation harness (30 labelled examples)
│   ├── case_logic.py     Deterministic echo-chamber case classifier
│   ├── news_fetcher.py   Google News RSS + NewsAPI fallback + relevance filter
│   ├── summarizer.py     GPT-3.5 summary + extractive fallback
│   ├── requirements.txt
│   └── models/           ← place fine-tuned model files here (local dev only;
│                            unused when MODEL_REPO_ID is set)
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
│       ├── sidebar.js    API calls + Chart.js rendering; BACKEND_URL defaults
│       │                   to the live HF Space, overridable via options page
│       ├── sidebar.css   Dark neomorphic theme (Manrope / JetBrains Mono)
│       └── chart.umd.min.js  Bundled Chart.js (MV3 CSP requires local)
│
├── testbench/            Load-testing harness (simulated users, concurrency
│                           sweeps, latency/cache-hit charts) — see
│                           specs/testbench/SPEC.md
│
├── cache/                SQLite databases (created automatically, local dev
│   │                       fallback only — production uses Redis)
│   ├── unblur.db         Response cache (24h TTL)
│   └── unblur_metrics.db Request metrics (7-day raw, then aggregated)
│
├── Dockerfile            Production container (python:3.11-slim, non-root),
│                           builds from repo root for HF Spaces Docker SDK
├── docker-compose.yml    Compose with volume mounts + healthcheck
├── .env.example          Environment variable template
└── evaluation_results.json  Last evaluation run output (git-tracked)
```

Model training lives at `model/`:
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

### Option A: local (venv)

```bash
# From repo root

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set NEWSAPI_KEY (free at https://newsapi.org)
#              optionally set OPENAI_API_KEY for GPT-3.5 summaries
#              optionally set REDIS_URL (else falls back to SQLite)
#              optionally set MODEL_REPO_ID + HF_TOKEN to pull the model from
#              a private HF Hub repo instead of local files

# Place model files in backend/models/ (see "Getting Model Files" below) —
# skip this if MODEL_REPO_ID is set

# Start the server
uvicorn backend.main:app --reload --port 8000
```

### Option B: Docker

```bash
# From repo root
cp .env.example .env   # fill in keys

docker compose up --build
# API available at http://localhost:8000
```

### Option C: Hugging Face Spaces (production)

The repo root is a Docker-SDK HF Space (this README's frontmatter is the Space card, the root `Dockerfile` is what it builds). To deploy your own:
1. Push this repo to a new HF Space (Docker SDK, matching `app_port: 8000` above)
2. Set secrets in the Space's **Settings → Repository secrets**: `REDIS_URL`, `MODEL_REPO_ID` + `HF_TOKEN` (or bake model files into the image), `NEWSAPI_KEY`, `OPENAI_API_KEY`
3. Spaces builds and starts the container automatically on push; check `/health` once it's live

### Getting Model Files

**Option 1: train your own (best accuracy)**

Open `model/UnblurNews_Training.ipynb` in Google Colab (T4 GPU, ~2–4 hours):
1. Run all cells top to bottom
2. The final cell downloads `UnBlur_model.zip`
3. Unzip into `UnBlur/backend/models/`

**Option 2: use the pre-trained checkpoint**

If `backend/models/` already contains `model.safetensors`, `task_heads.pt`, and `tokenizer.json`, you're ready to go.

**Option 3: export from an existing training checkpoint**

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

Liveness and readiness probe.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_error": null,
  "uptime_s": 3612.4,
  "cache_backend": "redis",
  "cache_db_bytes": 0
}
```

`cache_backend` is `"redis"` when `REDIS_URL` is set, otherwise `"sqlite"` (in which case `cache_db_bytes` reports the SQLite file size; it's always `0` for Redis, since size isn't tracked there).

---

## Extension Usage

1. Navigate to a news article
2. Click the **UnBlur** icon in the browser toolbar
3. The sidebar slides in from the right (article content shifts left)
4. Results appear in ~1–3 seconds (< 100 ms on a cache hit)

**Settings:** Right-click the icon → **Options** to set a custom backend URL. It ships pointed at the live HF Space (`https://kushal0532-unblur.hf.space`); point it at `http://localhost:8000` if you're running the backend locally.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEWSAPI_KEY` | No* | — | NewsAPI key (newsapi.org). Falls back to Google News RSS if unset. |
| `OPENAI_API_KEY` | No | — | Enables GPT-3.5 summaries. Falls back to extractive summary. |
| `MODEL_PATH` | No | `./backend/models` | Local model directory. Ignored if `MODEL_REPO_ID` is set. |
| `MODEL_REPO_ID` | No | — | Private HF Hub repo id (e.g. `you/unblur-model`) to download the model from at startup, instead of `MODEL_PATH`. |
| `HF_TOKEN` | Only with `MODEL_REPO_ID` on a private repo | — | Hugging Face access token for downloading the model. |
| `REDIS_URL` | No | — | Redis/Upstash connection URL (`rediss://` for TLS). If unset, falls back to the SQLite cache. |
| `CACHE_DB` | No | `./cache/unblur.db` | SQLite cache database path (used only when `REDIS_URL` is unset). |
| `METRICS_DB` | No | `./cache/unblur_metrics.db` | SQLite metrics database path |
| `PORT` | No | `8000` | Server port |

*NewsAPI is optional. Google News RSS works without any key.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Model backbone | ModernBERT-base | Faster CPU inference than BERT/RoBERTa, efficient attention |
| Multi-task heads | PyTorch `nn.Sequential` | Minimal, auditable; no framework lock-in |
| API server | FastAPI + uvicorn | Async, auto-generated OpenAPI docs, Pydantic validation |
| Caching | Redis (Upstash), SQLite fallback | Shared cache across Space restarts in prod; zero-dependency fallback for local dev |
| Metrics | SQLite WAL mode | Concurrent reads without blocking writes |
| Containerisation | Docker + Compose, deployed to HF Spaces (Docker SDK) | Reproducible environment; model weights mounted as volumes locally, pulled from HF Hub in prod |
| Extension | Vanilla JS + Manifest V3 | No build step; Chart.js bundled locally (MV3 CSP compliance) |
| Extension UI | Manrope + JetBrains Mono, dark neomorphic | Imported from a Claude Design mock, hand-ported into the existing sidebar.css/js (no framework) |
| Charts | Chart.js (scatter) | Lightweight, no React/Vue dependency |
| Summaries | OpenAI GPT-3.5 | Best quality; extractive fallback for offline use |
| News search | Google News RSS | Free, no API key, topic-specific (not generic top-news) |
| Weight format | safetensors | No-code-execution deserialization; memory-mapped loading |

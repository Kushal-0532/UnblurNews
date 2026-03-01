# ── UnBlur Backend — Dockerfile ──────────────────────────────────
#
# Design choices:
#   python:3.11-slim   — Debian base without GUI or dev headers (~55 MB).
#                        Smaller than the full image, larger than Alpine,
#                        but Alpine's musl libc causes subtle issues with
#                        PyTorch wheels that ship glibc-linked C extensions.
#
#   Non-root user      — Security best practice: limits blast radius if
#                        the app is ever exploited (container cannot write
#                        to system dirs or escalate privileges).
#
#   Layer caching      — requirements.txt copied first so Docker cache
#                        is invalidated only when deps change, not on every
#                        source code change.  Rebuilds stay fast.
#
#   --no-cache-dir     — Keeps the image smaller by not storing the pip
#                        download cache inside the layer.

FROM python:3.11-slim

# System deps needed by some PyTorch / transformers sub-packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies (cache-friendly layer) ───────────
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────
COPY backend/ ./backend/

# ── Persistent directories (mounted as volumes in compose) ────────
RUN mkdir -p cache backend/models

# ── Non-root user ─────────────────────────────────────────────────
RUN useradd -m -u 1001 unblur \
    && chown -R unblur:unblur /app
USER unblur

# ── Runtime ───────────────────────────────────────────────────────
ENV PYTHONPATH=/app
EXPOSE 8000

# --workers 1: single process because the model is held in-memory and
# multi-process would duplicate the ~900 MB footprint per worker.
# Use --workers N only if you add a shared cache (Redis) and model
# serving layer (TorchServe / Triton).
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

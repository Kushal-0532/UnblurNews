"""
analyzer.py — UnBlurAnalyzer: Singleton Inference Wrapper
=========================================================

Loads the fine-tuned MultiHeadModernBERT model and exposes a clean
`analyze(title, body)` API used by the FastAPI backend.

Expected model files in MODEL_PATH (default: ./models/):
    config.json          ModernBERT backbone architecture
    model.safetensors    ModernBERT backbone weights
    tokenizer.json       Tokenizer vocab / config
    task_heads.pt        State dict for the 3 classification heads

Head output classes:
    clickbait_head   2 classes  → [not-clickbait, clickbait]
    leaning_head     3 classes  → [left, center, right]
    sentiment_head   3 classes  → [negative, neutral, positive]
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ── Default model directory ──────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_BACKEND_DIR, "models")


# ═══════════════════════════════════════════════════════════════
#  INTERNAL MODEL DEFINITION  (inference-only, no training utils)
# ═══════════════════════════════════════════════════════════════

class _MultiHeadModel(nn.Module):
    """
    ModernBERT backbone + 3 classification heads.
    Architecture mirrors MultiHeadModernBERT from model/multi_head_model.py.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone

        def _build_head(num_classes: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        self.clickbait_head = _build_head(2)
        self.leaning_head   = _build_head(3)
        self.sentiment_head = _build_head(3)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] token
        return {
            "clickbait": self.clickbait_head(cls),
            "leaning":   self.leaning_head(cls),
            "sentiment": self.sentiment_head(cls),
        }


# ═══════════════════════════════════════════════════════════════
#  UnBlurAnalyzer — public API
# ═══════════════════════════════════════════════════════════════

class UnBlurAnalyzer:
    """
    Singleton wrapper around the fine-tuned MultiHeadModernBERT model.

    Usage
    -----
    analyzer = UnBlurAnalyzer.get_instance()
    result   = analyzer.analyze(title="...", body="...")

    The singleton is initialized once on first call to get_instance().
    model_loaded is False if initialization failed (missing files).
    """

    _instance: "UnBlurAnalyzer | None" = None

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self._model_path = model_path
        self._model: _MultiHeadModel | None = None
        self._tokenizer = None
        self._device = torch.device("cpu")
        self.model_loaded = False
        self._load_error: str | None = None
        self._try_load()

    # ── Singleton factory ────────────────────────────────────────
    @classmethod
    def get_instance(cls, model_path: str = DEFAULT_MODEL_PATH) -> "UnBlurAnalyzer":
        """Return the shared UnBlurAnalyzer instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    # ── Model loading ────────────────────────────────────────────
    def _try_load(self) -> None:
        """
        Attempt to load the model. Sets model_loaded=True on success.
        On failure, stores the error message and continues (graceful degradation).
        """
        try:
            self._load()
            self.model_loaded = True
        except FileNotFoundError as exc:
            self._load_error = str(exc)
            print(f"[UnBlurAnalyzer] WARNING — model not loaded: {exc}")
        except Exception as exc:
            self._load_error = f"Unexpected error loading model: {exc}"
            print(f"[UnBlurAnalyzer] ERROR: {self._load_error}")

    def _load(self) -> None:
        """
        Load backbone + task heads. Raises FileNotFoundError if files are missing.

        Supported formats (tried in order):
          1. HuggingFace dir + task_heads.pt  (spec format)
               models/config.json + model.safetensors + tokenizer.json + task_heads.pt
          2. Full training checkpoint  (produced by model/train_*.py)
               models/model_full.pt  →  {"model_state_dict": ..., "config": ...}
        """
        if not os.path.isdir(self._model_path):
            raise FileNotFoundError(
                f"Model directory not found: {self._model_path}\n"
                "Place fine-tuned model files there:\n"
                "  config.json, model.safetensors, tokenizer.json, task_heads.pt\n"
                "  OR model_full.pt (full training checkpoint)"
            )

        heads_path    = os.path.join(self._model_path, "task_heads.pt")
        full_ckpt_path = os.path.join(self._model_path, "model_full.pt")
        has_hf_backbone = os.path.exists(os.path.join(self._model_path, "config.json"))

        # ── Format 1: HuggingFace backbone + task_heads.pt ──────
        if has_hf_backbone and os.path.exists(heads_path):
            print(f"[UnBlurAnalyzer] Loading HuggingFace backbone from {self._model_path} ...")
            backbone = AutoModel.from_pretrained(self._model_path)
            hidden_size = backbone.config.hidden_size
            self._model = _MultiHeadModel(backbone, hidden_size)

            print(f"[UnBlurAnalyzer] Loading task heads from {heads_path} ...")
            state = torch.load(heads_path, map_location=self._device, weights_only=True)

            # Strip "model_state_dict" wrapper if present
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            missing, _ = self._model.load_state_dict(state, strict=False)
            if missing:
                print(f"[UnBlurAnalyzer] Note: {len(missing)} keys not in checkpoint")

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        # ── Format 2: full training checkpoint (.pt) ─────────────
        elif os.path.exists(full_ckpt_path):
            print(f"[UnBlurAnalyzer] Loading full checkpoint from {full_ckpt_path} ...")
            checkpoint = torch.load(full_ckpt_path, map_location=self._device, weights_only=False)
            cfg   = checkpoint.get("config", {})
            state = checkpoint.get("model_state_dict", checkpoint)

            model_name = cfg.get("model_name", "answerdotai/ModernBERT-base")
            print(f"[UnBlurAnalyzer] Loading backbone '{model_name}' from HuggingFace ...")
            backbone = AutoModel.from_pretrained(model_name)
            hidden_size = backbone.config.hidden_size
            self._model = _MultiHeadModel(backbone, hidden_size)

            missing, _ = self._model.load_state_dict(state, strict=False)
            if missing:
                print(f"[UnBlurAnalyzer] Note: {len(missing)} keys not in checkpoint")

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        else:
            raise FileNotFoundError(
                f"No loadable model found in {self._model_path}.\n"
                "Expected one of:\n"
                "  • config.json + model.safetensors + tokenizer.json + task_heads.pt\n"
                "  • model_full.pt  (from model/train_multitask.py)"
            )

        # ── Set eval mode on CPU ─────────────────────────────────
        self._model.to(self._device)
        self._model.eval()
        print("[UnBlurAnalyzer] Model loaded successfully.")

    # ── Public inference API ─────────────────────────────────────
    def analyze(self, title: str, body: str) -> dict:
        """
        Analyze a news article.

        Parameters
        ----------
        title : str
            Article headline.
        body  : str
            Article body text (first ~1500 chars recommended for speed).

        Returns
        -------
        dict:
            clickbait_pct   : float   0–100   (higher = more clickbait)
            political_score : float  -1 to +1  (-1 = left, +1 = right)
            sentiment_score : float  -1 to +1  (-1 = negative, +1 = positive)

        Raises
        ------
        RuntimeError if the model was not loaded successfully.
        """
        if not self.model_loaded:
            raise RuntimeError(
                f"Model is not loaded. {self._load_error or ''}\n"
                "Check that model files exist at: " + self._model_path
            )

        # Encode as "title [SEP] body", max 512 tokens
        text = f"{title} [SEP] {body}"
        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self._model(tokens["input_ids"], tokens["attention_mask"])

        # ── Clickbait: P(class=1) × 100 ─────────────────────────
        cb_probs = torch.softmax(outputs["clickbait"], dim=1)
        clickbait_pct = cb_probs[0, 1].item() * 100

        # ── Political: weighted avg  -1×P(left) + 0×P(center) + 1×P(right)
        lean_probs = torch.softmax(outputs["leaning"], dim=1)
        political_score = (
            -1.0 * lean_probs[0, 0].item()
            +  0.0 * lean_probs[0, 1].item()
            +  1.0 * lean_probs[0, 2].item()
        )

        # ── Sentiment: weighted avg  -1×P(neg) + 0×P(neu) + 1×P(pos)
        sent_probs = torch.softmax(outputs["sentiment"], dim=1)
        sentiment_score = (
            -1.0 * sent_probs[0, 0].item()
            +  0.0 * sent_probs[0, 1].item()
            +  1.0 * sent_probs[0, 2].item()
        )

        return {
            "clickbait_pct":    round(clickbait_pct,    2),
            "political_score":  round(political_score,  4),
            "sentiment_score":  round(sentiment_score,  4),
        }


# ═══════════════════════════════════════════════════════════════
#  Quick smoke test (run directly: python backend/analyzer.py)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  UnBlurAnalyzer — Smoke Test")
    print("=" * 60)

    analyzer = UnBlurAnalyzer.get_instance()
    print(f"\nmodel_loaded = {analyzer.model_loaded}")

    if not analyzer.model_loaded:
        print(f"\nSkipping inference test — {analyzer._load_error}")
        print("\nTo run inference, place model files in:")
        print(f"  {DEFAULT_MODEL_PATH}")
        sys.exit(0)

    SAMPLES = [
        {
            "title": "You Won't BELIEVE What This Politician Just Said!",
            "body":  "Fans are shocked after the latest viral moment left everyone speechless.",
            "desc":  "Expected: high clickbait, ~neutral leaning",
        },
        {
            "title": "GOP Tax Cuts Devastate Working Families",
            "body":  "Republican spending proposals disproportionately harm low-income households.",
            "desc":  "Expected: left-leaning, negative sentiment",
        },
        {
            "title": "Federal Reserve Holds Rates Steady",
            "body":  "The Fed maintained current interest rates citing stable inflation and moderate growth.",
            "desc":  "Expected: center, neutral sentiment",
        },
    ]

    print("\n  Testing inference on 3 sample articles ...\n")
    for i, s in enumerate(SAMPLES, 1):
        print(f"  ── Sample {i} ─────────────────────────────────")
        print(f"  Title:    {s['title']}")
        print(f"  Expected: {s['desc']}")
        result = analyzer.analyze(s["title"], s["body"])
        print(f"  Result:   {json.dumps(result, indent=2)}")
        print()

    print("=" * 60)
    print("  Smoke test complete.")
    print("=" * 60)

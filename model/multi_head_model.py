"""
multi_head_model.py — Multi-Head ModernBERT Architecture
========================================================

This file defines the core neural network.  One shared ModernBERT backbone
feeds three lightweight classification "heads":

    Head 1  →  Clickbait        (2 classes: Not-Clickbait / Clickbait)
    Head 2  →  Political Leaning (3 classes: Left / Center / Right)
    Head 3  →  Sentiment         (3 classes: Negative / Neutral / Positive)

                    ┌─────────────────┐
                    │  Input Text     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  ModernBERT     │   ← shared backbone
                    │  (encoder)      │
                    └────────┬────────┘
                             │  [CLS] token
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │ Clickbait │ │ Political │ │ Sentiment │
        │   Head    │ │   Head    │ │   Head    │
        └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
              │              │              │
         0 or 1        L / C / R     Neg / Neu / Pos
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class MultiHeadModernBERT(nn.Module):
    """
    A single ModernBERT backbone with three task-specific classification heads.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. "answerdotai/ModernBERT-base".
    num_clickbait_classes : int
        Number of output classes for the clickbait head (default 2).
    num_leaning_classes : int
        Number of output classes for the political-leaning head (default 3).
    num_sentiment_classes : int
        Number of output classes for the sentiment head (default 3).
    dropout : float
        Dropout probability used inside each head (default 0.1).
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        num_clickbait_classes: int = 2,
        num_leaning_classes: int = 3,
        num_sentiment_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── 1.  Load the shared backbone ────────────────────────
        # AutoModel gives us the raw encoder (no classification layer).
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # e.g. 768

        # ── 2.  Build the three classification heads ────────────
        #        Each head is:  Dropout → Linear → ReLU → Dropout → Linear
        self.clickbait_head = self._build_head(hidden_size, num_clickbait_classes, dropout)
        self.leaning_head   = self._build_head(hidden_size, num_leaning_classes,   dropout)
        self.sentiment_head = self._build_head(hidden_size, num_sentiment_classes, dropout)

    # ────────────────────────────────────────────────────────────
    #  Helper: create one classification head
    # ────────────────────────────────────────────────────────────
    @staticmethod
    def _build_head(input_size: int, num_classes: int, dropout: float) -> nn.Sequential:
        """
        Build a small feed-forward classifier on top of the backbone.

        Architecture:
            Dropout → Linear(768 → 256) → ReLU → Dropout → Linear(256 → num_classes)
        """
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    # ────────────────────────────────────────────────────────────
    #  Get [CLS] representation from the backbone
    # ────────────────────────────────────────────────────────────
    def get_cls_embedding(self, input_ids, attention_mask):
        """
        Run text through the backbone and return the [CLS] token vector.

        The [CLS] token is always the **first** token in the sequence.
        ModernBERT (like BERT) aggregates whole-sequence information there
        during pre-training, so it works well for classification.
        """
        # outputs.last_hidden_state has shape (batch, seq_len, hidden_size)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Take the first token ([CLS]) for each example in the batch
        cls_vector = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        return cls_vector

    # ────────────────────────────────────────────────────────────
    #  Forward pass  —  choose which head to use
    # ────────────────────────────────────────────────────────────
    def forward(self, input_ids, attention_mask, task="clickbait"):
        """
        Run a forward pass through the backbone + one (or all) heads.

        Parameters
        ----------
        input_ids : Tensor  (batch, seq_len)
        attention_mask : Tensor  (batch, seq_len)
        task : str
            "clickbait"  → only the clickbait head
            "leaning"    → only the political-leaning head
            "sentiment"  → only the sentiment head
            "all"        → run ALL three heads, return a dict

        Returns
        -------
        Tensor or dict[str, Tensor]
            Raw logits (not softmaxed) for the chosen task(s).
        """
        # Step 1 — shared backbone
        cls_vector = self.get_cls_embedding(input_ids, attention_mask)

        # Step 2 — route to the correct head
        if task == "clickbait":
            return self.clickbait_head(cls_vector)

        elif task == "leaning":
            return self.leaning_head(cls_vector)

        elif task == "sentiment":
            return self.sentiment_head(cls_vector)

        elif task == "all":
            return {
                "clickbait": self.clickbait_head(cls_vector),
                "leaning":   self.leaning_head(cls_vector),
                "sentiment": self.sentiment_head(cls_vector),
            }

        else:
            raise ValueError(f"Unknown task '{task}'. Use: clickbait, leaning, sentiment, all")

    # ────────────────────────────────────────────────────────────
    #  Utility: freeze / unfreeze backbone weights
    # ────────────────────────────────────────────────────────────
    def freeze_backbone(self):
        """
        Freeze all backbone parameters so only the heads are trained.
        Useful when you want to train a new head without disturbing
        the backbone or previously-trained heads.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen — only head weights will be updated.")

    def unfreeze_backbone(self):
        """Unfreeze the backbone so it can be fine-tuned together with heads."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen — all weights will be updated.")

    def freeze_head(self, head_name: str):
        """Freeze a specific head so its weights stay fixed."""
        head = {"clickbait": self.clickbait_head,
                "leaning":  self.leaning_head,
                "sentiment": self.sentiment_head}[head_name]
        for param in head.parameters():
            param.requires_grad = False
        print(f"✓ {head_name} head frozen.")

    def unfreeze_head(self, head_name: str):
        """Unfreeze a specific head."""
        head = {"clickbait": self.clickbait_head,
                "leaning":  self.leaning_head,
                "sentiment": self.sentiment_head}[head_name]
        for param in head.parameters():
            param.requires_grad = True
        print(f"✓ {head_name} head unfrozen.")


# ──────────────────────────────────────────────────────────────
#  Quick sanity check  (run this file directly to verify shapes)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from transformers import AutoTokenizer

    MODEL = "answerdotai/ModernBERT-base"
    print(f"Loading tokenizer and model: {MODEL} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = MultiHeadModernBERT(model_name=MODEL)

    # Tokenize a sample headline
    sample = "You Won't Believe What Happened Next!"
    tokens = tokenizer(sample, return_tensors="pt", padding=True, truncation=True)

    # Run through each head
    model.eval()
    with torch.no_grad():
        cb  = model(tokens["input_ids"], tokens["attention_mask"], task="clickbait")
        pol = model(tokens["input_ids"], tokens["attention_mask"], task="leaning")
        sen = model(tokens["input_ids"], tokens["attention_mask"], task="sentiment")
        all_out = model(tokens["input_ids"], tokens["attention_mask"], task="all")

    print(f"\nClickbait  logits shape: {cb.shape}")   # (1, 2)
    print(f"Leaning    logits shape: {pol.shape}")     # (1, 3)
    print(f"Sentiment  logits shape: {sen.shape}")     # (1, 3)
    print(f"All-heads  keys:         {list(all_out.keys())}")
    print("\n✓ Model architecture looks good!")

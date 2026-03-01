"""
train_clickbait.py — Day 3–4: Train the Clickbait Detection Head
================================================================

What this script does
---------------------
1. Loads the clickbait dataset  (HuggingFace or CSV)
2. Initializes the multi-head ModernBERT model
3. Trains ONLY the clickbait head + backbone
4. Evaluates on a validation split  (target: ≥ 85 % accuracy)
5. Saves the model checkpoint

Run
---
    python model/train_clickbait.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ── Imports from this package ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from multi_head_model import MultiHeadModernBERT
from datasets_loader import get_dataloaders


def train_clickbait():
    """
    Full training pipeline for the clickbait detection head.
    """

    print("=" * 60)
    print("  Day 3–4: Training Clickbait Detection Head")
    print("=" * 60)

    # ────────────────────────────────────────────────────────
    # STEP 1 — Choose device (GPU if available, else CPU)
    # ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")

    # ────────────────────────────────────────────────────────
    # STEP 2 — Load tokenizer
    # ────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {config.MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    print("✓ Tokenizer loaded.")

    # ────────────────────────────────────────────────────────
    # STEP 3 — Load and prepare clickbait data
    # ────────────────────────────────────────────────────────
    print("\nLoading clickbait dataset ...")
    train_loader, val_loader, _ = get_dataloaders(
        task="clickbait",
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
        batch_size=config.BATCH_SIZE,
        train_ratio=config.TRAIN_RATIO,
    )

    # ────────────────────────────────────────────────────────
    # STEP 4 — Initialize the multi-head model
    # ────────────────────────────────────────────────────────
    print(f"\nInitializing model: {config.MODEL_NAME} ...")
    model = MultiHeadModernBERT(model_name=config.MODEL_NAME)
    model.to(device)
    print("✓ Model loaded and moved to device.")

    # ────────────────────────────────────────────────────────
    # STEP 5 — Set up optimizer, scheduler, and loss function
    # ────────────────────────────────────────────────────────

    # We train the backbone AND the clickbait head together.
    # The other two heads are randomly initialized but won't receive
    # gradients because we only call task="clickbait" in forward().
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Total training steps = batches_per_epoch × number_of_epochs
    total_steps = len(train_loader) * config.NUM_EPOCHS_CLICKBAIT

    # Linear warmup then linear decay (common practice for fine-tuning)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    # Cross-entropy loss for binary classification  (2 classes)
    criterion = nn.CrossEntropyLoss()

    # ────────────────────────────────────────────────────────
    # STEP 6 — Training loop
    # ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Training Configuration")
    print("─" * 60)
    print(f"  Batch size      : {config.BATCH_SIZE}")
    print(f"  Learning rate   : {config.LEARNING_RATE}")
    print(f"  Epochs          : {config.NUM_EPOCHS_CLICKBAIT}")
    print(f"  Total steps     : {total_steps}")
    print(f"  Warmup steps    : {int(total_steps * config.WARMUP_RATIO)}")
    print("─" * 60)

    best_accuracy = 0.0

    for epoch in range(1, config.NUM_EPOCHS_CLICKBAIT + 1):

        # ── Train phase ─────────────────────────────────────
        model.train()
        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"  Epoch {epoch}/{config.NUM_EPOCHS_CLICKBAIT} [Train]",
            leave=True,
        )

        for batch in progress:
            # Move data to device (GPU / CPU)
            input_ids      = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["labels"].to(device)

            # Forward pass — only through the clickbait head
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, task="clickbait")

            # Compute loss
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            # Update weights
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # ── Validation phase ────────────────────────────────
        val_accuracy = evaluate_clickbait(model, val_loader, device)

        print(f"  → Epoch {epoch}  |  Train Loss: {avg_train_loss:.4f}  |  Val Accuracy: {val_accuracy:.2%}")

        # ── Save best model ─────────────────────────────────
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, config.CLICKBAIT_MODEL_PATH)
            print(f"    ✓ New best! Model saved. (accuracy: {best_accuracy:.2%})")

    # ────────────────────────────────────────────────────────
    # STEP 7 — Final summary
    # ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Training complete!  Best validation accuracy: {best_accuracy:.2%}")
    if best_accuracy >= 0.85:
        print("  ✓ Target of ≥ 85 % ACHIEVED!")
    else:
        print("  ⚠ Below 85 % target — consider more data or more epochs.")
    print(f"  Model saved to: {config.CLICKBAIT_MODEL_PATH}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  Evaluation helper
# ═══════════════════════════════════════════════════════════════

def evaluate_clickbait(model, dataloader, device):
    """
    Run the model on a validation set and return accuracy.

    Parameters
    ----------
    model      : MultiHeadModernBERT
    dataloader : DataLoader for the validation set
    device     : torch.device

    Returns
    -------
    accuracy : float   (0.0 – 1.0)
    """
    model.eval()          # switch to evaluation mode  (disables dropout)
    all_preds  = []
    all_labels = []

    with torch.no_grad():  # no gradient computation needed for eval
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, task="clickbait")

            # argmax converts logits → predicted class  (0 or 1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


# ═══════════════════════════════════════════════════════════════
#  Save / Load helpers
# ═══════════════════════════════════════════════════════════════

def save_model(model, path):
    """Save model weights and config to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": config.MODEL_NAME,
                "num_clickbait_classes": 2,
                "num_leaning_classes":   3,
                "num_sentiment_classes": 3,
            },
        },
        path,
    )


def load_model(path, device=None):
    """Load a saved multi-head model from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = MultiHeadModernBERT(
        model_name=cfg["model_name"],
        num_clickbait_classes=cfg["num_clickbait_classes"],
        num_leaning_classes=cfg["num_leaning_classes"],
        num_sentiment_classes=cfg["num_sentiment_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_clickbait()

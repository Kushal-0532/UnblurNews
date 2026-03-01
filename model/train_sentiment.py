"""
train_sentiment.py — Day 7 (part 1): Train the Sentiment Head
=============================================================

What this script does
---------------------
1. Loads the model checkpoint from Day 5–6 (clickbait + political heads trained)
2. Loads sentiment data:
     • tweet_eval sentiment subset  (from HuggingFace), OR
     • distill labels from cardiffnlp/twitter-roberta-base-sentiment-latest, OR
     • local CSV
3. Trains the sentiment head (with combined loss to protect earlier heads)
4. Saves the updated model

Run
---
    python model/train_sentiment.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ── Imports from this package ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from multi_head_model import MultiHeadModernBERT
from datasets_loader import get_dataloaders
from train_clickbait import load_model, save_model, evaluate_clickbait
from train_political import evaluate_task


def train_sentiment():
    """
    Full training pipeline for the sentiment analysis head.
    """

    print("=" * 60)
    print("  Day 7 (part 1): Training Sentiment Analysis Head")
    print("=" * 60)

    # ────────────────────────────────────────────────────────
    # STEP 1 — Device
    # ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")

    # ────────────────────────────────────────────────────────
    # STEP 2 — Load tokenizer
    # ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    print("✓ Tokenizer loaded.")

    # ────────────────────────────────────────────────────────
    # STEP 3 — Load pretrained model  (clickbait + political)
    # ────────────────────────────────────────────────────────
    # Try loading the most recent checkpoint
    if os.path.exists(config.POLITICAL_MODEL_PATH):
        print(f"\nLoading model from: {config.POLITICAL_MODEL_PATH}")
        model = load_model(config.POLITICAL_MODEL_PATH, device)
    elif os.path.exists(config.CLICKBAIT_MODEL_PATH):
        print(f"\nLoading model from: {config.CLICKBAIT_MODEL_PATH}")
        model = load_model(config.CLICKBAIT_MODEL_PATH, device)
    else:
        print("\n⚠ No checkpoint found — initializing fresh model.")
        model = MultiHeadModernBERT(model_name=config.MODEL_NAME)
        model.to(device)

    # ────────────────────────────────────────────────────────
    # STEP 4 — Load sentiment data
    # ────────────────────────────────────────────────────────
    print("\nLoading sentiment dataset ...")
    sent_train_loader, sent_val_loader, _ = get_dataloaders(
        task="sentiment",
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
        batch_size=config.BATCH_SIZE,
    )

    # Also load clickbait data for combined training
    print("\nLoading clickbait dataset (for combined loss) ...")
    cb_train_loader, cb_val_loader, _ = get_dataloaders(
        task="clickbait",
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
        batch_size=config.BATCH_SIZE,
    )

    # ────────────────────────────────────────────────────────
    # STEP 5 — Optimizer & scheduler
    # ────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    total_steps = len(sent_train_loader) * config.NUM_EPOCHS_SENTIMENT
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    # ────────────────────────────────────────────────────────
    # STEP 6 — Training loop  (with combined loss)
    # ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Training Configuration")
    print("─" * 60)
    print(f"  Batch size      : {config.BATCH_SIZE}")
    print(f"  Learning rate   : {config.LEARNING_RATE}")
    print(f"  Epochs          : {config.NUM_EPOCHS_SENTIMENT}")
    print("─" * 60)

    best_sent_accuracy = 0.0

    for epoch in range(1, config.NUM_EPOCHS_SENTIMENT + 1):

        model.train()
        running_loss = 0.0
        cb_iter = iter(cb_train_loader)

        progress = tqdm(
            sent_train_loader,
            desc=f"  Epoch {epoch}/{config.NUM_EPOCHS_SENTIMENT} [Train]",
            leave=True,
        )

        for sent_batch in progress:
            optimizer.zero_grad()

            # ── Sentiment loss ──────────────────────────────
            sent_ids    = sent_batch["input_ids"].to(device)
            sent_mask   = sent_batch["attention_mask"].to(device)
            sent_labels = sent_batch["labels"].to(device)

            sent_logits = model(sent_ids, sent_mask, task="sentiment")
            sent_loss   = criterion(sent_logits, sent_labels)

            # ── Clickbait loss (prevent forgetting) ─────────
            try:
                cb_batch = next(cb_iter)
            except StopIteration:
                cb_iter = iter(cb_train_loader)
                cb_batch = next(cb_iter)

            cb_ids    = cb_batch["input_ids"].to(device)
            cb_mask   = cb_batch["attention_mask"].to(device)
            cb_labels = cb_batch["labels"].to(device)

            cb_logits = model(cb_ids, cb_mask, task="clickbait")
            cb_loss   = criterion(cb_logits, cb_labels)

            # Combined loss
            total_loss = sent_loss + cb_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += total_loss.item()
            progress.set_postfix(loss=f"{total_loss.item():.4f}")

        avg_loss = running_loss / len(sent_train_loader)

        # ── Validate sentiment ──────────────────────────────
        sent_accuracy = evaluate_task(model, sent_val_loader, device, task="sentiment")

        # ── Validate clickbait ──────────────────────────────
        cb_accuracy = evaluate_clickbait(model, cb_val_loader, device)

        print(f"  → Epoch {epoch}  |  Loss: {avg_loss:.4f}  |  "
              f"Sentiment Acc: {sent_accuracy:.2%}  |  Clickbait Acc: {cb_accuracy:.2%}")

        if cb_accuracy < 0.85:
            print("    ⚠ Clickbait accuracy dropped below 85 %!")

        # ── Save best model ─────────────────────────────────
        if sent_accuracy > best_sent_accuracy:
            best_sent_accuracy = sent_accuracy
            save_model(model, config.FULL_MODEL_PATH)
            print(f"    ✓ New best sentiment accuracy! Model saved.")

    # ────────────────────────────────────────────────────────
    # STEP 7 — Final summary
    # ────────────────────────────────────────────────────────
    final_cb = evaluate_clickbait(model, cb_val_loader, device)

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best sentiment accuracy : {best_sent_accuracy:.2%}")
    print(f"  Final clickbait accuracy: {final_cb:.2%}")
    print(f"  Model saved to: {config.FULL_MODEL_PATH}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_sentiment()

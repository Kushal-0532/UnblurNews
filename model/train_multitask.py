"""
train_multitask.py — Day 7 (part 2): Unified Multi-Task Training
================================================================

What this script does
---------------------
Trains ALL THREE heads simultaneously using a weighted combined loss:

    total_loss = w1 * clickbait_loss
               + w2 * leaning_loss
               + w3 * sentiment_loss

At each training step we sample one batch from each dataset, compute each
task's loss, combine them, and back-propagate through the shared backbone
and all three heads at once.

This is the final polishing step — run it AFTER the individual head scripts
if you want to further harmonize the shared backbone.

Run
---
    python model/train_multitask.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# ── Imports from this package ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from multi_head_model import MultiHeadModernBERT
from datasets_loader import get_dataloaders
from train_clickbait import load_model, save_model, evaluate_clickbait
from train_political import evaluate_task


def train_multitask():
    """
    Unified multi-task training across Clickbait, Political Leaning,
    and Sentiment datasets.
    """

    print("=" * 60)
    print("  Day 7: Unified Multi-Task Training")
    print("=" * 60)

    # ────────────────────────────────────────────────────────
    # STEP 1 — Device
    # ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")

    # ────────────────────────────────────────────────────────
    # STEP 2 — Tokenizer
    # ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    print("✓ Tokenizer loaded.")

    # ────────────────────────────────────────────────────────
    # STEP 3 — Load best existing checkpoint (if any)
    # ────────────────────────────────────────────────────────
    checkpoint_path = None
    for path in [config.FULL_MODEL_PATH, config.POLITICAL_MODEL_PATH, config.CLICKBAIT_MODEL_PATH]:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        model = load_model(checkpoint_path, device)
    else:
        print("\n⚠ No checkpoint found — training from scratch.")
        model = MultiHeadModernBERT(model_name=config.MODEL_NAME)
        model.to(device)

    # ────────────────────────────────────────────────────────
    # STEP 4 — Load ALL three datasets
    # ────────────────────────────────────────────────────────
    print("\nLoading all datasets ...")

    cb_train, cb_val, _ = get_dataloaders(
        "clickbait", tokenizer, max_length=config.MAX_LENGTH,
        batch_size=config.BATCH_SIZE,
    )
    pol_train, pol_val, _ = get_dataloaders(
        "leaning", tokenizer, max_length=config.MAX_LENGTH_ARTICLE,
        batch_size=config.BATCH_SIZE,
    )
    sent_train, sent_val, _ = get_dataloaders(
        "sentiment", tokenizer, max_length=config.MAX_LENGTH,
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

    # Steps per epoch = length of the LONGEST loader
    steps_per_epoch = max(len(cb_train), len(pol_train), len(sent_train))
    total_steps = steps_per_epoch * config.NUM_EPOCHS_MULTITASK

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    # ────────────────────────────────────────────────────────
    # STEP 6 — Training loop
    # ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Multi-Task Training Configuration")
    print("─" * 60)
    print(f"  Batch size       : {config.BATCH_SIZE}")
    print(f"  Learning rate    : {config.LEARNING_RATE}")
    print(f"  Epochs           : {config.NUM_EPOCHS_MULTITASK}")
    print(f"  Steps per epoch  : {steps_per_epoch}")
    print(f"  Loss weights     : clickbait={config.W_CLICKBAIT}, "
          f"leaning={config.W_LEANING}, sentiment={config.W_SENTIMENT}")
    print("─" * 60)

    best_combined_score = 0.0

    for epoch in range(1, config.NUM_EPOCHS_MULTITASK + 1):

        model.train()
        epoch_loss = 0.0

        # Create infinite iterators for each task
        # (When one dataset is exhausted, restart it)
        cb_iter   = iter(cb_train)
        pol_iter  = iter(pol_train)
        sent_iter = iter(sent_train)

        progress = tqdm(
            range(steps_per_epoch),
            desc=f"  Epoch {epoch}/{config.NUM_EPOCHS_MULTITASK} [Multi-Task]",
            leave=True,
        )

        for step in progress:
            optimizer.zero_grad()

            # ── 1. Clickbait loss ───────────────────────────
            try:
                cb_batch = next(cb_iter)
            except StopIteration:
                cb_iter = iter(cb_train)
                cb_batch = next(cb_iter)

            cb_logits = model(
                cb_batch["input_ids"].to(device),
                cb_batch["attention_mask"].to(device),
                task="clickbait",
            )
            cb_loss = criterion(cb_logits, cb_batch["labels"].to(device))

            # ── 2. Political-leaning loss ───────────────────
            try:
                pol_batch = next(pol_iter)
            except StopIteration:
                pol_iter = iter(pol_train)
                pol_batch = next(pol_iter)

            pol_logits = model(
                pol_batch["input_ids"].to(device),
                pol_batch["attention_mask"].to(device),
                task="leaning",
            )
            pol_loss = criterion(pol_logits, pol_batch["labels"].to(device))

            # ── 3. Sentiment loss ───────────────────────────
            try:
                sent_batch = next(sent_iter)
            except StopIteration:
                sent_iter = iter(sent_train)
                sent_batch = next(sent_iter)

            sent_logits = model(
                sent_batch["input_ids"].to(device),
                sent_batch["attention_mask"].to(device),
                task="sentiment",
            )
            sent_loss = criterion(sent_logits, sent_batch["labels"].to(device))

            # ── 4. Combined weighted loss ───────────────────
            total_loss = (
                config.W_CLICKBAIT * cb_loss
                + config.W_LEANING * pol_loss
                + config.W_SENTIMENT * sent_loss
            )

            # ── 5. Backward + update ───────────────────────
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            progress.set_postfix(loss=f"{total_loss.item():.4f}")

        avg_loss = epoch_loss / steps_per_epoch

        # ── Validate all three tasks ────────────────────────
        cb_acc   = evaluate_clickbait(model, cb_val, device)
        pol_acc  = evaluate_task(model, pol_val, device, task="leaning")
        sent_acc = evaluate_task(model, sent_val, device, task="sentiment")

        # Simple combined score (average of all three)
        combined = (cb_acc + pol_acc + sent_acc) / 3.0

        print(f"  → Epoch {epoch}  |  Loss: {avg_loss:.4f}")
        print(f"    Clickbait: {cb_acc:.2%}  |  Political: {pol_acc:.2%}  |  "
              f"Sentiment: {sent_acc:.2%}  |  Avg: {combined:.2%}")

        # ── Save best model ─────────────────────────────────
        if combined > best_combined_score:
            best_combined_score = combined
            save_model(model, config.FULL_MODEL_PATH)
            print(f"    ✓ New best combined score! Model saved.")

    # ────────────────────────────────────────────────────────
    # STEP 7 — Final report
    # ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Multi-Task Training Complete!")
    print("─" * 60)
    print(f"  Best combined accuracy: {best_combined_score:.2%}")
    print(f"  Model saved to: {config.FULL_MODEL_PATH}")
    print()
    print("  Next steps:")
    print("    • Run  python model/inference.py   to test predictions")
    print("    • Run  python model/pipeline.py    for end-to-end demo")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_multitask()

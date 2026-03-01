"""
train_political.py — Day 5–6: Train the Political Leaning Head
==============================================================

What this script does
---------------------
1. Loads the model checkpoint from Day 3–4 (clickbait head already trained)
2. Loads political-leaning data  (CSV or AllSides-mapped articles)
3. Offers **two training strategies** to prevent catastrophic forgetting:
     Option A — Freeze backbone; train only the leaning head
     Option B — Joint fine-tune backbone + both heads  (combined loss)
4. Validates that clickbait accuracy remains ≥ 85 %
5. Saves the updated model

Run
---
    python model/train_political.py                   # defaults to Option B
    python model/train_political.py --freeze-backbone  # Option A
"""

import os
import sys
import argparse
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


def train_political(freeze_backbone: bool = False):
    """
    Train the political-leaning head on top of the shared ModernBERT backbone.

    Parameters
    ----------
    freeze_backbone : bool
        True  → Option A: backbone is frozen, only leaning head trains.
        False → Option B: backbone + clickbait head + leaning head all train
                          using a combined loss to prevent forgetting.
    """

    strategy = "A (freeze backbone)" if freeze_backbone else "B (joint fine-tune)"

    print("=" * 60)
    print("  Day 5–6: Training Political Leaning Head")
    print(f"  Strategy: Option {strategy}")
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
    # STEP 3 — Load pretrained model (with clickbait head)
    # ────────────────────────────────────────────────────────
    if os.path.exists(config.CLICKBAIT_MODEL_PATH):
        print(f"\nLoading pretrained model from: {config.CLICKBAIT_MODEL_PATH}")
        model = load_model(config.CLICKBAIT_MODEL_PATH, device)
    else:
        print("\n⚠ No clickbait checkpoint found — initializing fresh model.")
        model = MultiHeadModernBERT(model_name=config.MODEL_NAME)
        model.to(device)

    # ────────────────────────────────────────────────────────
    # STEP 4 — Load political-leaning data
    # ────────────────────────────────────────────────────────
    print("\nLoading political-leaning dataset ...")
    pol_train_loader, pol_val_loader, _ = get_dataloaders(
        task="leaning",
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH_ARTICLE,    # articles are longer than headlines
        batch_size=config.BATCH_SIZE,
    )

    # Also load clickbait data (needed for Option B and for validation)
    print("\nLoading clickbait dataset (for combined training / validation) ...")
    cb_train_loader, cb_val_loader, _ = get_dataloaders(
        task="clickbait",
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
        batch_size=config.BATCH_SIZE,
    )

    # ────────────────────────────────────────────────────────
    # STEP 5 — Optionally freeze backbone (Option A)
    # ────────────────────────────────────────────────────────
    if freeze_backbone:
        model.freeze_backbone()
        # Also freeze the clickbait head so it doesn't change
        model.freeze_head("clickbait")

    # ────────────────────────────────────────────────────────
    # STEP 6 — Optimizer & scheduler
    # ────────────────────────────────────────────────────────

    # Only optimize parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(trainable_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    total_steps = len(pol_train_loader) * config.NUM_EPOCHS_POLITICAL
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    # ────────────────────────────────────────────────────────
    # STEP 7 — Training loop
    # ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Training Configuration")
    print("─" * 60)
    print(f"  Batch size      : {config.BATCH_SIZE}")
    print(f"  Learning rate   : {config.LEARNING_RATE}")
    print(f"  Epochs          : {config.NUM_EPOCHS_POLITICAL}")
    print(f"  Backbone frozen : {freeze_backbone}")
    print("─" * 60)

    best_pol_accuracy = 0.0

    for epoch in range(1, config.NUM_EPOCHS_POLITICAL + 1):

        model.train()
        running_loss = 0.0

        # For Option B, we also iterate clickbait data
        if not freeze_backbone:
            cb_iter = iter(cb_train_loader)

        progress = tqdm(
            pol_train_loader,
            desc=f"  Epoch {epoch}/{config.NUM_EPOCHS_POLITICAL} [Train]",
            leave=True,
        )

        for pol_batch in progress:
            optimizer.zero_grad()

            # ── Political-leaning loss ──────────────────────
            pol_ids   = pol_batch["input_ids"].to(device)
            pol_mask  = pol_batch["attention_mask"].to(device)
            pol_labels = pol_batch["labels"].to(device)

            pol_logits = model(pol_ids, pol_mask, task="leaning")
            pol_loss   = criterion(pol_logits, pol_labels)

            # ── Combined loss (Option B only) ───────────────
            if not freeze_backbone:
                # Get a clickbait batch  (recycle if exhausted)
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

                # Combined loss keeps the backbone good at both tasks
                total_loss = pol_loss + cb_loss
            else:
                total_loss = pol_loss

            # ── Backward + update ───────────────────────────
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += total_loss.item()
            progress.set_postfix(loss=f"{total_loss.item():.4f}")

        avg_loss = running_loss / len(pol_train_loader)

        # ── Validate political leaning ──────────────────────
        pol_accuracy = evaluate_task(model, pol_val_loader, device, task="leaning")

        # ── Validate clickbait (check for forgetting) ───────
        cb_accuracy = evaluate_clickbait(model, cb_val_loader, device)

        print(f"  → Epoch {epoch}  |  Loss: {avg_loss:.4f}  |  "
              f"Political Acc: {pol_accuracy:.2%}  |  Clickbait Acc: {cb_accuracy:.2%}")

        if cb_accuracy < 0.85:
            print("    ⚠ Clickbait accuracy dropped below 85 % — catastrophic forgetting risk!")

        # ── Save best model ─────────────────────────────────
        if pol_accuracy > best_pol_accuracy:
            best_pol_accuracy = pol_accuracy
            save_model(model, config.POLITICAL_MODEL_PATH)
            print(f"    ✓ New best political accuracy! Model saved.")

    # ────────────────────────────────────────────────────────
    # STEP 8 — Final check
    # ────────────────────────────────────────────────────────
    final_cb = evaluate_clickbait(model, cb_val_loader, device)
    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best political leaning accuracy: {best_pol_accuracy:.2%}")
    print(f"  Final clickbait accuracy:        {final_cb:.2%}")
    if final_cb >= 0.85:
        print("  ✓ Clickbait accuracy still ≥ 85 % — no catastrophic forgetting!")
    else:
        print("  ⚠ Clickbait accuracy < 85 % — consider using Option A (freeze backbone).")
    print(f"  Model saved to: {config.POLITICAL_MODEL_PATH}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  Evaluation helper  (generic — works for any 3-class task)
# ═══════════════════════════════════════════════════════════════

def evaluate_task(model, dataloader, device, task="leaning"):
    """
    Evaluate accuracy for any task head.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, mask, task=task)
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return accuracy_score(all_labels, all_preds)


# ═══════════════════════════════════════════════════════════════
#  Manual spot-check helper
# ═══════════════════════════════════════════════════════════════

def spot_check_political(model, tokenizer, device=None):
    """
    Run the model on 10 hand-picked headlines and print predictions.
    Use this to eyeball whether scores make sense.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LABEL_NAMES = {0: "Left", 1: "Center", 2: "Right"}

    test_headlines = [
        "Trump rallies supporters in key swing state ahead of election",
        "Reuters: Global markets steady amid trade talks",
        "Fox News: Biden's border crisis worsens as migrants flood in",
        "NPR: Study finds healthcare costs rising for middle-class families",
        "MSNBC: GOP tax cuts benefit wealthy while workers struggle",
        "AP: Senate passes bipartisan infrastructure bill",
        "Breitbart: Big Tech censorship reaches alarming new levels",
        "The Hill: Both parties claim victory after budget negotiations",
        "CNN: Climate experts warn of irreversible damage without action",
        "Wall Street Journal: Federal Reserve signals cautious rate path",
    ]

    model.eval()
    print("\n  Manual Spot-Check — Political Leaning Predictions")
    print("  " + "─" * 56)

    for headline in test_headlines:
        tokens = tokenizer(headline, return_tensors="pt", padding=True,
                           truncation=True, max_length=config.MAX_LENGTH)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            logits = model(tokens["input_ids"], tokens["attention_mask"], task="leaning")
            probs  = torch.softmax(logits, dim=1).squeeze()
            pred   = torch.argmax(probs).item()

        label = LABEL_NAMES[pred]
        conf  = probs[pred].item()
        short = headline[:55] + "..." if len(headline) > 55 else headline
        print(f"  {label:>6} ({conf:.0%}) │ {short}")


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train political leaning head")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Option A: freeze backbone and clickbait head (default: Option B — joint fine-tune)",
    )
    args = parser.parse_args()

    train_political(freeze_backbone=args.freeze_backbone)

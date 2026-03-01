"""
export_for_backend.py — Export Trained Model for UnBlur Backend
==================================================================

Converts a full training checkpoint (model/saved/model_full.pt) into
the backend's expected format:

    backend/models/
        config.json        ← backbone config
        model.safetensors  ← backbone weights (HuggingFace format)
        tokenizer.json     ← tokenizer
        task_heads.pt      ← classification head weights only

Run
---
    python model/export_for_backend.py
    python model/export_for_backend.py --checkpoint model/saved/model_full.pt
    python model/export_for_backend.py --out /path/to/output/dir
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Try to import torch; fail clearly
try:
    import torch
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: PyTorch / transformers not installed.")
    print("Run: pip install torch transformers safetensors")
    sys.exit(1)

from multi_head_model import MultiHeadModernBERT
from train_clickbait import load_model


def export(checkpoint_path: str, output_dir: str) -> None:
    print("=" * 60)
    print("  UnBlur Model Exporter")
    print("=" * 60)

    # ── Find checkpoint ──────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
        print("Train a model first:")
        print("  python model/train_multitask.py")
        sys.exit(1)

    print(f"\nLoading checkpoint: {checkpoint_path}")
    device = torch.device("cpu")
    model = load_model(checkpoint_path, device)
    model.eval()
    print("✓ Model loaded.")

    # ── Prepare output directory ──────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nExporting to: {output_dir}")

    # ── Save backbone in HuggingFace format ───────────────────────
    backbone_path = output_dir
    print("  Saving backbone (HuggingFace format) ...")
    model.backbone.save_pretrained(backbone_path)
    print(f"  ✓ Backbone saved (config.json + model.safetensors)")

    # ── Save tokenizer ───────────────────────────────────────────
    print("  Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.save_pretrained(backbone_path)
    print(f"  ✓ Tokenizer saved")

    # ── Save task heads ───────────────────────────────────────────
    heads_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("backbone.")
    }
    heads_path = os.path.join(output_dir, "task_heads.pt")
    torch.save(heads_state, heads_path)
    print(f"  ✓ Task heads saved → {heads_path}")
    print(f"    Keys: {list(heads_state.keys())[:6]} ...")

    print("\n✓ Export complete.")
    print(f"\nPlace the contents of '{output_dir}' at:")
    print("  unblur/backend/models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained model for UnBlur backend")
    parser.add_argument(
        "--checkpoint",
        default=config.FULL_MODEL_PATH,
        help=f"Path to .pt checkpoint (default: {config.FULL_MODEL_PATH})"
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "unblur", "backend", "models"),
        help="Output directory (default: unblur/backend/models/)"
    )
    args = parser.parse_args()
    export(args.checkpoint, args.out)

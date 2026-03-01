"""
config.py — Central Configuration for UnblurNews Model
=======================================================

All hyperparameters, file paths, thresholds, and dataset settings live here.
Change values in this file to customize training and inference behavior.
"""

import os

# ──────────────────────────────────────────────────────────────
# PATH SETUP  (works no matter where you run the scripts from)
# ──────────────────────────────────────────────────────────────

# Folder that contains THIS file  →  .../UnblurNews/model/
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root  →  .../UnblurNews/
PROJECT_ROOT = os.path.dirname(MODEL_DIR)

# Where we save trained model weights
SAVE_DIR = os.path.join(MODEL_DIR, "saved")

# Where raw / processed dataset CSVs can be placed
DATA_DIR = os.path.join(MODEL_DIR, "data")


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────

# The shared backbone for all three heads
MODEL_NAME = "answerdotai/ModernBERT-base"

# Maximum tokens for headlines (short text)
MAX_LENGTH = 128

# Maximum tokens for full articles (long text)
MAX_LENGTH_ARTICLE = 512


# ──────────────────────────────────────────────────────────────
# TRAINING  HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────

BATCH_SIZE = 16                 # How many samples per training step
LEARNING_RATE = 2e-5            # Adam learning rate
WEIGHT_DECAY = 0.01             # L2 regularization strength
WARMUP_RATIO = 0.1              # Fraction of steps used for LR warmup
MAX_GRAD_NORM = 1.0             # Gradient clipping threshold

# Epochs per task (used by the single-head training scripts)
NUM_EPOCHS_CLICKBAIT = 5
NUM_EPOCHS_POLITICAL = 5
NUM_EPOCHS_SENTIMENT = 5

# Epochs for the unified multi-task script
NUM_EPOCHS_MULTITASK = 3

# Train / Validation split ratio
TRAIN_RATIO = 0.8   # 80 % training
VAL_RATIO   = 0.2   # 20 % validation


# ──────────────────────────────────────────────────────────────
# MULTI-TASK  LOSS  WEIGHTS
# ──────────────────────────────────────────────────────────────
# total_loss = W1 * clickbait_loss + W2 * leaning_loss + W3 * sentiment_loss

W_CLICKBAIT  = 1.0
W_LEANING    = 1.0
W_SENTIMENT  = 1.0


# ──────────────────────────────────────────────────────────────
# INFERENCE  THRESHOLDS  (used by determine_case)
# ──────────────────────────────────────────────────────────────

CLICKBAIT_THRESHOLD  = 0.5     # Above this → flagged as clickbait
LEANING_THRESHOLD    = 0.3     # |score| above this → flagged as biased
SENTIMENT_THRESHOLD  = 0.3     # |score| above this → flagged as emotional


# ──────────────────────────────────────────────────────────────
# SAVED  MODEL  FILENAMES
# ──────────────────────────────────────────────────────────────

CLICKBAIT_MODEL_PATH  = os.path.join(SAVE_DIR, "model_clickbait.pt")
POLITICAL_MODEL_PATH  = os.path.join(SAVE_DIR, "model_clickbait_political.pt")
FULL_MODEL_PATH       = os.path.join(SAVE_DIR, "model_full.pt")


# ──────────────────────────────────────────────────────────────
# DATASET  SOURCES
# ──────────────────────────────────────────────────────────────

# --- Clickbait ---
# HuggingFace dataset name (tried first).
# "christophsonntag/clickbait" contains ~24 k clickbait and ~24 k
# non-clickbait headlines split into two columns.
CLICKBAIT_HF_DATASET = "christophsonntag/clickbait"

# Fallback: local CSV with columns  text, label  (0 / 1)
CLICKBAIT_CSV = os.path.join(DATA_DIR, "clickbait.csv")

# --- Political Leaning ---
# Local CSV with columns  text, label  (0=Left, 1=Center, 2=Right)
POLITICAL_CSV = os.path.join(DATA_DIR, "political_leaning.csv")

# --- Sentiment ---
# HuggingFace dataset tried first  (tweet_eval → sentiment subset)
SENTIMENT_HF_DATASET     = "tweet_eval"
SENTIMENT_HF_SUBSET      = "sentiment"

# Teacher model used for knowledge distillation (fallback)
SENTIMENT_TEACHER_MODEL  = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Local CSV with columns  text, label  (0=Negative, 1=Neutral, 2=Positive)
SENTIMENT_CSV = os.path.join(DATA_DIR, "sentiment.csv")


# ──────────────────────────────────────────────────────────────
# ALLSIDES  SOURCE → LEANING  MAP
# ──────────────────────────────────────────────────────────────
# Used to auto-label articles by their publisher.
# Scale: -1.0 = Left,  -0.5 = Lean Left,  0 = Center,
#         0.5 = Lean Right,  1.0 = Right
#
# For 3-class training we bucket:
#   score < -0.25  →  0 (Left)
#   -0.25 ≤ score ≤ 0.25  →  1 (Center)
#   score > 0.25  →  2 (Right)

ALLSIDES_RATINGS = {
    # ---- LEFT (-1.0) ----
    "daily kos":            -1.0,
    "democracy now":        -1.0,
    "the intercept":        -1.0,
    "jacobin":              -1.0,
    "mother jones":         -1.0,
    "slate":                -1.0,

    # ---- LEAN LEFT (-0.5) ----
    "cnn":                  -0.5,
    "msnbc":                -0.5,
    "the new york times":   -0.5,
    "washington post":      -0.5,
    "the washington post":  -0.5,
    "nbc news":             -0.5,
    "abc news":             -0.5,
    "cbs news":             -0.5,
    "politico":             -0.5,
    "buzzfeed news":        -0.5,
    "buzzfeed":             -0.5,
    "vox":                  -0.5,
    "the guardian":         -0.5,
    "huffpost":             -0.5,
    "huffington post":      -0.5,
    "the atlantic":         -0.5,
    "npr":                  -0.5,
    "time":                 -0.5,
    "bloomberg":            -0.5,

    # ---- CENTER (0.0) ----
    "reuters":               0.0,
    "associated press":      0.0,
    "ap news":               0.0,
    "bbc":                   0.0,
    "bbc news":              0.0,
    "the hill":              0.0,
    "usa today":             0.0,
    "axios":                 0.0,
    "the wall street journal": 0.0,
    "forbes":                0.0,

    # ---- LEAN RIGHT (0.5) ----
    "new york post":         0.5,
    "the washington times":  0.5,
    "washington times":      0.5,
    "the daily mail":        0.5,
    "daily mail":            0.5,
    "fox business":          0.5,
    "national review":       0.5,
    "washington examiner":   0.5,
    "the epoch times":       0.5,

    # ---- RIGHT (1.0) ----
    "fox news":              1.0,
    "breitbart":             1.0,
    "the daily wire":        1.0,
    "daily wire":            1.0,
    "newsmax":               1.0,
    "the blaze":             1.0,
    "one america news":      1.0,
    "the federalist":        1.0,
}

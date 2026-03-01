"""
datasets_loader.py — Load & Preprocess Data for All Three Tasks
===============================================================

Provides loading functions and a generic PyTorch Dataset class.

Datasets
--------
1. **Clickbait**
   • Primary : HuggingFace  "christophsonntag/clickbait"
   • Fallback: local CSV at  model/data/clickbait.csv  (columns: text, label)

2. **Political Leaning**
   • Primary : local CSV at  model/data/political_leaning.csv  (columns: text, label)
   • Helper  : create_political_dataset_from_articles() labels articles by
               their source using the AllSides rating map in config.py

3. **Sentiment**
   • Primary : HuggingFace  "tweet_eval" → sentiment subset
   • Fallback: local CSV at  model/data/sentiment.csv  (columns: text, label)
   • Fallback: distill labels from cardiffnlp teacher model
"""

import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer

# -- Make imports work whether you run this file directly or import it --------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ═══════════════════════════════════════════════════════════════
#  GENERIC  PYTORCH  DATASET  (shared by all three tasks)
# ═══════════════════════════════════════════════════════════════

class TextClassificationDataset(Dataset):
    """
    A simple PyTorch Dataset for text classification.

    It stores pre-tokenized tensors so the DataLoader doesn't
    need to tokenize on the fly (faster training).

    Parameters
    ----------
    texts : list[str]      – raw text strings
    labels : list[int]     – integer class labels
    tokenizer              – HuggingFace tokenizer
    max_length : int       – maximum number of tokens
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Return one tokenized sample as a dict of tensors."""
        text  = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",       # pad to max_length
            truncation=True,            # cut if longer
            return_tensors="pt",        # return PyTorch tensors
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),   # (max_length,)
            "labels":         torch.tensor(label, dtype=torch.long),  # scalar
        }


# ═══════════════════════════════════════════════════════════════
#  HELPER: Train / Validation split
# ═══════════════════════════════════════════════════════════════

def train_val_split(dataset, train_ratio=0.8):
    """
    Split a PyTorch Dataset into training and validation subsets.

    Returns
    -------
    train_dataset, val_dataset
    """
    total   = len(dataset)
    n_train = int(total * train_ratio)
    n_val   = total - n_train

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),   # reproducible split
    )
    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════
#  1.  CLICKBAIT  DATA  LOADER
# ═══════════════════════════════════════════════════════════════

def load_clickbait_data():
    """
    Load clickbait headlines and labels.

    Tries (in order):
      1. HuggingFace dataset  (christophsonntag/clickbait)
      2. Local CSV file       (model/data/clickbait.csv)

    Returns
    -------
    texts : list[str]    – headline strings
    labels : list[int]   – 0 = not clickbait, 1 = clickbait
    """
    texts, labels = [], []

    # ── Attempt 1: HuggingFace ──────────────────────────────
    try:
        from datasets import load_dataset

        print(f"  Downloading clickbait data from HuggingFace: {config.CLICKBAIT_HF_DATASET} ...")
        hf = load_dataset(config.CLICKBAIT_HF_DATASET, trust_remote_code=True)

        # The christophsonntag/clickbait dataset has two columns:
        #   "clickbait"     — list of clickbait headlines
        #   "not_clickbait" — list of non-clickbait headlines
        # It presents them inside a single split (usually "train").

        split_name = list(hf.keys())[0]   # grab the first (often only) split
        data = hf[split_name]
        columns = data.column_names

        # --- Format A: two separate text columns ---
        if "clickbait" in columns and "not_clickbait" in columns:
            cb_texts     = [str(t) for t in data["clickbait"]     if t]
            non_cb_texts = [str(t) for t in data["not_clickbait"] if t]
            texts  = cb_texts + non_cb_texts
            labels = [1] * len(cb_texts) + [0] * len(non_cb_texts)

        # --- Format B: text + label columns ---
        else:
            # Auto-discover the text column
            text_col = None
            for col in ["text", "headline", "title", "content", "article"]:
                if col in columns:
                    text_col = col
                    break
            # Auto-discover the label column
            label_col = None
            for col in ["label", "clickbait", "is_clickbait", "class"]:
                if col in columns:
                    label_col = col
                    break

            if text_col and label_col:
                texts  = [str(t) for t in data[text_col]]
                labels = [int(l) for l in data[label_col]]
            else:
                raise ValueError(f"Cannot auto-detect columns. Found: {columns}")

        print(f"  ✓ Loaded {len(texts)} samples from HuggingFace.")
        return texts, labels

    except Exception as e:
        print(f"  ⚠ HuggingFace load failed ({e}). Trying local CSV ...")

    # ── Attempt 2: local CSV ────────────────────────────────
    if os.path.exists(config.CLICKBAIT_CSV):
        df = pd.read_csv(config.CLICKBAIT_CSV)
        texts  = df["text"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()
        print(f"  ✓ Loaded {len(texts)} samples from {config.CLICKBAIT_CSV}")
        return texts, labels

    # ── Nothing worked ──────────────────────────────────────
    raise FileNotFoundError(
        "\n╔══════════════════════════════════════════════════════╗\n"
        "║  Could not load clickbait data!                      ║\n"
        "║                                                      ║\n"
        "║  Option 1 — pip install datasets, then re-run.       ║\n"
        "║  Option 2 — Place a CSV at:                          ║\n"
        f"║    {config.CLICKBAIT_CSV:<52} ║\n"
        "║  with columns: text, label  (0 or 1)                 ║\n"
        "╚══════════════════════════════════════════════════════╝"
    )


# ═══════════════════════════════════════════════════════════════
#  2.  POLITICAL  LEANING  DATA  LOADER
# ═══════════════════════════════════════════════════════════════

def source_to_label(source_name: str) -> int:
    """
    Convert a source name to a 3-class label using AllSides ratings.

    Returns
    -------
    0 = Left,  1 = Center,  2 = Right
    -1 if the source is not in the map (unknown).
    """
    score = config.ALLSIDES_RATINGS.get(source_name.lower().strip(), None)
    if score is None:
        return -1            # unknown source

    if score < -0.25:
        return 0             # Left
    elif score > 0.25:
        return 2             # Right
    else:
        return 1             # Center


def create_political_dataset_from_articles(articles):
    """
    Build a political-leaning dataset from news articles that have a "source" field.

    For each article whose source appears in the AllSides map (config.py),
    we assign a label automatically.

    Parameters
    ----------
    articles : list[dict]
        Each dict should have at least "title", "description", "source".

    Returns
    -------
    texts : list[str]
    labels : list[int]   0=Left, 1=Center, 2=Right
    """
    texts, labels = [], []

    for article in articles:
        source = article.get("source", "")
        label  = source_to_label(source)

        if label == -1:
            continue  # skip unknown sources

        # Combine title + description for richer text
        title = article.get("title", "")
        desc  = article.get("description", "")
        text  = f"{title}. {desc}".strip()

        if len(text) > 5:  # skip empty entries
            texts.append(text)
            labels.append(label)

    return texts, labels


def load_political_data():
    """
    Load political-leaning text and labels.

    Tries:
      1. Local CSV at  model/data/political_leaning.csv
         (columns: text, label — 0=Left, 1=Center, 2=Right)

    Returns
    -------
    texts : list[str],  labels : list[int]
    """
    # ── Attempt: local CSV ──────────────────────────────────
    if os.path.exists(config.POLITICAL_CSV):
        df = pd.read_csv(config.POLITICAL_CSV)
        texts  = df["text"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()
        print(f"  ✓ Loaded {len(texts)} political-leaning samples from CSV.")
        return texts, labels

    # ── Nothing found ───────────────────────────────────────
    raise FileNotFoundError(
        "\n╔══════════════════════════════════════════════════════════╗\n"
        "║  Could not load political-leaning data!                   ║\n"
        "║                                                           ║\n"
        "║  Option 1 — Place a CSV at:                               ║\n"
        f"║    {config.POLITICAL_CSV:<57} ║\n"
        "║  Columns: text, label  (0=Left, 1=Center, 2=Right)       ║\n"
        "║                                                           ║\n"
        "║  Option 2 — Use create_political_dataset_from_articles()  ║\n"
        "║  to auto-label articles by their source (AllSides map).   ║\n"
        "╚══════════════════════════════════════════════════════════╝"
    )


# ═══════════════════════════════════════════════════════════════
#  3.  SENTIMENT  DATA  LOADER
# ═══════════════════════════════════════════════════════════════

def load_sentiment_data():
    """
    Load sentiment text and labels.

    Tries (in order):
      1. Local CSV  (model/data/sentiment.csv)
      2. HuggingFace  tweet_eval → sentiment
      3. Print instructions for distilling from CardiffNLP model

    Labels: 0 = Negative,  1 = Neutral,  2 = Positive

    Returns
    -------
    texts : list[str],  labels : list[int]
    """
    # ── Attempt 1: local CSV ────────────────────────────────
    if os.path.exists(config.SENTIMENT_CSV):
        df = pd.read_csv(config.SENTIMENT_CSV)
        texts  = df["text"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()
        print(f"  ✓ Loaded {len(texts)} sentiment samples from CSV.")
        return texts, labels

    # ── Attempt 2: HuggingFace tweet_eval ───────────────────
    try:
        from datasets import load_dataset

        print(f"  Downloading sentiment data from HuggingFace: {config.SENTIMENT_HF_DATASET} ...")
        hf = load_dataset(config.SENTIMENT_HF_DATASET, config.SENTIMENT_HF_SUBSET,
                          trust_remote_code=True)

        # tweet_eval sentiment has "text" and "label" columns
        # label: 0=negative, 1=neutral, 2=positive
        train_data = hf["train"]
        texts  = [str(t) for t in train_data["text"]]
        labels = [int(l) for l in train_data["label"]]

        # Also include validation split for more data
        if "validation" in hf:
            val_data = hf["validation"]
            texts  += [str(t) for t in val_data["text"]]
            labels += [int(l) for l in val_data["label"]]

        print(f"  ✓ Loaded {len(texts)} sentiment samples from HuggingFace.")
        return texts, labels

    except Exception as e:
        print(f"  ⚠ HuggingFace load failed ({e}).")

    # ── Nothing worked ──────────────────────────────────────
    raise FileNotFoundError(
        "\n╔══════════════════════════════════════════════════════════╗\n"
        "║  Could not load sentiment data!                           ║\n"
        "║                                                           ║\n"
        "║  Option 1 — pip install datasets, then re-run.            ║\n"
        "║  Option 2 — Place a CSV at:                               ║\n"
        f"║    {config.SENTIMENT_CSV:<57} ║\n"
        "║  Columns: text, label  (0=Neg, 1=Neutral, 2=Pos)         ║\n"
        "║                                                           ║\n"
        "║  Option 3 — Run distill_sentiment_labels() to create      ║\n"
        "║  labels from the CardiffNLP teacher model.                ║\n"
        "╚══════════════════════════════════════════════════════════╝"
    )


# ═══════════════════════════════════════════════════════════════
#  SENTIMENT  DISTILLATION  (teacher → student labels)
# ═══════════════════════════════════════════════════════════════

def distill_sentiment_labels(texts, teacher_model=None, batch_size=64):
    """
    Generate sentiment labels using a pre-trained teacher model.
    This is called **knowledge distillation**: we use the teacher's
    predictions as training labels for our ModernBERT student.

    Parameters
    ----------
    texts : list[str]
        Texts to label (e.g., news headlines collected from other datasets).
    teacher_model : str or None
        HuggingFace model name.  Defaults to the CardiffNLP model.
    batch_size : int
        How many texts to process at once.

    Returns
    -------
    labels : list[int]   0=Negative, 1=Neutral, 2=Positive
    """
    from transformers import pipeline as hf_pipeline

    if teacher_model is None:
        teacher_model = config.SENTIMENT_TEACHER_MODEL

    print(f"  Loading teacher model: {teacher_model} ...")
    device_id = 0 if torch.cuda.is_available() else -1
    teacher = hf_pipeline(
        "sentiment-analysis",
        model=teacher_model,
        top_k=None,           # return scores for ALL classes
        device=device_id,
        truncation=True,
        max_length=512,
    )

    # Label mapping for cardiffnlp model (labels may appear as strings or IDs)
    LABEL_MAP = {
        "negative": 0, "Negative": 0, "LABEL_0": 0,
        "neutral":  1, "Neutral":  1, "LABEL_1": 1,
        "positive": 2, "Positive": 2, "LABEL_2": 2,
    }

    labels = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch = texts[i : i + batch_size]

        # Teacher returns a list of lists:  [[{label, score}, ...], ...]
        results = teacher(batch)

        for result in results:
            # Pick the label with the highest score
            best = max(result, key=lambda r: r["score"])
            label = LABEL_MAP.get(best["label"], 1)   # default neutral
            labels.append(label)

        if batch_num % 10 == 0 or batch_num == total_batches:
            print(f"    Distilled {min(i + batch_size, len(texts))}/{len(texts)} samples ...")

    print(f"  ✓ Distilled {len(labels)} sentiment labels from teacher model.")
    return labels


# ═══════════════════════════════════════════════════════════════
#  CONVENIENCE: get ready-to-use DataLoaders
# ═══════════════════════════════════════════════════════════════

def get_dataloaders(task, tokenizer, max_length=None, batch_size=None, train_ratio=None):
    """
    One-stop function: load data → build Dataset → split → wrap in DataLoaders.

    Parameters
    ----------
    task : str   "clickbait", "leaning", or "sentiment"
    tokenizer    HuggingFace tokenizer
    max_length   Override config.MAX_LENGTH
    batch_size   Override config.BATCH_SIZE
    train_ratio  Override config.TRAIN_RATIO

    Returns
    -------
    train_loader, val_loader, full_dataset
    """
    max_length  = max_length  or config.MAX_LENGTH
    batch_size  = batch_size  or config.BATCH_SIZE
    train_ratio = train_ratio or config.TRAIN_RATIO

    # 1. Load raw texts + labels
    if task == "clickbait":
        texts, labels = load_clickbait_data()
    elif task == "leaning":
        texts, labels = load_political_data()
    elif task == "sentiment":
        texts, labels = load_sentiment_data()
    else:
        raise ValueError(f"Unknown task: {task}")

    # 2. Build a PyTorch Dataset
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length)

    # 3. Split into train / val
    train_ds, val_ds = train_val_split(dataset, train_ratio)

    # 4. Wrap in DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"  ✓ {task} data ready — {len(train_ds)} train / {len(val_ds)} val samples")
    return train_loader, val_loader, dataset


# ═══════════════════════════════════════════════════════════════
#  Quick test  (run this file directly to check loading)
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Testing dataset loaders")
    print("=" * 55)

    # Try loading clickbait data
    try:
        texts, labels = load_clickbait_data()
        print(f"  Sample: '{texts[0][:80]}...'  label={labels[0]}")
    except FileNotFoundError as e:
        print(e)

    # Try loading sentiment data
    try:
        texts, labels = load_sentiment_data()
        print(f"  Sample: '{texts[0][:80]}...'  label={labels[0]}")
    except FileNotFoundError as e:
        print(e)

    print("\nDone!")

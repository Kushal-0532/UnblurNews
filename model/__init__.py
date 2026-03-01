"""
UnblurNews — Multi-Head ModernBERT Model Package
=================================================

This package contains everything needed to train and run a multi-head
ModernBERT model for three tasks:

  1. Clickbait Detection   (binary classification)
  2. Political Leaning      (3-class: Left / Center / Right)
  3. Sentiment Analysis     (3-class: Negative / Neutral / Positive)

Quick start (inference):
    from model.inference import classify_article

    result = classify_article("Breaking: Scientists discover shocking truth!")
    print(result)
"""

from .multi_head_model import MultiHeadModernBERT

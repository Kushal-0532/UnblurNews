---
title: "One model, three jobs"
---

For sometime now I've been working on a fairly simple browser extension 'UnBlur'. It scores any news article for clickbait, political leaning, and sentiment, then shows you how the same story is being covered across the spectrum. The naive way to do the classification is to have three separate judgments about the same piece of text, a clickbait classifier, a leaning classifier, a sentiment classifier, each trained and run independently.

That's not what I've decided on doing though, here's why.

## The naive path

Three tasks, three models: train each one against its own dataset, ship each one separately, simple. Nothing about it is exactly wrong, but it has two costs that only show up once you try to run it in prod.

First, memory constraints. Loading a model isn't free per se. Loading a ModernBERT-scale backbone from disk takes a couple of seconds, and it sits in memory for the life of the process. Three independent models means three copies of that backbone resident at once, on a CPU-only inference box(I'm using the free tier), for every request. 

Second, and not so obvious: the three tasks aren't actually independent. Clickbait and political framing both live in *how* a headline is written, not only what it's about. A model that's only ever seen clickbait labels has no reason to learn anything beyond clickbait stuff like political framing etc.A human reading the same headline uses overlapping cues for both. Training three models separately throws away that overlap that could benefit accuracy.

## What I'm doing instead

One shared backbone — `answerdotai/ModernBERT-base` — feeds three lightweight heads:

```
Input text  ──►  ModernBERT-base backbone  ──►  [CLS] embedding (768-dim)
                                                        │
                    ┌───────────────────────────────────┤
                    │                │                  │
              clickbait_head    leaning_head       sentiment_head
              (768→256→2)      (768→256→3)         (768→256→3)
```

Each head is just `Dropout → Linear(768→256) → ReLU → Dropout → Linear(256→n_classes)` — small enough that adding two more heads barely moves the parameter count. The expensive part, the backbone, is shared thankfully.

Choosing ModernBERT was a deliberate choice. Alternating local/global attention and RoPE positional embeddings give it a 512-token efficient context window with meaningfully faster CPU inference than a standard BERT of similar size, very necessary when every `/analyze` call is a real-time forward pass, not a batch job.

## Training it without the tasks affecting each other

Multi-task training has an obvious failure mode: if you throw all three losses at the backbone from scratch, whichever task has the most or noisiest data can end up dominating early training and drag the shared representation somewhere that hurts the others.

I've avoided that with a two-phase schedule:

1. **Independent phase** — each head is trained on its own task first, backbone included, one task at a time. This lets the backbone adapt to news text generally before any task has to share it.
2. **Joint phase** — all three heads are then fine-tuned together, at half the learning rate, so the shared backbone updates from three loss signals at once without any single task overwriting what the others already learned.

That ordering is the whole point: independent-then-joint gets you a backbone that's already reasonably good at "understanding news text" before multi-task training starts asking it to serve three masters simultaneously.

## Turning three classifiers into a scatter chart

One more thing falls out of this almost for free. The leaning and sentiment heads output class probabilities (left/center/right, negative/neutral/positive), but UnBlur wants continuous scores for the sidebar's 2D bias map. Instead of taking the argmax class, it takes a weighted average over the class probabilities:

```
leaning_score = −1×P(left) + 0×P(center) + 1×P(right)   →  range [−1, +1]
```

This preserves the ordinality that a hard classification would throw away — "slightly right" and "very right" both round to the same class under argmax, but land in different places on the scatter chart under this scoring. That chart is only meaningful because the underlying scores are continuous, which is only cheap because both heads already share a backbone that's producing well-calibrated class probabilities from the same forward pass used for clickbait detection.

## The trade-off

None of this is free. A shared backbone means the three tasks are coupled — if you want to retrain just the sentiment head on a new dataset, you're now deciding whether to also re-run the joint phase, or accept some drift between the sentiment head and a backbone it was no longer fine-tuned alongside. Three independent models don't have that problem; they're trivially retrainable one at a time.

For UnBlur, one backbone in memory and shared representations across genuinely related tasks was worth that coupling. If the tasks were less related, or if I cared more about retraining each one independently than about memory, three separate models would win instead.

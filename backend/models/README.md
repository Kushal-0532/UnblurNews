# UnBlur ModernBERT — Multi-Task Model

Fine-tuned from `answerdotai/ModernBERT-base`.

## Validation Accuracy
- clickbait: 98.09%
- leaning: 33.33%
- sentiment: 76.47%

## Files
- `config.json` + `model.safetensors` — backbone weights (HuggingFace format)
- `tokenizer.json` + tokenizer files  — tokenizer
- `task_heads.pt`  — classification head weights only
- `model_full.pt`  — full model checkpoint (backbone + heads)

## Usage
Copy this folder to `backend/models/` and start the backend.

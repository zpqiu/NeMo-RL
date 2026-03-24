# Cross-Tokenizer On-Policy Distillation

Knowledge distillation between models with **different tokenizers/vocabularies**.

## Approach

- **Text-space alignment**: Align teacher and student tokens via byte offsets of decoded text
- **Chunk-level KL**: Aggregate token logprobs into chunks, compute KL divergence at chunk level
- **On-policy**: Student generates rollouts, text is decoded, then re-tokenized for teacher

## Quick Start

```bash
cd research/cross_tokenizer_distillation
uv run python run_cross_distillation.py --config configs/cross_distill_math.yaml
```

## Structure

- `cross_tokenizer_distillation/token_alignment.py` — Byte-offset token alignment
- `cross_tokenizer_distillation/cross_tokenizer_loss.py` — Chunk-level KL loss
- `cross_tokenizer_distillation/algorithm.py` — Setup + train loop
- `tests/` — Unit and integration tests

# On-Policy Distillation Research

Research project investigating loss function design for on-policy knowledge distillation in LLMs.

## Context

On-policy distillation trains a student model on its own generated responses, with a teacher model providing token-level supervision. The core design question is: what loss function best leverages teacher feedback when (a) sequences come from the student and (b) teacher information may be partial (top-k logits rather than full distribution)?

## Contents

- [analysis.md](analysis.md) — Comparative analysis of three recent approaches:
  - **Single-token reverse KL via RL** ([Thinking Machines Lab blog](https://thinkingmachines.ai/blog/on-policy-distillation/))
  - **Top-K local support matching** ([Fu et al., arXiv:2603.25562](https://arxiv.org/abs/2603.25562))
  - **ACP (Adaptive Continuous Penalty)** ([GitHub commit](https://github.com/ShuoYangtum/RL/commit/d1b0b723eb356a31db1edce8d667a7a6ea3935d0))

## Key Findings

1. **Two orthogonal design axes**: support selection (single-token vs top-K vs full-vocab) and divergence direction (forward vs reverse KL). Existing methods each pick different positions; the full design space is underexplored.
2. **Setup may dominate loss design**: teacher quality, tokenizer compatibility, and rollout strategy likely matter more than loss function details. No controlled comparison exists.
3. **Top-K alone can hurt**: Fu et al.'s ablation shows top-K matching without constrained rollouts degrades performance, suggesting rollout control (top-p) does the heavy lifting.
4. **Single-token and distributional signals are complementary**: Fu et al.'s best result includes both.

## Status

Analysis phase — no experiments yet.

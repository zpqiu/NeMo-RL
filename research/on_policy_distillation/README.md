# On-Policy Distillation Research

Research project investigating loss function design for on-policy knowledge distillation in LLMs.

## Context

On-policy distillation trains a student model on its own generated responses, with a teacher model providing token-level supervision. The core design question is: what loss function best leverages teacher feedback when (a) sequences come from the student and (b) teacher information may be partial (top-k logits rather than full distribution)?

## Contents

- [analysis.md](analysis.md) — Comparative analysis of three recent approaches:
  - **Single-token reverse KL via RL** ([Thinking Machines Lab blog](https://thinkingmachines.ai/blog/on-policy-distillation/))
  - **Top-K local support matching** ([Fu et al., arXiv:2603.25562](https://arxiv.org/abs/2603.25562))
  - **ACP (Adaptive Continuous Penalty)** ([GitHub commit](https://github.com/ShuoYangtum/RL/commit/d1b0b723eb356a31db1edce8d667a7a6ea3935d0))

## Baseline: Single-Token OPD

Implementation of the Thinking Machines Lab on-policy distillation approach as a baseline.

### Algorithm

1. Student generates responses on-policy
2. Teacher provides per-token logprobs on student-generated sequences
3. Per-token advantage = `log p_teacher(y_t) - log q_student(y_t)` (negative reverse KL)
4. Student updated via importance-sampling policy gradient (no clipping, no baseline)

### How to Run

```bash
python research/on_policy_distillation/run_opd.py \
    --config research/on_policy_distillation/configs/opd_math.yaml
```

Override config values:
```bash
python research/on_policy_distillation/run_opd.py \
    --config research/on_policy_distillation/configs/opd_math.yaml \
    policy.model_name=Qwen/Qwen3-1.7B-Base \
    teacher.model_name=Qwen/Qwen3-4B \
    distillation.max_num_steps=500
```

### Files

```
on_policy_distillation/
├── run_opd.py                              # Entry point
├── configs/
│   └── opd_math.yaml                       # Config (math reasoning)
├── on_policy_distillation/
│   ├── __init__.py
│   └── opd.py                              # Training loop + loss config
├── analysis.md                             # Literature analysis
└── README.md
```

### Design Notes

- Reuses `nemo_rl.algorithms.distillation.setup()` for student/teacher/generation initialization
- Uses `ClippedPGLossFn` with IS ratio and no clipping (equivalent to vanilla IS policy gradient)
- Teacher provides logprobs via `get_logprobs()` (not top-k logits)
- Single training epoch per rollout (γ=0 in the TM formulation)

### Relationship to Standard Distillation

| Aspect | Standard Distillation | This Baseline (OPD) |
|---|---|---|
| Teacher output | Top-k logits | Single-token logprobs |
| Loss type | Direct KL (forward/reverse/mixed) | IS policy gradient |
| Gradient source | Backprop through logits | Policy gradient (score function) |
| Tail handling | Renormalized (or ACP penalties) | Natural via sampling |

## Key Findings

1. **Two orthogonal design axes**: support selection (single-token vs top-K vs full-vocab) and divergence direction (forward vs reverse KL). Existing methods each pick different positions; the full design space is underexplored.
2. **Setup may dominate loss design**: teacher quality, tokenizer compatibility, and rollout strategy likely matter more than loss function details. No controlled comparison exists.
3. **Top-K alone can hurt**: Fu et al.'s ablation shows top-K matching without constrained rollouts degrades performance, suggesting rollout control (top-p) does the heavy lifting.
4. **Single-token and distributional signals are complementary**: Fu et al.'s best result includes both.

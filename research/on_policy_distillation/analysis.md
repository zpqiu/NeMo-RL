# On-Policy Distillation: Loss Design Analysis

## Problem Setting

- Student policy $q_\theta(y_t | y_{<t})$ generates responses on-policy
- Teacher policy $p(y_t | y_{<t})$ provides token-level supervision on student-generated prefixes
- Teacher may only expose partial distribution info (top-k logits, single token logprob, etc.)
- Goal: train student to match teacher's behavior

## Methods Compared

### 1. Single-Token Reverse KL via RL (Thinking Machines Lab)

**Reference:** [On-Policy Distillation (blog)](https://thinkingmachines.ai/blog/on-policy-distillation/)

**Objective:**

$$J(\theta) = \mathbb{E}_{y \sim q_\theta} \sum_t \left[\log p(y_t|y_{<t}) - \log q_\theta(y_t|y_{<t})\right] = -\sum_t \mathbb{E}_{y_{<t} \sim q_\theta}\left[D_{KL}(q_\theta(\cdot|y_{<t}) \| p(\cdot|y_{<t}))\right]$$

Optimizes full-vocab reverse KL via single-sample Monte Carlo estimation.

**Implementation (Tinker API):**

- Loss: `importance_sampling` — IS-corrected policy gradient, no clipping
- Advantage: raw per-token KL `log p(y_t) - log q(y_t)`, no normalization
- Discount: γ = 0 (purely token-level, no future-reward coupling)
- Essentially vanilla REINFORCE + IS correction

**Properties:**

| Dimension | Assessment |
|---|---|
| Theoretical soundness | Strong — optimizes exact full-vocab reverse KL |
| On-policy alignment | Perfect — reverse KL + student sampling |
| Teacher info needed | Minimal — 1 logprob per token |
| Per-position gradient | Stochastic (single sample), scalar reward |
| Tail/probability leak | None — full-vocab coverage via sampling |
| Tokenizer robustness | Weak — single-token comparison |

### 2. Top-K Local Support Matching (Fu et al.)

**Reference:** [Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562) (unpublished preprint)

**Objective:**

$$\mathcal{L}_{\text{LSM}} = \mathbb{E}\left[\frac{1}{\sum|o_i|} \sum_i \sum_t \sum_{v \in \mathcal{S}(c_{i,t})} \hat{\pi}_\theta(v|c_{i,t}) \log \frac{\hat{\pi}_\theta(v|c_{i,t})}{\hat{q}(v|c_{i,t})}\right]$$

where $\mathcal{S} = \text{TopK}_q$ and $\hat{\pi}_\theta, \hat{q}$ are renormalized on the support.

Truncated reverse KL on teacher-selected top-K support, with direct backprop.

**Practical requirements:** top-p rollout sampling + special-token masking.

**Properties:**

| Dimension | Assessment |
|---|---|
| Theoretical soundness | Medium — truncated surrogate (authors acknowledge) |
| On-policy alignment | Good — reverse KL direction correct |
| Teacher info needed | Top-K logits per token |
| Per-position gradient | Deterministic, K-dimensional |
| Tail/probability leak | Not addressed (renormalized) |
| Tokenizer robustness | Good — distributional dilutes single-token artifacts |

### 3. ACP — Adaptive Continuous Penalty

**Reference:** [GitHub commit d1b0b72](https://github.com/ShuoYangtum/RL/commit/d1b0b723eb356a31db1edce8d667a7a6ea3935d0) (unpublished)

**Objective:**

$$L_{\text{ACP}} = \underbrace{\sum_{i \in V_k} p_i \log \frac{p_i}{q_i}}_{\text{FKL-k (actual probs)}} + \underbrace{\lambda_1 \sum_{i \notin V_k} q_i^2}_{\text{square penalty}} + \underbrace{\lambda_2 \left[\max(0, U_{\text{out}} - B)\right]^2}_{\text{excess penalty}}$$

where $q_i$ from full-vocab softmax, $p_i$ actual (un-renormalized) teacher probabilities, $U_{\text{out}} = 1 - \sum_{i \in V_k} q_i$, $B$ = teacher tail probability.

**Properties:**

| Dimension | Assessment |
|---|---|
| Theoretical soundness | Weak — not a standard divergence |
| On-policy alignment | Poor — forward KL with on-policy sampling is mismatched |
| Teacher info needed | Top-K logits + tail probability |
| Per-position gradient | Deterministic, full-vocab via softmax |
| Tail/probability leak | Explicitly addressed via penalties |
| Tokenizer robustness | Good — distributional |

### 4. Renormalized Top-K Forward/Reverse KL (Baseline)

Standard approach in many codebases. Compute KL divergence on top-K tokens after renormalizing both teacher and student distributions to sum to 1 within the support.

**Properties:**

| Dimension | Assessment |
|---|---|
| Theoretical soundness | Medium — valid divergence on restricted support |
| Tail/probability leak | Not addressed — zero gradient outside top-K |
| Simplicity | High |

---

## Key Analysis Dimensions

### Dimension 1: Signal Quality Per Position

The gradient estimator quality varies significantly across methods:

| Method | Per-position info | Sampling noise in loss |
|---|---|---|
| Single-token RL | 1 scalar reward | High (depends on sampled token) |
| Top-K distributional | K-dimensional gradient | None (deterministic) |
| ACP | Full-vocab gradient | None (deterministic) |
| Full-vocab KL | \|V\|-dimensional gradient | None (deterministic) |

Fu et al.'s core argument: single-token reduces distribution matching to a point estimate, losing the "balanced" positive/negative signal that distributional methods provide.

**Counter-evidence:** ThinkingMachines achieves strong results (74.4% AIME24) with single-token + no advantage normalization, suggesting the imbalanced signal may not be a binding constraint in practice.

### Dimension 2: Tail Distribution Control

| Method | Tail handling |
|---|---|
| Single-token RL | Natural — sampling covers full vocab |
| Top-K renormalized | None — probability leak |
| ACP | Explicit penalties (heuristic) |
| Full-vocab KL | Natural — gradient for all tokens |

### Dimension 3: On-Policy Consistency

The sequence-level on-policy objective is:

$$J(\theta) = \mathbb{E}_{y \sim q_\theta}\left[\sum_t D_{KL}(q_\theta(\cdot|y_{<t}) \| p(\cdot|y_{<t}))\right]$$

- **Reverse KL** (ThinkingMachines, Fu et al.) is the natural token-level divergence for on-policy: $q$-weighted, aligned with student sampling.
- **Forward KL** (ACP) weights by teacher $p$ while sampling from student $q$ — a mismatch.

### Dimension 4: Robustness to Practical Issues

Fu et al. identify three failure modes of single-token OPD:

**FM1: Imbalanced signal.** Most sampled tokens receive negative rewards ($\mathbb{E}[r_t] = -D_{KL}(q\|p) \leq 0$). Positive signal concentrates on few tokens.

- This is mathematically inherent to reverse KL, not a bug.
- Standard RL handles this via baselines/advantage normalization.
- ThinkingMachines succeeds WITHOUT normalization, suggesting FM1 severity is setup-dependent.

**FM2: Unreliable teacher on OOD prefixes.** Teacher may "agree" with degenerate patterns (repetition loops) token-by-token.

- Real problem, but top-K distributional matching doesn't systematically solve it — if teacher's top-K distribution also "agrees" with repetition, distributional comparison is equally fooled.
- Root cause is the teacher being queried on OOD contexts, not the comparison granularity.

**FM3: Tokenizer/special-token mismatch.** Single-token comparison confuses semantic disagreement with tokenization artifacts.

- Real engineering issue.
- Solvable orthogonally via masking (works for both single-token and distributional methods).
- Fu et al.'s own results show masking alone gets most of the benefit (Table 1: 36.4 → 40.7).

---

## Critical Observations

### 1. Fu et al.'s top-K alone hurts performance

Table 3 (AIME24 avg@32):

| Method | Score |
|---|---|
| Sampled-token OPD | 20.4 |
| + teacher top-K (truncated reverse-KL) | 17.7 (worse) |
| + teacher top-K + top-p | 23.6 |

Top-K matching requires top-p rollout to be effective. This suggests top-p (constraining the student's sampling space) does most of the work, not the loss reformulation.

### 2. Including the sampled token helps

Table 4 (support variants):

| Support | Avg |
|---|---|
| Teacher top-K | 41.0 |
| Student top-K | 41.9 |
| Teacher top-K + sampled token | 42.9 |

The best result includes the sampled token, undermining the "replace single-token with distributional" narrative. Single-token and distributional signals appear complementary.

### 3. ACP's forward KL is misaligned with on-policy

ACP uses forward KL (teacher-weighted) on top-K while sampling on-policy from student. The gradient at $j \notin V_k$ is $S_k \cdot q_j$ — very weak for tokens with small $q_j$. The penalty terms compensate but are heuristic.

A more natural design would be top-K reverse KL (Fu et al.'s direction) + tail control (ACP's contribution).

### 4. Setup differences dominate loss design differences

ThinkingMachines (R1-level teacher, possibly same tokenizer family, potentially shorter horizons) gets 74.4% AIME24. Fu et al. (Qwen2.5-7B → OpenThinker3-7B, different model families) gets ~41% avg on math benchmarks. Direct comparison is impossible — the teacher quality and setup differences likely matter more than the loss function choice.

---

## Two Orthogonal Design Axes

On-policy distillation loss design has two independent dimensions:

**Axis 1: Support selection** (which tokens contribute to the loss)

```
Single token ←——→ Top-K ←——→ Full vocabulary
(least info,       (balanced,    (most info,
 highest variance)  moderate)     potentially noisy on teacher tail)
```

**Axis 2: Divergence direction**

```
Forward KL (teacher-weighted) ←——→ Reverse KL (student-weighted)
(mean-seeking, mismatched       (mode-seeking, natural for
 with on-policy sampling)        on-policy sampling)
```

Current methods occupy different positions:

| Method | Axis 1 | Axis 2 |
|---|---|---|
| ThinkingMachines | Single token | Reverse KL |
| Fu et al. | Top-K (renormalized) | Reverse KL |
| ACP | Top-K + penalty | Forward KL |
| Standard SFT/distillation | Full vocab | Forward KL |

An unexplored and potentially optimal combination: **Top-K reverse KL + tail control** (distributional signal quality from Axis 1, on-policy alignment from Axis 2, plus explicit tail handling).

---

## Open Questions

1. **How much does teacher quality dominate loss design?** All three approaches use different teachers. A controlled comparison (same teacher, same data, same compute) would be needed to isolate the loss function effect.

2. **Is advantage normalization the real fix for FM1?** Fu et al. didn't test sampled-token OPD with proper advantage normalization. If normalized single-token OPD matches top-K, the entire top-K argument weakens.

3. **What is the right tail handling?** ACP's penalties are heuristic. Fu et al. ignores the tail. A principled approach might assume uniform teacher tail: $p_i \approx B / (|V| - K)$ for $i \notin V_k$, then compute standard KL. This would preserve tail matching instead of suppression.

4. **Does the RL vs direct-backprop distinction matter more than the loss form?** ThinkingMachines uses RL-style policy gradient; Fu et al. and ACP use direct backprop. The optimization pathway may interact with the loss design in ways not captured by analyzing the loss alone.

5. **Practical: top-K reverse KL + excess penalty?** Combine Fu et al.'s truncated reverse KL (good on-policy alignment, balanced signal within top-K) with ACP's excess penalty (tail mass control). This would address both signal quality and probability leak without the heuristic square penalty.

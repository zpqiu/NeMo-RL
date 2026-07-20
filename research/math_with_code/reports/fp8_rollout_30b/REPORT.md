# FP8 Rollout for Multi-Turn Agentic RL: Qwen3-30B-A3B on math-with-code

> **Status**: source material for the multi-turn section of
> `low-precision-rl-tech-report` (which covers single-turn FP8 RL on
> Qwen3-8B/30B-Base with DAPO). This document follows that report's structure
> and figure conventions (orange = BF16, blue = FP8, green = ablation) so the
> results can be merged directly; the BF16 reference training curves below are
> the ones to preserve through the merge.

## Settings

**Training Setup.** We extend the tech report's FP8 W8A8 linear-rollout
evaluation from single-turn math RL to a multi-turn agentic task: boxed-answer
math problems where the policy iterates Python code in an isolated per-trial
container (nemo-gym Harbor harness, `MathCodeHarborAgent`, up to 16
model-call steps per rollout — each step may execute multiple Python tool
calls, and the loop ends early once the model answers without one).
Model: Qwen3-30B-A3B-Instruct-2507 (MoE, non-thinking). Framework: NeMo-RL
async GRPO on 4 GB200 nodes (4 GPUs/node) — 2 generation nodes (vLLM TP1, 8
replicas) and 2 training nodes (Megatron-Core TP4 x EP8), non-colocated.
Training data: 6389 DAPO prompts Qwen3-8B did not solve 8/8, with ReTool-style
tool-use reward shaping baked into train tasks; online validation on AIME 2025
(30 tasks x 16 repeats, shaping-free).

**Hyperparameters.** Prompt batch 64 with n=16 responses per prompt; async
GRPO with trajectory age at most 1 step and 64 in-flight prompt groups.
Token-level GRPO loss (clips 0.2/0.28, dual-clip c=10) with token-level
truncated importance sampling, C=2, applied to all arms — matching the tech
report's rollout-correction recipe. Maximum sequence length 20K. Rollout FP8
is on-the-fly blockwise W8A8 applied at each weight sync; training stays BF16
throughout.

**Metrics.** As in the tech report: (i) *validation accuracy* on AIME 2025;
(ii) *training reward*; (iii) *response length* (generated tokens per sample);
(iv) *mismatch KL* between rollout and training policy logprobs. Multi-turn
adds two behavioral metrics with no single-turn counterpart: (v) *policy
entropy* and (vi) *tool calls per rollout* — the task exhibits a phase
transition into heavy tool use, and whether that transition fires is the most
sensitive indicator of rollout-precision side effects we observed. Rollout
speed is measured by (vii) *time per output token* (ms/token) during
generation, as in the tech report: mean model-call wall-clock seconds ÷ mean
generated tokens per call (the batch-mean ratio equals Σseconds/Σtokens; the
wall clock is the HTTP call covering vLLM queueing, prefill, and decode,
excluding tool execution). It measures per-request serving speed at matched
load — lower is better — without baking concurrency into the number.

## Results: FP8 W8A8 Rollout with Token-Level TIS

Configurations: (i) BF16 reference (orange) and (ii) FP8 W8A8 with the MoE
router excluded from quantization (blue) — the recommended configuration,
matching the official Qwen FP8 checkpoint's `modules_to_not_convert`. The
router-precision choice is ablated in the next section.

![results dynamics](figures/results_dynamics.svg)

*Figure 1: Training dynamics for BF16 (orange) and FP8 rollout with router in
BF16 (blue). Noisy per-step series show a centered rolling median (w=9) over
the raw trace.*

**Training effectiveness.** FP8 rollout tracks the BF16 reference across all
training metrics. Validation accuracy stays on the reference trajectory
throughout and ends above it (best 0.877 / last 0.873 at step 200 vs BF16's
0.858 at step 220); reward curves overlap; response length grows through the
same multi-turn expansion. Mismatch KL sits at ~0.0045 — elevated over BF16's
~0.0015 by quantization as expected, but *flat* across 200 steps, mirroring
the stability the tech report observed for the single-turn 30B MoE with TIS
enabled. Policy entropy runs below the reference (~0.29 vs ~0.35, with a dip
before the tool-use transition) with no observed accuracy cost.

**Multi-turn observations.** The agentic task adds a behavioral dimension
absent from single-turn RL: around step 130 the BF16 policy transitions into a
heavy-tool-use regime (~7 to ~14 calls/rollout) that drives the late accuracy
gains. FP8 rollout undergoes the same transition roughly 50 steps later and
converges to ~15 calls/rollout — rollout quantization delays but does not
suppress the behavioral phase transition, provided the router stays in BF16.

**Rollout performance.**

![rollout performance](figures/rollout_perf.svg)

*Figure 2: Time per output token during generation (ms/token, tool time
excluded; lower is better). Centered rolling median (w=9) over the raw
trace.*

| window | BF16 | FP8 | Δ |
|---|---|---|---|
| steps 10–60 (early behavior: 1–2 model calls, ~6.5k-token trajectories) | 12.9 ms/token | 10.8 ms/token | **−16.2%** |
| steps 150+ (late behavior: heavy tool use, ~12k-token trajectories) | 16.1 ms/token | 14.2 ms/token | −12.1% |

FP8 rollout generates each token ~16% faster at matched early-training load.
The metric is behavior-dependent — it degrades for both arms as trajectories
shift toward many short model calls, because a growing share of wall time
goes to costs FP8 does not accelerate: per-call queueing/scheduling overhead
and KV-cache attention traffic (the KV cache stays BF16 in both arms).
Prefill compute does benefit from FP8 GEMMs, but over long multi-turn
histories prefill itself becomes increasingly KV-bound. Cross-arm deltas are
therefore only meaningful while the arms remain behaviorally matched; after
the FP8 arm's later tool-use transition (~step 180) the two curves converge
as expected.

Validation adds an end-to-end cross-check. Every 10 steps both arms run the
identical AIME burst — 480 rollouts (30 tasks x 16) submitted together — and
we time the whole burst from start until the *last* rollout finishes, so
stragglers count, exactly as they gate a training step. While the two
policies still behave nearly identically (steps 0–40, mean response lengths
within 2%), this is an apples-to-apples end-to-end comparison: FP8 finishes
the burst in a median 178 s vs BF16's 202 s (−12%), consistent with the
per-token speedup.

## Ablation: Router Precision

The tech report's single-turn router ablation (its §"Ablation: Router
Precision for MoE FP8 Rollout") established that quantizing the router
produces the highest mismatch KL and that BF16 ≈ FP32 for the router. The
multi-turn setting shows what that routing inconsistency does to the *policy*
over a longer horizon. Both arms below run identical FP8 W8A8 rollout and
differ only in router precision.

![router ablation](figures/ablation_router.svg)

*Figure 3: Router-precision ablation under FP8 rollout: router in BF16 (blue)
vs router quantized to FP8 (green). Mismatch KL and tool use show a centered
rolling median (w=9) over the raw trace.*

With the router quantized (green), mismatch KL grows steadily 0.004 → 0.008
instead of holding flat; the tool-use phase transition never fires (pinned at
~7.4 calls/rollout while the router-BF16 arm climbs to ~15); and validation
accuracy detaches after step ~120, plateauing near 0.80 with the residual gap
concentrated in the hard problems solved only through heavy tool iteration.
Excluding the router from quantization (blue) arrests all three at no
measurable cost.

Mechanism, consistent with the single-turn KL evidence: vLLM's on-the-fly
`Fp8Config` quantizes the router gate (`ReplicatedLinear`) by default, and
router noise flips top-8 expert selection near decision boundaries — a
discrete computation-path perturbation rather than smooth numeric error. The
RL loop then amplifies the sharpened sampling distribution, and importance
sampling cannot recover trajectories that were never explored. Fix (NeMo-RL):
`quantization_ignored_layer_kws: ["mlp.gate"]`. The official
Qwen3-30B-A3B-Instruct-2507-FP8 checkpoint likewise excludes all 48 `mlp.gate`
layers, and the Megatron trainer runs the router in FP32
(`moe_router_dtype`).

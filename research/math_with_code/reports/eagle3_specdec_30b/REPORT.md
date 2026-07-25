# EAGLE-3 Speculative Decoding for Multi-Turn Agentic RL: Qwen3-30B-A3B on math-with-code

> **Status**: companion to `fp8_rollout_30b/REPORT.md`. Same model, harness,
> cluster and operating point, so the two rollout-acceleration levers (FP8
> weight quantization and speculative decoding) are directly comparable. Data
> through 40 matched training steps; the arms were still running at cutoff.

## Settings

**Task and setup.** Identical to the FP8 study: boxed-answer math problems
where the policy iterates Python in an isolated per-trial container (nemo-gym
Harbor harness, `MathCodeHarborAgent`, up to 16 model-call steps per rollout).
Model: Qwen3-30B-A3B-Instruct-2507 (MoE, non-thinking). NeMo-RL async GRPO on
4 GB200 nodes — 2 generation nodes (vLLM TP1, 8 replicas) and 2 training nodes
(Megatron-Core TP4 x EP8), non-colocated. Harbor concurrency 512, sampling
temperature 1.0, 20K max sequence length. Token-level GRPO with TIS (C=2).

**Drafter.** `lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex`
— an EAGLE-3 head trained for exactly this base model on generic chat data
(open-perfectblend). One decoder layer, hidden 2048, draft vocab 32000.
vLLM runs it with `num_speculative_tokens=3`, `draft_tensor_parallel_size=1`.

> **Checkpoint gotcha.** The published checkpoint ships
> `max_position_embeddings=2048` (its training length) and vLLM clamps the
> drafter's rope to it. Past position 2048 the drafter keeps proposing but
> acceptance collapses to 1.012 (rate 0.4%) — pure draft overhead on 20K
> multi-turn contexts, and it presents as "speculative decoding does not help
> here" rather than as an error. A local copy with that field raised to 40960
> restores acceptance 2.184 at a 6875-token prompt. All results below use the
> patched copy.

**Arms.** Two otherwise byte-identical configurations:

| arm | `policy.draft.enabled` | drafter during RL |
|---|---|---|
| online | true | trained with the policy; `draft.*` weights refit into vLLM each step |
| frozen | false | fixed at the initial checkpoint while the policy trains |

Online draft training requires the Megatron backend and forbids sequence
packing, so **packing is disabled in both arms** to keep that cost common-mode.
The draft loss is a forward-KL distillation against the policy's own detached
logits, added as `L_total = L_policy + λ·L_draft` with λ=1.0.

**Metrics.** vLLM's speculative-decoding counters, aggregated per training
step and closed before validation so the window covers only
training-distribution rollouts: *acceptance length* (mean tokens emitted per
draft, = 1 + accepted/drafts) and per-position acceptance rate. Acceptance
length is the quantity that maps to decode speedup.

> **Analyze paired, per matched step.** Each arm's step-to-step standard
> deviation is 0.09-0.11, which swamps the effect; but that noise is almost
> entirely common-mode (both arms see the same step-indexed data and warmup
> fluctuations), so the paired difference has sd 0.025. The paired design is
> roughly 4x more sensitive, and comparing the two curves directly hides the
> result for the first ~15 steps.

## Results

Speculation was already established as worthwhile at this operating point in a
prior offline-drafter window: +82% per-stream generation throughput and -29%
validation wall against a no-speculation control at conc512, accuracy neutral,
with acceptance 2.10. That measurement answers "does speculation pay when
sampling at temperature 1.0 under high concurrency" — it does, because MoE
decode here is weight-traffic bound. This report answers the follow-on
question that only exists in RL: **what happens to the drafter as the policy
moves underneath it.**

The paired difference is positive at **all 40 matched steps** (mean +0.1299,
sign test p < 1e-11). Decomposed into 10-step blocks:

| steps | online mean (slope/step) | frozen mean (slope/step) | delta |
|---|---|---|---|
| 1-10 | 2.231 (+0.0139) | 2.205 (+0.0072) | +0.027 |
| 11-20 | 2.329 (+0.0198) | 2.218 (+0.0116) | +0.111 |
| 21-30 | 2.356 (+0.0045) | 2.192 (+0.0018) | +0.163 |
| 31-40 | 2.380 (-0.0020) | 2.161 (-0.0009) | +0.219 |

**Two mechanisms fire in sequence, and only the second is the one usually
advertised.**

*Steps 1-30 — domain adaptation.* The online drafter gains +0.149 acceptance
length overall while the frozen drafter does **not** decay; it drifts slightly
*upward* through step 20. The early advantage is therefore not staleness
resistance: it is a generic-chat drafter learning our multi-turn math-code
distribution. Both arms rising together is consistent with RL sharpening the
policy's output distribution, which makes any drafter's job easier.

*After step ~30 — staleness.* The online arm plateaus (block slope turns
slightly negative) while the frozen arm turns down from its block-2 peak of
2.218 to 2.161. The continued widening of the delta in the last block is
driven by frozen decay, not by further online gain. So the staleness effect is
real, but it is slow and mild — invisible for the first 30 steps, and worth
about -0.06 acceptance by step 40.

**Net effect at cutoff**: +10.1% tokens per draft against the frozen drafter
over the last 10 steps, and +13.3% against the offline-drafter baseline
(2.0997). The adaptation component has saturated; the staleness component is
still opening.

## Cost

Median non-validation step time is **848s (online) vs 789s (frozen)**, so
training the drafter costs about **7%** of step time. This is the price of the
draft forward plus the distillation loss, which is not chunked and whose
cross-entropy up-casts both sides to fp32 and retains two vocab-sized tensors
for backward — reintroducing the allocation `defer_fp32_logits` exists to
avoid. Both figures are far above the 307s of the packed Phase-1 configuration:
**disabling sequence packing, not draft training, is the dominant cost** of
this feature on a MoE policy, since it forfeits permute fusion.

## Practical notes

- Draft parameters land in the policy's optimizer group (lr 2e-6, shared
  `clip_grad`). That is two orders of magnitude below a typical drafter LR, yet
  it produced the full effect above — no separate group was needed. Raising
  `policy.draft.loss_weight` is *not* a substitute for a higher LR: Adam is
  scale-invariant for parameters whose gradient comes only from that loss, and
  a larger draft gradient consumes more of the shared clip-norm budget.
- `draft_loss` oscillates around 0.73 rather than falling. The teacher is
  non-stationary — the drafter is chasing a moving policy — so a flat draft
  loss while the frozen arm falls behind is the mechanism working, not
  evidence of no learning.
- Checkpoint/resume carries the drafter: `draft_model.eagle_module.*` is
  present in the distributed checkpoint (54 entries) and chained segments
  resume with it, so drafter training accumulates across the 4-hour wall.

## Caveats

- Single seed per arm. The paired design controls common-mode noise but not
  seed-level differences in the policy trajectory.
- Consecutive steps are autocorrelated, so the nominal p-values overstate
  significance; the block structure and the all-positive sign pattern are the
  robust statements.
- Acceptance is measured over a wall-clock window in async mode, so a step's
  reported acceptance covers generations that will be trained on slightly
  later. This is common to both arms.
- The staleness component is characterized over 40 steps only. Its trajectory
  beyond that — whether it continues linearly, accelerates when the policy
  undergoes the heavy-tool phase transition seen in the FP8 study around step
  120, or flattens — is not established here.

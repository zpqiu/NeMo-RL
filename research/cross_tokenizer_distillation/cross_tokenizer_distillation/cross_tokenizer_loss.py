# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-tokenizer distillation loss: chunk-level KL divergence.

Two implementations:

1. ``CrossTokenizerDistillationLossFn`` — standalone version used in unit tests.
   Called directly with lists of chunk logprobs.

2. ``CrossTokenizerTrainLossFn`` — NeMo RL ``LossFunction``-compatible version.
   Used with ``Policy.train()``. Receives ``next_token_logprobs`` from the
   forward pass and alignment/teacher data from the ``data`` dict.
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch

try:
    from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType, LossType
    from nemo_rl.algorithms.utils import masked_mean
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    _HAS_NEMO_RL = True
except ImportError:
    import enum

    class LossType(enum.Enum):
        TOKEN_LEVEL = "token_level"
        SEQUENCE_LEVEL = "sequence_level"

    class LossInputType(enum.Enum):
        LOGIT = "logit"
        LOGPROB = "logprob"
        DISTILLATION = "distillation"

    LossFunction = object  # type: ignore[assignment, misc]
    BatchedDataDict = dict  # type: ignore[assignment, misc]
    _HAS_NEMO_RL = False

    def masked_mean(values, mask, global_normalization_factor=None):
        if global_normalization_factor is not None:
            return (values * mask).sum() / global_normalization_factor
        return (values * mask).sum() / mask.sum().clamp(min=1)


# ===================================================================
# Configuration
# ===================================================================

class CrossTokenizerDistillationLossConfig(TypedDict):
    kl_type: str
    mixed_kl_weight: float
    nll_anchor_weight: float  # Weight for NLL anchor loss (0 = disabled, >0 = blended)
    terminal_eos_weight: float  # Weight for terminal stop supervision.
    clip_epsilon: float  # PPO clip range for importance sampling (kl_type="is")
    advantage_normalization: str  # none|center|standardize for IS advantages
    negative_advantage_weight: float  # Scale factor for negative IS advantages


# ===================================================================
# Core KL computation
# ===================================================================

def _compute_chunk_kl(
    teacher_chunk_logprobs: torch.Tensor,
    student_chunk_logprobs: torch.Tensor,
    kl_type: str,
    mixed_kl_weight: float = 0.5,
) -> torch.Tensor:
    """Per-chunk divergence using logprob difference.

    Forward KL: teacher_lp - student_lp (positive when student worse)
    Reverse KL: student_lp - teacher_lp (negative of forward)
    Mixed KL: |teacher_lp - student_lp| (symmetric, always non-negative)
    MSE: (teacher_lp - student_lp)^2 (bidirectional gradient — pushes
         student_lp toward teacher_lp from both sides)
    """
    diff = teacher_chunk_logprobs - student_chunk_logprobs

    if kl_type == "forward":
        return diff
    elif kl_type == "reverse":
        return -diff
    elif kl_type == "mse":
        return diff ** 2
    else:
        # Mixed: use absolute difference (symmetric loss)
        return diff.abs()


def _compute_is_ratio_terms(
    current_student_logprob: torch.Tensor,
    old_student_logprob: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unclipped and clipped IS ratios for the current event."""
    log_ratio = current_student_logprob - old_student_logprob.detach()
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    return ratio, clipped_ratio


def _compute_is_loss_from_advantage(
    advantage: torch.Tensor,
    ratio: torch.Tensor,
    clipped_ratio: torch.Tensor,
    negative_advantage_weight: float = 1.0,
) -> torch.Tensor:
    """Clipped importance-weighted surrogate from a precomputed advantage."""
    if negative_advantage_weight < 1.0:
        advantage = torch.where(
            advantage >= 0,
            advantage,
            advantage * negative_advantage_weight,
        )
    surr1 = advantage * ratio
    surr2 = advantage * clipped_ratio
    return -torch.where(
        advantage >= 0,
        torch.min(surr1, surr2),
        torch.max(surr1, surr2),
    )


def _normalize_advantages(
    advantages: list[torch.Tensor],
    mode: str,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Normalize detached IS advantages across the current batch of valid events."""
    if not advantages:
        zero = torch.tensor(0.0)
        one = torch.tensor(1.0)
        return [], zero, one

    stacked = torch.stack(advantages)
    mean = stacked.mean()
    if mode == "none":
        normalized = advantages
        scale = torch.tensor(1.0, device=mean.device, dtype=mean.dtype)
    elif mode == "center":
        normalized = [adv - mean for adv in advantages]
        scale = torch.tensor(1.0, device=mean.device, dtype=mean.dtype)
    elif mode == "standardize":
        scale = stacked.std(unbiased=False).clamp(min=1e-6)
        normalized = [(adv - mean) / scale for adv in advantages]
    else:
        raise ValueError(f"Unsupported advantage_normalization: {mode}")
    return normalized, mean, scale


def _token_position_to_logprob_index(
    token_position: int,
    token_seq_len: int,
    logprob_seq_len: int,
) -> int | None:
    """Map a token position in the input sequence to its logprob tensor index."""
    if token_position < 0 or token_position >= token_seq_len:
        return None
    if logprob_seq_len >= token_seq_len:
        return token_position
    if logprob_seq_len == token_seq_len - 1:
        if token_position == 0:
            return None
        return token_position - 1
    raise ValueError(
        f"Unsupported logprob shape: token_seq_len={token_seq_len}, logprob_seq_len={logprob_seq_len}"
    )


# ===================================================================
# 1) Standalone loss
# ===================================================================

class CrossTokenizerDistillationLossFn:
    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, cfg: CrossTokenizerDistillationLossConfig):
        self.kl_type = cfg["kl_type"]
        self.mixed_kl_weight = cfg.get("mixed_kl_weight", 0.5)
        assert self.kl_type in ("forward", "reverse", "mixed", "mse")
        assert 0 <= self.mixed_kl_weight <= 1

    def compute_chunk_kl(self, teacher_chunk_logprobs, student_chunk_logprobs):
        return _compute_chunk_kl(
            teacher_chunk_logprobs, student_chunk_logprobs,
            self.kl_type, self.mixed_kl_weight,
        )

    def __call__(
        self,
        teacher_chunk_logprobs_batch: list[torch.Tensor],
        student_chunk_logprobs_batch: list[torch.Tensor],
        chunk_masks_batch: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        total_kl = torch.tensor(0.0)
        total_chunks = 0
        per_sample_kl: list[float] = []

        for i, (t_lp, s_lp) in enumerate(
            zip(teacher_chunk_logprobs_batch, student_chunk_logprobs_batch)
        ):
            if t_lp.numel() == 0:
                continue
            device = t_lp.device
            if total_kl.device != device:
                total_kl = total_kl.to(device)

            chunk_kl = self.compute_chunk_kl(t_lp, s_lp)
            if chunk_masks_batch is not None and chunk_masks_batch[i] is not None:
                mask = chunk_masks_batch[i].to(device)
                chunk_kl = chunk_kl * mask
                n_valid = mask.sum().item()
            else:
                n_valid = chunk_kl.numel()

            total_kl = total_kl + chunk_kl.sum()
            total_chunks += n_valid
            per_sample_kl.append(chunk_kl.sum().item() / max(n_valid, 1))

        loss = total_kl / max(total_chunks, 1)
        return loss, {
            "loss": loss.item(),
            "num_chunks": total_chunks,
            "mean_per_sample_kl": sum(per_sample_kl) / max(len(per_sample_kl), 1),
            "num_samples": len(teacher_chunk_logprobs_batch),
        }


# ===================================================================
# 2) NeMo RL-compatible loss
# ===================================================================

class CrossTokenizerTrainLossFn:
    """Cross-tokenizer distillation loss compatible with ``Policy.train()``.

    **Expected data dict keys** (all padded to (B, S)):
    - ``xalign_teacher_chunk_logprobs``: (B, S) teacher chunk logprobs
    - ``xalign_chunk_student_start``: (B, S) start index of student tokens per chunk
    - ``xalign_chunk_mask``: (B, S) valid chunk mask
    - ``xalign_num_student_toks``: (B, S) count of consecutive student tokens per chunk
    - ``xalign_num_teacher_toks``: (B, S) count of teacher tokens per chunk
    - ``xalign_prompt_lengths``: (B,) prompt lengths in student token space
    - ``xalign_teacher_terminal_eos_logprob``: (B,) teacher EOS logprobs at terminal stop
    - ``xalign_student_terminal_eos_token_pos``: (B,) student terminal EOS positions
    - ``xalign_terminal_eos_mask``: (B,) valid terminal stop mask
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, cfg: CrossTokenizerDistillationLossConfig):
        self.kl_type = cfg["kl_type"]
        self.mixed_kl_weight = cfg.get("mixed_kl_weight", 0.5)
        self.nll_anchor_weight = cfg.get("nll_anchor_weight", 0.0)
        self.terminal_eos_weight = cfg.get("terminal_eos_weight", 1.0)
        self.clip_epsilon = cfg.get("clip_epsilon", 0.2)
        self.advantage_normalization = cfg.get("advantage_normalization", "center")
        self.negative_advantage_weight = cfg.get("negative_advantage_weight", 1.0)
        assert self.kl_type in ("forward", "reverse", "mixed", "mse", "is")
        assert 0 <= self.mixed_kl_weight <= 1
        assert self.terminal_eos_weight >= 0
        assert self.advantage_normalization in ("none", "center", "standardize")
        assert 0 <= self.negative_advantage_weight <= 1

    def __call__(
        self,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        next_token_logprobs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        assert next_token_logprobs is not None

        batch_size = next_token_logprobs.shape[0]
        device = next_token_logprobs.device

        teacher_chunk_lps = data["xalign_teacher_chunk_logprobs"].to(device)
        chunk_student_start = data["xalign_chunk_student_start"].to(device)
        chunk_mask = data["xalign_chunk_mask"].to(device)
        num_s_toks = data["xalign_num_student_toks"].to(device)
        num_t_toks = data["xalign_num_teacher_toks"].to(device)
        input_lengths = data["input_lengths"].to(device)
        prompt_lengths = data.get("xalign_prompt_lengths", input_lengths).to(device)
        token_mask = data["token_mask"].to(device)

        # Find max valid chunks to avoid iterating over all S positions
        max_valid = int(chunk_mask.sum(dim=1).max().item()) if chunk_mask.sum() > 0 else 0

        sample_mask = data.get("sample_mask")
        if sample_mask is None:
            sample_mask = torch.ones(batch_size, dtype=torch.float32, device=device)
        else:
            sample_mask = sample_mask.to(device=device, dtype=torch.float32)
        prev_lps = data.get("prev_logprobs")
        if self.kl_type == "is":
            assert prev_lps is not None, "prev_logprobs are required when kl_type='is'"
            prev_lps = prev_lps.to(device)
            # NeMo get_logprobs preserves sequence length by prepending a dummy
            # logprob for token 0, while next_token_logprobs is length S-1.
            if prev_lps.shape[1] == next_token_logprobs.shape[1] + 1:
                prev_lps = prev_lps[:, 1:]
            elif prev_lps.shape[1] != next_token_logprobs.shape[1]:
                raise ValueError(
                    "prev_logprobs must be either sequence-length aligned "
                    "(B, S) or next-token aligned (B, S-1); got "
                    f"{tuple(prev_lps.shape)} vs next_token_logprobs "
                    f"{tuple(next_token_logprobs.shape)}"
                )

        # Aggregate per-sample chunk loss and average over valid samples.
        sample_chunk_losses = [next_token_logprobs.sum() * 0.0 for _ in range(batch_size)]
        sample_chunk_counts = [0 for _ in range(batch_size)]
        is_chunk_events: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        is_terminal_events: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        all_student_tok_counts = []
        all_teacher_tok_counts = []
        # Debug stats for teacher/student logprob health
        _n_total_chunks = 0
        _n_t_zero = 0  # chunks where teacher per-token logprob == 0
        _n_s_zero = 0  # chunks where student per-token logprob == 0
        _t_lp_sum = 0.0
        _s_lp_sum = 0.0
        is_raw_advantages: list[float] = []
        is_advantages: list[float] = []
        is_ratios: list[float] = []
        is_clipped_ratios: list[float] = []
        is_chunk_losses: list[float] = []

        for b in range(batch_size):
            prompt_len = int(prompt_lengths[b].item())
            for c in range(max_valid):
                if chunk_mask[b, c] == 0:
                    continue
                n_s_toks = int(num_s_toks[b, c].item())
                n_t_toks = int(num_t_toks[b, c].item())
                if n_s_toks == 0 or n_t_toks == 0:
                    continue
                start_idx = int(chunk_student_start[b, c].item())
                # Map generation token indices to logprob indices
                lp_start = prompt_len - 1 + start_idx
                lp_end = lp_start + n_s_toks
                max_lp_idx = next_token_logprobs.shape[1]
                lp_start = max(0, min(lp_start, max_lp_idx - 1))
                lp_end = max(lp_start + 1, min(lp_end, max_lp_idx))
                curr_s_chunk_lp = next_token_logprobs[b, lp_start:lp_end].sum()
                t_chunk_lp = teacher_chunk_lps[b, c]
                curr_s_per_tok = curr_s_chunk_lp / n_s_toks
                t_per_tok = t_chunk_lp / n_t_toks

                # Note: zero teacher logprobs (fp32 softmax saturation) are
                # NOT skipped.  For IS loss, they produce a mild positive
                # advantage (0 - student_lp > 0) that balances the negative
                # advantages from chunks where the teacher disagrees.

                if self.kl_type == "is":
                    # Importance-sampling loss over the whole aligned chunk
                    # event, not tokenizer-specific per-token averages:
                    #   advantage = teacher_chunk_lp - old_student_chunk_lp
                    #   ratio     = exp(current_chunk_lp - old_chunk_lp)
                    #   loss      = -advantage * clipped_ratio
                    assert prev_lps is not None
                    max_prev_idx = prev_lps.shape[1]
                    prev_start = max(0, min(lp_start, max_prev_idx - 1))
                    prev_end = max(prev_start + 1, min(lp_end, max_prev_idx))
                    old_s_chunk_lp = prev_lps[b, prev_start:prev_end].sum()
                    raw_advantage = (t_chunk_lp - old_s_chunk_lp).detach()
                    ratio, clipped_ratio = _compute_is_ratio_terms(
                        current_student_logprob=curr_s_chunk_lp,
                        old_student_logprob=old_s_chunk_lp,
                        clip_epsilon=self.clip_epsilon,
                    )
                else:
                    # MSE / forward / reverse / mixed KL (existing path)
                    chunk_loss = _compute_chunk_kl(
                        t_per_tok.unsqueeze(0),
                        curr_s_per_tok.unsqueeze(0),
                        self.kl_type,
                        self.mixed_kl_weight,
                    ).squeeze(0)

                all_student_tok_counts.append(n_s_toks)
                all_teacher_tok_counts.append(n_t_toks)

                _n_total_chunks += 1
                _t_val = t_per_tok.item()
                _s_val = curr_s_per_tok.item()
                if _t_val == 0.0:
                    _n_t_zero += 1
                if _s_val == 0.0:
                    _n_s_zero += 1
                _t_lp_sum += _t_val
                _s_lp_sum += _s_val
                if self.kl_type == "is":
                    is_chunk_events.append((b, raw_advantage, ratio, clipped_ratio))
                    is_raw_advantages.append(float(raw_advantage.item()))
                else:
                    sample_chunk_losses[b] = sample_chunk_losses[b] + chunk_loss
                    sample_chunk_counts[b] += 1

        terminal_eos_loss = next_token_logprobs.sum() * 0.0
        num_valid_terminal_eos = 0
        sample_terminal_losses = [next_token_logprobs.sum() * 0.0 for _ in range(batch_size)]
        valid_terminal_sample_mask = torch.zeros(batch_size, dtype=torch.float32, device=device)
        if self.terminal_eos_weight > 0:
            teacher_terminal_eos = data.get("xalign_teacher_terminal_eos_logprob")
            student_terminal_eos_pos = data.get("xalign_student_terminal_eos_token_pos")
            terminal_eos_mask = data.get("xalign_terminal_eos_mask")

            if (
                teacher_terminal_eos is not None
                and student_terminal_eos_pos is not None
                and terminal_eos_mask is not None
            ):
                teacher_terminal_eos = teacher_terminal_eos.to(device)
                student_terminal_eos_pos = student_terminal_eos_pos.to(device)
                terminal_eos_mask = terminal_eos_mask.to(device)

                for b in range(batch_size):
                    if terminal_eos_mask[b] == 0:
                        continue
                    token_position = int(student_terminal_eos_pos[b].item())
                    seq_len = int(input_lengths[b].item())
                    lp_idx = _token_position_to_logprob_index(
                        token_position=token_position,
                        token_seq_len=seq_len,
                        logprob_seq_len=next_token_logprobs.shape[1],
                    )
                    if lp_idx is None or lp_idx >= next_token_logprobs.shape[1]:
                        continue
                    if self.kl_type == "is":
                        assert prev_lps is not None
                        if lp_idx >= prev_lps.shape[1]:
                            continue
                        raw_advantage = (teacher_terminal_eos[b] - prev_lps[b, lp_idx]).detach()
                        terminal_ratio, terminal_clipped_ratio = _compute_is_ratio_terms(
                            current_student_logprob=next_token_logprobs[b, lp_idx],
                            old_student_logprob=prev_lps[b, lp_idx],
                            clip_epsilon=self.clip_epsilon,
                        )
                        is_terminal_events.append((b, raw_advantage, terminal_ratio, terminal_clipped_ratio))
                        is_raw_advantages.append(float(raw_advantage.item()))
                    else:
                        terminal_eos_kl = _compute_chunk_kl(
                            teacher_terminal_eos[b].unsqueeze(0),
                            next_token_logprobs[b, lp_idx].unsqueeze(0),
                            self.kl_type,
                            self.mixed_kl_weight,
                        ).squeeze(0)
                    if self.kl_type != "is":
                        sample_terminal_losses[b] = terminal_eos_kl
                        valid_terminal_sample_mask[b] = 1.0

        is_terminal_losses: list[float] = []
        raw_adv_mean_value = 0.0
        raw_adv_scale_value = 1.0
        if self.kl_type == "is":
            raw_advantage_tensors = [event[1] for event in is_chunk_events] + [event[1] for event in is_terminal_events]
            normalized_advantages, raw_adv_mean, raw_adv_scale = _normalize_advantages(
                raw_advantage_tensors,
                self.advantage_normalization,
            )
            raw_adv_mean_value = float(raw_adv_mean.item())
            raw_adv_scale_value = float(raw_adv_scale.item())

            chunk_advantages = normalized_advantages[:len(is_chunk_events)]
            terminal_advantages = normalized_advantages[len(is_chunk_events):]

            for (sample_idx, _, ratio, clipped_ratio), advantage in zip(is_chunk_events, chunk_advantages):
                chunk_loss = _compute_is_loss_from_advantage(
                    advantage,
                    ratio,
                    clipped_ratio,
                    negative_advantage_weight=self.negative_advantage_weight,
                )
                sample_chunk_losses[sample_idx] = sample_chunk_losses[sample_idx] + chunk_loss
                sample_chunk_counts[sample_idx] += 1
                is_advantages.append(float(advantage.item()))
                is_ratios.append(float(ratio.item()))
                is_clipped_ratios.append(float(clipped_ratio.item()))
                is_chunk_losses.append(float(chunk_loss.item()))

            for (sample_idx, _, ratio, clipped_ratio), advantage in zip(is_terminal_events, terminal_advantages):
                terminal_eos_kl = _compute_is_loss_from_advantage(
                    advantage,
                    ratio,
                    clipped_ratio,
                    negative_advantage_weight=self.negative_advantage_weight,
                )
                sample_terminal_losses[sample_idx] = terminal_eos_kl
                valid_terminal_sample_mask[sample_idx] = 1.0
                is_advantages.append(float(advantage.item()))
                is_ratios.append(float(ratio.item()))
                is_clipped_ratios.append(float(clipped_ratio.item()))
                is_terminal_losses.append(float(terminal_eos_kl.item()))

            num_valid_terminal_eos = int(valid_terminal_sample_mask.sum().item())
            if num_valid_terminal_eos > 0:
                terminal_eos_loss = masked_mean(
                    torch.stack(sample_terminal_losses),
                    valid_terminal_sample_mask * sample_mask,
                    global_normalization_factor=None,
                )

        # Average over chunks within each sample (not sum) so that loss
        # does not scale with sequence length / number of chunks.
        for b in range(batch_size):
            if sample_chunk_counts[b] > 0:
                sample_chunk_losses[b] = sample_chunk_losses[b] / sample_chunk_counts[b]

        sample_chunk_loss_tensor = torch.stack(sample_chunk_losses)
        valid_chunk_sample_mask = torch.tensor(
            [count > 0 for count in sample_chunk_counts],
            dtype=torch.float32,
            device=device,
        )

        if valid_chunk_sample_mask.sum() == 0:
            trajectory_gap_loss = next_token_logprobs.sum() * 0.0
            s_tok_counts = torch.zeros(0, dtype=torch.float32, device=device)
            t_tok_counts = torch.zeros(0, dtype=torch.float32, device=device)
        else:
            s_tok_counts = torch.tensor(all_student_tok_counts, dtype=torch.float32, device=device)
            t_tok_counts = torch.tensor(all_teacher_tok_counts, dtype=torch.float32, device=device)
            trajectory_gap_loss = masked_mean(
                sample_chunk_loss_tensor,
                valid_chunk_sample_mask * sample_mask,
                global_normalization_factor=None,
            )

        # NLL anchor loss: standard next-token prediction loss on generated tokens
        # This prevents the student from drifting away from being a good language model
        nll_loss = torch.tensor(0.0, device=device)
        if self.nll_anchor_weight > 0:
            # next_token_logprobs are already log p(x_t | x_{<t})
            # NLL = -mean(log p(x_t | x_{<t})) over valid generation tokens
            # token_mask marks which positions are generation (assistant) tokens
            gen_logprobs = next_token_logprobs * token_mask[:, :next_token_logprobs.shape[1]]
            n_gen_tokens = token_mask[:, :next_token_logprobs.shape[1]].sum().clamp(min=1)
            nll_loss = -gen_logprobs.sum() / n_gen_tokens

        valid_sample_mask = torch.maximum(valid_chunk_sample_mask, valid_terminal_sample_mask) * sample_mask
        if valid_sample_mask.sum() == 0:
            # No text-alignment signal and no valid terminal stop signal.
            loss = next_token_logprobs.sum() * 0.0
            return loss, {
                "loss": 0.0,
                "distill_loss": 0.0,
                "chunk_distill_loss": 0.0,
                "terminal_eos_loss": 0.0,
                "nll_loss": nll_loss.item(),
                "num_chunks": 0,
                "num_valid_terminal_eos": 0,
                "num_valid_samples": batch_size,
            }

        combined_sample_distill = sample_chunk_loss_tensor + self.terminal_eos_weight * torch.stack(sample_terminal_losses)
        combined_distill_loss = masked_mean(combined_sample_distill, valid_sample_mask, global_normalization_factor=None)

        # Blend: total_loss = (1 - α) * distill + α * nll_loss
        alpha = self.nll_anchor_weight
        loss = (1.0 - alpha) * combined_distill_loss + alpha * nll_loss

        metrics = {
            "loss": loss.item(),
            "distill_loss": combined_distill_loss.item(),
            "chunk_distill_loss": trajectory_gap_loss.item(),
            "terminal_eos_loss": terminal_eos_loss.item(),
            "nll_loss": nll_loss.item(),
            "num_chunks": int(sum(sample_chunk_counts)),
            "num_valid_terminal_eos": num_valid_terminal_eos,
            "num_valid_samples": int(valid_sample_mask.sum().item()),
            "mean_student_toks_per_chunk": s_tok_counts.mean().item() if s_tok_counts.numel() > 0 else 0.0,
            "mean_teacher_toks_per_chunk": t_tok_counts.mean().item() if t_tok_counts.numel() > 0 else 0.0,
            "teacher_zero_pct": _n_t_zero / max(_n_total_chunks, 1) * 100,
            "student_zero_pct": _n_s_zero / max(_n_total_chunks, 1) * 100,
            "mean_teacher_pertok_lp": _t_lp_sum / max(_n_total_chunks, 1),
            "mean_student_pertok_lp": _s_lp_sum / max(_n_total_chunks, 1),
        }
        if self.kl_type == "is":
            metrics.update({
                "is_mean_raw_advantage": sum(is_raw_advantages) / max(len(is_raw_advantages), 1),
                "is_raw_pos_adv_frac": sum(1 for x in is_raw_advantages if x > 0) / max(len(is_raw_advantages), 1),
                "is_mean_advantage": sum(is_advantages) / max(len(is_advantages), 1),
                "is_pos_adv_frac": sum(1 for x in is_advantages if x > 0) / max(len(is_advantages), 1),
                "is_advantage_center": raw_adv_mean_value,
                "is_advantage_scale": raw_adv_scale_value,
                "is_negative_advantage_weight": self.negative_advantage_weight,
                "is_mean_ratio": sum(is_ratios) / max(len(is_ratios), 1),
                "is_mean_clipped_ratio": sum(is_clipped_ratios) / max(len(is_clipped_ratios), 1),
                "is_clip_frac": (
                    sum(1 for r, cr in zip(is_ratios, is_clipped_ratios) if abs(r - cr) > 1e-6)
                    / max(len(is_ratios), 1)
                ),
                "is_mean_chunk_loss": sum(is_chunk_losses) / max(len(is_chunk_losses), 1),
                "is_mean_terminal_loss": sum(is_terminal_losses) / max(len(is_terminal_losses), 1),
            })

        return loss, metrics

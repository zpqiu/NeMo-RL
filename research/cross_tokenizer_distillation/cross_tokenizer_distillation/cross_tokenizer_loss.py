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


# ===================================================================
# Core KL computation
# ===================================================================

def _compute_chunk_kl(
    teacher_chunk_logprobs: torch.Tensor,
    student_chunk_logprobs: torch.Tensor,
    kl_type: str,
    mixed_kl_weight: float = 0.5,
) -> torch.Tensor:
    """Per-chunk KL divergence using logprob difference.

    Forward KL: teacher_lp - student_lp (positive when student worse)
    Reverse KL: student_lp - teacher_lp (negative of forward)
    Mixed KL: |teacher_lp - student_lp| (symmetric, always non-negative)
    """
    diff = teacher_chunk_logprobs - student_chunk_logprobs

    if kl_type == "forward":
        kl = diff
    elif kl_type == "reverse":
        kl = -diff
    else:
        # Mixed: use absolute difference (symmetric loss)
        kl = diff.abs()
    return kl


# ===================================================================
# 1) Standalone loss
# ===================================================================

class CrossTokenizerDistillationLossFn:
    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, cfg: CrossTokenizerDistillationLossConfig):
        self.kl_type = cfg["kl_type"]
        self.mixed_kl_weight = cfg.get("mixed_kl_weight", 0.5)
        assert self.kl_type in ("forward", "reverse", "mixed")
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
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, cfg: CrossTokenizerDistillationLossConfig):
        self.kl_type = cfg["kl_type"]
        self.mixed_kl_weight = cfg.get("mixed_kl_weight", 0.5)
        assert self.kl_type in ("forward", "reverse", "mixed")
        assert 0 <= self.mixed_kl_weight <= 1

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

        # Find max valid chunks to avoid iterating over all S positions
        max_valid = int(chunk_mask.sum(dim=1).max().item()) if chunk_mask.sum() > 0 else 0

        # Aggregate student logprobs into chunks with gradient flow
        # Build student_chunk_lps as a list → stack to maintain autograd graph
        all_student_chunk_lps = []
        all_teacher_chunk_lps = []
        all_student_tok_counts = []
        all_teacher_tok_counts = []

        for b in range(batch_size):
            prompt_len = int(input_lengths[b].item())
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
                # Sum student logprobs for this chunk (preserves grad)
                s_chunk_lp = next_token_logprobs[b, lp_start:lp_end].sum()
                t_chunk_lp = teacher_chunk_lps[b, c]

                all_student_chunk_lps.append(s_chunk_lp)
                all_teacher_chunk_lps.append(t_chunk_lp)
                all_student_tok_counts.append(n_s_toks)
                all_teacher_tok_counts.append(n_t_toks)

        if len(all_student_chunk_lps) == 0:
            # No valid chunks — return zero loss
            loss = next_token_logprobs.sum() * 0.0
            return loss, {"loss": 0.0, "num_chunks": 0, "num_valid_samples": batch_size}

        # Stack all chunks into flat tensors (grad flows through student side)
        student_flat = torch.stack(all_student_chunk_lps)  # (N_chunks,)
        teacher_flat = torch.stack(all_teacher_chunk_lps)  # (N_chunks,)
        s_tok_counts = torch.tensor(all_student_tok_counts, dtype=torch.float32, device=device)
        t_tok_counts = torch.tensor(all_teacher_tok_counts, dtype=torch.float32, device=device)

        # Length-normalize: convert sum(logprobs) to mean(logprobs) per chunk
        # This makes teacher and student logprobs comparable regardless of
        # how many tokens each tokenizer uses for the same text span.
        student_normalized = student_flat / s_tok_counts
        teacher_normalized = teacher_flat / t_tok_counts

        # Compute KL on normalized per-token logprobs
        chunk_kl = _compute_chunk_kl(
            teacher_normalized, student_normalized,
            self.kl_type, self.mixed_kl_weight,
        )

        loss = chunk_kl.mean()

        metrics = {
            "loss": loss.item(),
            "num_chunks": len(all_student_chunk_lps),
            "num_valid_samples": batch_size,
            "mean_student_toks_per_chunk": s_tok_counts.mean().item(),
            "mean_teacher_toks_per_chunk": t_tok_counts.mean().item(),
        }

        return loss, metrics

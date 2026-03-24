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
    # Standalone mode — define minimal stubs for testing without nemo_rl
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
    """Configuration for cross-tokenizer distillation loss."""

    kl_type: str  # "forward", "reverse", or "mixed"
    mixed_kl_weight: float  # weight for forward KL in mixed mode (0-1)


# ===================================================================
# Core KL computation (shared between both implementations)
# ===================================================================

def _compute_chunk_kl(
    teacher_chunk_logprobs: torch.Tensor,
    student_chunk_logprobs: torch.Tensor,
    kl_type: str,
    mixed_kl_weight: float = 0.5,
) -> torch.Tensor:
    """Per-chunk KL divergence.

    Args:
        teacher_chunk_logprobs: ``(num_chunks,)``
        student_chunk_logprobs: ``(num_chunks,)``
        kl_type: "forward", "reverse", or "mixed"
        mixed_kl_weight: weight for forward KL in mixed mode

    Returns:
        ``(num_chunks,)`` — per-chunk KL values.
    """
    teacher_p = teacher_chunk_logprobs.exp()
    student_p = student_chunk_logprobs.exp()

    if kl_type == "forward":
        kl = teacher_p * (teacher_chunk_logprobs - student_chunk_logprobs)
    elif kl_type == "reverse":
        kl = student_p * (student_chunk_logprobs - teacher_chunk_logprobs)
    else:
        fwd = teacher_p * (teacher_chunk_logprobs - student_chunk_logprobs)
        rev = student_p * (student_chunk_logprobs - teacher_chunk_logprobs)
        kl = mixed_kl_weight * fwd + (1.0 - mixed_kl_weight) * rev
    return kl


# ===================================================================
# 1) Standalone loss — for unit tests and external use
# ===================================================================

class CrossTokenizerDistillationLossFn:
    """Chunk-level KL divergence for cross-tokenizer distillation.

    Standalone version — called directly with lists of pre-computed chunk
    logprobs.  Not compatible with ``Policy.train()``.
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, cfg: CrossTokenizerDistillationLossConfig):
        self.kl_type = cfg["kl_type"]
        self.mixed_kl_weight = cfg.get("mixed_kl_weight", 0.5)
        assert self.kl_type in ("forward", "reverse", "mixed")
        assert 0 <= self.mixed_kl_weight <= 1

    def compute_chunk_kl(
        self,
        teacher_chunk_logprobs: torch.Tensor,
        student_chunk_logprobs: torch.Tensor,
    ) -> torch.Tensor:
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
# 2) NeMo RL-compatible loss — for Policy.train()
# ===================================================================

class CrossTokenizerTrainLossFn:
    """Cross-tokenizer distillation loss compatible with ``Policy.train()``.

    This loss function receives ``next_token_logprobs`` (student per-token
    logprobs from the forward pass) and aggregates them into chunk-level
    logprobs using alignment info packed into the ``data`` dict.

    **Expected data dict keys** (packed by the algorithm before calling
    ``Policy.train()``):

    - ``input_ids``: (B, S) student input ids
    - ``input_lengths``: (B,) prompt lengths
    - ``token_mask``: (B, S) mask for generated tokens
    - ``sample_mask``: (B,) per-sample mask
    - ``xalign_teacher_chunk_logprobs``: (B, S) padded teacher chunk logprobs
    - ``xalign_chunk_student_start``: (B, S) start index of student tokens per chunk
    - ``xalign_chunk_mask``: (B, S) which positions are valid chunks
    - ``xalign_num_student_toks``: (B, S) number of consecutive student tokens per chunk
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
        """Compute cross-tokenizer chunk-level KL loss.

        Args:
            data: BatchedDataDict with alignment info + teacher chunk logprobs.
            global_valid_seqs: global normalization factor for sequence-level.
            global_valid_toks: global normalization factor for token-level.
            next_token_logprobs: (B, S-1) student per-token logprobs from forward pass.

        Returns:
            (loss, metrics)
        """
        assert next_token_logprobs is not None, "next_token_logprobs is required"

        batch_size = next_token_logprobs.shape[0]
        device = next_token_logprobs.device

        # Alignment data is (B, S) padded — chunk info in first positions
        teacher_chunk_lps = data["xalign_teacher_chunk_logprobs"].to(device)  # (B, S)
        chunk_student_start = data["xalign_chunk_student_start"].to(device)  # (B, S)
        chunk_mask = data["xalign_chunk_mask"].to(device)  # (B, S)
        num_s_toks = data["xalign_num_student_toks"].to(device)  # (B, S)
        input_lengths = data["input_lengths"].to(device)  # (B,)

        max_chunks = teacher_chunk_lps.shape[1]

        # Aggregate student logprobs into chunks
        # next_token_logprobs is (B, S-1) — logprob of token[t+1] given [0..t]
        # For generated token at position p (0-indexed in generation), the
        # logprob is at next_token_logprobs[:, input_length + p - 1]
        student_chunk_lps = torch.zeros_like(teacher_chunk_lps)  # (B, S)
        seq_dim = teacher_chunk_lps.shape[1]

        for b in range(batch_size):
            prompt_len = input_lengths[b].item()
            for c in range(seq_dim):
                if chunk_mask[b, c] == 0:
                    continue
                n_toks = int(num_s_toks[b, c].item())
                if n_toks == 0:
                    continue
                # Student tokens are consecutive starting at chunk_student_start
                start_idx = int(chunk_student_start[b, c].item())
                s_indices = torch.arange(start_idx, start_idx + n_toks, device=device)
                # Map to logprob indices: gen token at position p has logprob
                # at next_token_logprobs[b, prompt_len - 1 + p]
                lp_indices = (prompt_len - 1 + s_indices).long()
                # Clamp to valid range
                max_lp_idx = next_token_logprobs.shape[1] - 1
                lp_indices = lp_indices.clamp(0, max_lp_idx)
                student_chunk_lps[b, c] = next_token_logprobs[b, lp_indices].sum()

        # Compute KL divergence
        chunk_kl = _compute_chunk_kl(
            teacher_chunk_lps, student_chunk_lps,
            self.kl_type, self.mixed_kl_weight,
        )  # (B, max_C)

        # Masked mean over valid chunks
        total_valid_chunks = chunk_mask.sum()
        if total_valid_chunks > 0:
            loss = (chunk_kl * chunk_mask).sum() / total_valid_chunks
        else:
            loss = chunk_kl.sum() * 0.0  # zero loss but keep grad

        metrics = {
            "loss": loss.item(),
            "num_chunks": int(total_valid_chunks.item()),
            "num_valid_samples": batch_size,
        }

        return loss, metrics

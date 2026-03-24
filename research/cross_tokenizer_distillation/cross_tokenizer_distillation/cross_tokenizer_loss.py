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

"""Cross-tokenizer distillation loss: chunk-level KL divergence."""

from __future__ import annotations

from typing import Any, TypedDict

import torch

try:
    from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType, LossType
    from nemo_rl.algorithms.utils import masked_mean
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

    def masked_mean(values, mask, global_normalization_factor=None):
        if global_normalization_factor is not None:
            return (values * mask).sum() / global_normalization_factor
        return (values * mask).sum() / mask.sum().clamp(min=1)


class CrossTokenizerDistillationLossConfig(TypedDict):
    """Configuration for cross-tokenizer distillation loss."""

    kl_type: str  # "forward", "reverse", or "mixed"
    mixed_kl_weight: float  # weight for forward KL in mixed mode (0-1)


class CrossTokenizerDistillationLossFn:
    """Chunk-level KL divergence for cross-tokenizer distillation.

    Instead of comparing token-level logprobs directly (which requires shared
    vocabulary), this loss:

    1. Groups tokens into alignment *chunks* — contiguous text regions that
       map to whole tokens on both the teacher and student sides.
    2. Sums per-token log-probs within each chunk to get chunk-level log-probs.
    3. Computes KL divergence at the chunk level.

    The loss is computed *outside* the standard NeMo RL loss wrapper pipeline
    because cross-tokenizer distillation requires custom data flow (teacher and
    student have different sequence lengths/vocab). It is called directly from
    the algorithm's train step.
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, cfg: CrossTokenizerDistillationLossConfig):
        self.kl_type = cfg["kl_type"]
        self.mixed_kl_weight = cfg.get("mixed_kl_weight", 0.5)

        assert self.kl_type in ("forward", "reverse", "mixed"), (
            f"Invalid kl_type: {self.kl_type}"
        )
        assert 0 <= self.mixed_kl_weight <= 1, (
            f"mixed_kl_weight must be in [0, 1], got {self.mixed_kl_weight}"
        )

    def compute_chunk_kl(
        self,
        teacher_chunk_logprobs: torch.Tensor,
        student_chunk_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-chunk KL divergence.

        Args:
            teacher_chunk_logprobs: ``(num_chunks,)`` — sum of teacher token
                log-probs within each chunk.
            student_chunk_logprobs: ``(num_chunks,)`` — sum of student token
                log-probs within each chunk.

        Returns:
            Per-chunk KL divergence tensor of shape ``(num_chunks,)``.
        """
        teacher_p = teacher_chunk_logprobs.exp()
        student_p = student_chunk_logprobs.exp()

        if self.kl_type == "forward":
            # KL(teacher || student) = sum teacher_p * (log teacher_p - log student_p)
            kl = teacher_p * (teacher_chunk_logprobs - student_chunk_logprobs)
        elif self.kl_type == "reverse":
            # KL(student || teacher) = sum student_p * (log student_p - log teacher_p)
            kl = student_p * (student_chunk_logprobs - teacher_chunk_logprobs)
        else:
            # mixed
            fwd = teacher_p * (teacher_chunk_logprobs - student_chunk_logprobs)
            rev = student_p * (student_chunk_logprobs - teacher_chunk_logprobs)
            kl = self.mixed_kl_weight * fwd + (1.0 - self.mixed_kl_weight) * rev

        return kl

    def __call__(
        self,
        teacher_chunk_logprobs_batch: list[torch.Tensor],
        student_chunk_logprobs_batch: list[torch.Tensor],
        chunk_masks_batch: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute batch-level cross-tokenizer distillation loss.

        Args:
            teacher_chunk_logprobs_batch: List of tensors, one per sample.
                Each tensor has shape ``(num_chunks_i,)``.
            student_chunk_logprobs_batch: Same structure as teacher.
            chunk_masks_batch: Optional list of boolean masks per sample.
                Shape ``(num_chunks_i,)`` each. If None, all chunks are valid.

        Returns:
            (loss, metrics) tuple.
        """
        total_kl = torch.tensor(0.0)
        total_chunks = 0
        per_sample_kl: list[float] = []

        for i, (t_lp, s_lp) in enumerate(
            zip(teacher_chunk_logprobs_batch, student_chunk_logprobs_batch)
        ):
            if t_lp.numel() == 0:
                continue

            # Ensure same device
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
            per_sample_kl.append(
                chunk_kl.sum().item() / max(n_valid, 1)
            )

        if total_chunks > 0:
            loss = total_kl / total_chunks
        else:
            loss = total_kl

        metrics = {
            "loss": loss.item(),
            "num_chunks": total_chunks,
            "mean_per_sample_kl": sum(per_sample_kl) / max(len(per_sample_kl), 1),
            "num_samples": len(teacher_chunk_logprobs_batch),
        }

        return loss, metrics

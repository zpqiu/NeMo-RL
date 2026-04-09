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

"""GOLD (General On-Policy Logit Distillation) loss for cross-tokenizer distillation.

Ported from TRL's experimental GOLD implementation. Key components:
- VocabMapping: string-identity matching between tokenizer vocabularies
- Generalized JSD loss for matched vocabulary tokens
- Sorted L1 (ULD) loss for unmatched vocabulary tokens
- Hybrid combination with configurable weights

Reference: https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

try:
    from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType, LossType
    from nemo_rl.algorithms.utils import masked_mean
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.model_utils import (
        _get_tokens_on_this_cp_rank,
        distributed_vocab_topk,
        gather_logits_at_global_indices,
    )

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
# Vocabulary mapping
# ===================================================================


@dataclass
class VocabMapping:
    """Pre-computed string-identity mapping between two tokenizer vocabularies.

    For each token string that appears in both vocabularies, records the
    corresponding IDs on each side.
    """

    # Parallel lists: matched_student_ids[i] <-> matched_teacher_ids[i]
    matched_student_ids: list[int] = field(default_factory=list)
    matched_teacher_ids: list[int] = field(default_factory=list)

    # Boolean masks over the full vocabulary: True if the token is matched
    student_matched_mask: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    teacher_matched_mask: torch.Tensor = field(default_factory=lambda: torch.tensor([]))

    # teacher_id -> student_id for matched tokens
    teacher_to_student_map: dict[int, int] = field(default_factory=dict)

    # Tensor mapping: mapping_tensor[teacher_id] = student_id (or -1 if unmatched)
    mapping_tensor: torch.Tensor = field(default_factory=lambda: torch.tensor([]))

    num_matched: int = 0
    student_vocab_size: int = 0
    teacher_vocab_size: int = 0
    jaccard_index: float = 0.0


def build_vocab_mapping(
    student_tokenizer,
    teacher_tokenizer,
) -> VocabMapping:
    """Build string-identity vocabulary mapping between two tokenizers.

    For each token string that exists in both vocabularies, records the
    matched IDs. This is the same approach used in TRL's GOLD ULDLoss.
    """
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()

    student_token_to_id = dict(student_vocab.items())

    teacher_to_student: dict[int, int] = {}
    teacher_matched: list[int] = []
    student_matched: list[int] = []

    for token_str, teacher_id in teacher_vocab.items():
        if token_str in student_token_to_id:
            student_id = student_token_to_id[token_str]
            teacher_to_student[teacher_id] = student_id
            teacher_matched.append(teacher_id)
            student_matched.append(student_id)

    student_vocab_size = len(student_vocab)
    teacher_vocab_size = len(teacher_vocab)

    # Boolean masks
    student_mask = torch.zeros(student_vocab_size, dtype=torch.bool)
    teacher_mask = torch.zeros(teacher_vocab_size, dtype=torch.bool)
    for sid in student_matched:
        student_mask[sid] = True
    for tid in teacher_matched:
        teacher_mask[tid] = True

    # Dense mapping tensor: teacher_id -> student_id
    if teacher_matched:
        max_teacher_id = max(teacher_matched)
        mapping_tensor = torch.full((max_teacher_id + 1,), -1, dtype=torch.long)
        for tid, sid in teacher_to_student.items():
            mapping_tensor[tid] = sid
    else:
        mapping_tensor = torch.zeros(0, dtype=torch.long)

    num_matched = len(teacher_matched)
    # Jaccard index: |intersection| / |union|
    all_student_strings = set(student_vocab.keys())
    all_teacher_strings = set(teacher_vocab.keys())
    union_size = len(all_student_strings | all_teacher_strings)
    jaccard = num_matched / max(union_size, 1)

    return VocabMapping(
        matched_student_ids=student_matched,
        matched_teacher_ids=teacher_matched,
        student_matched_mask=student_mask,
        teacher_matched_mask=teacher_mask,
        teacher_to_student_map=teacher_to_student,
        mapping_tensor=mapping_tensor,
        num_matched=num_matched,
        student_vocab_size=student_vocab_size,
        teacher_vocab_size=teacher_vocab_size,
        jaccard_index=jaccard,
    )


# ===================================================================
# Loss helper functions
# ===================================================================


def generalized_jsd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Compute generalized Jensen-Shannon Divergence.

    Ported from TRL GOLDTrainer.generalized_jsd_loss.

    Args:
        student_log_probs: Log-probabilities [*, V] (already log-softmax'd or log(probs))
        teacher_log_probs: Log-probabilities [*, V]
        beta: Interpolation coefficient. 0=forward KL, 0.5=symmetric, 1=reverse KL.

    Returns:
        Scalar JSD loss (mean over positions and vocab).
    """
    if beta == 0.0:
        # Forward KL: KL(teacher || student)
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1.0:
        # Reverse KL: KL(student || teacher)
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([
                student_log_probs + torch.log1p(-beta_t),
                teacher_log_probs + torch.log(beta_t),
            ]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1.0 - beta_t) * kl_student

    # Sum over vocab, mean over positions
    return jsd.sum(-1).mean()


def sorted_l1_loss(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> torch.Tensor:
    """Sorted L1 distance between two probability distributions (ULD approach).

    Both distributions are sorted in descending order, padded to the same
    vocab size, then L1 distance is computed.

    Args:
        student_probs: [num_positions, V_student] probability distributions
        teacher_probs: [num_positions, V_teacher] probability distributions

    Returns:
        Scalar L1 loss averaged over positions.
    """
    student_sorted = student_probs.sort(dim=-1, descending=True).values
    teacher_sorted = teacher_probs.sort(dim=-1, descending=True).values

    sv = student_sorted.size(-1)
    tv = teacher_sorted.size(-1)
    max_v = max(sv, tv)

    if sv < max_v:
        student_sorted = F.pad(student_sorted, (0, max_v - sv))
    if tv < max_v:
        teacher_sorted = F.pad(teacher_sorted, (0, max_v - tv))

    l1 = F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
    num_positions = student_probs.size(0)
    return l1 / max(num_positions, 1)


# ===================================================================
# Configuration
# ===================================================================


class GoldLossConfig(TypedDict):
    jsd_beta: float          # JSD interpolation: 0=forward KL, 0.5=symmetric, 1=reverse KL
    matched_weight: float    # Weight for matched-token JSD component
    unmatched_weight: float  # Weight for unmatched-token sorted L1 component
    temperature: float       # Softmax temperature for both student and teacher


# ===================================================================
# GOLD loss function
# ===================================================================


class GoldTrainLossFn:
    """GOLD hybrid loss: JSD for matched vocab + sorted L1 for unmatched vocab.

    Uses ``LossInputType.DISTILLATION`` to leverage nemo_rl's distributed
    top-k logprob extraction (handles DTensor/TP without gathering full logits).

    The framework extracts ``student_topk_logprobs`` at teacher's top-k indices
    via ``get_distillation_topk_logprobs_from_logits``, which performs a
    distributed log_softmax. The loss then splits the top-k into matched and
    unmatched vocabulary tokens.

    **Expected data dict keys**:
    - ``teacher_topk_logits``: (B, S, k) teacher top-k logits at student positions
    - ``teacher_topk_indices``: (B, S, k) teacher top-k global vocab indices
    - ``gold_position_mask``: (B, S) valid alignment group mask (1 at positions
      corresponding to the first student token of each group, 0 elsewhere)
    - ``gold_teacher_cond_factor``: (B, S) teacher conditional factor per position
    - ``gold_student_cond_factor``: (B, S) student conditional factor per position
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DISTILLATION

    # Required by the DISTILLATION path in prepare_loss_input
    zero_outside_topk = True
    kl_type = "forward"

    def __init__(self, cfg: GoldLossConfig, vocab_mapping: VocabMapping):
        self.jsd_beta = cfg.get("jsd_beta", 0.0)
        self.matched_weight = cfg.get("matched_weight", 1.0)
        self.unmatched_weight = cfg.get("unmatched_weight", 1.0)
        self.temperature = cfg.get("temperature", 1.0)
        self.vocab_mapping = vocab_mapping
        self._device_initialized = False

    def _ensure_device(self, device: torch.device):
        """Move vocab mapping tensors to the correct device once."""
        if not self._device_initialized:
            self.vocab_mapping.teacher_matched_mask = self.vocab_mapping.teacher_matched_mask.to(device)
            self.vocab_mapping.student_matched_mask = self.vocab_mapping.student_matched_mask.to(device)
            self.vocab_mapping.mapping_tensor = self.vocab_mapping.mapping_tensor.to(device)
            self._device_initialized = True

    def _resolve_parallel_context(
        self,
        student_logits: torch.Tensor,
        vocab_parallel_rank: Optional[int],
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup],
        context_parallel_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[
        torch.Tensor,
        Optional[torch.distributed.ProcessGroup],
        Optional[torch.distributed.ProcessGroup],
        int,
        int,
    ]:
        """Return local logits and parallel metadata for TP/CP-aware gathers."""
        cp_group = context_parallel_group
        if isinstance(student_logits, DTensor):
            device_mesh = student_logits.device_mesh
            tp_group = device_mesh.get_group("tp")
            local_logits = student_logits.to_local()
            tp_rank = tp_group.rank()
            vocab_local = int(local_logits.shape[-1])
            vocab_start = tp_rank * vocab_local
            if (
                device_mesh.mesh_dim_names is not None
                and "cp" in device_mesh.mesh_dim_names
            ):
                cp_group = device_mesh.get_group("cp")
            return local_logits, tp_group, cp_group, vocab_start, vocab_start + vocab_local

        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be set when vocab_parallel_group is provided"
            )
            local_logits = student_logits
            vocab_local = int(local_logits.shape[-1])
            vocab_start = vocab_parallel_rank * vocab_local
            return (
                local_logits,
                vocab_parallel_group,
                cp_group,
                vocab_start,
                vocab_start + vocab_local,
            )

        local_logits = student_logits
        vocab_end = int(local_logits.shape[-1])
        return local_logits, None, cp_group, 0, vocab_end

    def _localize_sequence_tensor(
        self,
        tensor: torch.Tensor,
        *,
        device: torch.device,
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Convert CP-sharded DTensors to local tensors aligned with local logits."""
        if isinstance(tensor, DTensor):
            return tensor.to_local().to(device)
        tensor = tensor.to(device)
        if cp_group is None or torch.distributed.get_world_size(cp_group) == 1:
            return tensor
        cp_rank = torch.distributed.get_rank(cp_group)
        return _get_tokens_on_this_cp_rank(tensor, cp_rank, torch.distributed.get_world_size(cp_group), seq_dim=1)

    def _gather_student_logits_at_indices(
        self,
        student_logits: torch.Tensor,
        global_indices: torch.Tensor,
        vocab_parallel_rank: Optional[int],
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup],
        context_parallel_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Differentiably gather logits at global vocab ids for local sequence positions."""
        local_logits, tp_group, cp_group, vocab_start, vocab_end = self._resolve_parallel_context(
            student_logits,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
        )
        indices_local = self._localize_sequence_tensor(
            global_indices,
            device=local_logits.device,
            cp_group=cp_group,
        )
        if tp_group is not None or cp_group is not None:
            return gather_logits_at_global_indices(
                local_logits,
                indices_local,
                tp_group=tp_group,
                cp_group=cp_group,
                vocab_start_index=vocab_start,
                vocab_end_index=vocab_end,
            )
        return local_logits.gather(dim=-1, index=indices_local)

    def _topk_student_unmatched_indices(
        self,
        student_logits: torch.Tensor,
        k: int,
        vocab_parallel_rank: Optional[int],
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup],
        context_parallel_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Select student unmatched top-k ids per position for the current forward pass."""
        available_unmatched = max(
            int(self.vocab_mapping.student_vocab_size - self.vocab_mapping.num_matched),
            0,
        )
        k = min(k, available_unmatched)
        local_logits, tp_group, cp_group, vocab_start, vocab_end = self._resolve_parallel_context(
            student_logits,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
        )
        if k <= 0:
            shape = (*local_logits.shape[:2], 0)
            return torch.zeros(shape, dtype=torch.long, device=local_logits.device)

        logits_for_topk = local_logits.detach().to(torch.float32)
        student_matched_mask = self.vocab_mapping.student_matched_mask.to(local_logits.device)
        local_student_matched = student_matched_mask[vocab_start:vocab_end]
        logits_for_topk = logits_for_topk.masked_fill(
            local_student_matched.view(1, 1, -1),
            float("-inf"),
        )

        if tp_group is not None:
            _, topk_indices = distributed_vocab_topk(
                logits_for_topk,
                k=k,
                tp_group=tp_group,
                vocab_start_index=vocab_start,
                vocab_end_index=vocab_end,
            )
            return topk_indices

        k_eff = min(k, int(logits_for_topk.shape[-1]))
        if k_eff <= 0:
            shape = (*logits_for_topk.shape[:2], 0)
            return torch.zeros(shape, dtype=torch.long, device=logits_for_topk.device)
        topk_indices = torch.topk(logits_for_topk, k=k_eff, dim=-1).indices
        return topk_indices + vocab_start

    def prepare_distillation_loss_input(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> dict[str, torch.Tensor | None]:
        """Prepare GOLD-specific distillation tensors from current student logits."""
        local_logits, _, cp_group, _, _ = self._resolve_parallel_context(
            logits,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
        )
        device = local_logits.device
        self._ensure_device(device)

        teacher_topk_logits = self._localize_sequence_tensor(
            data["teacher_topk_logits"],
            device=device,
            cp_group=cp_group,
        )
        teacher_topk_indices_orig = self._localize_sequence_tensor(
            data["gold_teacher_topk_indices_original"],
            device=device,
            cp_group=cp_group,
        )

        max_teacher_idx = max(int(self.vocab_mapping.teacher_matched_mask.shape[0]) - 1, 0)
        teacher_match_indices = teacher_topk_indices_orig.clamp(0, max_teacher_idx)
        teacher_is_matched = self.vocab_mapping.teacher_matched_mask[teacher_match_indices]

        mapped_student_indices = torch.zeros_like(teacher_topk_indices_orig)
        if self.vocab_mapping.mapping_tensor.numel() > 0:
            safe_mapping_idx = teacher_topk_indices_orig.clamp(
                0, int(self.vocab_mapping.mapping_tensor.shape[0]) - 1
            )
            mapping_valid = (teacher_topk_indices_orig >= 0) & (
                teacher_topk_indices_orig < int(self.vocab_mapping.mapping_tensor.shape[0])
            )
            mapped_values = self.vocab_mapping.mapping_tensor[safe_mapping_idx].clamp(min=0)
            mapped_student_indices = torch.where(
                mapping_valid,
                mapped_values,
                mapped_student_indices,
            )

        student_teacher_view_logits = self._gather_student_logits_at_indices(
            logits,
            mapped_student_indices,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
        )

        unmatched_counts = (~teacher_is_matched).sum(dim=-1)
        max_unmatched_k = int(unmatched_counts.max().item()) if unmatched_counts.numel() > 0 else 0
        if max_unmatched_k > 0:
            student_unmatched_indices = self._topk_student_unmatched_indices(
                logits,
                k=max_unmatched_k,
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
            )
            student_unmatched_logits = self._gather_student_logits_at_indices(
                logits,
                student_unmatched_indices,
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
            )
            student_unmatched_topk_logprobs = torch.log_softmax(student_unmatched_logits, dim=-1)
        else:
            student_unmatched_topk_logprobs = teacher_topk_logits.new_zeros(
                teacher_topk_logits.shape[0],
                teacher_topk_logits.shape[1],
                0,
            )

        return {
            "student_topk_logprobs": torch.log_softmax(student_teacher_view_logits, dim=-1),
            "teacher_topk_logprobs": torch.log_softmax(teacher_topk_logits, dim=-1),
            "student_unmatched_topk_logprobs": student_unmatched_topk_logprobs,
            "H_all": None,
        }

    def __call__(
        self,
        student_topk_logprobs: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
        student_unmatched_topk_logprobs: torch.Tensor | None,
        H_all: torch.Tensor | None,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute GOLD hybrid loss from top-k logprobs.

        Args:
            student_topk_logprobs: [B, S, k] student log-probs at remapped teacher top-k indices.
            teacher_topk_logprobs: [B, S, k] teacher log-probs at top-k indices.
            student_unmatched_topk_logprobs: [B, S, k_u] student log-probs at the
                student's own unmatched top-k indices for each position.
            H_all: Student entropy (unused, may be None).
            data: Data dict with alignment masks and conditional factors.
            global_valid_seqs: Number of valid sequences for normalization.
            global_valid_toks: Number of valid tokens for normalization.

        Returns:
            (loss, metrics) tuple.
        """
        device = student_topk_logprobs.device
        self._ensure_device(device)

        batch_size = student_topk_logprobs.shape[0]
        seq_len = student_topk_logprobs.shape[1]

        # Unpack data dict
        teacher_topk_indices_orig = data["gold_teacher_topk_indices_original"].to(device)
        if teacher_topk_indices_orig.shape[1] > seq_len:
            teacher_topk_indices_orig = teacher_topk_indices_orig[:, :seq_len, :]

        position_mask = data["gold_position_mask"].to(device)
        if position_mask.shape[1] > seq_len:
            position_mask = position_mask[:, :seq_len]
        teacher_cond_factor = data["gold_teacher_cond_factor"].to(device)
        if teacher_cond_factor.shape[1] > seq_len:
            teacher_cond_factor = teacher_cond_factor[:, :seq_len]
        student_cond_factor = data["gold_student_cond_factor"].to(device)
        if student_cond_factor.shape[1] > seq_len:
            student_cond_factor = student_cond_factor[:, :seq_len]

        sample_mask = data.get("sample_mask")
        if sample_mask is None:
            sample_mask = torch.ones(batch_size, dtype=torch.float32, device=device)
        else:
            sample_mask = sample_mask.to(device=device, dtype=torch.float32)

        # Apply temperature scaling to logprobs before converting to probs.
        # The DISTILLATION path computes log_softmax at the default temperature;
        # re-scale here so that a non-default temperature config takes effect.
        if self.temperature != 1.0:
            student_topk_logprobs = student_topk_logprobs / self.temperature
            teacher_topk_logprobs = teacher_topk_logprobs / self.temperature
            # Re-normalize within top-k after rescaling
            student_topk_logprobs = student_topk_logprobs - student_topk_logprobs.logsumexp(dim=-1, keepdim=True)
            teacher_topk_logprobs = teacher_topk_logprobs - teacher_topk_logprobs.logsumexp(dim=-1, keepdim=True)

        # Convert logprobs to probs
        student_topk_probs = student_topk_logprobs.exp()
        teacher_topk_probs = teacher_topk_logprobs.exp()
        if student_unmatched_topk_logprobs is None:
            student_unmatched_topk_probs = student_topk_probs.new_zeros(batch_size, seq_len, 0)
        else:
            if self.temperature != 1.0:
                student_unmatched_topk_logprobs = student_unmatched_topk_logprobs / self.temperature
                student_unmatched_topk_logprobs = (
                    student_unmatched_topk_logprobs
                    - student_unmatched_topk_logprobs.logsumexp(dim=-1, keepdim=True)
                )
            student_unmatched_topk_probs = student_unmatched_topk_logprobs.exp()

        # Apply conditional factors (multiply probs by scalar per position)
        eps = 1e-8
        s_cond_scale = torch.exp(student_cond_factor).clamp(min=eps).unsqueeze(-1)
        t_cond_scale = torch.exp(teacher_cond_factor).clamp(min=eps).unsqueeze(-1)
        student_topk_probs = student_topk_probs * s_cond_scale
        teacher_topk_probs = teacher_topk_probs * t_cond_scale
        student_unmatched_topk_probs = student_unmatched_topk_probs * s_cond_scale

        # Identify matched/unmatched using ORIGINAL teacher indices
        teacher_matched_mask_full = self.vocab_mapping.teacher_matched_mask
        max_idx = teacher_matched_mask_full.shape[0]
        safe_indices = teacher_topk_indices_orig.clamp(0, max_idx - 1)
        topk_is_matched = teacher_matched_mask_full[safe_indices]

        # ---- Matched tokens: JSD ----
        matched_count = topk_is_matched.float()
        s_matched = student_topk_probs * matched_count
        t_matched = teacher_topk_probs * matched_count
        s_matched_sum = s_matched.sum(-1, keepdim=True).clamp(min=eps)
        t_matched_sum = t_matched.sum(-1, keepdim=True).clamp(min=eps)
        s_log = torch.log((s_matched / s_matched_sum).clamp(min=eps))
        t_log = torch.log((t_matched / t_matched_sum).clamp(min=eps))

        if self.jsd_beta == 0.0:
            per_pos_jsd = F.kl_div(s_log, t_log, reduction="none", log_target=True)
        elif self.jsd_beta == 1.0:
            per_pos_jsd = F.kl_div(t_log, s_log, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(self.jsd_beta, dtype=s_log.dtype, device=device)
            mixture = torch.logsumexp(
                torch.stack([s_log + torch.log1p(-beta_t), t_log + torch.log(beta_t)]), dim=0
            )
            kl_t = F.kl_div(mixture, t_log, reduction="none", log_target=True)
            kl_s = F.kl_div(mixture, s_log, reduction="none", log_target=True)
            per_pos_jsd = beta_t * kl_t + (1.0 - beta_t) * kl_s

        per_pos_jsd = (per_pos_jsd * matched_count).sum(-1)
        matched_present = (matched_count.sum(-1) > 0).float()
        # In this top-k approximation we only observe a truncated matched subset,
        # so using unnormalized chunk masses directly in KL is unstable. Keep the
        # JSD on normalized matched distributions and inject chunk-probability
        # information as a symmetric per-position weight.
        matched_weight_scale = torch.exp(0.5 * (student_cond_factor + teacher_cond_factor)).clamp(min=eps)
        matched_loss = per_pos_jsd * position_mask * matched_present * matched_weight_scale

        # ---- Unmatched tokens: sorted L1 ----
        unmatched_count = (~topk_is_matched).float()
        t_unmatched = teacher_topk_probs * unmatched_count

        s_unsorted = student_unmatched_topk_probs.sort(dim=-1, descending=True).values
        t_unsorted = t_unmatched.sort(dim=-1, descending=True).values
        sv = s_unsorted.size(-1)
        tv = t_unsorted.size(-1)
        max_v = max(sv, tv)
        if sv < max_v:
            s_unsorted = F.pad(s_unsorted, (0, max_v - sv))
        if tv < max_v:
            t_unsorted = F.pad(t_unsorted, (0, max_v - tv))
        per_pos_l1 = (s_unsorted - t_unsorted).abs().sum(-1)
        unmatched_present = (unmatched_count.sum(-1) > 0).float()
        unmatched_loss = per_pos_l1 * position_mask * unmatched_present

        # ---- Combine ----
        per_pos_loss = self.matched_weight * matched_loss + self.unmatched_weight * unmatched_loss
        # Average over valid positions per sample
        n_groups_per_sample = position_mask.sum(-1)  # [B] actual group count
        n_valid_per_sample = n_groups_per_sample.clamp(min=1)  # [B] for safe division
        per_sample_loss = per_pos_loss.sum(-1) / n_valid_per_sample  # [B]

        # Only include samples that actually have alignment groups
        valid_sample_mask = (n_groups_per_sample > 0).float() * sample_mask
        if valid_sample_mask.sum() == 0:
            loss = student_topk_logprobs.sum() * 0.0
            return loss, {
                "loss": 0.0,
                "matched_jsd_loss": 0.0,
                "unmatched_l1_loss": 0.0,
                "num_groups": 0,
                "num_valid_samples": 0,
            }

        loss = masked_mean(per_sample_loss, valid_sample_mask, global_normalization_factor=None)

        n_valid = int(valid_sample_mask.sum().item())
        total_groups = int(position_mask.sum().item())
        avg_matched = masked_mean(
            matched_loss.sum(-1) / n_valid_per_sample, valid_sample_mask, global_normalization_factor=None
        ).item()
        avg_unmatched = masked_mean(
            unmatched_loss.sum(-1) / n_valid_per_sample, valid_sample_mask, global_normalization_factor=None
        ).item()

        metrics = {
            "loss": loss.item(),
            "matched_jsd_loss": avg_matched,
            "unmatched_l1_loss": avg_unmatched,
            "num_groups": total_groups,
            "num_valid_samples": n_valid,
            "mean_groups_per_sample": total_groups / max(n_valid, 1),
            "vocab_overlap_pct": self.vocab_mapping.jaccard_index * 100,
        }

        return loss, metrics

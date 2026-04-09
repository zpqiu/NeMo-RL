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
from typing import Any, TypedDict

import torch
import torch.nn.functional as F

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
            self.vocab_mapping.mapping_tensor = self.vocab_mapping.mapping_tensor.to(device)
            self._device_initialized = True

    def __call__(
        self,
        student_topk_logprobs: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
        H_all: torch.Tensor | None,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute GOLD hybrid loss from top-k logprobs.

        Args:
            student_topk_logprobs: [B, S-1, k] student log-probs at teacher's top-k indices.
            teacher_topk_logprobs: [B, S-1, k] teacher log-probs at top-k indices.
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
        seq_len = student_topk_logprobs.shape[1]  # S-1
        topk_k = student_topk_logprobs.shape[2]

        # Unpack data dict
        # Use ORIGINAL teacher indices for matched/unmatched classification
        # (the "teacher_topk_indices" key has been remapped to student vocab
        # for the DISTILLATION gather path, so use the original for classification)
        teacher_topk_indices_orig = data["gold_teacher_topk_indices_original"].to(device)  # (B, S, k)
        # Align to S-1 (the DISTILLATION path shifts by 1)
        if teacher_topk_indices_orig.shape[1] > seq_len:
            teacher_topk_indices_orig = teacher_topk_indices_orig[:, :seq_len, :]

        position_mask = data["gold_position_mask"].to(device)  # (B, S)
        if position_mask.shape[1] > seq_len:
            position_mask = position_mask[:, :seq_len]
        teacher_cond_factor = data["gold_teacher_cond_factor"].to(device)  # (B, S)
        if teacher_cond_factor.shape[1] > seq_len:
            teacher_cond_factor = teacher_cond_factor[:, :seq_len]
        student_cond_factor = data["gold_student_cond_factor"].to(device)  # (B, S)
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
        student_topk_probs = student_topk_logprobs.exp()  # [B, S-1, k]
        teacher_topk_probs = teacher_topk_logprobs.exp()  # [B, S-1, k]

        # Apply conditional factors (multiply probs by scalar per position)
        eps = 1e-8
        s_cond_scale = torch.exp(student_cond_factor).clamp(min=eps).unsqueeze(-1)  # [B, S, 1]
        t_cond_scale = torch.exp(teacher_cond_factor).clamp(min=eps).unsqueeze(-1)
        student_topk_probs = student_topk_probs * s_cond_scale
        teacher_topk_probs = teacher_topk_probs * t_cond_scale

        # Identify matched/unmatched using ORIGINAL teacher indices
        teacher_matched_mask_full = self.vocab_mapping.teacher_matched_mask  # [V_teacher]
        max_idx = teacher_matched_mask_full.shape[0]
        safe_indices = teacher_topk_indices_orig.clamp(0, max_idx - 1)
        topk_is_matched = teacher_matched_mask_full[safe_indices]  # [B, S-1, k] bool

        # ---- Matched tokens: JSD ----
        # Compute per-position JSD over matched top-k tokens
        matched_count = topk_is_matched.float()  # [B, S-1, k]
        # Mask out unmatched tokens for JSD
        s_matched = student_topk_probs * matched_count
        t_matched = teacher_topk_probs * matched_count
        # Renormalize within matched tokens per position
        s_matched_sum = s_matched.sum(-1, keepdim=True).clamp(min=eps)
        t_matched_sum = t_matched.sum(-1, keepdim=True).clamp(min=eps)
        s_matched_norm = s_matched / s_matched_sum
        t_matched_norm = t_matched / t_matched_sum

        s_log = torch.log(s_matched_norm.clamp(min=eps))
        t_log = torch.log(t_matched_norm.clamp(min=eps))

        # Generalized JSD per position
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

        # Mask to matched tokens only and sum over vocab dim
        per_pos_jsd = (per_pos_jsd * matched_count).sum(-1)  # [B, S-1]
        # Mask to valid alignment positions
        matched_loss = per_pos_jsd * position_mask  # [B, S-1]

        # ---- Unmatched tokens: sorted L1 ----
        unmatched_count = (~topk_is_matched).float()
        s_unmatched = student_topk_probs * unmatched_count
        t_unmatched = teacher_topk_probs * unmatched_count

        # Sort unmatched probs per position and compute L1
        s_unsorted = s_unmatched.sort(dim=-1, descending=True).values
        t_unsorted = t_unmatched.sort(dim=-1, descending=True).values
        per_pos_l1 = (s_unsorted - t_unsorted).abs().sum(-1)  # [B, S-1]
        unmatched_loss = per_pos_l1 * position_mask  # [B, S-1]

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

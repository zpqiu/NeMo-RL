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

    Compatible with ``Policy.train()`` via ``LossInputType.LOGIT``.

    **Expected data dict keys** (all padded to (B, S)):
    - ``gold_teacher_topk_logits``: (B, S, k) teacher top-k logits per alignment group
    - ``gold_teacher_topk_indices``: (B, S, k) teacher top-k vocab indices per group
    - ``gold_teacher_cond_factor``: (B, S) sum of teacher logprobs for multi-token groups
    - ``gold_student_cond_factor``: (B, S) sum of student logprobs for multi-token groups
    - ``gold_group_student_pos``: (B, S) student token position for each alignment group
    - ``gold_group_mask``: (B, S) valid alignment group mask
    - ``gold_prompt_lengths``: (B,) prompt lengths in student tokens
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGIT

    def __init__(self, cfg: GoldLossConfig, vocab_mapping: VocabMapping):
        self.jsd_beta = cfg.get("jsd_beta", 0.0)
        self.matched_weight = cfg.get("matched_weight", 1.0)
        self.unmatched_weight = cfg.get("unmatched_weight", 1.0)
        self.temperature = cfg.get("temperature", 1.0)
        self.vocab_mapping = vocab_mapping

        # Pre-compute index tensors for efficiency (will be moved to device on first call)
        self._matched_student_ids_t = torch.tensor(
            vocab_mapping.matched_student_ids, dtype=torch.long
        )
        self._matched_teacher_ids_t = torch.tensor(
            vocab_mapping.matched_teacher_ids, dtype=torch.long
        )
        self._device_initialized = False

    def _ensure_device(self, device: torch.device):
        """Move pre-computed tensors to the correct device once."""
        if not self._device_initialized:
            self._matched_student_ids_t = self._matched_student_ids_t.to(device)
            self._matched_teacher_ids_t = self._matched_teacher_ids_t.to(device)
            self._device_initialized = True

    def __call__(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute GOLD hybrid loss.

        Args:
            logits: Student logits [B, S, V_student] from the training forward pass.
            data: Data dict with alignment info and teacher top-k data.
            global_valid_seqs: Number of valid sequences for normalization.
            global_valid_toks: Number of valid tokens for normalization.

        Returns:
            (loss, metrics) tuple.
        """
        # Handle DTensor (tensor-parallel sharded logits): gather the full
        # vocab on each rank so we can index with global vocab IDs.
        # With TP, the vocab dim is padded to be divisible by TP size.
        # After gathering, truncate to the original vocab size.
        try:
            from torch.distributed.tensor import DTensor
            if isinstance(logits, DTensor):
                logits = logits.full_tensor()
                # Truncate TP padding on the vocab dimension
                student_vocab = self.vocab_mapping.student_vocab_size
                if logits.shape[-1] > student_vocab:
                    logits = logits[..., :student_vocab]
        except ImportError:
            pass

        device = logits.device
        self._ensure_device(device)

        batch_size = logits.shape[0]

        # Unpack data dict
        teacher_topk_logits = data["gold_teacher_topk_logits"].to(device)  # (B, S, k)
        teacher_topk_indices = data["gold_teacher_topk_indices"].to(device)  # (B, S, k)
        teacher_cond_factor = data["gold_teacher_cond_factor"].to(device)  # (B, S)
        student_cond_factor = data["gold_student_cond_factor"].to(device)  # (B, S)
        group_student_pos = data["gold_group_student_pos"].to(device)  # (B, S)
        group_mask = data["gold_group_mask"].to(device)  # (B, S)
        prompt_lengths = data["gold_prompt_lengths"].to(device)  # (B,)

        sample_mask = data.get("sample_mask")
        if sample_mask is None:
            sample_mask = torch.ones(batch_size, dtype=torch.float32, device=device)
        else:
            sample_mask = sample_mask.to(device=device, dtype=torch.float32)

        # Find max valid groups to limit iteration
        max_valid = int(group_mask.sum(dim=1).max().item()) if group_mask.sum() > 0 else 0

        # Per-sample loss accumulators
        sample_matched_losses: list[torch.Tensor] = []
        sample_unmatched_losses: list[torch.Tensor] = []
        sample_group_counts: list[int] = []
        valid_sample_mask = torch.zeros(batch_size, dtype=torch.float32, device=device)

        # Debug accumulators
        total_matched_loss = 0.0
        total_unmatched_loss = 0.0
        total_groups = 0

        for b in range(batch_size):
            if sample_mask[b] == 0:
                sample_matched_losses.append(logits.sum() * 0.0)
                sample_unmatched_losses.append(logits.sum() * 0.0)
                sample_group_counts.append(0)
                continue

            prompt_len = int(prompt_lengths[b].item())
            b_matched_losses = []
            b_unmatched_losses = []

            for g in range(max_valid):
                if group_mask[b, g] == 0:
                    continue

                # Student logits at the group's representative position
                s_pos = int(group_student_pos[b, g].item())
                # Map generation-relative position to sequence position
                logit_pos = prompt_len + s_pos
                if logit_pos >= logits.shape[1]:
                    continue

                student_logits_at_pos = logits[b, logit_pos, :]  # [V_student]

                # Teacher top-k at this group
                t_topk_logits = teacher_topk_logits[b, g, :]  # [k]
                t_topk_indices = teacher_topk_indices[b, g, :]  # [k]

                # Conditional factors for multi-token groups
                s_cond = student_cond_factor[b, g]  # scalar logprob sum
                t_cond = teacher_cond_factor[b, g]  # scalar logprob sum

                matched_loss, unmatched_loss = self._compute_group_loss(
                    student_logits_at_pos=student_logits_at_pos,
                    teacher_topk_logits=t_topk_logits,
                    teacher_topk_indices=t_topk_indices,
                    student_cond_factor=s_cond,
                    teacher_cond_factor=t_cond,
                )

                b_matched_losses.append(matched_loss)
                b_unmatched_losses.append(unmatched_loss)

            n_groups = len(b_matched_losses)
            if n_groups > 0:
                b_matched = torch.stack(b_matched_losses).mean()
                b_unmatched = torch.stack(b_unmatched_losses).mean()
                valid_sample_mask[b] = 1.0
                total_matched_loss += b_matched.item()
                total_unmatched_loss += b_unmatched.item()
                total_groups += n_groups
            else:
                b_matched = logits.sum() * 0.0
                b_unmatched = logits.sum() * 0.0

            sample_matched_losses.append(b_matched)
            sample_unmatched_losses.append(b_unmatched)
            sample_group_counts.append(n_groups)

        # Combine losses
        if valid_sample_mask.sum() == 0:
            loss = logits.sum() * 0.0
            return loss, {
                "loss": 0.0,
                "matched_jsd_loss": 0.0,
                "unmatched_l1_loss": 0.0,
                "num_groups": 0,
                "num_valid_samples": 0,
            }

        matched_loss_tensor = torch.stack(sample_matched_losses)
        unmatched_loss_tensor = torch.stack(sample_unmatched_losses)
        combined = (
            self.matched_weight * matched_loss_tensor
            + self.unmatched_weight * unmatched_loss_tensor
        )
        loss = masked_mean(combined, valid_sample_mask * sample_mask, global_normalization_factor=None)

        n_valid = int(valid_sample_mask.sum().item())
        metrics = {
            "loss": loss.item(),
            "matched_jsd_loss": total_matched_loss / max(n_valid, 1),
            "unmatched_l1_loss": total_unmatched_loss / max(n_valid, 1),
            "num_groups": total_groups,
            "num_valid_samples": n_valid,
            "mean_groups_per_sample": total_groups / max(n_valid, 1),
            "vocab_overlap_pct": self.vocab_mapping.jaccard_index * 100,
        }

        return loss, metrics

    def _compute_group_loss(
        self,
        student_logits_at_pos: torch.Tensor,
        teacher_topk_logits: torch.Tensor,
        teacher_topk_indices: torch.Tensor,
        student_cond_factor: torch.Tensor,
        teacher_cond_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GOLD loss for a single alignment group.

        Args:
            student_logits_at_pos: [V_student] student logits at the group's position
            teacher_topk_logits: [k] teacher top-k logits at the group's position
            teacher_topk_indices: [k] teacher top-k global vocab indices
            student_cond_factor: scalar, sum of student logprobs for tokens 2..N in group
            teacher_cond_factor: scalar, sum of teacher logprobs for tokens 2..N in group

        Returns:
            (matched_jsd_loss, unmatched_l1_loss) tuple.
        """
        device = student_logits_at_pos.device
        eps = 1e-8

        # --- Student probabilities (full distribution) ---
        student_log_probs = F.log_softmax(student_logits_at_pos / self.temperature, dim=-1)
        student_probs = student_log_probs.exp()

        # --- Teacher probabilities (from top-k) ---
        # Apply temperature and compute softmax over top-k
        teacher_topk_probs = F.softmax(teacher_topk_logits / self.temperature, dim=-1)
        # Note: teacher_topk_probs sums to ~1 over the top-k tokens only.
        # For tokens outside top-k, we assume negligible probability.

        # --- Apply conditional factors for multi-token groups ---
        # These scale the entire distribution by the product of conditional probs
        # for subsequent tokens in the group (chain rule).
        s_cond_scale = torch.exp(student_cond_factor).clamp(min=eps)
        t_cond_scale = torch.exp(teacher_cond_factor).clamp(min=eps)

        # Scale student probs (full distribution scaled by scalar)
        student_probs_scaled = student_probs * s_cond_scale

        # Scale teacher top-k probs
        teacher_topk_probs_scaled = teacher_topk_probs * t_cond_scale

        # --- Split into matched / unmatched ---

        # Find which teacher top-k tokens are matched
        # For each top-k index, check if it's in matched_teacher_ids
        teacher_matched_mask = self.vocab_mapping.teacher_matched_mask.to(device)
        topk_is_matched = teacher_matched_mask[teacher_topk_indices]  # [k] bool

        # 1) Matched tokens: JSD loss
        matched_loss = student_logits_at_pos.sum() * 0.0  # zero with grad
        if topk_is_matched.any() and self.vocab_mapping.num_matched > 0:
            # Extract matched teacher probs and their corresponding student probs
            matched_teacher_topk_idx = topk_is_matched.nonzero(as_tuple=True)[0]
            matched_teacher_global_ids = teacher_topk_indices[matched_teacher_topk_idx]
            matched_teacher_probs = teacher_topk_probs_scaled[matched_teacher_topk_idx]

            # Map teacher IDs to student IDs via the mapping tensor
            mapping = self.vocab_mapping.mapping_tensor.to(device)
            # Clamp indices to valid range for the mapping tensor
            safe_indices = matched_teacher_global_ids.clamp(max=mapping.shape[0] - 1)
            matched_student_global_ids = mapping[safe_indices]

            # Filter out any that failed to map (shouldn't happen but be safe)
            valid = matched_student_global_ids >= 0
            if valid.any():
                matched_student_global_ids = matched_student_global_ids[valid]
                matched_teacher_probs = matched_teacher_probs[valid]

                matched_student_probs = student_probs_scaled[matched_student_global_ids]

                # Renormalize for JSD computation (the matched subset doesn't sum to 1)
                s_matched_sum = matched_student_probs.sum().clamp(min=eps)
                t_matched_sum = matched_teacher_probs.sum().clamp(min=eps)
                s_matched_normalized = matched_student_probs / s_matched_sum
                t_matched_normalized = matched_teacher_probs / t_matched_sum

                s_log = torch.log(s_matched_normalized.clamp(min=eps))
                t_log = torch.log(t_matched_normalized.clamp(min=eps))

                # JSD over matched vocab (unsqueeze to add position dim)
                matched_loss = generalized_jsd_loss(
                    s_log.unsqueeze(0),
                    t_log.unsqueeze(0),
                    beta=self.jsd_beta,
                )

        # 2) Unmatched tokens: sorted L1 loss
        unmatched_loss = student_logits_at_pos.sum() * 0.0  # zero with grad

        # Student unmatched: all tokens NOT in matched set
        student_matched_mask = self.vocab_mapping.student_matched_mask.to(device)
        student_unmatched_mask = ~student_matched_mask
        student_unmatched_probs = student_probs_scaled[student_unmatched_mask]

        # Teacher unmatched: top-k tokens NOT in matched set
        teacher_unmatched_probs = teacher_topk_probs_scaled[~topk_is_matched]

        if student_unmatched_probs.numel() > 0 and teacher_unmatched_probs.numel() > 0:
            unmatched_loss = sorted_l1_loss(
                student_unmatched_probs.unsqueeze(0),
                teacher_unmatched_probs.unsqueeze(0),
            )

        return matched_loss, unmatched_loss

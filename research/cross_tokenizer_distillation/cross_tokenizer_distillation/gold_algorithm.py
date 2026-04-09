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

"""GOLD (General On-Policy Logit Distillation) algorithm for cross-tokenizer distillation.

Ported from TRL's experimental GOLD implementation, adapted for nemo_rl.
Shares the on-policy generation + cross-tokenizer alignment pipeline with the
existing ``algorithm.py`` but replaces the IS-based loss with GOLD's hybrid
JSD (matched vocab) + sorted-L1 (unmatched vocab) loss.

Reference: https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, TypedDict, TypeVar, cast

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import _should_use_async_rollouts, refit_policy_generation
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig, print_message_log_samples
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

from cross_tokenizer_distillation.gold_loss import (
    GoldLossConfig,
    GoldTrainLossFn,
    VocabMapping,
    build_vocab_mapping,
)
from cross_tokenizer_distillation.token_alignment import (
    AlignmentResult,
    align_tokens_from_original_student_ids_with_stats,
    merge_alignment_chunks,
)

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class GoldDistillationConfig(TypedDict):
    """GOLD distillation training configuration."""

    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_rollout_turns: int
    max_num_steps: int
    max_num_epochs: int
    val_batch_size: int
    val_period: int
    val_at_start: bool
    val_at_end: bool
    max_val_samples: int
    seed: int
    teacher_topk_k: int  # top-k for teacher logits (default 1024)


class GoldDistillSaveState(TypedDict):
    total_steps: int
    current_epoch: int
    current_step: int
    val_reward: NotRequired[float]
    consumed_samples: int
    total_valid_tokens: int


def _default_save_state() -> GoldDistillSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


def _mean_metric_value(value: Any) -> float:
    """Convert worker/micro-batch metric payloads to a single scalar mean."""
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float("nan")
        return float(np.mean(value))
    if isinstance(value, (list, tuple)):
        scalar_values = [_mean_metric_value(v) for v in value]
        scalar_values = [v for v in scalar_values if not np.isnan(v)]
        if not scalar_values:
            return float("nan")
        return float(sum(scalar_values) / len(scalar_values))
    return float(value)


def _truncate_message_log_at_first_assistant(
    message_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the prompt and the first assistant response, dropping post-response env turns."""
    truncated: list[dict[str, Any]] = []
    for message in message_log:
        truncated.append(message)
        if message["role"] == "assistant":
            break
    return truncated


def _extract_token_logprob(
    token_logprobs: torch.Tensor,
    token_seq_len: int,
    token_position: int,
) -> torch.Tensor | None:
    """Extract logprob for a single token position."""
    logprob_seq_len = token_logprobs.shape[0]
    if token_position < 0 or token_position >= token_seq_len:
        return None
    if logprob_seq_len >= token_seq_len:
        lp_idx = token_position
    elif logprob_seq_len == token_seq_len - 1:
        if token_position == 0:
            return None
        lp_idx = token_position - 1
    else:
        return None
    if lp_idx >= token_logprobs.shape[0]:
        return None
    return token_logprobs[lp_idx]


def _extract_logprob_span_for_token_range(
    token_logprobs: torch.Tensor,
    token_seq_len: int,
    start_token_pos: int,
    end_token_pos_exclusive: int,
) -> torch.Tensor:
    """Extract logprobs for a range of token positions."""
    gathered: list[torch.Tensor] = []
    for token_pos in range(start_token_pos, end_token_pos_exclusive):
        lp = _extract_token_logprob(
            token_logprobs=token_logprobs,
            token_seq_len=token_seq_len,
            token_position=token_pos,
        )
        if lp is not None:
            gathered.append(lp)
    if not gathered:
        return token_logprobs.new_zeros(0)
    return torch.stack(gathered)


class GoldDistillMasterConfig(TypedDict):
    """Main configuration for GOLD cross-tokenizer distillation."""

    policy: PolicyConfig
    teacher: PolicyConfig
    loss_fn: GoldLossConfig
    env: dict[str, Any]
    data: DataConfig
    gold_distillation: GoldDistillationConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Text-space alignment (reuses existing token_alignment module)
# ===============================================================================


def decode_and_align(
    generated_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
    min_chunk_bytes: int = 0,
) -> tuple[list[str], list[AlignmentResult], dict[str, int]]:
    """Decode student-generated text and compute cross-tokenizer alignment.

    Identical to the function in algorithm.py — reused here to keep GOLD
    self-contained.
    """
    batch_size = generated_ids.shape[0]
    decoded_texts: list[str] = []
    alignments: list[AlignmentResult] = []
    alignment_stats: dict[str, int] = {
        "student_fast_path_hits": 0,
        "student_fast_path_misses": 0,
        "student_visible_piece_path_hits": 0,
        "piece_greedy_hits": 0,
        "piece_span_fallback_hits": 0,
    }

    for i in range(batch_size):
        prompt_len = int(input_lengths[i].item())
        gen_ids = generated_ids[i, prompt_len:]

        # Strip padding
        if student_tokenizer.pad_token_id is not None:
            mask = gen_ids != student_tokenizer.pad_token_id
            gen_ids = gen_ids[mask]

        if gen_ids.numel() == 0:
            decoded_texts.append("")
            alignments.append(AlignmentResult(text="", teacher_token_ids=[], student_token_ids=[]))
            continue

        # Strip trailing special tokens
        special_ids = set(student_tokenizer.all_special_ids or [])
        while gen_ids.numel() > 0 and int(gen_ids[-1].item()) in special_ids:
            gen_ids = gen_ids[:-1]

        if gen_ids.numel() == 0:
            decoded_texts.append("")
            alignments.append(AlignmentResult(text="", teacher_token_ids=[], student_token_ids=[]))
            continue

        gen_id_list = gen_ids.tolist()
        text = student_tokenizer.decode(
            gen_id_list,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_texts.append(text)

        alignment, sample_stats = align_tokens_from_original_student_ids_with_stats(
            text=text,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            original_student_token_ids=gen_id_list,
        )
        for key, value in sample_stats.items():
            alignment_stats[key] = alignment_stats.get(key, 0) + value
        alignment = merge_alignment_chunks(alignment, min_bytes=min_chunk_bytes)
        alignments.append(alignment)

    return decoded_texts, alignments, alignment_stats


# ===============================================================================
# GOLD-specific packing
# ===============================================================================


def pack_gold_alignment_into_data(
    alignments: list[AlignmentResult],
    teacher_topk_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_gen_logprobs: list[torch.Tensor],
    student_prev_logprobs: torch.Tensor,
    teacher_input_lengths: list[int],
    student_prompt_lengths: list[int],
    seq_len: int,
    topk_k: int,
    vocab_mapping: VocabMapping,
) -> dict[str, torch.Tensor]:
    """Pack alignment + teacher top-k into tensors indexed by student positions.

    The DISTILLATION loss path gathers student logprobs at specified indices
    via ``get_distillation_topk_logprobs_from_logits``. For cross-tokenizer
    GOLD, teacher top-k indices are in the teacher's vocabulary space — we
    must remap matched tokens to the student's vocabulary so the gathered
    student logprobs correspond to semantically equivalent tokens.

    Remapping strategy:
    - Matched teacher tokens → replaced with the corresponding student ID
    - Unmatched teacher tokens → kept as-is (their student logprobs are
      not semantically meaningful, but the sorted L1 loss only uses
      relative ordering anyway)

    All tensors are padded to ``[B, seq_len]`` (student sequence length).

    Args:
        alignments: Per-sample alignment results.
        teacher_topk_logits: [B, S_teacher, k] teacher top-k logits.
        teacher_topk_indices: [B, S_teacher, k] teacher top-k vocab indices.
        teacher_gen_logprobs: Per-sample teacher token logprobs for cond factors.
        student_prev_logprobs: [B, S_student-1] student logprobs for cond factors.
        teacher_input_lengths: Per-sample teacher prompt lengths.
        student_prompt_lengths: Per-sample student prompt lengths.
        seq_len: Student sequence length for padding.
        topk_k: Number of top-k.
        vocab_mapping: Vocabulary mapping for teacher→student ID remapping.

    Returns:
        Dict of tensors to merge into train_data.
    """
    batch_size = len(alignments)

    # Pre-build the teacher→student remapping tensor
    mapping = vocab_mapping.mapping_tensor  # teacher_id → student_id (or -1)
    max_teacher_id = mapping.shape[0]

    # Tensors indexed by STUDENT sequence position
    packed_teacher_topk_logits = torch.zeros(batch_size, seq_len, topk_k)
    packed_student_gather_indices = torch.zeros(batch_size, seq_len, topk_k, dtype=torch.long)
    packed_teacher_topk_indices = torch.zeros(batch_size, seq_len, topk_k, dtype=torch.long)
    gold_position_mask = torch.zeros(batch_size, seq_len)
    gold_teacher_cond_factor = torch.zeros(batch_size, seq_len)
    gold_student_cond_factor = torch.zeros(batch_size, seq_len)

    for b in range(batch_size):
        alignment = alignments[b]
        if alignment.num_chunks == 0:
            continue

        t_prompt_len = teacher_input_lengths[b]
        s_prompt_len = student_prompt_lengths[b]
        t_gen_lps = teacher_gen_logprobs[b]
        t_topk_seq_len = teacher_topk_logits.shape[1]

        for chunk in alignment.chunks:
            if not chunk.teacher_token_indices or not chunk.student_token_indices:
                continue

            t_first_gen_pos = chunk.teacher_token_indices[0]
            t_seq_pos = t_prompt_len + t_first_gen_pos
            if t_seq_pos >= t_topk_seq_len:
                continue

            s_first_gen_pos = chunk.student_token_indices[0]
            s_seq_pos = s_prompt_len + s_first_gen_pos
            if s_seq_pos >= seq_len:
                continue

            # Teacher top-k logits (stay in teacher space, used for teacher probs)
            packed_teacher_topk_logits[b, s_seq_pos, :] = teacher_topk_logits[b, t_seq_pos, :]

            # Original teacher indices (for matched/unmatched classification)
            t_indices = teacher_topk_indices[b, t_seq_pos, :]
            packed_teacher_topk_indices[b, s_seq_pos, :] = t_indices

            # Remap teacher indices to student vocab for gathering student logprobs:
            # matched tokens → student ID; unmatched → keep teacher ID as-is
            remapped = t_indices.clone()
            for j in range(topk_k):
                tid = int(t_indices[j].item())
                if 0 <= tid < max_teacher_id and mapping[tid].item() >= 0:
                    remapped[j] = mapping[tid]
            packed_student_gather_indices[b, s_seq_pos, :] = remapped

            gold_position_mask[b, s_seq_pos] = 1.0

            # Teacher conditional factor
            if len(chunk.teacher_token_indices) > 1:
                t_cond = 0.0
                for t_idx in chunk.teacher_token_indices[1:]:
                    if t_idx < t_gen_lps.shape[0]:
                        t_cond += t_gen_lps[t_idx].item()
                gold_teacher_cond_factor[b, s_seq_pos] = t_cond

            # Student conditional factor
            if len(chunk.student_token_indices) > 1:
                s_cond = 0.0
                for s_idx in chunk.student_token_indices[1:]:
                    lp_idx = s_prompt_len - 1 + s_idx
                    if 0 <= lp_idx < student_prev_logprobs.shape[1]:
                        s_cond += student_prev_logprobs[b, lp_idx].item()
                gold_student_cond_factor[b, s_seq_pos] = s_cond

    return {
        "teacher_topk_logits": packed_teacher_topk_logits,
        # Remapped indices for the DISTILLATION path to gather student logprobs
        "teacher_topk_indices": packed_student_gather_indices,
        # Original teacher indices for matched/unmatched classification
        "gold_teacher_topk_indices_original": packed_teacher_topk_indices,
        "gold_position_mask": gold_position_mask,
        "gold_teacher_cond_factor": gold_teacher_cond_factor,
        "gold_student_cond_factor": gold_student_cond_factor,
    }


# ===============================================================================
# Setup
# ===============================================================================


def setup(
    master_config: GoldDistillMasterConfig,
    student_tokenizer: TokenizerType,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,  # student_policy
    ColocatablePolicyInterface,  # teacher_policy
    Optional[GenerationInterface],  # student_generation
    PreTrainedTokenizerBase,  # teacher_tokenizer
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    GoldTrainLossFn,
    Logger,
    CheckpointManager,
    GoldDistillSaveState,
    GoldDistillMasterConfig,
    VocabMapping,
]:
    """Setup GOLD cross-tokenizer distillation."""
    policy_config = master_config["policy"]
    teacher_config = master_config["teacher"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    distill_config = master_config["gold_distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None

    set_seed(distill_config["seed"])

    # ==========================
    #     Teacher Tokenizer
    # ==========================
    print("\n[GOLD] Loading teacher tokenizer...", flush=True)
    teacher_model_name = teacher_config["model_name"]
    if os.path.isdir(teacher_model_name):
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_name, trust_remote_code=True, local_files_only=True
        )
    else:
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_name, trust_remote_code=True
        )
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if not hasattr(teacher_tokenizer, "chat_template") or teacher_tokenizer.chat_template is None:
        teacher_tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    print(
        f"  Teacher tokenizer: vocab_size={len(teacher_tokenizer)}, "
        f"model={teacher_config['model_name']}",
        flush=True,
    )
    print(
        f"  Student tokenizer: vocab_size={len(student_tokenizer)}, "
        f"model={policy_config['model_name']}",
        flush=True,
    )

    # ==========================
    #     Vocab Mapping
    # ==========================
    print("\n[GOLD] Building vocabulary mapping...", flush=True)
    vocab_mapping = build_vocab_mapping(student_tokenizer, teacher_tokenizer)
    print(
        f"  Matched tokens: {vocab_mapping.num_matched} "
        f"(Jaccard={vocab_mapping.jaccard_index:.3f})",
        flush=True,
    )
    print(
        f"  Student unmatched: {vocab_mapping.student_vocab_size - vocab_mapping.num_matched}, "
        f"Teacher unmatched: {vocab_mapping.teacher_vocab_size - vocab_mapping.num_matched}",
        flush=True,
    )

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    save_state: Optional[GoldDistillSaveState] = cast(
        Optional[GoldDistillSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if save_state is None:
        save_state = _default_save_state()

    # ==========================
    #           Data
    # ==========================
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distill_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
    )

    if last_checkpoint_path:
        dl_state = torch.load(os.path.join(last_checkpoint_path, "train_dataloader.pt"))
        dataloader.load_state_dict(dl_state)

    val_dataloader: Optional[StatefulDataLoader] = None
    if distill_config["val_period"] > 0 or distill_config.get("val_at_start") or distill_config.get("val_at_end"):
        assert val_dataset is not None
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distill_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n[GOLD] Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]

    if colocated_inference:
        cluster = RayVirtualCluster(
            name="gold_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]] * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1 if generation_config["backend"] == "megatron" else 3,
        )
        train_cluster = cluster
        inference_cluster = cluster
    else:
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]
        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        if cluster_config["num_nodes"] == 1:
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            train_nodes -= inference_nodes

        train_cluster = RayVirtualCluster(
            name="gold_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        inference_cluster = RayVirtualCluster(
            name="gold_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=3,
        )

    # ==========================
    #      Teacher Policy
    # ==========================
    print("\n[GOLD] Setting up teacher policy...", flush=True)
    teacher_policy = Policy(
        name_prefix="teacher",
        cluster=train_cluster,
        config=teacher_config,
        tokenizer=teacher_tokenizer,
        weights_path=None,
        optimizer_path=None,
        init_optimizer=False,
        init_reference_model=False,
    )
    teacher_policy.offload_after_refit()

    # ==========================
    #    Student Generation
    # ==========================
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]

    student_generation: Optional[GenerationInterface] = None
    if backend == "megatron":
        student_generation = None
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        if "vllm_cfg" in generation_config:
            generation_config["vllm_cfg"]["hf_overrides"] = policy_config.get("hf_config_overrides", {})
        student_generation = VllmGeneration(cluster=inference_cluster, config=generation_config)
        student_generation.finish_generation()

    # ==========================
    #      Student Policy
    # ==========================
    print("\n[GOLD] Setting up student policy...", flush=True)
    weights_path = None
    optimizer_path = None
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"

    student_policy = Policy(
        name_prefix="student",
        cluster=train_cluster,
        config=policy_config,
        tokenizer=student_tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
    )

    if student_generation is not None:
        state_dict_info = student_policy.prepare_refit_info()
        student_generation.prepare_refit_info(state_dict_info)

    if not colocated_inference and student_generation is not None:
        ip, port = train_cluster.get_master_address_and_port()
        train_world_size = train_cluster.world_size()
        world_size = train_world_size + inference_nodes * inference_gpus_per_node
        futures_train = student_policy.init_collective(ip, port, world_size, train_world_size=train_world_size)
        futures_inference = student_generation.init_collective(ip, port, world_size, train_world_size=train_world_size)
        ray.get(futures_train + futures_inference)

    loss_fn = GoldTrainLossFn(loss_config, vocab_mapping)

    print("\n" + "=" * 60)
    print(" " * 12 + "GOLD DISTILLATION SETUP COMPLETE")
    print("=" * 60 + "\n", flush=True)

    return (
        student_policy,
        teacher_policy,
        student_generation,
        teacher_tokenizer,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        save_state,
        master_config,
        vocab_mapping,
    )


# ===============================================================================
# Validation (identical to algorithm.py)
# ===============================================================================


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: GoldDistillMasterConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation."""
    if val_dataloader is None or val_task_to_env is None:
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        total_rewards = []
        total_lengths = []
        all_message_logs = []

        distill_config = master_config["gold_distillation"]
        max_batches = distill_config["max_val_samples"] // distill_config["val_batch_size"]

        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            if _should_use_async_rollouts(master_config):
                val_batch, gen_metrics = run_async_multi_turn_rollout(
                    policy_generation, val_batch, tokenizer, val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=distill_config["max_rollout_turns"],
                    greedy=False,
                )
            else:
                val_batch, gen_metrics = run_multi_turn_rollout(
                    policy_generation, val_batch, tokenizer, val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=distill_config["max_rollout_turns"],
                    greedy=False,
                )

            total_rewards.extend(val_batch["total_reward"].tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])
            to_env = [
                get_keys_from_message_log(val_batch["message_log"][i], ["role", "content"])
                for i in range(len(val_batch["message_log"]))
            ]
            all_message_logs.extend(to_env)

        accuracy = sum(total_rewards) / len(total_rewards) if total_rewards else 0
        avg_length = sum(total_lengths) / len(total_lengths) if total_lengths else 0

        val_metrics = {"accuracy": accuracy, "avg_length": avg_length}

        try:
            print_message_log_samples(
                all_message_logs, total_rewards,
                num_samples=min(master_config["logger"]["num_val_samples_to_print"], len(all_message_logs)),
                step=step,
            )
        except Exception as e:
            print(f"  Warning: Error displaying message samples: {str(e)}", flush=True)

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    print(f"\n  Validation: accuracy={accuracy:.4f}, avg_length={avg_length:.1f}", flush=True)
    timer.reset()
    return val_metrics, timing_metrics


# ===============================================================================
# Training
# ===============================================================================


def gold_distillation_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    student_generation: Optional[GenerationInterface],
    student_tokenizer: TokenizerType,
    teacher_tokenizer: PreTrainedTokenizerBase,
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    loss_fn: GoldTrainLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    save_state: GoldDistillSaveState,
    master_config: GoldDistillMasterConfig,
    vocab_mapping: VocabMapping,
) -> None:
    """Run GOLD cross-tokenizer distillation training.

    Pipeline per step:
    1. Student generates text on-policy
    2. Text is decoded + cross-tokenizer alignment computed
    3. Teacher top-k logits are computed on re-tokenized text
    4. Alignment + teacher top-k are packed into train_data
    5. Policy.train() with GoldTrainLossFn (LossInputType.LOGIT)
    """
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    if student_generation is None:
        student_generation = student_policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True
    assert student_generation is not None

    current_epoch = save_state["current_epoch"]
    current_step = save_state["current_step"]
    total_steps = save_state["total_steps"]
    consumed_samples = save_state["consumed_samples"]
    total_valid_tokens = save_state["total_valid_tokens"]
    distill_config = master_config["gold_distillation"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    max_epochs = distill_config["max_num_epochs"]
    max_steps = distill_config["max_num_steps"]
    max_seq_len = master_config["policy"]["max_total_sequence_length"]
    val_period = distill_config["val_period"]
    val_at_start = distill_config.get("val_at_start", False)
    val_at_end = distill_config.get("val_at_end", False)
    teacher_topk_k = distill_config.get("teacher_topk_k", 1024)

    # Validation at start
    if val_at_start and total_steps == 0:
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(student_policy, student_generation, colocated_inference)
            POLICY_GENERATION_STALE = False
        else:
            student_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            student_generation, val_dataloader, student_tokenizer,
            val_task_to_env, step=total_steps, master_config=master_config,
        )
        student_generation.finish_generation()
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    batch: BatchedDataDict[DatumSpec]

    while total_steps < max_steps and current_epoch < max_epochs:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}", flush=True)

        for batch in dataloader:
            print(f"\n{'=' * 25} [GOLD] Step {current_step + 1} (total: {total_steps + 1}) {'=' * 25}", flush=True)
            maybe_gpu_profile_step(student_policy, total_steps + 1)

            with timer.time("total_step_time"):
                # ---- 1) Prepare batch ----
                with timer.time("data_processing"):
                    repeated_batch = batch.repeat_interleave(distill_config["num_generations_per_prompt"])

                # ---- 2) Student on-policy generation ----
                print("[GOLD] Generating student responses...", flush=True)
                with timer.time("prepare_for_generation"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(student_policy, student_generation, colocated_inference, timer=timer)
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()

                with timer.time("generation"):
                    if _should_use_async_rollouts(master_config):
                        repeated_batch, rollout_metrics = run_async_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=student_tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=max_seq_len,
                            max_rollout_turns=distill_config["max_rollout_turns"],
                            greedy=False,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=student_tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=max_seq_len,
                            max_rollout_turns=distill_config["max_rollout_turns"],
                            greedy=False,
                        )
                    student_generation.finish_generation()

                # ---- 3) Flatten student output ----
                with timer.time("data_processing"):
                    distill_message_logs = [
                        _truncate_message_log_at_first_assistant(message_log)
                        for message_log in repeated_batch["message_log"]
                    ]

                    for message_log in distill_message_logs:
                        for message in message_log:
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(message["token_ids"])
                            else:
                                message["token_loss_mask"] = torch.zeros_like(message["token_ids"])

                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        distill_message_logs,
                        pad_value_dict={"token_ids": student_tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"]["make_sequence_length_divisible_by"],
                    )

                    student_input_ids = flat_messages["token_ids"]
                    student_token_mask = flat_messages["token_loss_mask"]

                # ---- 4) Text-space alignment ----
                print("[GOLD] Computing cross-tokenizer alignment...", flush=True)
                with timer.time("alignment"):
                    prompt_lengths = []
                    for msg_log in distill_message_logs:
                        plen = 0
                        for msg in msg_log:
                            if msg["role"] != "assistant":
                                plen += msg["token_ids"].shape[0]
                            else:
                                break
                        prompt_lengths.append(plen)
                    prompt_lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.long)

                    decoded_texts, alignments, alignment_path_stats = decode_and_align(
                        generated_ids=student_input_ids,
                        input_lengths=prompt_lengths_tensor,
                        student_tokenizer=student_tokenizer,
                        teacher_tokenizer=teacher_tokenizer,
                    )

                    n_nonempty = sum(1 for t in decoded_texts if t.strip())
                    n_chunks_total = sum(a.num_chunks for a in alignments)
                    print(f"  Decoded {len(decoded_texts)} texts, {n_nonempty} non-empty, {n_chunks_total} total groups", flush=True)

                # ---- 5) Build teacher input ----
                print("[GOLD] Computing teacher top-k logits...", flush=True)
                with timer.time("teacher_data_prep"):
                    teacher_all_ids = []
                    teacher_input_lengths_list = []
                    teacher_sequence_lengths = []
                    teacher_token_masks = []

                    for idx, (text, alignment) in enumerate(zip(decoded_texts, alignments)):
                        if not text or alignment.num_chunks == 0:
                            t_ids = torch.tensor([teacher_tokenizer.bos_token_id or 0], dtype=torch.long)
                            teacher_all_ids.append(t_ids)
                            teacher_input_lengths_list.append(1)
                            teacher_sequence_lengths.append(1)
                            teacher_token_masks.append(torch.zeros(1, dtype=torch.long))
                            continue

                        prompt_msgs = [m for m in distill_message_logs[idx] if m["role"] != "assistant"]

                        if hasattr(teacher_tokenizer, "apply_chat_template"):
                            messages = [{"role": msg["role"], "content": msg["content"]} for msg in prompt_msgs]
                            messages.append({"role": "assistant", "content": text})
                            full_ids = teacher_tokenizer.apply_chat_template(
                                messages, add_generation_prompt=False, return_tensors="pt"
                            ).squeeze(0)
                            prompt_only = messages[:-1]
                            prompt_ids = teacher_tokenizer.apply_chat_template(
                                prompt_only, add_generation_prompt=True, return_tensors="pt"
                            ).squeeze(0)
                            prompt_len = prompt_ids.shape[0]
                        else:
                            prompt_text = "".join(m.get("content", "") for m in prompt_msgs)
                            prompt_enc = teacher_tokenizer(prompt_text, add_special_tokens=True)
                            gen_enc = teacher_tokenizer(text, add_special_tokens=False)
                            full_ids = torch.tensor(prompt_enc["input_ids"] + gen_enc["input_ids"])
                            prompt_len = len(prompt_enc["input_ids"])

                        if full_ids.shape[0] > max_seq_len:
                            full_ids = full_ids[:max_seq_len]

                        t_mask = torch.zeros(full_ids.shape[0], dtype=torch.long)
                        t_mask[prompt_len:] = 1

                        teacher_all_ids.append(full_ids)
                        teacher_input_lengths_list.append(prompt_len)
                        teacher_sequence_lengths.append(full_ids.shape[0])
                        teacher_token_masks.append(t_mask)

                        # Fix teacher alignment indices (same offset logic as algorithm.py)
                        teacher_standalone_ids = teacher_tokenizer(text, add_special_tokens=False)["input_ids"]
                        teacher_ic_ids = full_ids[prompt_len:].tolist()
                        t_offset = 0
                        if teacher_standalone_ids:
                            match_len = min(5, len(teacher_standalone_ids))
                            for o in range(len(teacher_ic_ids) - match_len + 1):
                                if teacher_ic_ids[o:o + match_len] == teacher_standalone_ids[:match_len]:
                                    t_offset = o
                                    break
                        if t_offset > 0:
                            for chunk in alignment.chunks:
                                chunk.teacher_token_indices = [i + t_offset for i in chunk.teacher_token_indices]

                    # Pad teacher inputs
                    t_max_len = max(ids.shape[0] for ids in teacher_all_ids) if teacher_all_ids else 1
                    t_pad_id = teacher_tokenizer.pad_token_id or 0
                    t_padded_ids = torch.full((len(teacher_all_ids), t_max_len), t_pad_id, dtype=torch.long)
                    t_padded_masks = torch.zeros(len(teacher_all_ids), t_max_len, dtype=torch.long)

                    for i, (ids, mask) in enumerate(zip(teacher_all_ids, teacher_token_masks)):
                        t_padded_ids[i, :ids.shape[0]] = ids
                        t_padded_masks[i, :mask.shape[0]] = mask

                    teacher_train_data = BatchedDataDict({
                        "input_ids": t_padded_ids,
                        "input_lengths": torch.tensor(teacher_sequence_lengths, dtype=torch.long),
                        "token_mask": t_padded_masks,
                        "sample_mask": torch.ones(len(teacher_all_ids), dtype=torch.float32),
                    })
                    teacher_train_data.to("cpu")

                # ---- 6) Get teacher top-k logits ----
                with timer.time("teacher_topk_inference_prep"):
                    teacher_policy.prepare_for_lp_inference()

                with timer.time("teacher_topk_inference"):
                    teacher_topk = teacher_policy.get_topk_logits(
                        teacher_train_data,
                        k=teacher_topk_k,
                        timer=timer,
                    )
                    t_topk_logits = teacher_topk["topk_logits"]   # [B, S_teacher, k]
                    t_topk_indices = teacher_topk["topk_indices"]  # [B, S_teacher, k]

                # We also need teacher per-token logprobs for conditional factors
                with timer.time("teacher_logprob_inference"):
                    teacher_logprob_result = teacher_policy.get_logprobs(teacher_train_data, timer=timer)
                    teacher_token_logprobs = teacher_logprob_result["logprobs"]

                # Extract teacher generation logprobs
                teacher_gen_logprobs = []
                for i in range(teacher_token_logprobs.shape[0]):
                    t_prompt_len = teacher_input_lengths_list[i]
                    t_seq_len = teacher_sequence_lengths[i]
                    t_gen_lps = _extract_logprob_span_for_token_range(
                        token_logprobs=teacher_token_logprobs[i],
                        token_seq_len=t_seq_len,
                        start_token_pos=t_prompt_len,
                        end_token_pos_exclusive=t_seq_len,
                    )
                    teacher_gen_logprobs.append(t_gen_lps)

                # ---- 7) Student prev_logprobs for conditional factors ----
                print("[GOLD] Computing student prev logprobs...", flush=True)
                with timer.time("student_logprob_inference"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_lp_inference()
                    logprob_data = BatchedDataDict({
                        "input_ids": student_input_ids,
                        "input_lengths": input_lengths,
                        "token_mask": student_token_mask,
                        "sample_mask": repeated_batch["loss_multiplier"],
                    })
                    logprob_data.to("cpu")
                    prev_logprobs = student_policy.get_logprobs(logprob_data, timer=timer)["logprobs"]

                # ---- 8) Pack GOLD alignment data ----
                print("[GOLD] Packing alignment data...", flush=True)
                with timer.time("gold_packing"):
                    alignment_data = pack_gold_alignment_into_data(
                        alignments=alignments,
                        teacher_topk_logits=t_topk_logits,
                        teacher_topk_indices=t_topk_indices,
                        teacher_gen_logprobs=teacher_gen_logprobs,
                        student_prev_logprobs=prev_logprobs,
                        teacher_input_lengths=teacher_input_lengths_list,
                        student_prompt_lengths=prompt_lengths,
                        seq_len=student_input_ids.shape[1],
                        topk_k=teacher_topk_k,
                        vocab_mapping=vocab_mapping,
                    )

                # ---- 9) Train student ----
                print("[GOLD] Training student policy...", flush=True)
                with timer.time("training_prep"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_training()
                    POLICY_GENERATION_STALE = True

                train_data = BatchedDataDict({
                    "input_ids": student_input_ids,
                    "input_lengths": input_lengths,
                    "token_mask": student_token_mask,
                    "sample_mask": repeated_batch["loss_multiplier"],
                    # Standard keys for DISTILLATION path (remapped to student vocab)
                    "teacher_topk_logits": alignment_data["teacher_topk_logits"],
                    "teacher_topk_indices": alignment_data["teacher_topk_indices"],
                    # GOLD-specific: original teacher indices for matched/unmatched
                    "gold_teacher_topk_indices_original": alignment_data["gold_teacher_topk_indices_original"],
                    "gold_position_mask": alignment_data["gold_position_mask"],
                    "gold_teacher_cond_factor": alignment_data["gold_teacher_cond_factor"],
                    "gold_student_cond_factor": alignment_data["gold_student_cond_factor"],
                })
                train_data.update(flat_messages.get_multimodal_dict(as_tensors=False))
                train_data.to("cpu")

                with timer.time("policy_training"):
                    train_results = student_policy.train(train_data, loss_fn, timer=timer)

                # ---- 10) Metrics ----
                mb_metrics = train_results.get("all_mb_metrics", {})
                global_loss = _mean_metric_value(train_results["loss"])
                logged_loss = _mean_metric_value(mb_metrics.get("loss", global_loss))
                metrics = {
                    "loss": logged_loss,
                    "global_loss": global_loss,
                    "grad_norm": _mean_metric_value(train_results["grad_norm"]),
                }
                for k, v in mb_metrics.items():
                    if k not in metrics:
                        metrics[k] = v
                metrics.update(rollout_metrics)

                total_valid_tokens += int(alignment_data["gold_position_mask"].sum().item())

                # Checkpointing
                consumed_samples += distill_config["num_prompts_per_step"]
                timeout.mark_iteration()

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                should_save = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"] == 0
                    or timeout.check_save()
                )

                should_validate = (
                    (val_period > 0 and (total_steps + 1) % val_period == 0)
                    or (val_at_end and is_last_step)
                )
                if should_validate:
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(student_policy, student_generation, colocated_inference)
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        student_generation, val_dataloader, student_tokenizer,
                        val_task_to_env, step=total_steps + 1, master_config=master_config,
                    )
                    student_generation.finish_generation()
                    POLICY_GENERATION_STALE = True
                    logger.log_metrics(val_metrics, total_steps + 1, prefix="validation")
                    logger.log_metrics(validation_timings, total_steps + 1, prefix="timing/validation")

                if master_config["checkpointing"]["enabled"] and should_save:
                    student_policy.prepare_for_training()
                    save_state.update({
                        "current_epoch": current_epoch,
                        "current_step": current_step + 1,
                        "total_steps": total_steps + 1,
                        "total_valid_tokens": total_valid_tokens,
                        "consumed_samples": consumed_samples,
                    })
                    with timer.time("checkpointing"):
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, save_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                            optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer"),
                            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            timing_metrics = timer.get_timing_metrics(reduction_op="sum")
            total_time = timing_metrics.get("total_step_time", 0)

            train_loss = metrics.get("loss", float("nan"))
            matched_jsd = _mean_metric_value(metrics.get("matched_jsd_loss", 0))
            unmatched_l1 = _mean_metric_value(metrics.get("unmatched_l1_loss", 0))
            n_groups = _mean_metric_value(metrics.get("num_groups", 0))

            print(f"\n  [GOLD] Step {total_steps + 1} Results:")
            print(f"    Loss: {train_loss:.4f}")
            print(f"    Matched JSD: {matched_jsd:.4f}  |  Unmatched L1: {unmatched_l1:.4f}")
            print(f"    Groups: {int(n_groups)}  |  Mean gen length: {rollout_metrics.get('mean_gen_tokens_per_sample', 0):.1f}")
            print(f"    Time: {total_time:.2f}s", flush=True)

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1

            if total_steps >= max_steps:
                print("[GOLD] Max steps reached.", flush=True)
                return

        current_epoch += 1
        current_step = 0

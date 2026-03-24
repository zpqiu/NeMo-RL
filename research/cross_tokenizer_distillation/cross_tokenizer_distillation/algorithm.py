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

"""Cross-tokenizer on-policy distillation algorithm.

Based on ``nemo_rl.algorithms.distillation`` but removes the shared-tokenizer
requirement and introduces text-space alignment between teacher and student.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

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

from cross_tokenizer_distillation.cross_tokenizer_loss import (
    CrossTokenizerDistillationLossConfig,
    CrossTokenizerTrainLossFn,
)
from cross_tokenizer_distillation.token_alignment import (
    AlignmentResult,
    align_tokens_by_byte_offset,
    compute_chunk_logprobs,
)

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class CrossDistillationConfig(TypedDict):
    """Cross-tokenizer distillation specific configuration."""

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


class CrossDistillSaveState(TypedDict):
    total_steps: int
    current_epoch: int
    current_step: int
    val_reward: NotRequired[float]
    consumed_samples: int
    total_valid_tokens: int


def _default_save_state() -> CrossDistillSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


class CrossDistillMasterConfig(TypedDict):
    """Main configuration for cross-tokenizer distillation."""

    policy: PolicyConfig  # Student model configuration
    teacher: PolicyConfig  # Teacher model configuration (different tokenizer!)
    loss_fn: CrossTokenizerDistillationLossConfig
    env: dict[str, Any]
    data: DataConfig
    cross_distillation: CrossDistillationConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Text-space alignment utilities
# ===============================================================================


def decode_and_align(
    generated_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[str], list[AlignmentResult]]:
    """Decode student-generated text and compute cross-tokenizer alignment.

    Args:
        generated_ids: (batch, seq_len) — full sequence (prompt + generation).
        input_lengths: (batch,) — length of prompt for each sample.
        student_tokenizer: Student's tokenizer.
        teacher_tokenizer: Teacher's tokenizer.

    Returns:
        (decoded_texts, alignments) — one AlignmentResult per sample.
    """
    batch_size = generated_ids.shape[0]
    decoded_texts: list[str] = []
    alignments: list[AlignmentResult] = []

    for i in range(batch_size):
        prompt_len = int(input_lengths[i].item())
        gen_ids = generated_ids[i, prompt_len:]

        # Strip padding
        if student_tokenizer.pad_token_id is not None:
            mask = gen_ids != student_tokenizer.pad_token_id
            gen_ids = gen_ids[mask]

        if gen_ids.numel() == 0:
            decoded_texts.append("")
            alignments.append(
                AlignmentResult(text="", teacher_token_ids=[], student_token_ids=[])
            )
            continue

        text = student_tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
        decoded_texts.append(text)

        alignment = align_tokens_by_byte_offset(text, teacher_tokenizer, student_tokenizer)
        alignments.append(alignment)

    return decoded_texts, alignments


def pack_alignment_into_data(
    alignments: list[AlignmentResult],
    teacher_gen_logprobs: list[torch.Tensor],
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Pack alignment info + teacher chunk logprobs into tensors for Policy.train().

    All tensors are padded to (B, seq_len) to pass check_sequence_dim
    and shard_by_batch_size validation.

    Args:
        alignments: List of AlignmentResult, one per sample.
        teacher_gen_logprobs: List of teacher per-token logprobs for generated tokens.
        seq_len: Sequence length to pad to (matches input_ids shape[1]).

    Returns:
        Dict of tensors to merge into train_data.
    """
    batch_size = len(alignments)

    # Compute teacher chunk logprobs and collect student indices
    all_teacher_chunk_lps: list[torch.Tensor] = []
    all_student_indices: list[list[list[int]]] = []

    for i, (alignment, t_gen_lps) in enumerate(zip(alignments, teacher_gen_logprobs)):
        if alignment.num_chunks == 0:
            all_teacher_chunk_lps.append(torch.zeros(0))
            all_student_indices.append([])
            continue

        t_chunk_lps = compute_chunk_logprobs(t_gen_lps, alignment.chunks, "teacher")
        all_teacher_chunk_lps.append(t_chunk_lps)

        sample_indices = []
        for chunk in alignment.chunks:
            sample_indices.append(chunk.student_token_indices)
        all_student_indices.append(sample_indices)

    # Max chunks and max tokens per chunk
    max_chunks = max((lps.numel() for lps in all_teacher_chunk_lps), default=0)
    max_toks = 1
    for sample_idx in all_student_indices:
        for chunk_idx in sample_idx:
            max_toks = max(max_toks, len(chunk_idx))

    if max_chunks == 0:
        max_chunks = 1

    # Pad to (B, seq_len) — alignment data goes in first max_chunks positions
    # Using seq_len ensures check_sequence_dim passes
    pad_dim = seq_len  # Match sequence dimension

    teacher_chunk_logprobs = torch.zeros(batch_size, pad_dim)
    chunk_mask = torch.zeros(batch_size, pad_dim)
    num_student_toks = torch.zeros(batch_size, pad_dim, dtype=torch.long)
    # chunk_student_indices needs to be (B, pad_dim, max_toks) but that would
    # fail check_sequence_dim. Instead, encode as (B, pad_dim) with a packed format.
    # We'll store the first student token index for each chunk in a (B, pad_dim) tensor
    # and the count in num_student_toks. For multi-token chunks, we store the start
    # index and assume consecutive tokens (which is true by alignment construction).
    chunk_student_start = torch.zeros(batch_size, pad_dim, dtype=torch.long)

    for i in range(batch_size):
        n_chunks = all_teacher_chunk_lps[i].numel()
        if n_chunks == 0:
            continue
        teacher_chunk_logprobs[i, :n_chunks] = all_teacher_chunk_lps[i]
        chunk_mask[i, :n_chunks] = 1.0
        for c in range(n_chunks):
            s_idx = all_student_indices[i][c]
            n_t = len(s_idx)
            num_student_toks[i, c] = n_t
            if n_t > 0:
                chunk_student_start[i, c] = s_idx[0]

    return {
        "xalign_teacher_chunk_logprobs": teacher_chunk_logprobs,  # (B, S)
        "xalign_chunk_student_start": chunk_student_start,  # (B, S)
        "xalign_chunk_mask": chunk_mask,  # (B, S)
        "xalign_num_student_toks": num_student_toks,  # (B, S)
    }


# ===============================================================================
# Setup
# ===============================================================================


def setup(
    master_config: CrossDistillMasterConfig,
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
    CrossTokenizerTrainLossFn,
    Logger,
    CheckpointManager,
    CrossDistillSaveState,
    CrossDistillMasterConfig,
]:
    """Setup cross-tokenizer distillation."""
    policy_config = master_config["policy"]
    teacher_config = master_config["teacher"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    distill_config = master_config["cross_distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None

    set_seed(distill_config["seed"])

    # ==========================
    #     Teacher Tokenizer
    # ==========================
    print("\n▶ Loading teacher tokenizer (different from student!)...", flush=True)
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
    # Ensure teacher tokenizer has a chat template (base models may not)
    if not hasattr(teacher_tokenizer, "chat_template") or teacher_tokenizer.chat_template is None:
        teacher_tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    print(
        f"  ✓ Teacher tokenizer: vocab_size={len(teacher_tokenizer)}, "
        f"model={teacher_config['model_name']}",
        flush=True,
    )
    print(
        f"  ✓ Student tokenizer: vocab_size={len(student_tokenizer)}, "
        f"model={policy_config['model_name']}",
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
    save_state: Optional[CrossDistillSaveState] = cast(
        Optional[CrossDistillSaveState],
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

    print(f"  ✓ Training dataloader: {len(train_dataset)} samples", flush=True)

    val_dataloader: Optional[StatefulDataLoader] = None
    if distill_config["val_period"] > 0 or distill_config.get("val_at_start") or distill_config.get("val_at_end"):
        assert val_dataset is not None
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distill_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )
        print(f"  ✓ Validation dataloader: {len(val_dataset)} samples", flush=True)

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]

    if colocated_inference:
        cluster = RayVirtualCluster(
            name="xdistill_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
            * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1
            if generation_config["backend"] == "megatron"
            else 3,
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
            name="xdistill_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        inference_cluster = RayVirtualCluster(
            name="xdistill_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=3,
        )

    # ==========================
    #      Teacher Policy
    # ==========================
    print("\n▶ Setting up teacher policy...", flush=True)
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
            generation_config["vllm_cfg"]["hf_overrides"] = policy_config.get(
                "hf_config_overrides", {}
            )
        student_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        student_generation.finish_generation()

    # ==========================
    #      Student Policy
    # ==========================
    print("\n▶ Setting up student policy...", flush=True)
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

    loss_fn = CrossTokenizerTrainLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 12 + "CROSS-TOKENIZER DISTILLATION SETUP COMPLETE")
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
    )


# ===============================================================================
# Training
# ===============================================================================


def cross_tokenizer_distillation_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    student_generation: Optional[GenerationInterface],
    student_tokenizer: TokenizerType,
    teacher_tokenizer: PreTrainedTokenizerBase,
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    loss_fn: CrossTokenizerTrainLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    save_state: CrossDistillSaveState,
    master_config: CrossDistillMasterConfig,
) -> None:
    """Run cross-tokenizer distillation training.

    The key difference from standard distillation:
    1. Student generates text on-policy
    2. Text is decoded back to string
    3. String is re-tokenized with teacher tokenizer
    4. Byte-offset alignment creates chunks
    5. Teacher logprobs are computed on re-tokenized text
    6. Teacher chunk logprobs + alignment info are packed into train_data
    7. Policy.train() does the student forward + backward + optimizer step
       using CrossTokenizerTrainLossFn which aggregates student logprobs
       into chunks and computes chunk-level KL
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
    distill_config = master_config["cross_distillation"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    max_epochs = distill_config["max_num_epochs"]
    max_steps = distill_config["max_num_steps"]
    max_seq_len = master_config["policy"]["max_total_sequence_length"]

    batch: BatchedDataDict[DatumSpec]

    while total_steps < max_steps and current_epoch < max_epochs:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}", flush=True)

        for batch in dataloader:
            print(f"\n{'=' * 25} Step {current_step + 1} (total: {total_steps + 1}) {'=' * 25}", flush=True)
            maybe_gpu_profile_step(student_policy, total_steps + 1)

            with timer.time("total_step_time"):
                # ---- 1) Prepare batch ----
                with timer.time("data_processing"):
                    repeated_batch = batch.repeat_interleave(
                        distill_config["num_generations_per_prompt"]
                    )

                # ---- 2) Student on-policy generation ----
                print("▶ Generating student responses...", flush=True)
                with timer.time("prepare_for_generation"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            student_policy, student_generation, colocated_inference, timer=timer,
                        )
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
                    for message_log in repeated_batch["message_log"]:
                        for message in message_log:
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(message["token_ids"])
                            else:
                                message["token_loss_mask"] = torch.zeros_like(message["token_ids"])

                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": student_tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"]["make_sequence_length_divisible_by"],
                    )

                    student_input_ids = flat_messages["token_ids"]  # (B, S)

                # ---- 4) Text-space alignment ----
                print("▶ Decoding text & computing cross-tokenizer alignment...", flush=True)
                with timer.time("alignment"):
                    decoded_texts, alignments = decode_and_align(
                        generated_ids=student_input_ids,
                        input_lengths=input_lengths,
                        student_tokenizer=student_tokenizer,
                        teacher_tokenizer=teacher_tokenizer,
                    )

                # ---- 5) Get teacher logprobs ----
                # Build teacher input: re-tokenize the student-generated text
                # with the teacher tokenizer, keeping the same prompts.
                print("▶ Computing teacher logprobs...", flush=True)

                with timer.time("teacher_data_prep"):
                    # Build teacher input by re-tokenizing decoded text
                    teacher_all_ids = []
                    teacher_input_lengths = []
                    teacher_token_masks = []

                    for idx, (text, alignment) in enumerate(zip(decoded_texts, alignments)):
                        if not text or alignment.num_chunks == 0:
                            # Minimal dummy input for empty generations
                            t_ids = torch.tensor([teacher_tokenizer.bos_token_id or 0], dtype=torch.long)
                            teacher_all_ids.append(t_ids)
                            teacher_input_lengths.append(1)
                            teacher_token_masks.append(torch.zeros(1, dtype=torch.long))
                            continue

                        # Get prompt messages
                        prompt_msgs = [m for m in repeated_batch["message_log"][idx] if m["role"] != "assistant"]

                        # Build teacher input with chat template
                        if hasattr(teacher_tokenizer, "apply_chat_template"):
                            messages = []
                            for msg in prompt_msgs:
                                messages.append({"role": msg["role"], "content": msg["content"]})
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
                        teacher_input_lengths.append(prompt_len)
                        teacher_token_masks.append(t_mask)

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
                        "input_lengths": torch.tensor(teacher_input_lengths, dtype=torch.long),
                        "token_mask": t_padded_masks,
                        "sample_mask": torch.ones(len(teacher_all_ids), dtype=torch.float32),
                    })
                    teacher_train_data.to("cpu")

                with timer.time("teacher_logprob_inference_prep"):
                    teacher_policy.prepare_for_lp_inference()

                with timer.time("teacher_logprob_inference"):
                    teacher_logprob_result = teacher_policy.get_logprobs(teacher_train_data, timer=timer)
                    teacher_token_logprobs = teacher_logprob_result["logprobs"]  # (B, S-1)

                # ---- 6) Pack alignment + teacher chunk logprobs into train_data ----
                print("▶ Packing alignment data...", flush=True)
                with timer.time("alignment_packing"):
                    # Extract teacher generation logprobs
                    teacher_gen_logprobs = []
                    for i in range(teacher_token_logprobs.shape[0]):
                        t_prompt_len = teacher_input_lengths[i]
                        # logprobs[t] = logprob of token[t+1] given [0..t]
                        # For generation starting at position t_prompt_len,
                        # the first gen token's logprob is at index t_prompt_len - 1
                        t_gen_lps = teacher_token_logprobs[i, t_prompt_len - 1:]
                        teacher_gen_logprobs.append(t_gen_lps)

                    alignment_data = pack_alignment_into_data(
                        alignments, teacher_gen_logprobs,
                        seq_len=student_input_ids.shape[1],
                    )

                # ---- 7) Build student train_data and call Policy.train() ----
                print("▶ Training student policy...", flush=True)
                with timer.time("training_prep"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_training()
                    POLICY_GENERATION_STALE = True

                train_data = BatchedDataDict({
                    "input_ids": student_input_ids,
                    "input_lengths": input_lengths,
                    "token_mask": flat_messages["token_loss_mask"],
                    "sample_mask": repeated_batch["loss_multiplier"],
                    # Cross-tokenizer alignment data (padded to seq_len for compatibility)
                    "xalign_teacher_chunk_logprobs": alignment_data["xalign_teacher_chunk_logprobs"],
                    "xalign_chunk_student_start": alignment_data["xalign_chunk_student_start"],
                    "xalign_chunk_mask": alignment_data["xalign_chunk_mask"],
                    "xalign_num_student_toks": alignment_data["xalign_num_student_toks"],
                })
                train_data.update(flat_messages.get_multimodal_dict(as_tensors=False))
                train_data.to("cpu")

                with timer.time("policy_training"):
                    train_results = student_policy.train(
                        train_data,
                        loss_fn,
                        timer=timer,
                    )

                # ---- 8) Metrics ----
                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                }
                for k, v in metrics.items():
                    if isinstance(v, np.ndarray):
                        metrics[k] = np.sum(v).item()
                metrics.update(train_results.get("all_mb_metrics", {}))
                metrics.update(rollout_metrics)

                total_valid_tokens += int(alignment_data["chunk_mask"].sum().item())

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
            print(f"\n📊 Step {total_steps + 1} Results:")
            print(f"  • Loss (chunk KL): {train_loss:.4f}")
            print(f"  • Chunks: {int(alignment_data['chunk_mask'].sum().item())}")
            print(f"  • Mean gen length: {rollout_metrics.get('mean_gen_tokens_per_sample', 0):.1f}")
            print(f"\n⏱️  Timing: {total_time:.2f}s total", flush=True)

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1

            if total_steps >= max_steps:
                print("Max steps reached.", flush=True)
                return

        current_epoch += 1
        current_step = 0

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

"""On-Policy Distillation via single-token reverse KL (Thinking Machines Lab approach).

Reference: https://thinkingmachines.ai/blog/on-policy-distillation/

Algorithm:
    1. Student generates responses on-policy
    2. Teacher provides per-token logprobs on student-generated sequences
    3. Per-token advantage = log p_teacher(y_t) - log q_student(y_t) (= negative reverse KL)
    4. Student is updated via importance-sampling policy gradient (no clipping by default)

This reuses the distillation infrastructure (setup, generation, teacher/student policies)
but replaces the distributional KL loss with a policy gradient loss driven by single-token
teacher feedback.
"""

import os
import warnings

import numpy as np
import torch

from nemo_rl.algorithms.distillation import (
    MasterConfig,
    _should_use_async_rollouts,
    refit_policy_generation,
)
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn, ClippedPGLossConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.rollout import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation import GenerationInterface
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.types import (
    DatumSpec,
    LLMMessageLogType,
    TokenizerType,
)
from nemo_rl.utils.batched_message_log_to_flat_message import (
    batched_message_log_to_flat_message,
)
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import Timer, TimeoutChecker
from nemo_rl.utils.gpu_profiling import maybe_gpu_profile_step
from torch.utils.data import DataLoader as StatefulDataLoader


# ===============================================================================
# Loss Function Configuration
# ===============================================================================


def create_opd_loss_fn(master_config: MasterConfig) -> ClippedPGLossFn:
    """Create ClippedPGLossFn configured for single-token OPD.

    Following the Thinking Machines Lab approach:
    - IS policy gradient (importance sampling ratio, no clipping by default)
    - No KL regularization to a reference policy
    - Token-level loss reduction
    - No advantage normalization (raw per-token KL as advantage)
    """
    opd_config = master_config.get("opd", {})

    cfg: ClippedPGLossConfig = {
        # No KL penalty to reference policy
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": None,
        "kl_output_clamp_value": None,
        # PPO ratio clipping — disabled by default for vanilla IS policy gradient.
        # Set ratio_clip_min/max in config to enable PPO-style clipping.
        "ratio_clip_min": opd_config.get("ratio_clip_min", 1e6),
        "ratio_clip_max": opd_config.get("ratio_clip_max", 1e6),
        "ratio_clip_c": None,
        # No on-policy KL approximation or IS correction (single epoch per rollout)
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        # Token-level loss (γ=0 in the TM formulation)
        "token_level_loss": True,
        # Use IS ratio (π_new / π_old), not REINFORCE
        "disable_ppo_ratio": False,
        "force_on_policy_ratio": False,
    }
    return ClippedPGLossFn(cfg)


# ===============================================================================
# Training
# ===============================================================================


def opd_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    student_generation: GenerationInterface | None,
    dataloader: StatefulDataLoader,
    val_dataloader: StatefulDataLoader | None,
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: dict[str, EnvironmentInterface] | None,
    logger: Logger,
    checkpointer: CheckpointManager,
    save_state: dict,
    master_config: MasterConfig,
) -> None:
    """Run on-policy distillation via single-token reverse KL policy gradient.

    This follows the Thinking Machines Lab approach:
    - Generate responses from the student (on-policy)
    - Compute teacher logprobs on student-generated sequences
    - Per-token advantage = log p_teacher(y_t) - log q_student(y_t)
    - Update student via importance-sampling policy gradient
    """
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    # Create the policy gradient loss function
    loss_fn = create_opd_loss_fn(master_config)

    NEED_REFIT = True
    if student_generation is None:
        student_generation = student_policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True
    assert student_generation is not None

    # Training state
    current_epoch = save_state["current_epoch"]
    current_step = save_state["current_step"]
    total_steps = save_state["total_steps"]
    consumed_samples = save_state["consumed_samples"]
    total_valid_tokens = save_state["total_valid_tokens"]

    distillation_config = master_config["distillation"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    max_epochs = distillation_config["max_num_epochs"]
    max_steps = distillation_config["max_num_steps"]

    batch: BatchedDataDict[DatumSpec]

    while total_steps < max_steps and current_epoch < max_epochs:
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}",
            flush=True,
        )

        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(student_policy, total_steps + 1)
            if student_policy != student_generation:
                maybe_gpu_profile_step(student_generation, total_steps + 1)

            with timer.time("total_step_time"):
                # ── 1. Prepare batch ──────────────────────────────────────
                print(">> Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            distillation_config["num_generations_per_prompt"]
                        )
                    )

                # ── 2. Generate responses from student (on-policy) ────────
                print(
                    f">> Generating responses (batch size {repeated_batch.size})...",
                    flush=True,
                )
                with timer.time("prepare_for_generation"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            student_policy,
                            student_generation,
                            colocated_inference,
                            timer=timer,
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()

                with timer.time("generation"):
                    if _should_use_async_rollouts(master_config):
                        repeated_batch, rollout_metrics = run_async_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=distillation_config["max_rollout_turns"],
                            greedy=False,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=distillation_config["max_rollout_turns"],
                            greedy=False,
                        )
                    student_generation.finish_generation()

                # ── 3. Flatten messages and extract generation logprobs ────
                with timer.time("data_processing"):
                    # Mark assistant tokens for loss computation
                    for message_log in repeated_batch["message_log"]:
                        for message in message_log:
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(
                                    message["token_ids"]
                                )
                            else:
                                message["token_loss_mask"] = torch.zeros_like(
                                    message["token_ids"]
                                )

                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Build base training data with generation logprobs
                    train_data = BatchedDataDict(
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    train_data.update(
                        flat_messages.get_multimodal_dict(as_tensors=False)
                    )
                    train_data.to("cpu")

                # ── 4. Compute student prev_logprobs (fresh forward pass) ─
                print(">> Computing student logprobs...", flush=True)
                with timer.time("student_logprob_inference"):
                    student_policy.prepare_for_lp_inference()
                    logprob_data = BatchedDataDict(
                        {
                            "input_ids": train_data["input_ids"],
                            "input_lengths": train_data["input_lengths"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    student_prev_logprobs = student_policy.get_logprobs(
                        logprob_data, timer=timer
                    )["logprobs"]

                # ── 5. Compute teacher logprobs ───────────────────────────
                print(">> Computing teacher logprobs...", flush=True)
                with timer.time("teacher_logprob_inference_prep"):
                    teacher_policy.prepare_for_lp_inference()

                with timer.time("teacher_logprob_inference"):
                    teacher_logprobs = teacher_policy.get_logprobs(
                        logprob_data, timer=timer
                    )["logprobs"]

                # ── 6. Compute per-token advantages ───────────────────────
                # ThinkingMachines: advantage_t = log p_teacher(y_t) - log q_student(y_t)
                # This is the negative of per-token reverse KL.
                with timer.time("advantage_computation"):
                    print(">> Computing advantages...", flush=True)
                    generation_logprobs = train_data["generation_logprobs"]
                    advantages = teacher_logprobs - generation_logprobs

                    # Expand advantages to match sequence length
                    # (advantages are [B, S], same shape as logprobs)
                    train_data["advantages"] = advantages
                    train_data["prev_logprobs"] = student_prev_logprobs
                    # reference_policy_logprobs required by data dict but unused (kl_penalty=0)
                    train_data["reference_policy_logprobs"] = generation_logprobs

                    del logprob_data

                # ── 7. Train student ──────────────────────────────────────
                print(">> Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_training()
                    POLICY_GENERATION_STALE = True

                print(">> Training policy...", flush=True)
                with timer.time("policy_training"):
                    train_results = student_policy.train(
                        train_data,
                        loss_fn,
                        timer=timer,
                    )

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                # ── 8. Validation ─────────────────────────────────────────
                val_metrics = None
                val_period = distillation_config.get("val_period", 0)
                val_at_end = distillation_config.get("val_at_end", False)

                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    if val_dataloader is not None and val_task_to_env is not None:
                        from nemo_rl.algorithms.distillation import validate

                        if NEED_REFIT and POLICY_GENERATION_STALE:
                            refit_policy_generation(
                                student_policy,
                                student_generation,
                                colocated_inference,
                            )
                            POLICY_GENERATION_STALE = False
                        else:
                            student_generation.prepare_for_generation()
                        val_metrics, validation_timings = validate(
                            student_generation,
                            val_dataloader,
                            tokenizer,
                            val_task_to_env,
                            step=total_steps + 1,
                            master_config=master_config,
                        )
                        student_generation.finish_generation()
                        logger.log_metrics(
                            val_metrics, total_steps + 1, prefix="validation"
                        )
                        logger.log_metrics(
                            validation_timings,
                            total_steps + 1,
                            prefix="timing/validation",
                        )

                # ── 9. Metrics ────────────────────────────────────────────
                # Compute OPD-specific metrics
                token_mask = train_data["token_mask"][:, 1:]
                sample_mask = train_data["sample_mask"]
                mask = token_mask * sample_mask.unsqueeze(-1)
                valid_count = mask.sum().clamp(min=1)

                gen_lp = generation_logprobs[:, 1:]
                teacher_lp = teacher_logprobs[:, 1:]
                per_token_kl = (gen_lp - teacher_lp) * mask
                mean_reverse_kl = per_token_kl.sum() / valid_count

                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "mean_reverse_kl": mean_reverse_kl.item(),
                    "mean_prompt_length": repeated_batch["length"].numpy(),
                    "total_num_tokens": input_lengths.numpy(),
                }
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {
                        "lr",
                        "wd",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                        "mean_reverse_kl",
                    }:
                        metrics[k] = np.mean(v).item() if hasattr(v, "__len__") else v
                    elif k not in {"mean_reverse_kl"}:
                        metrics[k] = np.sum(v).item() if hasattr(v, "__len__") else v
                metrics.update(rollout_metrics)
                total_valid_tokens += metrics.get("global_valid_toks", 0)

                # ── 10. Checkpointing ─────────────────────────────────────
                consumed_samples += distillation_config["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1)
                    % master_config["checkpointing"]["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    student_policy.prepare_for_training()
                    save_state.update(
                        {
                            "current_epoch": current_epoch,
                            "current_step": current_step + 1,
                            "total_steps": total_steps + 1,
                            "total_valid_tokens": total_valid_tokens,
                            "consumed_samples": consumed_samples,
                        }
                    )
                    if val_metrics is not None:
                        save_state["val_reward"] = val_metrics["accuracy"]

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        prefix_str, metric_name = full_metric_name.split(":", 1)
                        metrics_source = (
                            metrics if prefix_str == "train" else val_metrics
                        )
                        if metrics_source and metric_name in metrics_source:
                            save_state[full_metric_name] = metrics_source[metric_name]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, save_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            )
                            if checkpointer.save_optimizer
                            else None,
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # ── Logging ───────────────────────────────────────────────
            log_data = {"content": flat_messages["content"]}
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )

            print("\n--- Training Results:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Mean Reverse KL: {metrics['mean_reverse_kl']:.4f}")
            print(
                f"  Mean Gen Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
            )

            total_time = timing_metrics.get("total_step_time", 0)
            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            metrics["tokens_per_sec_per_gpu"] = (
                metrics["total_num_tokens"] / max(total_time, 1e-8) / total_num_gpus
            )

            print(f"\n--- Timing: total {total_time:.2f}s")
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    pct = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  {k}: {v:.2f}s ({pct:.1f}%)")

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout reached, stopping early", flush=True)
                return
            if total_steps >= max_steps:
                print("Max steps reached, stopping", flush=True)
                return

        # Epoch complete
        current_step = 0
        current_epoch += 1
        save_state["current_epoch"] = current_epoch
        save_state["current_step"] = 0

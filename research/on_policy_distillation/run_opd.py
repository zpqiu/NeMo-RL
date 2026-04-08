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

"""Entry point for on-policy distillation via single-token reverse KL.

Usage:
    python research/on_policy_distillation/run_opd.py \
        --config research/on_policy_distillation/configs/opd_math.yaml

    # Override config values:
    python research/on_policy_distillation/run_opd.py \
        --config research/on_policy_distillation/configs/opd_math.yaml \
        policy.model_name=Qwen/Qwen3-1.7B-Base \
        teacher.model_name=Qwen/Qwen3-4B
"""

import argparse
import os

from omegaconf import OmegaConf

from nemo_rl.algorithms.distillation import MasterConfig, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

from on_policy_distillation.opd import opd_train


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="On-policy distillation via single-token reverse KL"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    return parser.parse_known_args()


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "opd_math.yaml"
        )

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    if config["policy"]["generation"] is not None:
        config["policy"]["generation"] = configure_generation_config(
            config["policy"]["generation"], tokenizer
        )

    # Setup data (reuses distillation data pipeline)
    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
        tokenizer, config["data"], config["env"]
    )

    # Reuse distillation setup for student/teacher/generation initialization.
    # The returned loss_fn (DistillationLossFn) is ignored — we create our own
    # ClippedPGLossFn in opd_train().
    (
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        _distillation_loss_fn,  # not used
        logger,
        checkpointer,
        save_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    opd_train(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        student_generation=student_generation,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        val_task_to_env=val_task_to_env,
        logger=logger,
        checkpointer=checkpointer,
        save_state=save_state,
        master_config=master_config,
    )


if __name__ == "__main__":
    main()

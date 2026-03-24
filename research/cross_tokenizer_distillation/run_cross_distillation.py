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

"""Entry point for cross-tokenizer on-policy distillation."""

import argparse
import os

from omegaconf import OmegaConf

from cross_tokenizer_distillation.algorithm import (
    CrossDistillMasterConfig,
    cross_tokenizer_distillation_train,
    setup,
)
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


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run cross-tokenizer distillation training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    return parser.parse_known_args()


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "cross_distill_math.yaml"
        )

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    config: CrossDistillMasterConfig = OmegaConf.to_container(config, resolve=True)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])

    init_ray()

    # Student tokenizer — handle local paths
    tokenizer_name = config["policy"]["tokenizer"]["name"]
    if os.path.isdir(tokenizer_name):
        # Local path: load directly to avoid HF Hub validation issues
        from transformers import AutoTokenizer as _AT
        student_tokenizer = _AT.from_pretrained(tokenizer_name, trust_remote_code=True, local_files_only=True)
        if student_tokenizer.pad_token is None:
            student_tokenizer.pad_token = student_tokenizer.eos_token
        print(f"Loaded student tokenizer from local path: {tokenizer_name}")
    else:
        student_tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    if config["policy"]["generation"] is not None:
        config["policy"]["generation"] = configure_generation_config(
            config["policy"]["generation"], student_tokenizer
        )

    # Setup data with STUDENT tokenizer (student generates the rollouts)
    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
        student_tokenizer, config["data"], config["env"]
    )

    (
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
    ) = setup(config, student_tokenizer, dataset, val_dataset)

    cross_tokenizer_distillation_train(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        student_generation=student_generation,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        task_to_env=task_to_env,
        val_task_to_env=val_task_to_env,
        logger=logger,
        checkpointer=checkpointer,
        save_state=save_state,
        master_config=master_config,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#SBATCH --job-name=math-code-sif
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=batch
# The GB200 batch QOS requires a full four-GPU node even though this build is CPU-only.
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=math-code-sif-%j.out

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to the NeMo-RL .sqsh path}"
: "${REPO_ROOT:?Set REPO_ROOT to this NeMo-RL checkout}"
: "${USER_ROOT:?Set USER_ROOT to the shared user directory mounted by Pyxis}"
: "${SIF_OUT:?Set SIF_OUT to the persistent output .sif path}"

NODE_TMP="/tmp/${USER}/math-code-sif-${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
mkdir -p "$NODE_TMP"
trap 'rm -rf "$NODE_TMP"' EXIT

srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="$CONTAINER_IMAGE" \
    --no-container-mount-home \
    --container-mounts="$USER_ROOT:$USER_ROOT,$NODE_TMP:/tmp" \
    --container-workdir="$REPO_ROOT" \
    bash research/math_with_code/docker/build_math_code_sif_local.sh

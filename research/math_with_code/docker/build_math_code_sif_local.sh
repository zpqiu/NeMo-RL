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
# Build and smoke-test the math-code SIF from inside the NeMo-RL sqsh.
set -euo pipefail

: "${SIF_OUT:?Set SIF_OUT to the persistent output .sif path}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/tmp/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/apptainer-tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$SIF_OUT")"

echo "Build host architecture: $(uname -m)"
test "$(uname -m)" = "aarch64"
command -v apptainer
command -v singularity
readlink -f "$(command -v singularity)"
singularity --version

# Pyxis starts the enclosing NeMo-RL container as root. Apptainer rejects
# root + --fakeroot without /etc/subuid mappings; root does not need it.
fakeroot_args=()
if [[ "$(id -u)" -ne 0 ]]; then
    fakeroot_args=(--fakeroot)
fi
echo "Effective uid: $(id -u); fakeroot args: ${fakeroot_args[*]:-(none)}"

build_id="${SLURM_JOB_ID:-manual-$$}"
tmp_sif="/tmp/math-code-py312-aarch64-${build_id}.sif"
smoke_root="/tmp/hb-smoke-${build_id}"
trap 'rm -rf "$tmp_sif" "$smoke_root"' EXIT

singularity build "${fakeroot_args[@]}" "$tmp_sif" "$SCRIPT_DIR/math_code_aarch64.def"

# Exercise the exact isolation flags used by Gym's custom environment, plus
# the persistent Python tool transport and verifier import.
mkdir -p "$smoke_root/staging" "$smoke_root/agent" "$smoke_root/verifier"
cp "$SCRIPT_DIR/math_code_sif_smoke.sh" "$smoke_root/staging/smoke.sh"
cp "$PROJECT_ROOT/responses_api_agents/harbor_agent/custom_agents/math_code_session.py" \
    "$smoke_root/staging/math_code_session.py"
singularity exec \
    --no-mount home \
    --no-mount tmp \
    --no-mount bind-paths \
    --pwd /app \
    --writable-tmpfs \
    "${fakeroot_args[@]}" \
    --containall \
    --pid \
    -B "$smoke_root/staging:/staging" \
    -B "$smoke_root/agent:/logs/agent" \
    -B "$smoke_root/verifier:/logs/verifier" \
    "$tmp_sif" \
    bash /staging/smoke.sh

# Publish atomically only after both build and runtime validation succeed.
mv -f "$tmp_sif" "$SIF_OUT"
singularity inspect "$SIF_OUT"
sha256sum "$SIF_OUT"
ls -lh "$SIF_OUT"

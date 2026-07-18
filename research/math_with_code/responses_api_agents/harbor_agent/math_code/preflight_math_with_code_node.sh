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
# Fast per-node checks for the Harbor math-code training runtime.
# Intended for ray.sub's SETUP_COMMAND so it runs inside the training sqsh on
# every allocated node before Ray starts.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARBOR_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$HARBOR_ROOT/../.." && pwd)"
VENV_ROOT="${NEMO_GYM_VENV_DIR:?export NEMO_GYM_VENV_DIR to the shared Gym venv root}"
HARBOR_PYTHON="$VENV_ROOT/responses_api_agents/harbor_agent/.venv/bin/python"
TASK_DIR="${MATH_CODE_PREFLIGHT_TASK_DIR:-$HARBOR_ROOT/data/math_code/aime_2024/task_000000}"
MATH_CONFIG="$HARBOR_ROOT/configs/math_code_harbor_agent.yaml"
MATH_AGENT="$HARBOR_ROOT/custom_agents/math_code_harbor_agent.py"

log() {
    printf '[math-code-node-preflight] host=%s %s\n' "$(hostname)" "$*"
}

fail() {
    log "ERROR: $*" >&2
    exit 1
}

[[ -x "$HARBOR_PYTHON" ]] || fail "Harbor Python is not executable: $HARBOR_PYTHON"
[[ -f "$MATH_CONFIG" ]] || fail "current math-code config is missing: $MATH_CONFIG (check the /opt/nemo-rl live mount)"
[[ -f "$MATH_AGENT" ]] || fail "current math-code agent is missing: $MATH_AGENT (check the /opt/nemo-rl live mount)"
[[ -d "$TASK_DIR" ]] || fail "task directory is missing: $TASK_DIR"

# nemo-gym resolves server dirs cwd-first; without the repo-root symlink it
# silently falls back to the pristine Gym submodule's harbor_agent instead of
# this overlay (see README "How the overlay fork works").
REPO_ROOT="$(cd -- "$PROJECT_ROOT/../.." && pwd)"
OVERLAY_DIR="$(readlink -f "$PROJECT_ROOT/responses_api_agents")"
[[ "$(readlink -f "$REPO_ROOT/responses_api_agents")" == "$OVERLAY_DIR" ]] || \
    fail "repo-root responses_api_agents does not point at the overlay; run: ln -sfn research/math_with_code/responses_api_agents $REPO_ROOT/responses_api_agents"

SIF_PATH="$(sed -n 's/^docker_image = "\(.*\)"/\1/p' "$TASK_DIR/task.toml")"
[[ -n "$SIF_PATH" && -r "$SIF_PATH" ]] || fail "shared SIF is missing or unreadable: $SIF_PATH"

command -v singularity >/dev/null || fail "singularity is not installed in the training container"

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$HARBOR_PYTHON" - <<'PY'
import platform
import sys

import harbor
import nemo_gym
import pydantic
import ray

print(
    f"python={sys.version.split()[0]} machine={platform.machine()} "
    f"harbor={harbor.__file__} nemo_gym={nemo_gym.__file__} "
    f"ray={ray.__version__} pydantic={pydantic.__version__}"
)
PY

singularity exec --cleanenv "$SIF_PATH" python3 - <<'PY'
import fastapi
import math_verify
import numpy
import scipy
import uvicorn

print("SIF imports: OK")
PY

log "PASS harbor_python=$HARBOR_PYTHON sif=$SIF_PATH"

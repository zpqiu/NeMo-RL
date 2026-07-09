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
# Build the isolated NeMo Gym server venv for responses_api_agents/harbor_agent.
# Put NEMO_GYM_VENV_DIR on shared storage when Harbor Ray jobs may run on
# multiple nodes; the venv's bin/python path must resolve on every Ray node.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
AGENT_DIR="$OVERLAY_ROOT/responses_api_agents/harbor_agent"
BASE_PYTHON="${GYM_PYTHON:-/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python}"
VENV_ROOT="${NEMO_GYM_VENV_DIR:-/opt/gym_venvs}"
VENV_DIR="$VENV_ROOT/responses_api_agents/harbor_agent/.venv"
HARBOR_PYTHON="$VENV_DIR/bin/python"

log() {
    printf '[harbor-venv] %s\n' "$*"
}

die() {
    printf '[harbor-venv] ERROR: %s\n' "$*" >&2
    exit 1
}

command -v uv >/dev/null 2>&1 || die "uv is not on PATH"
command -v git >/dev/null 2>&1 || die "git is required by harbor_agent/requirements.txt"
[[ -x "$BASE_PYTHON" ]] || die "base NemoGym Python is not executable: $BASE_PYTHON"
[[ -f "$AGENT_DIR/requirements.txt" ]] || die "requirements.txt not found: $AGENT_DIR/requirements.txt"

ray_version="$($BASE_PYTHON -c 'import ray; print(ray.__version__)')"
openai_version="$($BASE_PYTHON -c 'import openai; print(openai.__version__)')"

log "overlay root: $OVERLAY_ROOT"
log "base NemoGym Python: $BASE_PYTHON"
log "target Harbor venv: $VENV_DIR"
log "pinning ray==$ray_version and openai==$openai_version to the parent runtime"

mkdir -p "$VENV_ROOT"
uv venv --seed --allow-existing --python "$BASE_PYTHON" "$VENV_DIR"

# requirements.txt contains a ../../ editable path, matching Gym's native
# setup_env_command, so installation must run from the agent directory.
cd "$AGENT_DIR"
uv pip install \
    --python "$HARBOR_PYTHON" \
    -r requirements.txt \
    "ray[default]==$ray_version" \
    "openai==$openai_version"

PYTHONPATH="$OVERLAY_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$HARBOR_PYTHON" - <<'PY'
import sys

import harbor
import nemo_gym
import openai
import ray

from responses_api_agents.harbor_agent.custom_agents.math_code_harbor_agent import MathCodeHarborAgent
from responses_api_agents.harbor_agent.custom_envs.singularity.singularity import SingularityEnvironment

print(f"python={sys.executable}")
print(f"harbor={getattr(harbor, '__file__', None)}")
print(f"nemo_gym={getattr(nemo_gym, '__file__', None)}")
print(f"ray={ray.__version__} openai={openai.__version__}")
print(f"agent={MathCodeHarborAgent.__module__}.{MathCodeHarborAgent.__name__}")
print(f"environment={SingularityEnvironment.__module__}.{SingularityEnvironment.__name__}")
PY

log "PASS"
log "validate with: HARBOR_PYTHON=$HARBOR_PYTHON $SCRIPT_DIR/validate_runtime.sh"

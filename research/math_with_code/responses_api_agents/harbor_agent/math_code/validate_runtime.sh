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
# Validate the installed Gym/Harbor runtime and one real math-code Harbor trial.
#
# This script never builds or modifies the SIF.  By default it uses an existing
# generated task, reads the shared SIF path from that task's task.toml, and runs
# a deterministic local fake Responses API so no policy model is required.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
source "$OVERLAY_ROOT/math_code_paths.sh"
DEFAULT_TASK_DIR="$OVERLAY_ROOT/responses_api_agents/harbor_agent/data/math_code/aime_2024/task_000000"
TASK_DIR="${MATH_CODE_TASK_DIR:-$DEFAULT_TASK_DIR}"
MATH_CONFIG="$OVERLAY_ROOT/responses_api_agents/harbor_agent/configs/math_code_harbor_agent.yaml"
FULL_TRIAL_TIMEOUT_SEC="${FULL_TRIAL_TIMEOUT_SEC:-180}"

log() {
    printf '[math-code-check] %s\n' "$*"
}

die() {
    printf '[math-code-check] ERROR: %s\n' "$*" >&2
    exit 1
}

find_harbor_python() {
    local candidate

    if [[ -n "${HARBOR_PYTHON:-}" ]]; then
        [[ -x "$HARBOR_PYTHON" ]] || die "HARBOR_PYTHON is not executable: $HARBOR_PYTHON"
        printf '%s\n' "$HARBOR_PYTHON"
        return
    fi

    # NeMo Gym intentionally gives each server its own venv.  Harbor is not
    # expected in /opt/ray_venvs/...NemoGym; it belongs in the harbor_agent
    # server venv under NEMO_GYM_VENV_DIR (normally /opt/gym_venvs, or a
    # cluster-shared override for multi-node jobs).
    local venv_root="$NEMO_GYM_VENV_DIR"
    local -a candidates=()
    for candidate in \
        "$venv_root/responses_api_agents/harbor_agent/.venv/bin/python" \
        /opt/gym_venvs/responses_api_agents/harbor_agent/.venv/bin/python \
        "$OVERLAY_ROOT/responses_api_agents/harbor_agent/.venv/bin/python" \
        "${GYM_PYTHON:-}"; do
        [[ -x "$candidate" ]] && candidates+=("$candidate")
    done
    if [[ -d "$venv_root" ]]; then
        while IFS= read -r candidate; do
            candidates+=("$candidate")
        done < <(find -L "$venv_root" -maxdepth 6 -type f \
            \( -path '*/bin/python' -o -path '*/bin/python3' \) 2>/dev/null | sort -u)
    fi

    for candidate in "${candidates[@]}"; do
        [[ -x "$candidate" ]] || continue
        if PYTHONPATH="$OVERLAY_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            "$candidate" -c 'import harbor, nemo_gym' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    die "Harbor agent venv not found under $venv_root; run math_code/build_harbor_venv.sh first"
}

PYTHON="$(find_harbor_python)"
export PYTHONPATH="$OVERLAY_ROOT${PYTHONPATH:+:$PYTHONPATH}"

log "overlay root: $OVERLAY_ROOT"
log "Harbor agent Python: $PYTHON"
log "Host: $(hostname); arch: $(uname -m); uid: $(id -u)"

log "checking required runtime imports"
"$PYTHON" - <<'PY'
import importlib.metadata
import platform
import sys

import harbor
import nemo_gym
from responses_api_agents.harbor_agent.custom_agents.math_code_harbor_agent import MathCodeHarborAgent
from responses_api_agents.harbor_agent.custom_envs.singularity.singularity import SingularityEnvironment


def version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "unknown"


print(f"python={sys.version.split()[0]} machine={platform.machine()} executable={sys.executable}")
print(f"harbor={version('harbor', 'harbor-ai')} module={getattr(harbor, '__file__', None)}")
print(f"nemo-gym={version('nemo-gym')} module={getattr(nemo_gym, '__file__', None)}")
print(f"ray={version('ray')} pydantic={version('pydantic')} httpx={version('httpx')}")
print(f"agent={MathCodeHarborAgent.__module__}.{MathCodeHarborAgent.__name__}")
print(f"environment={SingularityEnvironment.__module__}.{SingularityEnvironment.__name__}")
print("Harbor overlay imports: OK")
PY


if [[ "${SKIP_FULL_TRIAL:-0}" == "1" ]]; then
    log "SKIP_FULL_TRIAL=1; done after import/unit checks"
    exit 0
fi

[[ -d "$TASK_DIR" ]] || die "task directory not found: $TASK_DIR"
[[ -f "$TASK_DIR/task.toml" ]] || die "task.toml not found under: $TASK_DIR"
[[ -f "$TASK_DIR/tests/expected_answer.json" ]] || die "expected_answer.json not found under: $TASK_DIR"
[[ -f "$MATH_CONFIG" ]] || die "production Harbor config not found: $MATH_CONFIG"
command -v singularity >/dev/null 2>&1 || die "singularity is not on PATH inside the sqsh"
command -v timeout >/dev/null 2>&1 || die "GNU timeout is not on PATH"

log "singularity: $(command -v singularity)"
singularity --version
log "running one real Harbor trial with existing task: $TASK_DIR"

JOBS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/math-code-harbor-check.XXXXXX")"
cleanup() {
    local status=$?
    if [[ "$status" -eq 0 && "${KEEP_VALIDATION_ARTIFACTS:-0}" != "1" ]]; then
        rm -rf "$JOBS_DIR"
    else
        log "keeping validation artifacts for inspection: $JOBS_DIR"
    fi
}
trap cleanup EXIT

# The fake API returns two execute_python calls followed by the task's known
# boxed answer.  This validates the complete runtime path without starting a
# policy model: Harbor -> custom agent -> Singularity -> persistent Python ->
# ATIF trajectory -> verifier -> NeMo-Gym response conversion.
timeout --signal=TERM --kill-after=15s "${FULL_TRIAL_TIMEOUT_SEC}s" \
    "$PYTHON" - "$TASK_DIR" "$JOBS_DIR" "$MATH_CONFIG" <<'PY'
import asyncio
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

import responses_api_agents.harbor_agent.app as harbor_app_module
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.harbor_agent.app import (
    HarborAgent,
    HarborAgentConfig,
    HarborRunRequest,
)
from responses_api_agents.harbor_agent.custom_envs.singularity.singularity import (
    resolve_math_code_sif_path,
)
from responses_api_agents.harbor_agent.utils import HarborAgentUtils


task_dir = Path(sys.argv[1]).resolve()
jobs_dir = Path(sys.argv[2]).resolve()
math_config_path = Path(sys.argv[3]).resolve()
task_config = tomllib.loads((task_dir / "task.toml").read_text())
sif_path = resolve_math_code_sif_path(task_config["environment"]["docker_image"])
if not sif_path.is_file():
    raise FileNotFoundError(f"task.toml points to a missing SIF: {sif_path}")

production_config = yaml.safe_load(math_config_path.read_text())
try:
    production_agent = production_config["math_code_harbor_agent"][
        "responses_api_agents"
    ]["harbor_agent"]
except (KeyError, TypeError) as exc:
    raise RuntimeError(
        f"invalid production Harbor config: {math_config_path}"
    ) from exc

# Reuse production constructor kwargs so this smoke test fails if the custom
# agent contract and its real config drift apart. The local model server and
# short watchdogs below are intentional test-only substitutions.
agent_kwargs = dict(production_agent["harbor_agent_kwargs"])
environment_kwargs = dict(production_agent.get("harbor_environment_kwargs") or {})
environment_kwargs["singularity_image_cache_dir"] = str(jobs_dir / "sif-cache")
environment_kwargs["workdir"] = "/app"

verifier_import_probe = subprocess.run(
    [
        "singularity",
        "exec",
        str(sif_path),
        "python3",
        "-c",
        (
            "from math_verify import grader; "
            "from math_verify.errors import TimeoutException; "
            "from math_verify.metric import math_metric; "
            "from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig"
        ),
    ],
    text=True,
    capture_output=True,
)
if verifier_import_probe.returncode != 0:
    raise RuntimeError(
        "SIF is missing the boxed-answer verifier runtime required by tests/verify.py: "
        f"{sif_path}\nstdout={verifier_import_probe.stdout}\nstderr={verifier_import_probe.stderr}"
    )

expected_payload = json.loads((task_dir / "tests" / "expected_answer.json").read_text())
ground_truth = str(expected_payload["ground_truth"])
request_bodies: list[dict[str, Any]] = []


def response_payload(output: list[dict[str, Any]], request_index: int) -> dict[str, Any]:
    payload = HarborAgentUtils.get_default_response_object()
    payload.update(
        {
            "id": f"resp-runtime-smoke-{request_index}",
            "created_at": int(time.time()),
            "model": "math-code-runtime-smoke",
            "output": output,
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 6,
            },
        }
    )
    return payload


def function_call(call_number: int, code: str, prompt_token_ids: list[int]) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"function-{call_number}",
        "call_id": f"call-{call_number}",
        "name": "execute_python",
        "arguments": json.dumps({"code": code}),
        "status": "completed",
        "prompt_token_ids": prompt_token_ids,
        "generation_token_ids": [20 + call_number],
        "generation_log_probs": [-0.1],
    }


class FakeResponsesHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/v1/responses":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        request_bodies.append(body)

        tools = body.get("tools") or []
        if not any(tool.get("name") == "execute_python" for tool in tools):
            self.send_error(400, "execute_python tool definition is missing")
            return
        outputs_seen = sum(
            item.get("type") == "function_call_output" for item in body.get("input", [])
        )
        if outputs_seen == 0:
            output = [function_call(1, "counter = 40\ncounter", [10, 11])]
        elif outputs_seen == 1:
            output = [function_call(2, "counter += 2\ncounter", [10, 11, 21, 12, 13])]
        else:
            output = [
                {
                    "type": "message",
                    "id": "message-final",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"The checked final answer is \\boxed{{{ground_truth}}}.",
                            "annotations": [],
                        }
                    ],
                    "prompt_token_ids": [10, 11, 21, 12, 13, 22, 14, 15],
                    "generation_token_ids": [41, 42],
                    "generation_log_probs": [-0.1, -0.1],
                }
            ]

        encoded = json.dumps(response_payload(output, len(request_bodies))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeResponsesHandler)
server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
server_thread.start()


async def run_trial() -> tuple[Path, Any]:
    # Exercise the same HarborAgent.run handler used by Gym's /run endpoint.
    # Use the real server cwd and a relative dataset path so path resolution is
    # covered; only the Ray transport is replaced with a local thread future.
    harbor_root = task_dir.parents[3]
    os.chdir(harbor_root)
    config = HarborAgentConfig(
        name="math_code_runtime_check",
        host="127.0.0.1",
        port=0,
        entrypoint="",
        concurrency=1,
        model_server={"type": "responses_api_models", "name": "unused_fake_model"},
        harbor_datasets={
            "runtime_check": {
                "local_dataset_path": str(task_dir.parent.relative_to(harbor_root)),
                "workdir": "/app",
            }
        },
        harbor_agent_name=production_agent.get("harbor_agent_name"),
        harbor_agent_import_path=production_agent.get("harbor_agent_import_path"),
        harbor_agent_kwargs=agent_kwargs,
        harbor_environment_type=production_agent.get("harbor_environment_type"),
        harbor_environment_import_path=production_agent.get(
            "harbor_environment_import_path"
        ),
        harbor_environment_kwargs=environment_kwargs,
        harbor_agent_override_timeout=90,
        harbor_verifier_override_timeout=30,
        harbor_timeout_multiplier=1.0,
        harbor_job_timeout_sec=150,
        harbor_job_cooperative_timeout_sec=140,
        harbor_request_queue_timeout_sec=5,
        harbor_raise_on_job_error=True,
        harbor_validate_training_tokens=True,
        harbor_jobs_dir=str(jobs_dir),
        harbor_save_response_json=False,
        harbor_retain_successful_jobs=True,
    )
    agent_server = HarborAgent.model_construct(config=config, server_client=None, sem=None)
    agent_server.sem = asyncio.Semaphore(1)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    class LocalRunnerRemote:
        @staticmethod
        def remote(runner: Any, params: dict[str, Any]) -> concurrent.futures.Future:
            return executor.submit(runner, **params)

    original_runner = harbor_app_module.runner_ray_remote
    original_global_config = harbor_app_module.get_global_config_dict
    original_ray_get = harbor_app_module.ray.get
    harbor_app_module.runner_ray_remote = LocalRunnerRemote
    harbor_app_module.get_global_config_dict = lambda: {
        "policy_model_name": "math-code-runtime-smoke",
        "unused_fake_model": {
            "responses_api_models": {
                "vllm_model": {
                    "host": "127.0.0.1",
                    "port": httpd.server_port,
                }
            }
        },
    }
    harbor_app_module.ray.get = (
        lambda future, timeout=None: future.result(timeout=timeout)
    )
    try:
        response = await agent_server.run(
            HarborRunRequest(
                instance_id=f"runtime_check::{task_dir.name}",
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[],
                    temperature=0.0,
                    top_p=1.0,
                ),
            )
        )
    finally:
        harbor_app_module.runner_ray_remote = original_runner
        harbor_app_module.get_global_config_dict = original_global_config
        harbor_app_module.ray.get = original_ray_get
        executor.shutdown(wait=True)

    # Harbor may write both job summaries and per-trial results. Identify the
    # latter by the artifact this smoke test consumes instead of relying on
    # result count or generated directory-name conventions.
    result_paths = [
        path
        for path in jobs_dir.rglob("result.json")
        if (path.parent / "agent" / "trajectory.json").is_file()
    ]
    if len(result_paths) != 1:
        raise AssertionError(
            f"expected one Harbor trial result with a trajectory, found {result_paths}"
        )
    return result_paths[0].parent.resolve(), response


try:
    trial_dir, endpoint_response = asyncio.run(run_trial())
finally:
    httpd.shutdown()
    httpd.server_close()
    server_thread.join(timeout=5)

result = json.loads((trial_dir / "result.json").read_text())
trajectory = json.loads((trial_dir / "agent" / "trajectory.json").read_text())
reward = HarborAgentUtils.extract_reward(result.get("verifier_result"))
if endpoint_response.reward != reward:
    raise AssertionError(
        f"Gym /run reward mismatch: endpoint={endpoint_response.reward}, Harbor={reward}"
    )
if reward < 0.5:
    def read_artifact(relative_path: str) -> str:
        path = trial_dir / relative_path
        if not path.exists():
            return "<missing>"
        return path.read_text(errors="replace")[-20_000:]

    diagnostics = {
        "details.json": read_artifact("verifier/details.json"),
        "test-stdout.txt": read_artifact("verifier/test-stdout.txt"),
        "test-stderr.txt": read_artifact("verifier/test-stderr.txt"),
        "exception.txt": read_artifact("exception.txt"),
        "trial.log": read_artifact("trial.log"),
    }
    raise AssertionError(
        f"verifier reward was {reward}; diagnostics={json.dumps(diagnostics, indent=2)}; result={result}"
    )

if len(request_bodies) != 3:
    raise AssertionError(f"expected three model calls, got {len(request_bodies)}")

agent_steps = [step for step in trajectory["steps"] if step.get("source") == "agent"]
observations = [
    json.loads(item["content"])
    for step in agent_steps
    for item in ((step.get("observation") or {}).get("results") or [])
]
if [item.get("result") for item in observations] != ["40", "42"]:
    raise AssertionError(f"Python session was not persistent: {observations}")
if not agent_steps[-1].get("message", "").endswith(f"\\boxed{{{ground_truth}}}."):
    raise AssertionError(f"unexpected final agent message: {agent_steps[-1].get('message')!r}")

converted = HarborAgentUtils.trial_result_to_responses(result, trajectory)
converted_types = [item["type"] if isinstance(item, dict) else item.type for item in converted]
for required_type in ("function_call", "function_call_output", "message"):
    if required_type not in converted_types:
        raise AssertionError(f"NeMo-Gym conversion lost {required_type}: {converted_types}")

# Mirror NeMo-RL's strict multi-turn token-contiguity invariant. Every later
# model call must begin with all prompt+generation tokens seen in earlier calls.
seen_token_ids: list[int] = []
for item in converted:
    item_dict = item if isinstance(item, dict) else item.model_dump(exclude_none=True)
    if "generation_token_ids" not in item_dict:
        continue
    prompt_token_ids = item_dict["prompt_token_ids"]
    if seen_token_ids != prompt_token_ids[: len(seen_token_ids)]:
        raise AssertionError(
            "converted rollout violates NeMo-RL token contiguity: "
            f"seen={seen_token_ids}, next_prompt={prompt_token_ids}"
        )
    seen_token_ids.extend(prompt_token_ids[len(seen_token_ids) :])
    seen_token_ids.extend(item_dict["generation_token_ids"])

print(
    json.dumps(
        {
            "status": "PASS",
            "sif": str(sif_path),
            "trial_dir": str(trial_dir),
            "reward": reward,
            "model_calls": len(request_bodies),
            "python_results": [item["result"] for item in observations],
            "trajectory_steps": len(trajectory["steps"]),
            "converted_output_types": converted_types,
            "contiguous_rollout_tokens": len(seen_token_ids),
        },
        indent=2,
    )
)
PY

log "PASS: Gym venv, Harbor API, custom agent, existing SIF, persistent Python, verifier, and output conversion all worked"

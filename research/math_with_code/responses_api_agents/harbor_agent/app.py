# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import re
import shutil
import sys
from asyncio import Semaphore
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional
from uuid import uuid4

# This harbor_agent is an overlay fork living under NeMo-RL's research/ tree.
# The editable nemo-gym install maps responses_api_agents.* to the pristine
# Gym submodule via a setuptools meta-path finder; PathFinder consults sys.path
# first, so putting the overlay root ahead makes every responses_api_agents.*
# import (here and in nemo-gym internals) resolve to this fork.
_OVERLAY_ROOT = str(Path(__file__).resolve().parents[2])
if _OVERLAY_ROOT not in sys.path:
    sys.path.insert(0, _OVERLAY_ROOT)

import ray
from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import (
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.harbor_agent.utils import HarborAgentUtils


class HarborDatasetSourceConfig(BaseModel):
    local_dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None
    workdir: Optional[str] = None


class HarborAgentConfig(BaseResponsesAPIAgentConfig):
    concurrency: int

    # --- Harbor agent settings ---
    # Name of a built-in Harbor agent (e.g. "terminus-2", "claude-code", "aider").
    harbor_agent_name: Optional[str] = "terminus-2"
    # Python import path for a custom agent class (e.g. "my_pkg.my_mod:MyAgent").
    # Overrides harbor_agent_name when set.
    harbor_agent_import_path: Optional[str] = None
    # Extra kwargs forwarded to the Harbor AgentConfig (e.g. collect_rollout_details,
    # model_info). See harbor_agent.yaml for examples.
    harbor_agent_kwargs: Optional[dict[str, Any]] = None

    # --- Dataset routing ---
    # Map of dataset aliases to source definitions. Each alias must define exactly
    # one source:
    # 1) local: {"local_dataset_path": "..."}
    # 2) registry: {"dataset_name": "...", "dataset_version": "..."} (version optional)
    # Requests must provide instance_id in the form "<dataset_alias>::<task_name>".
    harbor_datasets: dict[str, HarborDatasetSourceConfig]

    # --- Environment ---
    # Harbor environment type: "singularity", "docker", "daytona", "modal", etc.
    harbor_environment_type: Optional[str] = "singularity"
    # Python import path for a custom environment class (e.g. "my_pkg.my_mod:MyEnv").
    # Overrides harbor_environment_type when set.
    harbor_environment_import_path: Optional[str] = None
    # Extra kwargs forwarded to the Harbor EnvironmentConfig (e.g.
    # singularity_image_cache_dir, singularity_force_pull).
    harbor_environment_kwargs: Optional[dict[str, Any]] = None

    # --- Timeouts ---
    # Override agent timeout (seconds). Replaces the task's own timeout entirely.
    # Use this to set a fixed timeout for all tasks regardless of task.toml.
    harbor_agent_override_timeout: Optional[int] = None
    # Cap agent timeout (seconds). Uses the task's own timeout but clamps it
    # to this maximum. Respects shorter per-task timeouts unlike harbor_agent_override_timeout.
    harbor_agent_max_timeout: Optional[int] = None
    # Override verifier timeout (seconds). Replaces the task's own verifier timeout.
    harbor_verifier_override_timeout: Optional[int] = None
    # Cap verifier timeout (seconds). Uses the task's own verifier timeout but
    # clamps it to this maximum.
    harbor_verifier_max_timeout: Optional[int] = None
    # Multiplier applied to all Harbor timeouts after override/cap. None = 1.0.
    harbor_timeout_multiplier: Optional[float] = None
    # Final watchdog around the Ray-hosted Harbor job. Harbor's own phase
    # timeouts remain primary; this prevents a wedged cleanup/runtime from
    # blocking NeMo-RL rollout collection indefinitely. None disables it.
    harbor_job_timeout_sec: Optional[int] = None
    # Cooperative timeout inside the Ray worker. Keep this below the outer
    # watchdog so Harbor normally gets a chance to run environment cleanup.
    harbor_job_cooperative_timeout_sec: Optional[float] = None
    # Maximum time a request may wait for the per-server concurrency semaphore.
    # None preserves the unbounded queue used by older configs.
    harbor_request_queue_timeout_sec: Optional[float] = None
    # Training should fail loudly when a Harbor job cannot produce a rollout.
    # The legacy default keeps the previous reward=0/empty-output behavior.
    harbor_raise_on_job_error: bool = False
    # Enforce NeMo-RL's cumulative prompt/token/logprob contract before the
    # response leaves the Harbor server.
    harbor_validate_training_tokens: bool = False

    # --- Job output ---
    # Directory where Harbor writes job results and trial artifacts.
    harbor_jobs_dir: str = "jobs"
    # Full response copies and successful Harbor trial trees are useful for
    # debugging, but create substantial small-file pressure in RL training.
    harbor_save_response_json: bool = True
    harbor_retain_successful_jobs: bool = True
    # Drop duplicated rollout/config payloads from response metadata. The
    # trainable token details remain in response.output.
    harbor_compact_metadata: bool = False

    # --- ReTool-style tool-use shaping (arXiv:2504.11536) ---
    # Among FAILED rollouts only, add bonus_per_call per executed tool call,
    # capped below 1.0 so a correct answer always dominates. Within a GRPO
    # group this breaks reward ties on all-fail prompts and points the
    # advantage toward tool use. Applied only to the listed dataset aliases so
    # validation accuracy stays a pure correctness metric.
    harbor_tool_shaping_dataset_aliases: list[str] = []
    harbor_tool_shaping_bonus_per_call: float = 0.1
    harbor_tool_shaping_max_bonus: float = 0.4

    # --- Model routing ---
    # NeMo Gym model server reference used to resolve Harbor model base URL.
    model_server: ModelServerRef


class HarborRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    instance_id: str


class HarborVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    # Numeric fields are auto-aggregated into per-agent W&B metrics by NeMo-RL
    # (train/<agent>/raw_reward/mean, ...), so true correctness and tool-call
    # counts stay observable when `reward` carries the shaped value.
    raw_reward: float = 0.0
    num_tool_calls: int = 0
    # Pure rollout-speed metrics from the agent's wall-clock split (see
    # MathCodeHarborAgent._write_perf_metrics). model_generation_sec covers
    # only the model HTTP calls — tool execution is measured separately — so
    # gen_tokens_per_model_sec is a clean per-stream decode-rate for comparing
    # rollout-acceleration changes (FP8, cudagraphs, ...) across runs.
    model_generation_sec: float = 0.0
    tool_exec_sec: float = 0.0
    num_model_calls: int = 0
    generated_tokens: int = 0
    gen_tokens_per_model_sec: float = 0.0


async def run_harbor_job(job_config_dict: dict) -> str:
    """Runs a single Harbor Job and returns the trial directory path.

    The trial directory contains:
    - result.json: Summary result with reward, agent_result, verifier_result, etc.
    - agent/trajectory.json: Full ATIF trajectory with per-step messages, tool
      calls, observations, and per-token logprobs.

    Harbor writes result.json and trajectory.json to disk even when the trial
    fails (e.g. verifier timeout, reward file not found, OOM).  We recover the
    trial directory after an exception so the caller can still use the partial
    trajectory for training.
    """
    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    config = JobConfig(**job_config_dict)
    job = Job(config)

    job_error = None
    try:
        await job.run()
    except Exception as e:
        job_error = e

    # Find the trial directory from the job output directory.  Harbor writes
    # result.json before propagating most exceptions, so we can usually
    # recover the trial even when job.run() raised.
    job_dir = config.jobs_dir / config.job_name
    if job_dir.exists():
        for trial_dir in job_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            result_path = trial_dir / "result.json"
            if result_path.exists():
                return str(trial_dir.resolve())

    # No trial directory found — re-raise the original error if there was one,
    # otherwise raise FileNotFoundError.
    if job_error is not None:
        raise job_error
    raise FileNotFoundError(f"No trial result found in {job_dir}")


_RAY_WORKER_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _run_harbor_job_sync(
    job_config_dict: dict, cooperative_timeout_sec: Optional[float] = None
) -> str:
    """Synchronous wrapper for run_harbor_job for use in Ray remote.

    Ray workers are long-lived processes. Reusing a single event loop per worker
    avoids cross-loop issues with global async state (e.g., LiteLLM logging worker
    queues) when multiple jobs execute sequentially in the same process.
    """
    global _RAY_WORKER_EVENT_LOOP
    if _RAY_WORKER_EVENT_LOOP is None or _RAY_WORKER_EVENT_LOOP.is_closed():
        _RAY_WORKER_EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_RAY_WORKER_EVENT_LOOP)
    coroutine = run_harbor_job(job_config_dict)
    if cooperative_timeout_sec is not None:
        coroutine = asyncio.wait_for(coroutine, timeout=cooperative_timeout_sec)
    return _RAY_WORKER_EVENT_LOOP.run_until_complete(coroutine)


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
        # Harbor resolves harbor_agent_import_path/harbor_environment_import_path
        # inside this worker. Point responses_api_agents.* at the overlay fork,
        # not the pristine Gym submodule exposed by the editable install.
        "env_vars": {"PYTHONPATH": _OVERLAY_ROOT},
    },
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)


class HarborAgent(SimpleResponsesAPIAgent):
    config: HarborAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        return app

    @asynccontextmanager
    async def _request_slot(self) -> AsyncIterator[None]:
        """Acquire a bounded Harbor execution slot and report queue latency."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            if self.config.harbor_request_queue_timeout_sec is None:
                await self.sem.acquire()
            else:
                await asyncio.wait_for(
                    self.sem.acquire(),
                    timeout=self.config.harbor_request_queue_timeout_sec,
                )
        except TimeoutError as exc:
            waited = loop.time() - started
            raise TimeoutError(
                "Harbor request exceeded queue timeout of "
                f"{self.config.harbor_request_queue_timeout_sec} seconds "
                f"after waiting {waited:.1f} seconds"
            ) from exc

        waited = loop.time() - started
        print(f"Harbor request acquired execution slot after {waited:.3f}s")
        try:
            yield
        finally:
            self.sem.release()

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    async def run(self, body: HarborRunRequest) -> HarborVerifyResponse:
        async with self._request_slot():
            global_config_dict = get_global_config_dict()

            policy_model_name = global_config_dict["policy_model_name"]
            base_url = self._resolve_model_base_url(global_config_dict)
            run_timestamp = datetime.now(timezone.utc)
            run_id = self._build_run_id(run_timestamp)

            instance_id = body.instance_id
            dataset_alias, task_name = self._parse_instance_id(instance_id)

            output_file_dir = self._get_results_output_dir(policy_model_name, dataset_alias, run_timestamp)
            jobs_dir = self._get_jobs_output_dir(policy_model_name, dataset_alias, run_timestamp)
            job_name = self._build_job_name(run_id)

            responses_create_params = body.responses_create_params.model_dump(
                exclude_unset=True,
                exclude_none=True,
            )

            job_config_dict = self._build_job_config(
                dataset_alias,
                task_name,
                policy_model_name,
                base_url,
                job_name=job_name,
                jobs_dir=jobs_dir,
                responses_create_params=responses_create_params,
            )

            trial_dir = None
            try:
                harbor_job_timeout = False
                params = dict(
                    job_config_dict=job_config_dict,
                    cooperative_timeout_sec=self.config.harbor_job_cooperative_timeout_sec,
                )
                future = runner_ray_remote.remote(_run_harbor_job_sync, params)
                try:
                    trial_dir_path = await asyncio.to_thread(
                        ray.get,
                        future,
                        timeout=self.config.harbor_job_timeout_sec,
                    )
                except ray.exceptions.GetTimeoutError as exc:
                    harbor_job_timeout = True
                    ray.cancel(future, force=True)
                    raise TimeoutError(
                        f"Harbor job exceeded final watchdog of {self.config.harbor_job_timeout_sec} seconds"
                    ) from exc
                trial_dir = Path(trial_dir_path)

                # Read the trial result (summary: reward, agent_result, verifier_result)
                with open(trial_dir / "result.json", "r") as f:
                    trial_result = json.load(f)

                exception_info = trial_result.get("exception_info") or {}
                if exception_info:
                    exception_type = exception_info.get("exception_type", "unknown")
                    exception_message = exception_info.get(
                        "exception_message", "<no message>"
                    )
                    raise RuntimeError(
                        f"Harbor trial failed with {exception_type}: "
                        f"{exception_message}"
                    )

                # Read the ATIF trajectory (full conversation with per-token logprobs)
                trajectory = None
                trajectory_path = trial_dir / "agent" / "trajectory.json"
                if trajectory_path.exists():
                    with open(trajectory_path, "r") as f:
                        trajectory = json.load(f)

                # Read agent error flags written by the agent
                agent_error_flags = {}
                agent_error_flags_path = trial_dir / "agent" / "agent_error_flags.json"
                if agent_error_flags_path.exists():
                    with open(agent_error_flags_path, "r") as f:
                        agent_error_flags = json.load(f)

                agent_perf = {}
                agent_perf_path = trial_dir / "agent" / "agent_perf.json"
                if agent_perf_path.exists():
                    with open(agent_perf_path, "r") as f:
                        agent_perf = json.load(f)

                # Extract reward from verifier result
                verifier_result = trial_result.get("verifier_result")
                reward = HarborAgentUtils.extract_reward(verifier_result)

                # Convert Harbor outputs to NeMo Gym response items:
                # keep rich trajectory details, then overlay rollout token details when present.
                output_items = HarborAgentUtils.trial_result_to_responses(trial_result, trajectory)
                if self.config.harbor_validate_training_tokens:
                    HarborAgentUtils.validate_training_token_details(output_items)

                # Extract the initial instruction from the trajectory as input messages
                input_messages = HarborAgentUtils.extract_input_from_trajectory(trajectory)
                tool_definitions = ((trajectory or {}).get("agent") or {}).get("tool_definitions") or []

                # Populate usage from trajectory final_metrics or agent_result
                usage = HarborAgentUtils.extract_usage(trial_result, trajectory)

            except Exception as e:
                print(f"Error running Harbor job: {e}")
                if self.config.harbor_raise_on_job_error:
                    raise RuntimeError(
                        f"Harbor job failed for instance {instance_id!r}"
                    ) from e
                harbor_job_timeout = locals().get("harbor_job_timeout", False)
                trial_result = None
                trajectory = None
                agent_error_flags = {}
                agent_perf = {}
                output_items = []
                input_messages = []
                tool_definitions = []
                usage = None
                reward = 0.0

            num_tool_calls = sum(
                1 for item in output_items if item.get("type") == "function_call"
            )
            raw_reward = reward
            # Legacy path for task trees whose baked verify.py predates the
            # in-verifier shaping: those emit strictly binary rewards, so
            # shaping only exact-0 failures makes the two layers mutually
            # exclusive — a verifier-shaped failure arrives in (0, 1) and is
            # passed through untouched. The cap keeps every shaped failure
            # strictly below a correct answer.
            if (
                dataset_alias in self.config.harbor_tool_shaping_dataset_aliases
                and reward == 0.0
                and num_tool_calls > 0
            ):
                reward = min(
                    reward
                    + self.config.harbor_tool_shaping_bonus_per_call * num_tool_calls,
                    self.config.harbor_tool_shaping_max_bonus,
                )

            response = HarborAgentUtils.get_default_response_object()
            response["model"] = policy_model_name
            response["temperature"] = responses_create_params.get("temperature")
            response["top_p"] = responses_create_params.get("top_p")
            response["output"] = output_items
            response["tools"] = tool_definitions
            if usage:
                response["usage"] = usage

            # Update responses_create_params with the actual input sent to the agent
            updated_params = body.responses_create_params
            actual_params: dict[str, Any] = {}
            if input_messages:
                actual_params["input"] = input_messages
            if tool_definitions:
                actual_params["tools"] = tool_definitions
                actual_params["parallel_tool_calls"] = False
                actual_params["tool_choice"] = "auto"
            if actual_params:
                updated_params = body.responses_create_params.model_copy(update=actual_params)

            metadata = (
                self._compact_trial_metadata(trial_result)
                if self.config.harbor_compact_metadata
                else trial_result
            )
            model_generation_sec = float(agent_perf.get("model_generation_sec", 0.0))
            generated_tokens = int(agent_perf.get("generated_tokens", 0))
            verify_response = HarborVerifyResponse(
                responses_create_params=updated_params,
                reward=reward,
                raw_reward=raw_reward,
                num_tool_calls=num_tool_calls,
                model_generation_sec=model_generation_sec,
                tool_exec_sec=float(agent_perf.get("tool_exec_sec", 0.0)),
                num_model_calls=int(agent_perf.get("num_model_calls", 0)),
                generated_tokens=generated_tokens,
                gen_tokens_per_model_sec=(
                    generated_tokens / model_generation_sec
                    if model_generation_sec > 0
                    else 0.0
                ),
                response=response,
                instance_id=instance_id,
                metadata=metadata if metadata else {},
                context_length_exceeded_error=int(agent_error_flags.get("context_length_exceeded", False)),
                memory_limit_exceeded_error=int(agent_error_flags.get("memory_limit_exceeded", False)),
                agent_timeout_error=int(
                    harbor_job_timeout
                    or ((trial_result or {}).get("exception_info") or {}).get("exception_type")
                    == "AgentTimeoutError"
                ),
            )

            if self.config.harbor_save_response_json:
                # Save result to disk (folder = run_id, file = task name)
                output_path = output_file_dir / run_id
                output_path.mkdir(parents=True, exist_ok=True)

                safe_instance_id = self._sanitize_path_component(instance_id)
                with open(output_path / f"{safe_instance_id}.json", "w") as f:
                    json.dump(verify_response.model_dump(), f, indent=2)

            has_agent_error = any(
                bool(value) for value in agent_error_flags.values()
            )
            healthy_job = (
                trial_dir is not None
                and trial_result is not None
                and not trial_result.get("exception_info")
                and not has_agent_error
                and bool(output_items)
            )
            if healthy_job and not self.config.harbor_retain_successful_jobs:
                assert trial_dir is not None
                self._remove_job_artifacts(trial_dir, job_name)

            return verify_response

    def _get_results_output_dir(self, policy_model_name: str, dataset_alias: str, run_timestamp: datetime) -> Path:
        """Build immutable run output directory grouped by date/dataset/model."""
        date_key = run_timestamp.strftime("%Y%m%d")
        dataset_key = self._sanitize_path_component(dataset_alias)
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        return Path.cwd() / "results" / "runs" / date_key / dataset_key / model_key

    def _get_jobs_output_dir(self, policy_model_name: str, dataset_alias: str, run_timestamp: datetime) -> Path:
        """Build Harbor jobs directory grouped by date/dataset/model."""
        date_key = run_timestamp.strftime("%Y%m%d")
        dataset_key = self._sanitize_path_component(dataset_alias)
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        jobs_root = Path(self.config.harbor_jobs_dir).expanduser()
        if not jobs_root.is_absolute():
            jobs_root = Path.cwd() / jobs_root
        return jobs_root.resolve() / date_key / dataset_key / model_key

    @staticmethod
    def _compact_trial_metadata(trial_result: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Keep diagnostic summary fields without duplicating rollout token arrays."""
        if not trial_result:
            return {}
        keys = (
            "id",
            "task_name",
            "trial_name",
            "source",
            "agent_info",
            "verifier_result",
            "exception_info",
            "started_at",
            "finished_at",
        )
        return {key: trial_result[key] for key in keys if key in trial_result}

    @staticmethod
    def _remove_job_artifacts(trial_dir: Path, job_name: str) -> None:
        """Remove one completed single-trial Harbor job without widening scope."""
        job_dir = trial_dir.resolve().parent
        if job_dir.name != job_name:
            print(
                f"Refusing to remove unexpected Harbor job directory {job_dir}; "
                f"expected basename {job_name!r}"
            )
            return
        try:
            shutil.rmtree(job_dir)
        except OSError as exc:
            print(f"Failed to remove successful Harbor job artifacts {job_dir}: {exc}")

    @staticmethod
    def _parse_instance_id(instance_id: str) -> tuple[str, str]:
        """Parse instance id in the required form: <dataset_alias>::<task_name>."""
        dataset_alias, sep, task_name = instance_id.partition("::")
        dataset_alias = dataset_alias.strip()
        task_name = task_name.strip()
        if not sep or not dataset_alias or not task_name:
            raise ValueError(f"instance_id must be in the form '<dataset_alias>::<task_name>' (got: {instance_id!r})")
        return dataset_alias, task_name

    def _build_run_id(self, run_timestamp: datetime) -> str:
        """Build a compact run id (time + short hash) for immutable file naming."""
        time_key = run_timestamp.strftime("%H%M%S")
        return f"{time_key}_{uuid4().hex[:8]}"

    def _build_job_name(self, run_id: str) -> str:
        """Build a Harbor job name from run id only."""
        return run_id

    @staticmethod
    def _extract_model_name(policy_model_name: str) -> str:
        """Extract the final model name from a full path or HF-style identifier.

        '/lustre/.../nano-v3-sft-...-hf'  -> 'nano-v3-sft-...-hf'
        'Qwen/Qwen3-8B'                   -> 'Qwen3-8B'
        'my-model'                         -> 'my-model'
        """
        return Path(policy_model_name).name or policy_model_name

    def _sanitize_path_component(self, value: str) -> str:
        """Sanitize path components to avoid accidental nested directories."""
        sanitized = value.replace("/", "__").replace("\\", "__").replace(":", "__")
        sanitized = re.sub(r"\s+", "_", sanitized)
        sanitized = sanitized.strip("._")
        return sanitized or "unknown"

    def _resolve_model_base_url(self, global_config_dict: Any) -> str:
        """Resolve model base URL from required model_server reference."""
        server_name = self.config.model_server.name
        model_server_config = get_first_server_config_dict(
            global_config_dict,
            server_name,
        )
        return f"http://{model_server_config['host']}:{model_server_config['port']}/v1"

    def _build_job_config(
        self,
        dataset_alias: str,
        task_name: str,
        model_name: str,
        api_base: str,
        job_name: str,
        jobs_dir: Path,
        responses_create_params: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Build a Harbor JobConfig dict for a single task."""
        from harbor.models.job.config import (
            JobConfig,
            LocalDatasetConfig,
            OrchestratorConfig,
            RegistryDatasetConfig,
        )
        from harbor.models.registry import RemoteRegistryInfo
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            VerifierConfig,
        )

        agent_kwargs: dict[str, Any] = {"api_base": api_base}
        if responses_create_params:
            agent_kwargs["responses_create_params"] = responses_create_params
            # Terminus-2 accepts temperature as a top-level kwarg for trajectory metadata.
            if "temperature" in responses_create_params:
                agent_kwargs["temperature"] = responses_create_params["temperature"]
        if self.config.harbor_agent_kwargs:
            agent_kwargs.update(self.config.harbor_agent_kwargs)

        agent_config = AgentConfig(
            name=self.config.harbor_agent_name if not self.config.harbor_agent_import_path else None,
            import_path=self.config.harbor_agent_import_path,
            model_name=model_name,
            override_timeout_sec=(
                float(self.config.harbor_agent_override_timeout)
                if self.config.harbor_agent_override_timeout is not None
                else None
            ),
            max_timeout_sec=(
                float(self.config.harbor_agent_max_timeout)
                if self.config.harbor_agent_max_timeout is not None
                else None
            ),
            kwargs=agent_kwargs,
        )

        dataset_source = self.config.harbor_datasets.get(dataset_alias)
        if dataset_source is None:
            available = ", ".join(sorted(self.config.harbor_datasets.keys()))
            raise ValueError(
                f"Unknown dataset alias in instance_id: {dataset_alias!r}. Available aliases: [{available}]"
            )

        has_local = bool(dataset_source.local_dataset_path)
        has_registry = bool(dataset_source.dataset_name)
        if has_local == has_registry:
            raise ValueError(
                f"Dataset alias {dataset_alias!r} must define exactly one source: "
                "local_dataset_path OR dataset_name[/dataset_version]."
            )

        environment_kwargs = {}
        if self.config.harbor_environment_kwargs:
            environment_kwargs.update(self.config.harbor_environment_kwargs)
        cache_dir = environment_kwargs.get("singularity_image_cache_dir")
        if cache_dir:
            cache_path = Path(cache_dir).expanduser()
            if not cache_path.is_absolute():
                cache_path = Path.cwd() / cache_path
            environment_kwargs["singularity_image_cache_dir"] = str(
                cache_path.resolve()
            )
        # Dataset alias-level workdir overrides global harbor_environment_kwargs.workdir.
        if dataset_source.workdir is not None:
            environment_kwargs["workdir"] = dataset_source.workdir

        environment_config = EnvironmentConfig(
            type=self.config.harbor_environment_type if not self.config.harbor_environment_import_path else None,
            import_path=self.config.harbor_environment_import_path,
            kwargs=environment_kwargs,
        )

        verifier_config = VerifierConfig(
            override_timeout_sec=(
                float(self.config.harbor_verifier_override_timeout)
                if self.config.harbor_verifier_override_timeout is not None
                else None
            ),
            max_timeout_sec=(
                float(self.config.harbor_verifier_max_timeout)
                if self.config.harbor_verifier_max_timeout is not None
                else None
            ),
        )

        orchestrator_config = OrchestratorConfig(
            n_concurrent_trials=1,
            quiet=True,
        )

        if has_registry:
            dataset_config = RegistryDatasetConfig(
                registry=RemoteRegistryInfo(),
                name=dataset_source.dataset_name,
                version=dataset_source.dataset_version,
                task_names=[task_name],
            )
        else:
            dataset_path = Path(dataset_source.local_dataset_path).expanduser()
            if not dataset_path.is_absolute():
                dataset_path = Path.cwd() / dataset_path
            dataset_config = LocalDatasetConfig(
                path=dataset_path.resolve(),
                task_names=[task_name],
            )

        job_config = JobConfig(
            job_name=job_name,
            jobs_dir=jobs_dir,
            timeout_multiplier=(
                self.config.harbor_timeout_multiplier if self.config.harbor_timeout_multiplier is not None else 1.0
            ),
            orchestrator=orchestrator_config,
            environment=environment_config,
            verifier=verifier_config,
            agents=[agent_config],
            datasets=[dataset_config],
        )

        return job_config.model_dump(mode="json")


if __name__ == "__main__":
    HarborAgent.run_webserver()

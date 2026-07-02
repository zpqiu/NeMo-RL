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
"""Harbor agent for multi-turn math reasoning with an in-container Python tool."""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories.agent import Agent
from harbor.models.trajectories.final_metrics import FinalMetrics
from harbor.models.trajectories.metrics import Metrics
from harbor.models.trajectories.observation import Observation
from harbor.models.trajectories.observation_result import ObservationResult
from harbor.models.trajectories.step import Step
from harbor.models.trajectories.tool_call import ToolCall
from harbor.models.trajectories.trajectory import Trajectory
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


EXECUTE_PYTHON_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "execute_python",
    "description": (
        "Execute Python in a persistent per-problem session. Variables and imports survive "
        "between calls. NumPy (`np`), SciPy (`scipy`), and pandas (`pd`) are preloaded."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute. Print values you want to inspect.",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    "strict": True,
}


DEFAULT_SYSTEM_PROMPT = """You are a mathematical problem-solving assistant.
Solve the user's problem rigorously. You may call execute_python to check calculations,
explore patterns, or perform symbolic/numeric work. The Python session is persistent across
calls, but Python output is only scratch work and is not itself the final response.

End your final assistant response with exactly one final answer wrapped as \\boxed{...}.
Do not place the final answer only in Python output."""


class MathCodeHarborAgent(BaseAgent):
    """Responses-API agent that owns a stateful Python process inside the task SIF."""

    SUPPORTS_ATIF = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        api_base: str | None = None,
        responses_create_params: dict[str, Any] | None = None,
        max_steps: int = 4,
        code_timeout_sec: int = 10,
        code_memory_limit_mb: int = 1024,
        model_timeout_sec: float = 180.0,
        max_tool_output_chars: int | None = None,
        max_code_chars: int = 100_000,
        collect_rollout_details: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if not api_base:
            raise ValueError("MathCodeHarborAgent requires api_base")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if code_timeout_sec < 1:
            raise ValueError("code_timeout_sec must be at least 1")
        if code_memory_limit_mb < 1:
            raise ValueError("code_memory_limit_mb must be at least 1")
        if max_tool_output_chars is None:
            raise ValueError("MathCodeHarborAgent requires max_tool_output_chars")
        if max_tool_output_chars < 1:
            raise ValueError("max_tool_output_chars must be at least 1")
        self._api_base = api_base.rstrip("/")
        self._responses_create_params = dict(responses_create_params or {})
        self._max_steps = max_steps
        self._code_timeout_sec = code_timeout_sec
        self._code_memory_limit_mb = code_memory_limit_mb
        self._model_timeout_sec = model_timeout_sec
        self._max_tool_output_chars = max_tool_output_chars
        self._max_code_chars = max_code_chars
        self._collect_rollout_details = collect_rollout_details
        self._system_prompt = system_prompt
        self._session_script = "/app/.math_code_session.py"
        self._session_socket = "/app/.math_code_session.sock"
        self._steps: list[Step] = []
        self._turn_metrics: list[Metrics] = []
        self._run_error: str | None = None
        self._context_length_exceeded = False

    @staticmethod
    def name() -> str:
        return "math-code-harbor"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        helper_path = Path(__file__).with_name("math_code_session.py")
        await environment.upload_file(helper_path, self._session_script)
        command = (
            f"rm -f {shlex.quote(self._session_socket)}; "
            f"nohup python3 {shlex.quote(self._session_script)} serve "
            f"--socket {shlex.quote(self._session_socket)} "
            f"--timeout-sec {self._code_timeout_sec} "
            f"--memory-limit-mb {self._code_memory_limit_mb} "
            ">/app/.math_code_session.log 2>&1 </dev/null & "
            # Wait inside one environment.exec request. Retrying through the
            # HTTP command server generated a warning/traceback for every
            # expected pre-readiness ping and amplified load at high rollout
            # concurrency.
            "for _ in $(seq 1 100); do "
            f"if python3 {shlex.quote(self._session_script)} ping "
            f"--socket {shlex.quote(self._session_socket)} >/dev/null 2>&1; then exit 0; fi; "
            "sleep 0.1; "
            "done; "
            "tail -100 /app/.math_code_session.log; exit 1"
        )
        result = await environment.exec(command, cwd="/app", timeout_sec=15)
        if result.return_code != 0:
            raise RuntimeError(f"Failed to start Python session: {result.stdout or result.stderr}")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        session_id = uuid4().hex
        self._steps = [
            Step(step_id=1, source="system", message=self._system_prompt),
            Step(step_id=2, source="user", message=instruction),
        ]
        self._turn_metrics = []
        self._run_error = None
        self._context_length_exceeded = False
        generated_items: list[dict[str, Any]] = []
        initial_input = [
            {"type": "message", "role": "system", "content": self._system_prompt},
            {"type": "message", "role": "user", "content": instruction},
        ]

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._model_timeout_sec, connect=min(10.0, self._model_timeout_sec))
            ) as client:
                for _ in range(self._max_steps):
                    body = self._build_model_body(initial_input + generated_items)
                    response = await self._call_model(client, body)
                    metrics = self._extract_metrics(response)
                    if self._is_context_length_overflow(response, metrics):
                        self._context_length_exceeded = True
                        self._run_error = (
                            "Context length exceeded before the model generated token details"
                        )
                        self.logger.warning(self._run_error)
                        break

                    output_items = [item.model_dump(exclude_none=True) for item in response.output]
                    generated_items.extend(output_items)

                    self._turn_metrics.append(metrics)
                    step, function_calls = self._build_agent_step(response, metrics)
                    self._steps.append(step)

                    if not function_calls or response.incomplete_details:
                        break

                    observations: list[ObservationResult] = []
                    for function_call in function_calls:
                        tool_output, parsed_arguments = await self._execute_tool_call(environment, function_call)
                        generated_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": function_call.call_id,
                                "output": tool_output,
                            }
                        )
                        observations.append(
                            ObservationResult(source_call_id=function_call.call_id, content=tool_output)
                        )
                        # Arguments are parsed after the initial Step construction so malformed
                        # JSON can still be represented by a valid ATIF dictionary.
                        for tool_call in step.tool_calls or []:
                            if tool_call.tool_call_id == function_call.call_id:
                                tool_call.arguments = parsed_arguments
                    step.observation = Observation(results=observations)
                    self._write_trajectory(session_id)
                    self._populate_context(context)
        except Exception as exc:
            # A single failed rollout should yield a partial trajectory and reward 0,
            # rather than aborting collection of the entire batch.
            self._run_error = f"{type(exc).__name__}: {exc}"
            self.logger.exception("MathCodeHarborAgent stopped with a partial trajectory")
        finally:
            self._write_trajectory(session_id)
            self._populate_context(context)
            self._write_error_flags()

    def _build_model_body(self, inputs: list[dict[str, Any]]) -> NeMoGymResponseCreateParamsNonStreaming:
        params = dict(self._responses_create_params)
        for field in (
            "input",
            "instructions",
            "model",
            "parallel_tool_calls",
            "previous_response_id",
            "stream",
            "tool_choice",
            "tools",
        ):
            params.pop(field, None)
        params.update(
            {
                "input": inputs,
                "model": self.model_name,
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [EXECUTE_PYTHON_TOOL],
                "stream": False,
            }
        )
        return NeMoGymResponseCreateParamsNonStreaming.model_validate(params)

    async def _call_model(
        self,
        client: httpx.AsyncClient,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymResponse:
        response = await client.post(
            f"{self._api_base}/responses",
            json=body.model_dump(exclude_none=True, exclude_unset=True),
        )
        response.raise_for_status()
        try:
            return NeMoGymResponse.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            raise RuntimeError(f"Model server returned an invalid Responses payload: {response.text[:2000]}") from exc

    def _build_agent_step(self, response: NeMoGymResponse, metrics: Metrics) -> tuple[Step, list[Any]]:
        message_parts: list[str] = []
        reasoning_parts: list[str] = []
        function_calls: list[Any] = []
        tool_calls: list[ToolCall] = []

        for item in response.output:
            if item.type == "message" and item.role == "assistant":
                for content in item.content:
                    if content.type == "output_text":
                        message_parts.append(content.text)
                    elif content.type == "refusal":
                        message_parts.append(content.refusal)
            elif item.type == "reasoning":
                reasoning_parts.extend(summary.text for summary in item.summary)
            elif item.type == "function_call":
                function_calls.append(item)
                tool_calls.append(
                    ToolCall(
                        tool_call_id=item.call_id,
                        function_name=item.name,
                        arguments={"_raw": item.arguments},
                    )
                )

        step = Step(
            step_id=len(self._steps) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="agent",
            model_name=self.model_name,
            message="".join(message_parts),
            reasoning_content="\n".join(reasoning_parts) or None,
            tool_calls=tool_calls or None,
            metrics=metrics,
        )
        return step, function_calls

    async def _execute_tool_call(
        self,
        environment: BaseEnvironment,
        function_call: Any,
    ) -> tuple[str, dict[str, Any]]:
        try:
            arguments = json.loads(function_call.arguments)
            if not isinstance(arguments, dict):
                raise TypeError("tool arguments must be a JSON object")
        except (json.JSONDecodeError, TypeError) as exc:
            arguments = {"_raw": function_call.arguments}
            return self._format_tool_error(f"Invalid tool arguments: {type(exc).__name__}: {exc}"), arguments

        if function_call.name != EXECUTE_PYTHON_TOOL["name"]:
            return self._format_tool_error(f"Unknown tool: {function_call.name}"), arguments
        code = arguments.get("code")
        if not isinstance(code, str):
            return self._format_tool_error("execute_python requires a string `code` argument"), arguments
        if len(code) > self._max_code_chars:
            return self._format_tool_error(
                f"Python code exceeds the {self._max_code_chars}-character limit"
            ), arguments

        encoded = base64.urlsafe_b64encode(code.encode("utf-8")).decode("ascii")
        command = (
            f"python3 {shlex.quote(self._session_script)} execute "
            f"--socket {shlex.quote(self._session_socket)} "
            f"--code-base64 {shlex.quote(encoded)}"
        )
        try:
            result = await environment.exec(
                command,
                cwd="/app",
                timeout_sec=self._code_timeout_sec + 3,
            )
            if result.return_code != 0:
                payload: dict[str, Any] = {
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "error_message": f"Python tool process exited with code {result.return_code}",
                }
                if result.stdout or result.stderr:
                    payload["stderr"] = result.stdout or result.stderr
            else:
                payload = json.loads(result.stdout or "{}")
        except (asyncio.TimeoutError, json.JSONDecodeError, TypeError, ValueError) as exc:
            payload = {
                "success": False,
                "stdout": "",
                "stderr": "",
                "result": None,
                "error_message": f"Python tool transport error: {type(exc).__name__}: {exc}",
            }
        return self._truncate_tool_output(json.dumps(payload, ensure_ascii=False)), arguments

    @staticmethod
    def _format_tool_error(message: str) -> str:
        return json.dumps(
            {
                "success": False,
                "stdout": "",
                "stderr": "",
                "result": None,
                "error_message": message,
            },
            ensure_ascii=False,
        )

    def _truncate_tool_output(self, output: str) -> str:
        if len(output) <= self._max_tool_output_chars:
            return output
        omitted = len(output) - self._max_tool_output_chars
        return f"{output[: self._max_tool_output_chars]}\n... <truncated {omitted} characters>"

    @staticmethod
    def _extract_metrics(response: NeMoGymResponse) -> Metrics:
        token_item: Any = None
        for item in reversed(response.output):
            if hasattr(item, "prompt_token_ids"):
                token_item = item
                break
        usage = response.usage
        return Metrics(
            prompt_tokens=usage.input_tokens if usage else None,
            completion_tokens=usage.output_tokens if usage else None,
            cached_tokens=usage.input_tokens_details.cached_tokens if usage else None,
            prompt_token_ids=list(token_item.prompt_token_ids) if token_item is not None else None,
            completion_token_ids=list(token_item.generation_token_ids) if token_item is not None else None,
            logprobs=list(token_item.generation_log_probs) if token_item is not None else None,
        )

    @staticmethod
    def _is_context_length_overflow(response: NeMoGymResponse, metrics: Metrics) -> bool:
        """Detect the empty response emitted when vLLM rejects a full prompt."""
        return bool(
            response.incomplete_details
            and response.incomplete_details.reason == "max_output_tokens"
            and metrics.prompt_token_ids is None
            and metrics.completion_token_ids is None
            and metrics.logprobs is None
        )

    def _trajectory(self, session_id: str) -> Trajectory:
        return Trajectory(
            schema_version="ATIF-v1.5",
            session_id=session_id,
            agent=Agent(
                name=self.name(),
                version=self.version(),
                model_name=self.model_name,
                tool_definitions=[EXECUTE_PYTHON_TOOL],
                extra={
                    "max_steps": self._max_steps,
                    "code_timeout_sec": self._code_timeout_sec,
                    "code_memory_limit_mb": self._code_memory_limit_mb,
                },
            ),
            steps=self._steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=sum(metric.prompt_tokens or 0 for metric in self._turn_metrics),
                total_completion_tokens=sum(metric.completion_tokens or 0 for metric in self._turn_metrics),
                total_cached_tokens=sum(metric.cached_tokens or 0 for metric in self._turn_metrics),
                total_steps=len(self._steps),
                extra={"run_error": self._run_error} if self._run_error else None,
            ),
        )

    def _write_trajectory(self, session_id: str) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = self.logs_dir / "trajectory.json"
        temporary_path = trajectory_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(self._trajectory(session_id).to_json_dict(), ensure_ascii=False, indent=2)
        )
        temporary_path.replace(trajectory_path)

    def _populate_context(self, context: AgentContext) -> None:
        context.n_input_tokens = sum(metric.prompt_tokens or 0 for metric in self._turn_metrics)
        context.n_output_tokens = sum(metric.completion_tokens or 0 for metric in self._turn_metrics)
        context.n_cache_tokens = sum(metric.cached_tokens or 0 for metric in self._turn_metrics)
        context.metadata = {"run_error": self._run_error} if self._run_error else {}
        if self._collect_rollout_details:
            context.rollout_details = [
                {
                    "prompt_token_ids": [metric.prompt_token_ids or [] for metric in self._turn_metrics],
                    "completion_token_ids": [metric.completion_token_ids or [] for metric in self._turn_metrics],
                    "logprobs": [metric.logprobs or [] for metric in self._turn_metrics],
                }
            ]

    def _write_error_flags(self) -> None:
        flags = {
            "context_length_exceeded": self._context_length_exceeded,
            "memory_limit_exceeded": False,
            "run_error": self._run_error,
        }
        (self.logs_dir / "agent_error_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2))

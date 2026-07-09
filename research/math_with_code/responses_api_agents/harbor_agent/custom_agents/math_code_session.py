#!/usr/bin/env python3
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
"""Small persistent Python execution service used inside a Harbor task SIF.

The service deliberately uses only the Python standard library for its transport.
It keeps a worker process alive between tool calls so notebook-like state is
preserved, and replaces that worker if submitted code exceeds its deadline.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import multiprocessing as mp
import os
import reprlib
import resource
import signal
import socket
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any


MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_CAPTURE_CHARS = 200_000
MIB = 1024 * 1024

# task.toml declares one CPU, but the Singularity backend does not install a
# cgroup. Enforce the same contract in BLAS-backed libraries before importing
# NumPy/SciPy so concurrent trials cannot each fan out across the whole node.
for _thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_env_var] = "1"


class _BoundedStringIO(io.StringIO):
    """StringIO compatible writer that retains at most MAX_CAPTURE_CHARS."""

    def __init__(self, max_chars: int = MAX_CAPTURE_CHARS) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.truncated = False

    def write(self, value: str) -> int:
        remaining = max(0, self.max_chars - self.tell())
        if len(value) > remaining:
            self.truncated = True
        if remaining:
            super().write(value[:remaining])
        # TextIO.write reports consumed input, even when retention is bounded.
        return len(value)

    def getvalue(self) -> str:
        value = super().getvalue()
        if self.truncated:
            return value + "\n...[output truncated by math-code executor]"
        return value


def _bounded_repr(value: Any) -> str:
    formatter = reprlib.Repr()
    formatter.maxstring = MAX_CAPTURE_CHARS
    formatter.maxother = MAX_CAPTURE_CHARS
    formatter.maxlist = 1_000
    formatter.maxtuple = 1_000
    formatter.maxset = 1_000
    formatter.maxfrozenset = 1_000
    formatter.maxdict = 1_000
    return formatter.repr(value)


def _initial_namespace() -> dict[str, Any]:
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    # These imports are part of the math-code SIF contract. Keeping aliases in
    # the initial namespace matches the existing math_with_code executor.
    import numpy as np
    import pandas as pd
    import scipy

    namespace.update({"np": np, "numpy": np, "pd": pd, "pandas": pd, "scipy": scipy})
    return namespace


def _set_worker_memory_limit(memory_limit_mb: int) -> None:
    """Bound code allocations above the scientific-library preload baseline."""
    if memory_limit_mb < 1:
        raise ValueError("memory_limit_mb must be at least 1")

    try:
        virtual_pages = int(Path("/proc/self/statm").read_text().split()[0])
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError("Unable to read worker virtual memory baseline") from exc

    baseline_bytes = virtual_pages * os.sysconf("SC_PAGE_SIZE")
    requested_limit = baseline_bytes + memory_limit_mb * MIB
    _, inherited_hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    if inherited_hard_limit != resource.RLIM_INFINITY:
        requested_limit = min(requested_limit, inherited_hard_limit)
    if requested_limit <= baseline_bytes:
        raise RuntimeError(
            "Inherited address-space limit leaves no memory for Python execution"
        )

    resource.setrlimit(resource.RLIMIT_AS, (requested_limit, requested_limit))


def _execute_once(code: str, namespace: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    stdout = _BoundedStringIO()
    stderr = _BoundedStringIO()

    def _handle_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"Python execution exceeded {timeout_sec} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout_sec)
    try:
        module = ast.parse(code, mode="exec")
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result: Any = None
            # Evaluate a trailing expression exactly once, like a notebook.
            if module.body and isinstance(module.body[-1], ast.Expr):
                prefix = ast.Module(body=module.body[:-1], type_ignores=[])
                if prefix.body:
                    exec(compile(prefix, "<math-code-tool>", "exec"), namespace, namespace)
                expression = ast.Expression(module.body[-1].value)
                result = eval(compile(expression, "<math-code-tool>", "eval"), namespace, namespace)
            else:
                exec(compile(module, "<math-code-tool>", "exec"), namespace, namespace)
        return {
            "success": True,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "result": None if result is None else _bounded_repr(result),
            "error_message": None,
        }
    except BaseException as exc:
        reset_worker = isinstance(exc, (MemoryError, TimeoutError))
        if isinstance(exc, MemoryError):
            error_message = "MemoryError: Python execution exceeded its memory limit"
        else:
            error_message = f"{type(exc).__name__}: {exc}"
        return {
            "success": False,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "result": None,
            "error_message": error_message,
            "_reset_worker": reset_worker,
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _worker_main(
    connection: Any, *, timeout_sec: int, memory_limit_mb: int
) -> None:
    # Isolate submitted-code subprocesses in the worker's process group so a
    # timeout/reset can terminate the complete tree rather than only Python.
    os.setsid()
    namespace = _initial_namespace()
    _set_worker_memory_limit(memory_limit_mb)
    while True:
        request = connection.recv()
        if request.get("command") == "shutdown":
            return
        connection.send(_execute_once(str(request.get("code", "")), namespace, timeout_sec))


class SessionWorker:
    def __init__(self, *, timeout_sec: int, memory_limit_mb: int) -> None:
        self.timeout_sec = timeout_sec
        self.memory_limit_mb = memory_limit_mb
        self._process: mp.Process | None = None
        self._connection: Any = None
        self._start()

    def _start(self) -> None:
        parent, child = mp.get_context("fork").Pipe()
        process = mp.get_context("fork").Process(
            target=_worker_main,
            args=(child,),
            kwargs={
                "timeout_sec": self.timeout_sec,
                "memory_limit_mb": self.memory_limit_mb,
            },
            daemon=True,
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process

    def _replace(self) -> None:
        self.close()
        self._start()

    def execute(self, code: str) -> dict[str, Any]:
        try:
            self._connection.send({"command": "execute", "code": code})
            if self._connection.poll(self.timeout_sec + 1):
                result = self._connection.recv()
                reset_worker = bool(result.pop("_reset_worker", False))
                if reset_worker:
                    error_message = result.get("error_message") or "Python worker failed"
                    result["error_message"] = (
                        f"{error_message}; the Python session was reset"
                    )
                    self._replace()
                return result
        except (BrokenPipeError, EOFError, OSError):
            pass

        self._replace()
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "result": None,
            "error_message": (
                f"Python execution exceeded {self.timeout_sec} seconds; "
                "the Python session was reset"
            ),
        }

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                process.terminate()
            process.join(timeout=0.5)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                process.kill()
            process.join(timeout=0.5)
        try:
            self._connection.close()
        except OSError:
            pass
        self._process = None


def _read_request(connection: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise ValueError(f"request exceeds {MAX_REQUEST_BYTES} bytes")
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    payload = b"".join(chunks).split(b"\n", 1)[0]
    return json.loads(payload)


def serve(
    socket_path: str, *, timeout_sec: int, memory_limit_mb: int
) -> None:
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    worker = SessionWorker(
        timeout_sec=timeout_sec, memory_limit_mb=memory_limit_mb
    )
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))
        os.chmod(path, 0o600)
        server.listen(4)
        while True:
            connection, _ = server.accept()
            should_stop = False
            with connection:
                try:
                    request = _read_request(connection)
                    command = request.get("command")
                    if command == "ping":
                        response = {"success": True, "status": "ready"}
                    elif command == "shutdown":
                        response = {"success": True, "status": "stopping"}
                        should_stop = True
                    elif command == "execute":
                        response = worker.execute(str(request.get("code", "")))
                    else:
                        response = {"success": False, "error_message": f"unknown command: {command!r}"}
                except BaseException as exc:
                    response = {"success": False, "error_message": f"{type(exc).__name__}: {exc}"}
                connection.sendall(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
            if should_stop:
                break
    finally:
        worker.close()
        server.close()
        path.unlink(missing_ok=True)


def send_request(socket_path: str, request: dict[str, Any]) -> dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # The caller is already wrapped by Harbor environment.exec's hard timeout.
    # Keep this above the longest supported code timeout so the client does not
    # abandon a healthy worker while it is still calculating.
    client.settimeout(3600)
    try:
        client.connect(socket_path)
        client.sendall(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        return json.loads(b"".join(chunks).split(b"\n", 1)[0])
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--socket", required=True)
    serve_parser.add_argument("--timeout-sec", type=int, required=True)
    serve_parser.add_argument("--memory-limit-mb", type=int, required=True)

    for command in ("ping", "shutdown"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--socket", required=True)

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--socket", required=True)
    execute_parser.add_argument("--code-base64", required=True)

    args = parser.parse_args()
    if args.command == "serve":
        serve(
            args.socket,
            timeout_sec=args.timeout_sec,
            memory_limit_mb=args.memory_limit_mb,
        )
        return
    if args.command == "execute":
        code = base64.urlsafe_b64decode(args.code_base64.encode("ascii")).decode("utf-8")
        request = {"command": "execute", "code": code}
    else:
        request = {"command": args.command}
    print(json.dumps(send_request(args.socket, request), ensure_ascii=False))


if __name__ == "__main__":
    main()

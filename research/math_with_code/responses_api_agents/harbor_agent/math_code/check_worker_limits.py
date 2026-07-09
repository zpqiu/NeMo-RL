# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manually validate math-code worker runtime and memory isolation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_session_worker() -> type[Any]:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "custom_agents"
        / "math_code_session.py"
    )
    spec = importlib.util.spec_from_file_location(
        "math_code_session_limit_check", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load math-code session module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SessionWorker


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trigger memory and runtime limits, then verify that each worker "
            "reset leaves a healthy Python session."
        )
    )
    parser.add_argument("--timeout-sec", type=int, required=True)
    parser.add_argument("--memory-limit-mb", type=int, required=True)
    args = parser.parse_args()

    session_worker = _load_session_worker()
    worker = session_worker(
        timeout_sec=args.timeout_sec,
        memory_limit_mb=args.memory_limit_mb,
    )
    try:
        memory_result = worker.execute(
            "items = []\n"
            "while True:\n"
            "    items.append(bytearray(1024 * 1024))"
        )
        after_memory = worker.execute("40 + 2")
        timeout_result = worker.execute("while True:\n    pass")
        after_timeout = worker.execute("6 * 7")
    finally:
        worker.close()

    results = {
        "python": sys.executable,
        "memory": memory_result,
        "after_memory": after_memory,
        "timeout": timeout_result,
        "after_timeout": after_timeout,
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))

    assert memory_result["success"] is False
    assert "memory limit" in memory_result["error_message"]
    assert "session was reset" in memory_result["error_message"]
    assert after_memory["result"] == "42"
    assert timeout_result["success"] is False
    assert (
        f"exceeded {args.timeout_sec} seconds" in timeout_result["error_message"]
    )
    assert "session was reset" in timeout_result["error_message"]
    assert after_timeout["result"] == "42"
    print("math-code runtime and memory limits: PASS")


if __name__ == "__main__":
    main()

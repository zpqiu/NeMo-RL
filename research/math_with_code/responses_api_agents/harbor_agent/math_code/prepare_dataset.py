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
"""Convert boxed Hugging Face math datasets into local Harbor tasks."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from datasets import Dataset


@dataclass(frozen=True)
class ConversionResult:
    dataset_name: str
    dataset_alias: str
    split: str
    task_count: int
    seed: int | None
    sif_path: str
    tasks_dir: str
    jsonl_path: str


def _extract_instruction(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 1:
        raise ValueError("Expected `prompt` to contain exactly one chat message")
    message = prompt[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("Expected the sole prompt message to have role=user")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Prompt content must be a non-empty string")
    if "\\boxed" not in content:
        raise ValueError("Dataset prompt does not explicitly request a boxed answer")
    return content.strip() + "\n"


def _extract_ground_truth(row: dict[str, Any]) -> str:
    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
        raise ValueError("Expected reward_model.ground_truth in every dataset row")
    ground_truth = reward_model["ground_truth"]
    if ground_truth is None or not str(ground_truth).strip():
        raise ValueError("Ground truth must be non-empty")
    return str(ground_truth)


def _task_toml(sif_path: Path, dataset_name: str) -> str:
    escaped_sif = str(sif_path).replace("\\", "\\\\").replace('"', '\\"')
    escaped_dataset = dataset_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'''version = "1.0"
source = "{escaped_dataset}"

[metadata]
category = "math"
difficulty = "unknown"
tags = ["math", "python", "boxed-answer"]

[agent]
timeout_sec = 240.0

[verifier]
timeout_sec = 30.0

[environment]
docker_image = "{escaped_sif}"
cpus = 1
memory_mb = 2048
storage_mb = 1024
gpus = 0
allow_internet = true
'''


def convert_rows(
    rows: Iterable[tuple[int, dict[str, Any]]],
    *,
    dataset_name: str,
    dataset_alias: str,
    split: str,
    sif_path: Path,
    tasks_dir: Path,
    jsonl_path: Path,
    seed: int | None,
    resume: bool = False,
) -> ConversionResult:
    templates_dir = Path(__file__).with_name("templates")
    tasks_dir.mkdir(parents=True, exist_ok=resume)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    request_rows: list[dict[str, Any]] = []

    for task_index, (source_index, row) in enumerate(rows):
        task_name = f"task_{task_index:06d}"
        task_dir = tasks_dir / task_name
        environment_dir = task_dir / "environment"
        tests_dir = task_dir / "tests"
        required_paths = (
            task_dir / "instruction.md",
            task_dir / "task.toml",
            tests_dir / "test.sh",
            tests_dir / "verify.py",
            tests_dir / "expected_answer.json",
        )
        task_is_complete = resume and environment_dir.is_dir() and all(path.is_file() for path in required_paths)
        if not task_is_complete:
            if task_dir.exists():
                shutil.rmtree(task_dir)
            environment_dir.mkdir(parents=True)
            tests_dir.mkdir()

            (task_dir / "instruction.md").write_text(_extract_instruction(row))
            (task_dir / "task.toml").write_text(_task_toml(sif_path, dataset_name))
            shutil.copy2(templates_dir / "test.sh", tests_dir / "test.sh")
            shutil.copy2(templates_dir / "verify.py", tests_dir / "verify.py")
            (tests_dir / "test.sh").chmod(0o755)
            (tests_dir / "verify.py").chmod(0o755)
            (tests_dir / "expected_answer.json").write_text(
                json.dumps(
                    {
                        "ground_truth": _extract_ground_truth(row),
                        "dataset": dataset_name,
                        "split": split,
                        "source_index": source_index,
                        "reward_style": (row.get("reward_model") or {}).get("style"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        request_rows.append(
            {
                "id": task_index,
                "instance_id": f"{dataset_alias}::{task_name}",
                "agent_ref": {"name": "math_code_harbor_agent"},
                "responses_create_params": {"input": []},
            }
        )

    with jsonl_path.open("w") as handle:
        for request_row in request_rows:
            handle.write(json.dumps(request_row, ensure_ascii=False) + "\n")

    result = ConversionResult(
        dataset_name=dataset_name,
        dataset_alias=dataset_alias,
        split=split,
        task_count=len(request_rows),
        seed=seed,
        sif_path=str(sif_path),
        tasks_dir=str(tasks_dir),
        jsonl_path=str(jsonl_path),
    )
    (tasks_dir / "manifest.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return result


def select_rows(dataset: "Dataset", limit: int | None, seed: int | None) -> list[tuple[int, dict[str, Any]]]:
    indices = list(range(len(dataset)))
    if seed is not None:
        # Shuffle indices rather than the Dataset so source_index continues to
        # identify the original upstream row in verifier diagnostics.
        import random

        random.Random(seed).shuffle(indices)
    if limit is not None:
        indices = indices[:limit]
    return [(index, dict(dataset[index])) for index in indices]


def load_dataset_server_rows(paths: list[Path]) -> list[tuple[int, dict[str, Any]]]:
    """Load pages returned by Hugging Face's official Dataset Server API."""
    rows: list[tuple[int, dict[str, Any]]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        for wrapped_row in payload.get("rows", []):
            source_index = wrapped_row.get("row_idx")
            row = wrapped_row.get("row")
            if not isinstance(source_index, int) or not isinstance(row, dict):
                raise ValueError(f"Invalid Dataset Server row in {path}")
            rows.append((source_index, row))
    rows.sort(key=lambda item: item[0])
    return rows


def load_parquet_rows(paths: list[Path], limit: int | None, seed: int | None) -> list[tuple[int, dict[str, Any]]]:
    """Load locally cached parquet shards without contacting the Hub."""
    resolved_paths = [path.expanduser().resolve() for path in paths]
    for path in resolved_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Parquet shard not found: {path}")

    # datasets is optional for Dataset Server JSON input and intentionally
    # imported only for parquet/Hugging Face-backed conversion.
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in resolved_paths]},
        split="train",
    )
    return select_rows(dataset, limit, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-alias", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--dataset-server-json",
        type=Path,
        action="append",
        default=[],
        help="Offline page downloaded from Hugging Face's Dataset Server API; may be repeated.",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        action="append",
        default=[],
        help="Local parquet shard downloaded with `hf download`; may be repeated.",
    )
    parser.add_argument("--sif-path", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sif_path = args.sif_path.expanduser().resolve()
    if not sif_path.is_file():
        raise FileNotFoundError(f"SIF not found: {sif_path}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.dataset_server_json and args.parquet_path:
        raise ValueError("--dataset-server-json and --parquet-path are mutually exclusive")

    tasks_dir = args.tasks_dir.expanduser().resolve()
    jsonl_path = args.jsonl_path.expanduser().resolve()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if tasks_dir.exists():
        if not (args.overwrite or args.resume):
            raise FileExistsError(f"Tasks directory already exists: {tasks_dir}; pass --overwrite to replace it")
        if args.overwrite:
            shutil.rmtree(tasks_dir)
    if jsonl_path.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"JSONL already exists: {jsonl_path}; pass --overwrite to replace it")

    if args.dataset_server_json:
        rows = load_dataset_server_rows(args.dataset_server_json)
        if args.seed is not None:
            import random

            random.Random(args.seed).shuffle(rows)
        if args.limit is not None:
            rows = rows[: args.limit]
    elif args.parquet_path:
        rows = load_parquet_rows(args.parquet_path, args.limit, args.seed)
    else:
        # datasets is optional for offline Dataset Server JSON conversion.
        from datasets import load_dataset

        dataset = load_dataset(args.dataset, split=args.split)
        rows = select_rows(dataset, args.limit, args.seed)
    result = convert_rows(
        rows,
        dataset_name=args.dataset,
        dataset_alias=args.dataset_alias,
        split=args.split,
        sif_path=sif_path,
        tasks_dir=tasks_dir,
        jsonl_path=jsonl_path,
        seed=args.seed,
        resume=args.resume,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

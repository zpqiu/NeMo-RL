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
"""Convert a plain problem/answer HF dataset into prepare_dataset.py input.

Datasets like math-ai/aime25 ship bare `problem` and `answer` columns. This
tool fetches rows through Hugging Face's Dataset Server API (stdlib-only, so
it runs on login nodes without a venv) and re-emits them as a Dataset Server
JSON page in the boxed source format `prepare_dataset.py --dataset-server-json`
expects, appending the exact instruction line used by the existing train and
eval task trees so prompt templates stay aligned across datasets.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

# Must match the instruction line baked into dapo_math_17k* and aime_2024
# tasks; eval prompts diverging from the train template would skew accuracy.
BOXED_INSTRUCTION = 'Solve the problem step by step and provide the final answer in "\\boxed{...}"'
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100


def fetch_rows(dataset: str, config: str, split: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split,
             "offset": offset, "length": PAGE_SIZE}
        )
        with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=60) as resp:
            payload = json.load(resp)
        page = payload.get("rows", [])
        rows.extend(page)
        if offset + len(page) >= payload.get("num_rows_total", 0) or not page:
            return rows
        offset += len(page)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", required=True)
    parser.add_argument("--problem-field", default="problem")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw_rows = fetch_rows(args.dataset, args.config, args.split)
    if not raw_rows:
        raise SystemExit(f"Dataset Server returned no rows for {args.dataset}:{args.split}")

    converted = []
    for wrapped in raw_rows:
        row = wrapped["row"]
        problem = str(row.get(args.problem_field, "")).strip()
        answer = str(row.get(args.answer_field, "")).strip()
        if not problem or not answer:
            raise SystemExit(f"Row {wrapped.get('row_idx')} is missing problem or answer")
        converted.append(
            {
                "row_idx": wrapped["row_idx"],
                "row": {
                    "prompt": [
                        {
                            "role": "user",
                            "content": f"{problem}\n\n{BOXED_INSTRUCTION}",
                        }
                    ],
                    "reward_model": {"ground_truth": answer},
                },
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": converted}, ensure_ascii=False, indent=2))
    print(f"wrote {len(converted)} rows -> {args.out}")


if __name__ == "__main__":
    main()

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
"""Grade the final assistant answer in an ATIF trajectory with math-verify."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any

from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def _strip_math_delimiters(value: str) -> str:
    value = value.strip()
    if value.startswith("\\(") and value.endswith("\\)"):
        value = value[2:-2].strip()
    if value.startswith("$") and value.endswith("$") and len(value) > 1:
        value = value[1:-1].strip()
    return value


def _assistant_text(trajectory: dict[str, Any]) -> str:
    return "\n".join(
        str(step.get("message", ""))
        for step in trajectory.get("steps", [])
        if step.get("source") == "agent"
    )


def verify(expected: str, generated: str) -> tuple[float, str | None]:
    metric = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    gold = f"\\boxed{{{_strip_math_delimiters(expected)}}}"
    try:
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
            score, extracted = metric([gold], [generated])
        extracted_prediction: str | None = None
        if extracted is not None:
            extracted_gold, extracted_predictions = extracted
            for prediction in extracted_predictions:
                if any(grader.verify(gold_value, prediction) for gold_value in extracted_gold):
                    extracted_prediction = str(prediction)
                    break
            if extracted_prediction is None and extracted_predictions:
                extracted_prediction = str(extracted_predictions[0])
        return float(score), extracted_prediction
    except (Exception, TimeoutException):
        return 0.0, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    args = parser.parse_args()

    logging.getLogger("math_verify").setLevel(logging.CRITICAL)
    details: dict[str, Any] = {"reward": 0.0, "extracted_answer": None}
    try:
        expected_payload = json.loads(args.expected.read_text())
        trajectory = json.loads(args.trajectory.read_text())
        generated = _assistant_text(trajectory)
        reward, extracted = verify(str(expected_payload["ground_truth"]), generated)
        details.update(
            {
                "reward": reward,
                "extracted_answer": extracted,
                "dataset": expected_payload.get("dataset"),
                "source_index": expected_payload.get("source_index"),
            }
        )
    except Exception as exc:
        details["error"] = f"{type(exc).__name__}: {exc}"

    args.details.write_text(json.dumps(details, ensure_ascii=False, indent=2))
    return 0 if details["reward"] > 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())

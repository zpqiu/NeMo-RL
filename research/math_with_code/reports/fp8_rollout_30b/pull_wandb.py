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
"""Pull and stitch the 30B BF16-vs-FP8-rollout chains from wandb into CSVs.

Each 4h Slurm window logs as a separate wandb run under the SAME name, and a
requeued window resumes from the last checkpoint — so step ranges overlap
across segments and the last segment's value for a step is the one on the
surviving lineage. Stitching: order segments by created_at, then let later
segments overwrite earlier ones per step.

Needs WANDB_API_KEY in the environment (e.g. `source ~/.exp_env`).
"""

import csv
from pathlib import Path

import wandb

PROJECT = "nv-welcome/grpo-math-with-code"
CHAINS = {
    "bf16": "grpo-math-code-B64G16-qwen3-30ba3b-instruct-non8-async-v3-pty",
    "fp8_v3": "grpo-math-code-B64G16-qwen3-30ba3b-instruct-non8-async-fp8-v3-pty",
    "fp8_v4": "grpo-math-code-B64G16-qwen3-30ba3b-instruct-non8-async-fp8-v4-pty",
}
METRICS = [
    "validation/accuracy",
    "train/approx_entropy",
    "train/gen_kl_error",
    "train/reward",
    "train/math_code_harbor_agent/num_tool_calls/mean",
    "train/mean_gen_tokens_per_sample",
    "timing/train/total_step_time",
    "timing/train/exposed_generation",
    "performance/generation_tokens_per_sec",
    "performance/generation_tokens_per_sec_per_gpu",
]
OUT_DIR = Path(__file__).parent / "data"


def stitch_chain(api: wandb.Api, run_name: str) -> dict[int, dict[str, float]]:
    """Merge all same-name segments; later segments win on overlapping steps."""
    segments = sorted(
        (r for r in api.runs(PROJECT, per_page=200) if r.name == run_name),
        key=lambda r: r.created_at,
    )
    by_step: dict[int, dict[str, float]] = {}
    for seg in segments:
        for row in seg.scan_history(page_size=500):
            step = row.get("step", row.get("_step"))
            if step is None:
                continue
            dest = by_step.setdefault(int(step), {})
            for metric in METRICS:
                value = row.get(metric)
                if value is not None:
                    dest[metric] = value
    print(f"{run_name}: {len(segments)} segments, {len(by_step)} steps")
    return by_step


def main() -> None:
    import sys

    # One chain per process by default: wandb's local service process has been
    # observed dying after a long scan_history session, killing later chains.
    wanted = sys.argv[1:] or list(CHAINS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    for alias, run_name in ((a, CHAINS[a]) for a in wanted):
        by_step = stitch_chain(api, run_name)
        out_path = OUT_DIR / f"{alias}.csv"
        with out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", *METRICS])
            for step in sorted(by_step):
                row = by_step[step]
                writer.writerow([step, *[row.get(m, "") for m in METRICS]])
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

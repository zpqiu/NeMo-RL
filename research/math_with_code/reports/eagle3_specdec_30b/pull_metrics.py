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
"""Stitch the eagle3 online/frozen chains (and the BF16 reference) into CSVs.

Same stitching semantics as the FP8 report's `pull_wandb.py`: each 4h Slurm
window logs as a separate run under the SAME name and resumes from the last
checkpoint, so step ranges overlap and the later segment's value for a step is
the one on the surviving lineage.

Unlike that script this reads the LOCAL .wandb transaction logs rather than the
W&B API, because the API returned WandbApiFailedError on this campaign's
finished segment runs — with 13 segments per arm, a service failure would drop
whole spans of the record. The local logs carry the same history.

Note for anyone extending this: history items store the full slash-path in
``nested_key`` (a one-element list), not in ``key``. Matching on ``key`` alone
silently yields nothing.
"""

import csv
import glob
import json
import os
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

RESULTS = Path("/lustre/fsw/general_sa/alexq/RL-agent/results")
OUT_DIR = Path(__file__).parent / "data"
CHAINS = {
    "online": "eagle3-online-qwen3-30ba3b-c512-k3-pty",
    "frozen": "eagle3-frozen-qwen3-30ba3b-c512-k3-pty",
    "bf16": "grpo-math-code-B64G16-qwen3-30ba3b-instruct-non8-async-v3-pty",
    # Byte-identical replicate of the bf16 chain, same grpo.seed=42, run to
    # measure how far this MoE + workload reproduces its own trajectory. It also
    # carries the per-request timing split from step 1, which the original chain
    # only gained at step 227.
    "bf16_replicate":
        "grpo-math-code-B64G16-qwen3-30ba3b-instruct-non8-async-v3-rerun-pty",
}
METRICS = [
    "train/vllm/spec_acceptance_length",
    "train/vllm/spec_acceptance_rate",
    "train/draft_loss",
    "validation/accuracy",
    "train/approx_entropy",
    "train/gen_kl_error",
    "train/reward",
    "train/turns_per_sample/mean",
    "train/math_code_harbor_agent/num_tool_calls/mean",
    # Per-stream rate inputs: R = generated_tokens/mean / model_generation_sec/mean
    # (batch sum/sum identity; denominator is the HTTP wall clock — vLLM queue +
    # prefill + decode — with tool execution excluded).
    "train/math_code_harbor_agent/generated_tokens/mean",
    "train/math_code_harbor_agent/model_generation_sec/mean",
    "train/math_code_harbor_agent/num_model_calls/mean",
    # vLLM's own per-request split. The HTTP wall above accumulates per rollout
    # across every turn, so it inflates with tool-use depth even when decode
    # speed is unchanged; only the decode component is what speculation
    # accelerates. Landed partway through the original BF16 chain, so that one
    # only carries the split from step 227; the replicate has it throughout.
    "train/vllm/request_decode_time_mean_s",
    "train/vllm/request_prefill_time_mean_s",
    "train/vllm/request_queue_time_mean_s",
    "train/mean_gen_tokens_per_sample",
    "validation/timing/rollout/total",
    "timing/train/total_step_time",
    "timing/train/exposed_generation",
    "performance/generation_tokens_per_sec",
]


def stitch_chain(run_name: str) -> dict[int, dict[str, float]]:
    """Merge all of a chain's segments; later segments win on overlapping steps."""
    by_step: dict[int, dict[str, float]] = {}
    pattern = str(RESULTS / run_name / "training/exp_*/wandb/wandb/run-*/run-*.wandb")
    for path in sorted(glob.glob(pattern), key=os.path.getmtime):
        store = datastore.DataStore()
        try:
            store.open_for_scan(path)
        except Exception as exc:  # noqa: BLE001 - a torn segment must not drop the chain
            print(f"  skip {Path(path).name}: {type(exc).__name__}")
            continue
        while True:
            try:
                data = store.scan_data()
            except Exception:  # noqa: BLE001 - live segment has a truncated tail
                break
            if data is None:
                break
            record = pb.Record()
            try:
                record.ParseFromString(data)
            except Exception:  # noqa: BLE001
                continue
            if record.WhichOneof("record_type") != "history":
                continue
            dest = by_step.setdefault(record.history.step.num, {})
            for item in record.history.item:
                name = item.key or "/".join(item.nested_key)
                if name in METRICS:
                    try:
                        dest[name] = json.loads(item.value_json)
                    except Exception:  # noqa: BLE001
                        pass
    return by_step


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for alias, run_name in CHAINS.items():
        by_step = stitch_chain(run_name)
        out = OUT_DIR / f"{alias}.csv"
        with open(out, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step"] + METRICS)
            for step in sorted(by_step):
                row = by_step[step]
                writer.writerow([step] + [row.get(m, "") for m in METRICS])
        print(f"wrote {out} ({len(by_step)} steps)")


if __name__ == "__main__":
    main()

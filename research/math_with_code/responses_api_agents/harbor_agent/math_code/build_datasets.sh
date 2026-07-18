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
# Build all three supported math-code datasets in one pass:
#   dapo_math_17k_non8  (train, tool-shaped, 6389 tasks)
#   aime_2024           (val, 30 tasks)
#   aime_2025           (val, 30 tasks)
# The only input is MATH_CODE_SIF_PATH (its absolute path is written into
# every task.toml, so rerun this script if the SIF moves). Run inside the
# training container so uv and the datasets library are available; needs
# Hugging Face network access. For custom datasets, drive prepare_dataset.py
# directly instead.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARBOR_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="$HARBOR_ROOT/data/math_code"

SIF_PATH="${MATH_CODE_SIF_PATH:?set MATH_CODE_SIF_PATH to the shared math-code SIF path}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SOURCES_DIR="$HF_HOME/math_code_sources"

# Train source: DAPO-17k filtered to prompts Qwen3-8B did not solve 8/8.
TRAIN_REPO="alex-chiu/DAPO-Math-17k-Qwen3-8B-non8"
TRAIN_PARQUET="data/train-00000-of-00001.parquet"
TRAIN_ALIAS="dapo_math_17k_non8"
TRAIN_TASKS=6389

log() {
    printf '[math-code-datasets] %s\n' "$*"
}

fail() {
    log "ERROR: $*" >&2
    exit 1
}

check_dataset() {
    local alias="$1"
    local expected="$2"
    local actual

    actual="$(find "$DATA_ROOT/$alias" -mindepth 1 -maxdepth 1 -type d \
        -name 'task_[0-9][0-9][0-9][0-9][0-9][0-9]' -printf '.\n' | wc -l)"
    [[ "$actual" -eq "$expected" ]] || \
        fail "expected $expected tasks for $alias, found $actual"
    [[ "$(wc -l <"$DATA_ROOT/$alias.jsonl")" -eq "$expected" ]] || \
        fail "request JSONL for $alias does not contain $expected rows"
    log "OK $alias tasks=$expected"
}

[[ -f "$SIF_PATH" ]] || fail "math-code SIF is missing: $SIF_PATH"
command -v uv >/dev/null || fail "uv is missing; run this script inside the training container"
mkdir -p "$DATA_ROOT" "$SOURCES_DIR"
export HF_HOME
export HF_HUB_OFFLINE=0

log "[1/3] $TRAIN_ALIAS: download $TRAIN_REPO"
uv run hf download "$TRAIN_REPO" "$TRAIN_PARQUET" \
    --repo-type dataset \
    --local-dir "$SOURCES_DIR/$TRAIN_ALIAS"
uv run python "$SCRIPT_DIR/prepare_dataset.py" \
    --dataset "$TRAIN_REPO" \
    --dataset-alias "$TRAIN_ALIAS" \
    --split train \
    --parquet-path "$SOURCES_DIR/$TRAIN_ALIAS/$TRAIN_PARQUET" \
    --sif-path "$SIF_PATH" \
    --tasks-dir "$DATA_ROOT/$TRAIN_ALIAS" \
    --jsonl-path "$DATA_ROOT/$TRAIN_ALIAS.jsonl" \
    --tool-shaping \
    --overwrite
check_dataset "$TRAIN_ALIAS" "$TRAIN_TASKS"

log "[2/3] aime_2024: convert tongyx361/AIME-2024-Boxed"
uv run python "$SCRIPT_DIR/prepare_dataset.py" \
    --dataset tongyx361/AIME-2024-Boxed \
    --dataset-alias aime_2024 \
    --split train \
    --sif-path "$SIF_PATH" \
    --tasks-dir "$DATA_ROOT/aime_2024" \
    --jsonl-path "$DATA_ROOT/aime_2024.jsonl" \
    --overwrite
check_dataset aime_2024 30

log "[3/3] aime_2025: convert math-ai/aime25 (plain problem/answer format)"
uv run python "$SCRIPT_DIR/convert_plain_problem_answer.py" \
    --dataset math-ai/aime25 \
    --split test \
    --out "$SOURCES_DIR/aime_2025/rows.json"
uv run python "$SCRIPT_DIR/prepare_dataset.py" \
    --dataset math-ai/aime25 \
    --dataset-alias aime_2025 \
    --split test \
    --dataset-server-json "$SOURCES_DIR/aime_2025/rows.json" \
    --sif-path "$SIF_PATH" \
    --tasks-dir "$DATA_ROOT/aime_2025" \
    --jsonl-path "$DATA_ROOT/aime_2025.jsonl" \
    --overwrite
check_dataset aime_2025 30

log "PASS all datasets under $DATA_ROOT"

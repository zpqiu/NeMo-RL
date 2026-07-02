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
# Download the boxed DAPO parquet through the Hugging Face CLI, convert it on
# node-local storage, and publish only one archive plus one request JSONL.
# Run this inside nemo_rl.0627.sqsh so uv, datasets, and pyarrow are available.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
HARBOR_ROOT="$PROJECT_ROOT/responses_api_agents/harbor_agent"
DATA_ROOT="$HARBOR_ROOT/data/math_code"

DATASET_REPO="${MATH_CODE_DATASET_REPO:-tongyx361/DAPO-Math-Unique-Boxed-17k}"
DATASET_REVISION="${MATH_CODE_DATASET_REVISION:-refs/convert/parquet}"
DATASET_PARQUET="${MATH_CODE_DATASET_PARQUET:-default/train/0000.parquet}"
DATASET_ALIAS="${MATH_CODE_DATASET_ALIAS:-dapo_math_17k}"
EXPECTED_TASKS="${MATH_CODE_EXPECTED_TASKS:-17398}"
SIF_PATH="${MATH_CODE_SIF_PATH:?set MATH_CODE_SIF_PATH to the shared math-code SIF path}"
BUILD_ROOT="${MATH_CODE_BUILD_ROOT:-/tmp/math_code_17k_build_${SLURM_JOB_ID:-$$}}"
TASKS_DIR="$BUILD_ROOT/$DATASET_ALIAS"
JSONL_PATH="${MATH_CODE_JSONL_PATH:-$DATA_ROOT/${DATASET_ALIAS}.jsonl}"
ARCHIVE_PATH="${MATH_CODE_TASKS_ARCHIVE:-$DATA_ROOT/${DATASET_ALIAS}.tar.gz}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
DOWNLOAD_DIR="${MATH_CODE_DOWNLOAD_DIR:-$HF_HOME/math_code_sources/$DATASET_ALIAS}"

log() {
    printf '[math-code-prepare] %s\n' "$*"
}

fail() {
    log "ERROR: $*" >&2
    exit 1
}

cleanup() {
    if [[ "${MATH_CODE_KEEP_BUILD_TREE:-0}" != "1" ]]; then
        rm -rf -- "$BUILD_ROOT"
    fi
}
trap cleanup EXIT

[[ "$EXPECTED_TASKS" =~ ^[1-9][0-9]*$ ]] || fail "MATH_CODE_EXPECTED_TASKS must be a positive integer"
[[ -f "$SIF_PATH" ]] || fail "math-code SIF is missing: $SIF_PATH"
command -v uv >/dev/null || fail "uv is missing; run this script inside nemo_rl.0627.sqsh"
case "$ARCHIVE_PATH" in
    *.tar.gz | *.tgz) command -v gzip >/dev/null || fail "gzip is required to create $ARCHIVE_PATH" ;;
    *.tar.zst) command -v zstd >/dev/null || fail "zstd is required to create $ARCHIVE_PATH" ;;
    *) fail "unsupported archive extension: $ARCHIVE_PATH" ;;
esac
mkdir -p "$BUILD_ROOT" "$DATA_ROOT" "$HF_HOME" "$DOWNLOAD_DIR"

log "cache dataset=$DATASET_REPO revision=$DATASET_REVISION file=$DATASET_PARQUET"
export HF_HOME
export HF_HUB_OFFLINE=0
uv run hf download "$DATASET_REPO" "$DATASET_PARQUET" \
    --repo-type dataset \
    --revision "$DATASET_REVISION" \
    --local-dir "$DOWNLOAD_DIR"
PARQUET_PATH="$DOWNLOAD_DIR/$DATASET_PARQUET"
[[ -f "$PARQUET_PATH" ]] || fail "hf download did not return a parquet path: $PARQUET_PATH"

log "convert parquet=$PARQUET_PATH tasks=$EXPECTED_TASKS local_build=$TASKS_DIR"
uv run python "$HARBOR_ROOT/math_code/prepare_dataset.py" \
    --dataset "$DATASET_REPO" \
    --dataset-alias "$DATASET_ALIAS" \
    --split train \
    --parquet-path "$PARQUET_PATH" \
    --sif-path "$SIF_PATH" \
    --tasks-dir "$TASKS_DIR" \
    --jsonl-path "$JSONL_PATH" \
    --overwrite

ACTUAL_TASKS="$(find "$TASKS_DIR" -mindepth 1 -maxdepth 1 -type d \
    -name 'task_[0-9][0-9][0-9][0-9][0-9][0-9]' -printf '.\n' | wc -l)"
[[ "$ACTUAL_TASKS" -eq "$EXPECTED_TASKS" ]] || \
    fail "expected $EXPECTED_TASKS converted tasks, found $ACTUAL_TASKS"
[[ "$(wc -l <"$JSONL_PATH")" -eq "$EXPECTED_TASKS" ]] || \
    fail "request JSONL does not contain $EXPECTED_TASKS rows: $JSONL_PATH"

log "package archive=$ARCHIVE_PATH"
MATH_CODE_DATASET_ALIAS="$DATASET_ALIAS" \
MATH_CODE_TASKS_SOURCE_DIR="$TASKS_DIR" \
MATH_CODE_TASKS_ARCHIVE="$ARCHIVE_PATH" \
MATH_CODE_EXPECTED_TASKS="$EXPECTED_TASKS" \
    "$SCRIPT_DIR/package_math_code_dataset.sh"

log "PASS jsonl=$JSONL_PATH archive=$ARCHIVE_PATH"

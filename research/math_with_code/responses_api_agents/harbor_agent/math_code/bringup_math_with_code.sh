#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Run the complete non-training math-with-code bring-up inside a compute-node
# NeMo-RL container: build the shared Harbor venv, materialize/reuse datasets,
# then execute one real Harbor trial against the local fake Responses API.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
source "$PROJECT_ROOT/math_code_paths.sh"
SIF_PATH="$MATH_CODE_SIF_PATH"
VENV_ROOT="$NEMO_GYM_VENV_DIR"

log() {
    printf '[math-code-bringup] %s\n' "$*"
}

fail() {
    printf '[math-code-bringup] ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -f "$SIF_PATH" ]] || fail "math-code SIF is missing: $SIF_PATH"
SIF_PATH="$(readlink -f "$SIF_PATH")"
[[ "$SIF_PATH" == /* ]] || fail "math-code SIF must resolve to an absolute path: $SIF_PATH"
[[ -c /dev/fuse ]] || fail "/dev/fuse is unavailable; add /dev/fuse:/dev/fuse to the container mounts"
command -v uv >/dev/null 2>&1 || fail "uv is missing; run inside the NeMo-RL training container"
command -v singularity >/dev/null 2>&1 || fail "singularity is missing; use a recent NeMo-RL training image"

export MATH_CODE_SIF_PATH="$SIF_PATH"
export NEMO_GYM_VENV_DIR="$VENV_ROOT"

log "SIF: $MATH_CODE_SIF_PATH"
log "Harbor venv root: $NEMO_GYM_VENV_DIR"
log "[1/3] build or refresh Harbor venv"
"$SCRIPT_DIR/build_harbor_venv.sh"
log "[2/3] build or verify datasets"
"$SCRIPT_DIR/build_datasets.sh"
log "[3/3] execute one real Harbor runtime trial"
"$SCRIPT_DIR/validate_runtime.sh"
log "PASS"

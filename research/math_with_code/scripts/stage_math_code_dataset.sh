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
# Stage the inode-heavy Harbor math-code task tree on node-local storage.
# ray.sub runs this script once on every allocated node before Ray starts.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATASET_ALIAS="${MATH_CODE_DATASET_ALIAS:-dapo_math_17k}"
ARCHIVE="${MATH_CODE_TASKS_ARCHIVE:-$PROJECT_ROOT/responses_api_agents/harbor_agent/data/math_code/${DATASET_ALIAS}.tar.gz}"
TARGET_DIR="${MATH_CODE_TASKS_DIR:-/tmp/nemo_rl_math_code/$DATASET_ALIAS}"
EXPECTED_TASKS="${MATH_CODE_EXPECTED_TASKS:-17398}"

log() {
    printf '[math-code-stage] host=%s %s\n' "$(hostname)" "$*"
}

fail() {
    log "ERROR: $*" >&2
    exit 1
}

count_paths() {
    local root="$1"
    local min_depth="$2"
    local max_depth="$3"
    local path_type="$4"
    local name="$5"

    find "$root" -mindepth "$min_depth" -maxdepth "$max_depth" \
        -type "$path_type" -name "$name" -printf '.\n' | wc -l
}

validate_dataset() {
    local dataset_dir="$1"
    local task_count
    local path_count

    [[ -f "$dataset_dir/manifest.json" ]] || fail "manifest is missing from $dataset_dir"
    task_count="$(count_paths "$dataset_dir" 1 1 d 'task_[0-9][0-9][0-9][0-9][0-9][0-9]')"
    [[ "$task_count" -eq "$EXPECTED_TASKS" ]] || \
        fail "expected $EXPECTED_TASKS task directories in $dataset_dir, found $task_count"

    for required_file in instruction.md task.toml; do
        path_count="$(count_paths "$dataset_dir" 2 2 f "$required_file")"
        [[ "$path_count" -eq "$EXPECTED_TASKS" ]] || \
            fail "expected $EXPECTED_TASKS $required_file files, found $path_count"
    done
    path_count="$(count_paths "$dataset_dir" 2 2 d environment)"
    [[ "$path_count" -eq "$EXPECTED_TASKS" ]] || \
        fail "expected $EXPECTED_TASKS environment directories, found $path_count"
    path_count="$(count_paths "$dataset_dir" 2 2 d tests)"
    [[ "$path_count" -eq "$EXPECTED_TASKS" ]] || \
        fail "expected $EXPECTED_TASKS tests directories, found $path_count"
    for required_file in test.sh verify.py expected_answer.json; do
        path_count="$(count_paths "$dataset_dir" 3 3 f "$required_file")"
        [[ "$path_count" -eq "$EXPECTED_TASKS" ]] || \
            fail "expected $EXPECTED_TASKS tests/$required_file files, found $path_count"
    done
}

[[ "$EXPECTED_TASKS" =~ ^[1-9][0-9]*$ ]] || fail "MATH_CODE_EXPECTED_TASKS must be a positive integer"
[[ -r "$ARCHIVE" ]] || fail "dataset archive is missing or unreadable: $ARCHIVE"
command -v flock >/dev/null || fail "flock is required to serialize node-local staging"
command -v sha256sum >/dev/null || fail "sha256sum is required to validate the staged archive"
command -v tar >/dev/null || fail "tar is required to extract the staged archive"
case "$ARCHIVE" in
    *.tar.gz | *.tgz)
        command -v gzip >/dev/null || fail "gzip is required to extract $ARCHIVE"
        TAR_EXTRACT_ARGS=(-xzf)
        ;;
    *.tar.zst)
        command -v zstd >/dev/null || fail "zstd is required to extract $ARCHIVE"
        TAR_EXTRACT_ARGS=(--zstd -xf)
        ;;
    *)
        fail "unsupported archive extension: $ARCHIVE (expected .tar.gz, .tgz, or .tar.zst)"
        ;;
esac

TARGET_PARENT="$(dirname -- "$TARGET_DIR")"
TARGET_NAME="$(basename -- "$TARGET_DIR")"
[[ "$TARGET_NAME" == "$DATASET_ALIAS" ]] || \
    fail "MATH_CODE_TASKS_DIR must end with /$DATASET_ALIAS, got $TARGET_DIR"
mkdir -p "$TARGET_PARENT"

# SETUP_COMMAND can be retried if Ray fails to start. Serialize retries and use
# an archive hash marker so a healthy node-local copy is reused safely.
exec 9>"$TARGET_PARENT/.${DATASET_ALIAS}.stage.lock"
flock 9
ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
MARKER="$TARGET_DIR/.archive.sha256"
if [[ -f "$MARKER" ]] && [[ "$(<"$MARKER")" == "$ARCHIVE_SHA256" ]]; then
    validate_dataset "$TARGET_DIR"
    log "reuse target=$TARGET_DIR tasks=$EXPECTED_TASKS archive_sha256=$ARCHIVE_SHA256"
    exit 0
fi

AVAILABLE_INODES="$(df -Pi "$TARGET_PARENT" | awk 'NR == 2 {print $4}')"
REQUIRED_INODES=$((EXPECTED_TASKS * 8 + 16))
[[ "$AVAILABLE_INODES" -ge "$REQUIRED_INODES" ]] || \
    fail "not enough local inodes under $TARGET_PARENT: need at least $REQUIRED_INODES, have $AVAILABLE_INODES"

STAGING_DIR="$(mktemp -d "$TARGET_PARENT/.${DATASET_ALIAS}.stage.XXXXXX")"
STALE_DIR="$TARGET_PARENT/.${DATASET_ALIAS}.stale.$$"
cleanup() {
    rm -rf -- "$STAGING_DIR" "$STALE_DIR"
}
trap cleanup EXIT

log "extract archive=$ARCHIVE target=$TARGET_DIR archive_sha256=$ARCHIVE_SHA256"
tar "${TAR_EXTRACT_ARGS[@]}" "$ARCHIVE" -C "$STAGING_DIR"
EXTRACTED_DIR="$STAGING_DIR/$DATASET_ALIAS"
[[ -d "$EXTRACTED_DIR" ]] || \
    fail "archive must contain one top-level $DATASET_ALIAS directory"
validate_dataset "$EXTRACTED_DIR"
printf '%s\n' "$ARCHIVE_SHA256" >"$EXTRACTED_DIR/.archive.sha256"

if [[ -e "$TARGET_DIR" ]]; then
    mv -- "$TARGET_DIR" "$STALE_DIR"
fi
if ! mv -- "$EXTRACTED_DIR" "$TARGET_DIR"; then
    if [[ -e "$STALE_DIR" ]]; then
        mv -- "$STALE_DIR" "$TARGET_DIR"
    fi
    fail "could not atomically install staged dataset at $TARGET_DIR"
fi

log "PASS target=$TARGET_DIR tasks=$EXPECTED_TASKS archive_sha256=$ARCHIVE_SHA256"

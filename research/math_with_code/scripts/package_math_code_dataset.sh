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
# Package a generated Harbor task tree into one shared-filesystem inode.
set -euo pipefail

DATASET_ALIAS="${MATH_CODE_DATASET_ALIAS:-dapo_math_17k}"
SOURCE_DIR="${MATH_CODE_TASKS_SOURCE_DIR:?export MATH_CODE_TASKS_SOURCE_DIR to the generated task directory}"
ARCHIVE="${MATH_CODE_TASKS_ARCHIVE:?export MATH_CODE_TASKS_ARCHIVE to the output archive path}"
EXPECTED_TASKS="${MATH_CODE_EXPECTED_TASKS:-17398}"

fail() {
    printf '[math-code-package] ERROR: %s\n' "$*" >&2
    exit 1
}

[[ "$EXPECTED_TASKS" =~ ^[1-9][0-9]*$ ]] || fail "MATH_CODE_EXPECTED_TASKS must be a positive integer"
[[ -d "$SOURCE_DIR" ]] || fail "source task directory does not exist: $SOURCE_DIR"
[[ "$(basename -- "$SOURCE_DIR")" == "$DATASET_ALIAS" ]] || \
    fail "source directory must end with /$DATASET_ALIAS"
[[ -f "$SOURCE_DIR/manifest.json" ]] || fail "source manifest is missing: $SOURCE_DIR/manifest.json"
command -v tar >/dev/null || fail "tar is required to create $ARCHIVE"

case "$ARCHIVE" in
    *.tar.gz | *.tgz)
        command -v gzip >/dev/null || fail "gzip is required to create $ARCHIVE"
        TAR_CREATE_ARGS=(-czf)
        ;;
    *.tar.zst)
        command -v zstd >/dev/null || fail "zstd is required to create $ARCHIVE"
        TAR_CREATE_ARGS=(-I 'zstd -T0 -10' -cf)
        ;;
    *)
        fail "unsupported archive extension: $ARCHIVE (expected .tar.gz, .tgz, or .tar.zst)"
        ;;
esac

TASK_COUNT="$(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d \
    -name 'task_[0-9][0-9][0-9][0-9][0-9][0-9]' -printf '.\n' | wc -l)"
[[ "$TASK_COUNT" -eq "$EXPECTED_TASKS" ]] || \
    fail "expected $EXPECTED_TASKS tasks in $SOURCE_DIR, found $TASK_COUNT"

mkdir -p "$(dirname -- "$ARCHIVE")"
TEMP_ARCHIVE="${ARCHIVE}.tmp.$$"
cleanup() {
    rm -f -- "$TEMP_ARCHIVE"
}
trap cleanup EXIT

tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
    "${TAR_CREATE_ARGS[@]}" "$TEMP_ARCHIVE" \
    -C "$(dirname -- "$SOURCE_DIR")" "$DATASET_ALIAS"
case "$ARCHIVE" in
    *.tar.gz | *.tgz) gzip --test "$TEMP_ARCHIVE" ;;
    *.tar.zst) zstd --test --quiet "$TEMP_ARCHIVE" ;;
esac
mv -- "$TEMP_ARCHIVE" "$ARCHIVE"
printf '[math-code-package] PASS archive=%s tasks=%s sha256=%s\n' \
    "$ARCHIVE" "$EXPECTED_TASKS" "$(sha256sum "$ARCHIVE" | awk '{print $1}')"

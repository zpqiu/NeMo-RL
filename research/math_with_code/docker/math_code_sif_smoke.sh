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
set -euo pipefail

test "$(uname -m)" = "aarch64"
command -v timeout
python3 -c 'import fastapi, uvicorn, pydantic, numpy, scipy, pandas; from math_verify import grader; from math_verify.errors import TimeoutException; from math_verify.metric import math_metric; from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig; print("runtime smoke test ok")'
touch /app/write-test
test -f /app/write-test

# Exercise the same persistent Python session used by MathCodeHarborAgent.
SESSION_SCRIPT=/staging/math_code_session.py
SESSION_SOCKET=/app/.smoke-math-code.sock
SESSION_LOG=/app/.smoke-math-code.log
test -f "$SESSION_SCRIPT"
python3 "$SESSION_SCRIPT" serve \
    --socket "$SESSION_SOCKET" \
    --timeout-sec 3 \
    >"$SESSION_LOG" 2>&1 &
session_pid=$!
trap 'kill "$session_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
    if python3 "$SESSION_SCRIPT" ping --socket "$SESSION_SOCKET" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
python3 "$SESSION_SCRIPT" ping --socket "$SESSION_SOCKET"

first_code=$(printf 'x = 40\nprint(x)' | base64 -w0)
second_code=$(printf 'x + 2' | base64 -w0)
python3 "$SESSION_SCRIPT" execute --socket "$SESSION_SOCKET" --code-base64 "$first_code" \
    | python3 -c 'import json, sys; result=json.load(sys.stdin); assert result["stdout"] == "40\n"'
python3 "$SESSION_SCRIPT" execute --socket "$SESSION_SOCKET" --code-base64 "$second_code" \
    | python3 -c 'import json, sys; result=json.load(sys.stdin); assert result["result"] == "42"'
python3 "$SESSION_SCRIPT" shutdown --socket "$SESSION_SOCKET"
wait "$session_pid"
trap - EXIT
echo "persistent Python session smoke test ok"

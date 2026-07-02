#!/usr/bin/env bash
set -u

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

# A verifier timeout must never leave Harbor waiting forever or without a
# reward file. Harbor also applies its task-level timeout outside this process.
if timeout --signal=TERM 20s python3 /tests/verify.py \
    --expected /tests/expected_answer.json \
    --trajectory /logs/agent/trajectory.json \
    --details /logs/verifier/details.json; then
    echo 1 > /logs/verifier/reward.txt
fi

exit 0

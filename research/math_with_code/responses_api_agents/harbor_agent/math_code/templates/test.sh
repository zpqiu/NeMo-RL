#!/usr/bin/env bash
set -u

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

# verify.py writes the (possibly shaped, fractional) reward itself; the
# pre-written 0 stays in place if it crashes or times out. A verifier timeout
# must never leave Harbor waiting forever or without a reward file.
timeout --signal=TERM 20s python3 /tests/verify.py \
    --expected /tests/expected_answer.json \
    --trajectory /logs/agent/trajectory.json \
    --details /logs/verifier/details.json \
    --reward /logs/verifier/reward.txt

exit 0

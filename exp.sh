#!/bin/bash
# exp.sh — Unified experiment launcher
#
# Usage:
#   ./exp.sh <experiment.sh>                    # submit once
#   ./exp.sh <experiment.sh> -n 3               # chain 3 jobs
#   ./exp.sh <experiment.sh> --dep 9871363      # depend on existing job
#   ./exp.sh <experiment.sh> --dep 9871363 -n 3 # chain 3 after existing job
#   ./exp.sh <experiment.sh> -p batch          # use batch partition (4h)
#   ./exp.sh <experiment.sh> -p batch -n 3     # chain 3 on batch partition
#
# Log directory structure:
#   results/<EXP_NAME>/
#     ├── <JOBID>-logs/           ← per-job logs (created by ray.sub)
#     ├── latest -> <JOBID>-logs  ← symlink to most recent job
#     ├── jobs.log                ← job chain history
#     └── experiment.meta         ← git hash, submit time, etc.
#
# Experiment files must set: EXP_NAME, COMMAND
# Experiment files may set:  NUM_NODES, PARTITION, TIME, CONTAINER, MOUNTS, GPUS_PER_NODE
# See experiments/ for examples.

set -euo pipefail

usage() {
  echo "Usage: $0 <experiment.sh> [-n N_CALLS] [--dep JOBID] [-p PARTITION]"
  exit 1
}

[[ $# -lt 1 ]] && usage

EXP_FILE="$1"; shift
[[ ! -f "$EXP_FILE" ]] && { echo "Not found: $EXP_FILE"; exit 1; }

N_CALLS=1
DEP_JOBID=""
CLI_PARTITION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) N_CALLS="$2"; shift 2;;
    --dep) DEP_JOBID="$2"; shift 2;;
    -p) CLI_PARTITION="$2"; shift 2;;
    *) usage;;
  esac
done

# ── Load env/secrets (provides WANDB_API_KEY, ACCOUNT, CONTAINER, MOUNTS, etc.) ──
#
# Required variables in ~/.exp_env:
#   WANDB_API_KEY   — Weights & Biases API key
#   ACCOUNT         — SLURM account name
#   CONTAINER       — Default container image path (.sqsh)
#   MOUNTS          — Default bind mounts (src:dst)

[[ -f ~/.exp_env ]] && source ~/.exp_env

for var in WANDB_API_KEY ACCOUNT MOUNTS; do
  [[ -z "${!var:-}" ]] && { echo "Error: $var not set. Add it to ~/.exp_env"; exit 1; }
done

export WANDB_API_KEY

# ── Defaults (experiment file can override any of these) ──────────

NUM_NODES=1
PARTITION=batch
TIME="4:0:0"
GPUS_PER_NODE=8

EXP_NAME=""   # REQUIRED
COMMAND=""    # REQUIRED — the full command to run inside the container

# ── Load experiment definition ────────────────────────────────────
# Temporarily disable errexit: `read -r -d ''` returns non-zero at EOF,
# which is the normal way heredocs work for setting variables.

set +e
source "$EXP_FILE"
set -e

[[ -z "$EXP_NAME" ]] && { echo "Error: EXP_NAME not set in $EXP_FILE"; exit 1; }
[[ -z "$COMMAND" ]] && { echo "Error: COMMAND not set in $EXP_FILE"; exit 1; }
[[ -z "${CONTAINER:-}" ]] && { echo "Error: CONTAINER not set. Set it in $EXP_FILE or ~/.exp_env"; exit 1; }

# ── Partition override (CLI -p wins over experiment file) ─────────

if [[ -n "$CLI_PARTITION" ]]; then
  PARTITION="$CLI_PARTITION"
fi

# Default time limits per partition (only applied if TIME was not explicitly set in experiment file)
declare -A PARTITION_TIME=( [batch_short]="2:0:0" [batch]="4:0:0" )
if [[ -n "${PARTITION_TIME[$PARTITION]:-}" ]]; then
  TIME="${PARTITION_TIME[$PARTITION]}"
fi

export COMMAND

# ── Results / log directory ───────────────────────────────────────

RESULTS_DIR="results/${EXP_NAME}"
export BASE_LOG_DIR="$(readlink -f .)/${RESULTS_DIR}"
mkdir -p "$BASE_LOG_DIR"

# ── Save experiment metadata ──────────────────────────────────────

cat > "$BASE_LOG_DIR/experiment.meta" <<EOF
experiment: ${EXP_NAME}
source: ${EXP_FILE}
submitted: $(date -Iseconds)
git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
git_dirty: $(git diff --quiet 2>/dev/null && echo no || echo yes)
num_nodes: ${NUM_NODES}
partition: ${PARTITION}
time_limit: ${TIME}
container: ${CONTAINER}
---
command: |
$(echo "$COMMAND" | sed 's/^/  /')
EOF

# ── Submit job chain ──────────────────────────────────────────────

PREV_JOBID="$DEP_JOBID"

for (( i = 1; i <= N_CALLS; i++ )); do
  DEP_OPT=""
  [[ -n "$PREV_JOBID" ]] && DEP_OPT="--dependency=afterany:${PREV_JOBID}"

  echo "[${i}/${N_CALLS}] Submitting ${EXP_NAME}${PREV_JOBID:+ (after ${PREV_JOBID})}"

  OUTPUT=$(CONTAINER="$CONTAINER" \
    MOUNTS="$MOUNTS" \
    BASE_LOG_DIR="$BASE_LOG_DIR" \
    GPUS_PER_NODE="$GPUS_PER_NODE" \
    sbatch \
      ${DEP_OPT} \
      --nodes=${NUM_NODES} \
      --account=${ACCOUNT} \
      --job-name="nemo-rl.${EXP_NAME}" \
      --partition=${PARTITION} \
      --time=${TIME} \
      --gres=gpu:${GPUS_PER_NODE} \
      --output="${BASE_LOG_DIR}/%j-slurm.out" \
      ray.sub)

  PREV_JOBID=$(echo "$OUTPUT" | awk '{print $4}')
  echo "  -> Job ${PREV_JOBID}"

  # Append to job chain log
  echo "${PREV_JOBID} $(date -Iseconds)" >> "$BASE_LOG_DIR/jobs.log"
done

echo ""
echo "Results: ${RESULTS_DIR}/"
echo "Logs:    tail -f ${RESULTS_DIR}/latest/ray-driver.log"

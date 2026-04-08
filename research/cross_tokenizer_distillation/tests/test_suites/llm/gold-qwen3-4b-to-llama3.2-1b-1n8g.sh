#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../../../../..)

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=240
# ===== END CONFIG =====

set -eou pipefail

MODEL_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/alexq/models

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
CKPT_DIR=$EXP_DIR/ckpts
RUN_LOG=$EXP_DIR/run.log

# Clean previous checkpoints to start fresh
rm -rf $CKPT_DIR
mkdir -p $EXP_DIR $LOG_DIR $CKPT_DIR

export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/research/cross_tokenizer_distillation:${PYTHONPATH:-}
export NRL_FORCE_REBUILD_VENVS=true

cd $PROJECT_ROOT

uv run python research/cross_tokenizer_distillation/run_gold.py \
    --config research/cross_tokenizer_distillation/configs/gold_math.yaml \
    gold_distillation.max_num_steps=$MAX_STEPS \
    policy.model_name=$MODEL_DIR/meta-llama/Llama-3.2-1B-Instruct \
    teacher.model_name=$MODEL_DIR/Qwen/Qwen3-4B-Instruct-2507 \
    loss_fn.jsd_beta=0.0 \
    loss_fn.matched_weight=1.0 \
    loss_fn.unmatched_weight=1.0 \
    loss_fn.temperature=1.0 \
    gold_distillation.teacher_topk_k=1024 \
    policy.optimizer.kwargs.lr=1e-5 \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.tensorboard_enabled=False \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.num_nodes=$NUM_NODES \
    2>&1 | tee $RUN_LOG

echo ""
echo "============================================"
echo "  GOLD Distillation: Qwen3-4B -> Llama-3.2-1B"
echo "  Algorithm: JSD (matched) + Sorted L1 (unmatched)"
echo "============================================"

echo ""
echo "=== Validation Accuracy ==="
grep -E "(Accuracy|accuracy)" $RUN_LOG || echo "No accuracy found"

echo ""
echo "=== Training Loss ==="
grep "Loss:" $RUN_LOG || echo "No loss found"

echo ""
echo "=== GOLD Loss Components ==="
grep "Matched JSD" $RUN_LOG || echo "No GOLD loss components found"

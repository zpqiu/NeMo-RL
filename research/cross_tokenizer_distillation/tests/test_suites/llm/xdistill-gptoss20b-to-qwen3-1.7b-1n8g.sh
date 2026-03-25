#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../../../../..)

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=360
# ===== END CONFIG =====

set -eou pipefail

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

MODEL_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/alexq/models

cd $PROJECT_ROOT

uv run python research/cross_tokenizer_distillation/run_cross_distillation.py \
    --config research/cross_tokenizer_distillation/configs/cross_distill_gpt_oss_to_qwen3.yaml \
    cross_distillation.max_num_steps=$MAX_STEPS \
    cross_distillation.val_at_start=true \
    policy.model_name=$MODEL_DIR/Qwen/Qwen3-1.7B-Base \
    teacher.model_name=$MODEL_DIR/openai/gpt-oss-20b \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.tensorboard_enabled=False \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.num_nodes=$NUM_NODES \
    2>&1 | tee $RUN_LOG

echo ""
echo "============================================"
echo "  Cross-Tokenizer Distillation: gpt-oss-20b -> Qwen3-1.7B"
echo "  Experiment: Length-Normalized Chunk KL"
echo "============================================"

echo ""
echo "=== Validation Accuracy ==="
grep -E "(Accuracy|accuracy)" $RUN_LOG || echo "No accuracy found"

echo ""
echo "=== Training Loss ==="
grep "Loss (chunk KL)" $RUN_LOG || echo "No loss found"

echo ""
echo "=== Sample Outputs ==="
grep "Sample text" $RUN_LOG || echo "No sample text found"

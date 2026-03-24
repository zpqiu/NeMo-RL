#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../../../../..)

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=1
NUM_MINUTES=60
# ===== END CONFIG =====

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
CKPT_DIR=$EXP_DIR/ckpts
RUN_LOG=$EXP_DIR/run.log
JSON_METRICS=$EXP_DIR/metrics.json

mkdir -p $EXP_DIR $LOG_DIR $CKPT_DIR

export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/research/cross_tokenizer_distillation:${PYTHONPATH:-}

MODEL_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/alexq/models

cd $PROJECT_ROOT

# Run cross-tokenizer distillation
uv run python research/cross_tokenizer_distillation/run_cross_distillation.py \
    --config research/cross_tokenizer_distillation/configs/cross_distill_math.yaml \
    cross_distillation.max_num_steps=$MAX_STEPS \
    cross_distillation.num_prompts_per_step=16 \
    cross_distillation.val_period=0 \
    policy.model_name=$MODEL_DIR/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base \
    policy.tokenizer.name=$MODEL_DIR/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base \
    teacher.model_name=$MODEL_DIR/Qwen/Qwen3-4B-Instruct-2507 \
    teacher.tokenizer.name=$MODEL_DIR/Qwen/Qwen3-4B-Instruct-2507 \
    policy.max_total_sequence_length=2048 \
    policy.train_global_batch_size=16 \
    policy.generation_batch_size=16 \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=False \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    checkpointing.save_period=10 \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.num_nodes=$NUM_NODES \
    2>&1 | tee $RUN_LOG

echo ""
echo "============================================"
echo "  Cross-Tokenizer Distillation Smoke Test"
echo "============================================"

# Check for loss values in log
if grep -q "Loss (chunk KL):" $RUN_LOG; then
    echo "✅ Training produced loss values"
    grep "Loss (chunk KL):" $RUN_LOG | tail -5
else
    echo "❌ No loss values found in log"
    tail -50 $RUN_LOG
    exit 1
fi

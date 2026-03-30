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

uv run python research/cross_tokenizer_distillation/run_cross_distillation.py \
    --config research/cross_tokenizer_distillation/configs/cross_distill_qwen3_to_gemma3.yaml \
    cross_distillation.max_num_steps=$MAX_STEPS \
    policy.model_name=$MODEL_DIR/google/gemma-3-4b-it \
    loss_fn.kl_type=is \
    loss_fn.clip_epsilon=0.2 \
    loss_fn.terminal_eos_weight=0.0 \
    loss_fn.advantage_normalization=none \
    loss_fn.negative_advantage_weight=0.25 \
    policy.optimizer.kwargs.lr=1e-5 \
    teacher.model_name=$MODEL_DIR/Qwen/Qwen3-4B-Instruct-2507 \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.tensorboard_enabled=False \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.num_nodes=$NUM_NODES \
    2>&1 | tee $RUN_LOG

echo ""
echo "============================================"
echo "  Cross-Tokenizer Distillation: Qwen3-4B -> Gemma-3-4B"
echo "  Experiment: IS Loss + Terminal EOS + Per-Token Normalization"
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


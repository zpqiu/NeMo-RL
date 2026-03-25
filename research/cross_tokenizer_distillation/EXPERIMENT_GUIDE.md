# Cross-Tokenizer Distillation — Experiment Guide

## Quick Reference

| Item | Value |
|------|-------|
| **Cluster** | DFW (`ssh dfw`, user `alexq`) |
| **Remote CWD** | `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/alexq/RL-xopd` |
| **Container** | `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/alexq/nemo_rl.0228.sqsh` |
| **SLURM Account** | `coreai_dlalgo_nemorl` |
| **Partition** | `batch` |
| **GPUs per Node** | 8 (H100) |
| **Git Branch** | `research/cross-tokenizer-distill` |
| **Local Project Root** | `/Users/qiuzhaopeng/Projs/RL-xopd` |

### Models (on cluster)

| Model | Path |
|-------|------|
| Qwen3-4B-Base | `Qwen/Qwen3-4B-Base` (HuggingFace, auto-download) |
| Qwen3-1.7B-Base | `/lustre/fs1/.../models/Qwen/Qwen3-1.7B-Base` |
| Qwen3-0.6B-Base | `/lustre/fs1/.../models/Qwen/Qwen3-0.6B-Base` |
| Qwen3-8B-Base | `/lustre/fs1/.../models/Qwen/Qwen3-8B-Base` |
| Nemotron-9B-v2-Base | `/lustre/fs1/.../models/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` |
| gpt-oss-20b | `/lustre/fs1/.../models/openai/gpt-oss-20b` |

> Full models dir: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/alexq/models/`

---

## End-to-End Workflow

### 1. Make code changes locally

Edit files under `research/cross_tokenizer_distillation/`. The main files:

```
cross_tokenizer_distillation/token_alignment.py      # Byte-offset alignment
cross_tokenizer_distillation/cross_tokenizer_loss.py  # Chunk-level KL loss
cross_tokenizer_distillation/algorithm.py             # Training loop
configs/cross_distill_math.yaml                       # Default config
tests/test_suites/llm/cross-distill-smoke-1n8g.sh     # Experiment launch script
```

### 2. Commit & push

```bash
git add -A && git commit -m "description of change"
git push origin research/cross-tokenizer-distill -f
```

### 3. Submit to cluster (one-liner)

```bash
git push origin research/cross-tokenizer-distill -f 2>&1 && \
ssh dfw "cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/alexq/RL-xopd && \
  git fetch origin && git reset --hard origin/research/cross-tokenizer-distill && \
  source ~/.exp_env && \
  rm -rf code_snapshots/<EXP_NAME> && \
  CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/alexq/nemo_rl.0228.sqsh \
  ACCOUNT=coreai_dlalgo_nemorl \
  PARTITION=batch \
  bash tools/launch research/cross_tokenizer_distillation/tests/test_suites/llm/<SCRIPT_NAME>.sh" 2>&1
```

Replace `<EXP_NAME>` with the experiment name (= script filename without `.sh`) and `<SCRIPT_NAME>` with the actual script file.

### 4. Monitor job

```bash
# Check job status
ssh dfw "squeue -u alexq -h -o '%i %j %T %M %S' --state=RUNNING,PENDING"

# Watch training log (ray-driver.log is the main output)
ssh dfw "tail -50 /lustre/fsw/.../RL-xopd/code_snapshots/<EXP_NAME>/<JOBID>-logs/ray-driver.log"

# Grep key metrics
ssh dfw "grep -E '(Loss \(chunk KL\)|Chunks:|Mean gen length|Step .* Results)' \
  /lustre/fsw/.../RL-xopd/code_snapshots/<EXP_NAME>/<JOBID>-logs/ray-driver.log"

# Check for errors
ssh dfw "grep -i 'error\|Error\|traceback\|OOM' \
  /lustre/fsw/.../RL-xopd/code_snapshots/<EXP_NAME>/<JOBID>-logs/ray-driver.log | tail -20"

# Check slurm output (allocation-level logs)
ssh dfw "ls /lustre/fsw/.../RL-xopd/code_snapshots/<EXP_NAME>/slurm-*.out"
ssh dfw "tail -50 /lustre/fsw/.../RL-xopd/code_snapshots/<EXP_NAME>/slurm-*.out"
```

### 5. Cancel job

```bash
ssh dfw "scancel <JOBID>"
```

---

## How `tools/launch` Works

1. Reads `# ===== BEGIN CONFIG =====` ... `# ===== END CONFIG =====` section from the shell script to extract: `NUM_NODES`, `GPUS_PER_NODE`, `MAX_STEPS`, `NUM_RUNS`, `NUM_MINUTES`
2. Creates a code snapshot under `code_snapshots/<EXP_NAME>/` (git-tracked files only, via rsync)
3. Generates `code_snapshots/<EXP_NAME>/continue.sh` with the sbatch command
4. Submits via `sbatch` using `ray.sub` as the job template
5. `ray.sub` starts a Ray cluster (head + workers), then runs the experiment command as the Ray driver

### Key environment variables for `tools/launch`

| Env Var | Required | Description |
|---------|----------|-------------|
| `CONTAINER` | ✅ | Path to `.sqsh` container image |
| `ACCOUNT` | ✅ | SLURM account |
| `PARTITION` | ✅ | SLURM partition |
| `DRYRUN` | ❌ | `1` = print GPU hours only, `2` = create snapshot but don't submit |
| `WATCH` | ❌ | Set to track job completion |

---

## Experiment Script Template

Create a new script under `tests/test_suites/llm/`:

```bash
#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../../../../..)

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=50
MAX_STEPS=50
NUM_RUNS=1
NUM_MINUTES=180
# ===== END CONFIG =====

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
CKPT_DIR=$EXP_DIR/ckpts
RUN_LOG=$EXP_DIR/run.log

mkdir -p $EXP_DIR $LOG_DIR $CKPT_DIR

export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/research/cross_tokenizer_distillation:${PYTHONPATH:-}
export NRL_FORCE_REBUILD_VENVS=true

cd $PROJECT_ROOT

uv run python research/cross_tokenizer_distillation/run_cross_distillation.py \
    --config research/cross_tokenizer_distillation/configs/cross_distill_math.yaml \
    cross_distillation.max_num_steps=$MAX_STEPS \
    cross_distillation.num_prompts_per_step=32 \
    cross_distillation.val_period=0 \
    loss_fn.kl_type=mixed \
    policy.optimizer.kwargs.lr=1e-6 \
    policy.model_name=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base \
    teacher.model_name=Qwen/Qwen3-4B-Base \
    policy.max_total_sequence_length=2048 \
    policy.train_global_batch_size=32 \
    policy.generation_batch_size=32 \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=False \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    checkpointing.save_period=10 \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.num_nodes=$NUM_NODES \
    2>&1 | tee $RUN_LOG
```

### Required script config variables

| Variable | Description |
|----------|-------------|
| `NUM_NODES` | Number of SLURM nodes |
| `GPUS_PER_NODE` | GPUs per node (8 for H100) |
| `STEPS_PER_RUN` | Approx steps per job run (for chaining) |
| `MAX_STEPS` | Total training steps |
| `NUM_RUNS` | Number of chained jobs: `ceil(MAX_STEPS / STEPS_PER_RUN)` |
| `NUM_MINUTES` | Walltime per job in minutes |

---

## Output Format

Training logs print per-step results:

```
📊 Step 27 Results:
  • Loss (chunk KL): 22.6259
  • Chunks: 65397
  • Mean gen length: 537.1

⏱️  Timing: 30.26s total
```

### Key metrics to extract

```bash
# Loss values
grep "Loss (chunk KL):" ray-driver.log

# Chunk counts (mode collapse indicator: should stay high, e.g. >10K)
grep "Chunks:" ray-driver.log

# Generation length (collapse indicator: should stay reasonable)
grep "Mean gen length:" ray-driver.log
```

---

## Results Tracking

Results are tracked in `research/cross_tokenizer_distillation/results.tsv`:

```
commit	step	status	tests	description
```

| Column | Description |
|--------|-------------|
| `commit` | Short git hash (7 chars) |
| `step` | Experiment ID or step range |
| `status` | `keep` / `drop` / `crash` |
| `tests` | Number of steps or tests passed |
| `description` | What this experiment tried + key results |

### Autoresearch-style experiment loop

Following Karpathy's autoresearch methodology:

1. **Modify code** locally (only the algorithm/loss/alignment files)
2. **git commit** with descriptive message
3. **Push & submit** via the one-liner above
4. **Monitor** until completion (~3-30 min depending on steps)
5. **Extract results**: loss, chunks, gen length from logs
6. **Record** in `results.tsv`
7. **Decision**:
   - If improved → `keep`, advance (next experiment builds on this)
   - If same/worse → `drop`, `git reset --hard` to previous best commit
   - If crashed → `crash`, debug or move on
8. **Repeat**

---

## Filesystem Notes

- `/lustre/fsw/...` and `/lustre/fs1/...` resolve to the same physical path (`fsw` → `fs1` symlink)
- Code snapshots are created at `<remote_cwd>/code_snapshots/<EXP_NAME>/`
- **Delete old snapshots** before re-submitting: `rm -rf code_snapshots/<EXP_NAME>` (otherwise it reuses the old snapshot)
- Ray driver logs: `code_snapshots/<EXP_NAME>/<JOBID>-logs/ray-driver.log`
- SLURM output: `code_snapshots/<EXP_NAME>/slurm-*-<JOBID>-*.out`

## Common Issues

| Issue | Fix |
|-------|-----|
| `transformers` version mismatch | Set `NRL_FORCE_REBUILD_VENVS=true` in script |
| Mode collapse (chunks → 0) | Lower LR to `1e-6`, try `mixed` KL |
| SSH timeout | Re-check `ssh -O check dfw` |
| Old code running | Delete `code_snapshots/<EXP_NAME>/` before re-submitting |
| `PYTHONPATH` missing | Ensure both project root AND `research/cross_tokenizer_distillation` are in PYTHONPATH |

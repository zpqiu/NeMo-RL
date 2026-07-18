# Math with Code

Multi-turn RL on boxed-answer math problems where the policy can execute Python
inside an isolated per-trial Singularity container. Built on nemo-gym's Harbor
harness with a custom agent (`MathCodeHarborAgent`) that drives a persistent
in-container Python session, and trained with NeMo-RL async GRPO.

## Layout

```
research/math_with_code/
├── responses_api_agents/harbor_agent/  # Overlay fork of nemo-gym's harbor_agent server
│   ├── custom_agents/math_code_*.py    #   the math-code agent + persistent Python session
│   ├── math_code/                      #   all tooling: dataset build, node preflight, venv build, runtime validation
│   ├── configs/math_code_harbor_agent.yaml
│   └── data/math_code/                 #   generated datasets (gitignored)
├── configs/                            # NeMo-RL training config
├── eagle3/                             # EAGLE-3 speculative-decoding track (drafter smoke test)
├── docker/                             # SIF build, only for def changes/new arch — prefer the prebuilt pull (see Running)
├── experiments/                        # local-only launchers, gitignored (cluster-private paths)
└── archive/                            # local-only backup, gitignored (see below)
```

## How the overlay fork works

The Gym submodule (`3rdparty/Gym-workspace/Gym`) stays pristine at its pinned
commit. This project carries a forked copy of the entire
`responses_api_agents/harbor_agent` server directory, wired in via three
mechanisms:

1. **Server discovery** — nemo-gym checks `Path.cwd()/<server_rel_path>` before
   its install location (`nemo_gym/cli.py`). `experiments/exp.sh` creates the
   repo-root symlink `responses_api_agents ->
   research/math_with_code/responses_api_agents` on every submit so the fork
   wins; `config_paths` resolution works the same way. The symlink is a runtime
   artifact (listed in `.git/info/exclude`, never committed) — recreate it with
   the same `ln -sfn` if you launch without exp.sh.
2. **Python imports** — the editable nemo-gym install maps
   `responses_api_agents.*` to the pristine submodule via a setuptools
   meta-path finder. PathFinder consults `sys.path` first, so the fork's
   `app.py` prepends the overlay root to `sys.path` (server process) and to the
   Harbor Ray worker's `PYTHONPATH` (`runner_ray_remote` runtime env).
3. **Dependencies** — the fork's `requirements.txt` editable-installs nemo-gym
   from the submodule by relative path.

The trade-off: this directory no longer receives upstream Gym changes to
harbor_agent automatically; sync manually by diffing against the submodule.

## Bring-up

This assumes you have already run a single-turn GRPO job with NeMo-RL on your
cluster — container, `sbatch + ray.sub` submission, `uv` — per
[docs/cluster.md](../../docs/cluster.md) and
[docs/guides/grpo.md](../../docs/guides/grpo.md). Only the math-code deltas
are listed here.

**Prerequisites on top of stock single-turn GRPO:**

- A recent NeMo-RL image (nightly or >= 0.7): the recipe additionally needs
  `singularity`/`apptainer` and the prebuilt NemoGym venv inside the training
  container.
- The Gym submodule checked out: `git submodule update --init
  3rdparty/Gym-workspace/Gym` (the Harbor venv editable-installs it).
- `/dev/fuse:/dev/fuse` in the container mounts of **every** job below,
  including training. Harbor trials run `singularity exec` inside the
  training container and squashfuse needs the device; missing it fails with
  `squashfuse_ll exited: fuse: device not found`.
- Hugging Face network access for steps 2–3.

All three one-time steps run inside the training container **on the
compute-node architecture** (x86 login nodes won't do for aarch64 clusters:
the venv is arch-specific and the SIF pull needs apptainer) — e.g. via
`srun --container-image=<training image> --container-mounts=...,/dev/fuse:/dev/fuse`.
Outputs all land on shared storage.

```bash
# 1. Pull the per-trial SIF (x86 clusters: build via docker/ instead).
apptainer pull oras://ghcr.io/zpqiu/math-code-sif:py312-aarch64

# 2. Build the Harbor venv (NEMO_GYM_VENV_DIR on shared storage).
NEMO_GYM_VENV_DIR=/shared/gym_venvs \
    responses_api_agents/harbor_agent/math_code/build_harbor_venv.sh

# 3. Build all three datasets (non8 train + AIME 2024/2025 val).
MATH_CODE_SIF_PATH=$PWD/math-code-sif_py312-aarch64.sif \
    responses_api_agents/harbor_agent/math_code/build_datasets.sh

# Optional: end-to-end sanity check — one real Harbor trial against a fake
# Responses API, no GPUs or policy model needed.
NEMO_GYM_VENV_DIR=/shared/gym_venvs \
    responses_api_agents/harbor_agent/math_code/validate_runtime.sh
```

**Launch** is the standard `sbatch + ray.sub` flow from
[docs/cluster.md](../../docs/cluster.md) with two math-code additions: a
repo-root symlink that lets the overlay fork win nemo-gym's cwd-first server
discovery (see "How the overlay fork works"), and a per-node preflight in
`SETUP_COMMAND`:

```bash
cd <repo-root>
ln -sfn research/math_with_code/responses_api_agents responses_api_agents

SETUP_COMMAND="export NEMO_GYM_VENV_DIR=/shared/gym_venvs && \
    ./research/math_with_code/responses_api_agents/harbor_agent/math_code/preflight_math_with_code_node.sh" \
COMMAND="export NEMO_GYM_VENV_DIR=/shared/gym_venvs && \
    uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config research/math_with_code/configs/grpo_math_with_code_qwen3_8b_thinking_async.yaml" \
CONTAINER=<training image> \
MOUNTS="<your mounts>,/dev/fuse:/dev/fuse" \
GPUS_PER_NODE=4 \
    sbatch --nodes=3 --account=... --job-name=... ray.sub
```

The 8B config wants 3 nodes x 4 GPUs (2 generation + 1 training, see the
yaml's cluster block); the 30B config wants 4 nodes. `WANDB_API_KEY` in the
environment enables W&B logging as usual.

`experiments/` is a local-only (gitignored) convenience layer over exactly
this flow — launchers there embed cluster-private account and storage paths.

## Data

Nothing under `responses_api_agents/harbor_agent/data/` is committed — the
directory is fully gitignored.

The data model has two layers that must stay consistent:

- The **request JSONL** (`data.train.data_path` in the training config) is
  only the index NeMo-RL's dataloader iterates; each row references a task in
  one of the agent's `harbor_datasets`.
- The **task tree** (one directory per task: `instruction.md`, `task.toml`,
  `tests/`, `environment/`) is what Harbor actually executes. All trees live
  directly on shared storage under `data/math_code/<alias>/`. They cost ~8
  inodes/task, fine at the few-thousand-task scale used here; if you scale to
  a much larger set, revisit (git history has a tar.gz + node-local staging
  flow that was removed for simplicity).
- The dataset **alias** ties the layers together: it must match the
  `harbor_datasets` key in `configs/math_code_harbor_agent.yaml` (whose
  `local_dataset_path` points at the task tree) and the JSONL filename in
  `data.train.data_path`.

Datasets come from two places:

- **Filtered train source** (6389 DAPO prompts Qwen3-8B did not solve 8/8):
  published as
  [alex-chiu/DAPO-Math-17k-Qwen3-8B-non8](https://huggingface.co/datasets/alex-chiu/DAPO-Math-17k-Qwen3-8B-non8)
  (public), source-format rows. Publishing this subset matters because the
  filter cost a full difficulty-labeling campaign (8 rollouts x 17398 prompts).
- **Everything else rebuilds from one script.** `build_datasets.sh` (under
  `responses_api_agents/harbor_agent/math_code/`) builds the full supported
  set — non8 train + AIME 2024/2025 val — in one pass:

```bash
MATH_CODE_SIF_PATH=/shared/path/to/math-code.sif \
    responses_api_agents/harbor_agent/math_code/build_datasets.sh
```

That is the only input; sources, aliases, task counts, and reward shaping are
fixed inside the script. It converts the non8 subset (task ids renumbered
`task_000000..task_006388`, tree/JSONL pair self-consistent),
[tongyx361/AIME-2024-Boxed](https://huggingface.co/datasets/tongyx361/AIME-2024-Boxed),
and [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) (plain
problem/answer rows, adapted via `convert_plain_problem_answer.py` so prompts
match the train template).

The train set is built with `--tool-shaping`: ReTool-style reward shaping baked
into the tasks (failed answers earn 0.1 per executed tool call, capped at 0.4;
see `math_code/templates/verify.py`). Eval sets are built without it, so eval
accuracy stays a pure correctness metric.

For a custom dataset, drive `prepare_dataset.py` directly — it accepts a HF
dataset name, local parquet shards, or Dataset Server JSON, plus
`--sif-path/--tasks-dir/--jsonl-path` and optional `--tool-shaping`.

Harbor trial outputs land in `responses_api_agents/harbor_agent/jobs/`
(gitignored). Successful trials are deleted automatically; failed ones are kept
for diagnosis and are worth purging occasionally to protect shared-fs inodes.

## Residual diffs outside this directory

Kept in core because they are generic bug fixes (upstream candidates, not
research-specific):

- `nemo_rl/algorithms/async_utils/trajectory_collector.py` — propagate
  background rollout failures instead of hanging; drain-on-pause for
  validation; `max_inflight_prompt_groups` knob.
- `nemo_rl/algorithms/grpo.py` — failure-channel call sites + re-raise.
- `nemo_rl/environments/math_environment.py` — double-append reward fix on the
  verifier exception path.

## archive/ (local-only, not in git)

Backup of retired tooling, kept only on this cluster: the one-shot
difficulty-labeling scripts that produced the non8 filter (their labeled
outputs live under `results/`), `core_difficulty_support.patch` preserving the
reverted core NeMo-RL eval changes they relied on, and removed tests. Nothing
here is needed for training runs.

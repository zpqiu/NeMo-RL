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

## Running

```bash
# One-time: get the runtime SIF — pull the prebuilt aarch64 image
#     apptainer pull oras://ghcr.io/zpqiu/math-code-sif:py312-aarch64
# (x86 clusters: build from docker/math_code_aarch64.def instead, see docker/).
# Then build the Harbor venv
# (responses_api_agents/harbor_agent/math_code/build_harbor_venv.sh with
# NEMO_GYM_VENV_DIR on shared storage), and build all datasets
# (MATH_CODE_SIF_PATH=<sif> responses_api_agents/harbor_agent/math_code/build_datasets.sh).

# Launch training (submits via sbatch + ray.sub from the repo root):
research/math_with_code/experiments/exp.sh \
    research/math_with_code/experiments/grpo-math-code-qwen3-8b-async-3x4.sh
```

### Cluster prerequisites

- **`/dev/fuse` must be visible inside the training container.** Harbor
  trials run `singularity exec` on the per-trial SIF from inside the training
  container; squashfuse needs `/dev/fuse`, and pyxis/enroot does not expose
  it on every cluster. Symptom: `FATAL: container creation failed: ...
  squashfuse_ll exited: fuse: device not found`. Fix: add
  `/dev/fuse:/dev/fuse` to the container mounts of every job — the one-time
  setup jobs and the ray.sub training job alike.
- **Run every one-time step on the compute-node architecture.** On clusters
  with x86 login nodes and aarch64 compute (e.g. GB200), the SIF pull (needs
  apptainer), the Harbor venv build, and dataset prep (both need the training
  image's uv) must all run through `srun --container-image=<training sqsh>`
  on a compute node. The Harbor venv is arch-specific: build it where it will
  run, on shared storage.
- **Per-node setup contract**:
  `responses_api_agents/harbor_agent/math_code/preflight_math_with_code_node.sh`
  reads `NEMO_GYM_VENV_DIR` (and `MATH_CODE_PREFLIGHT_TASK_DIR` to check a
  non-default task). Export them in the scheduler wrapper so they reach
  ray.sub's `SETUP_COMMAND` environment on every node.

`~/.exp_env` must provide `WANDB_API_KEY`, `ACCOUNT`, `MOUNTS`, and optionally
`CONTAINER`. Results and Slurm logs land under the repo-root `results/` tree.

`experiments/` is local-only (gitignored) because the launchers embed
cluster-private account and storage paths. On another cluster, drive
`configs/` from your own scheduler wrapper: run the per-node
`responses_api_agents/harbor_agent/math_code/preflight_math_with_code_node.sh`
as setup, then `uv run python examples/nemo_gym/run_grpo_nemo_gym.py --config
research/math_with_code/configs/grpo_math_with_code_qwen3_8b_thinking_async.yaml`
from the repo root.

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

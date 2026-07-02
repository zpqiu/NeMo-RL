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
│   ├── math_code/                      #   dataset conversion, venv build, runtime validation
│   ├── configs/math_code_harbor_agent.yaml
│   └── data/math_code/                 #   generated datasets (gitignored)
├── configs/                            # NeMo-RL training config
├── scripts/                            # dataset prepare/package/stage + node preflight
├── docker/                             # math-code SIF build (aarch64)
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
# One-time: build the SIF (see docker/), build the Harbor venv
# (responses_api_agents/harbor_agent/math_code/build_harbor_venv.sh with
# NEMO_GYM_VENV_DIR on shared storage), and prepare the dataset
# (scripts/prepare_math_code_17k.sh).

# Launch training (submits via sbatch + ray.sub from the repo root):
research/math_with_code/experiments/exp.sh \
    research/math_with_code/experiments/grpo-math-code-qwen3-8b-async-2x4.sh
```

`~/.exp_env` must provide `WANDB_API_KEY`, `ACCOUNT`, `MOUNTS`, and optionally
`CONTAINER`. Results and Slurm logs land under the repo-root `results/` tree.

`experiments/` is local-only (gitignored) because the launchers embed
cluster-private account and storage paths. On another cluster, drive
`configs/` + `scripts/` from your own scheduler wrapper: run the per-node
`scripts/stage_math_code_dataset.sh` + `scripts/preflight_math_with_code_node.sh`
as setup, then `uv run python examples/nemo_gym/run_grpo_nemo_gym.py --config
research/math_with_code/configs/grpo_math_with_code_qwen3_8b_thinking_async.yaml`
from the repo root.

## Data

Nothing under `responses_api_agents/harbor_agent/data/` is committed — the
directory is fully gitignored. Datasets come from two places:

- **Filtered train source** (6389 DAPO prompts Qwen3-8B did not solve 8/8):
  published as
  [alex-chiu/DAPO-Math-17k-Qwen3-8B-non8](https://huggingface.co/datasets/alex-chiu/DAPO-Math-17k-Qwen3-8B-non8)
  (public), source-format rows. Publishing this subset matters because the
  filter cost a full difficulty-labeling campaign (8 rollouts x 17398 prompts).
- **Everything else rebuilds from scripts.** Full 17k (or the filtered set)
  via `scripts/prepare_math_code_17k.sh`; AIME tasks via
  `responses_api_agents/harbor_agent/math_code/prepare_dataset.py`; nodes stage
  the archive locally with `scripts/stage_math_code_dataset.sh`.

Rebuild the filtered task tree + request JSONL directly from the published
subset (task ids are renumbered `task_000000..task_006388`; the tree/JSONL pair
stays self-consistent):

```bash
MATH_CODE_DATASET_REPO=alex-chiu/DAPO-Math-17k-Qwen3-8B-non8 \
MATH_CODE_DATASET_REVISION=main \
MATH_CODE_DATASET_PARQUET=data/train-00000-of-00001.parquet \
MATH_CODE_DATASET_ALIAS=dapo_math_17k_non8 \
MATH_CODE_EXPECTED_TASKS=6389 \
    scripts/prepare_math_code_17k.sh
```

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

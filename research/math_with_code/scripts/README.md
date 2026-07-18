# scripts/

Dataset lifecycle and per-node runtime setup for the math-code Harbor tasks.
Two groups, in execution order:

## One-time dataset build (run inside the training sqsh, compute-node arch)

| Script | Role |
|--------|------|
| `prepare_math_code_17k.sh` | End-to-end train-set build: HF download → `math_code/prepare_dataset.py` conversion on node-local disk → request JSONL + task archive on shared storage. Calls `package_math_code_dataset.sh` for the archive step. |
| `package_math_code_dataset.sh` | Standalone (re-)packaging of an already generated task tree into one reproducible `tar.gz`/`tar.zst` inode. Only needed directly when re-archiving without a full rebuild. |

Small eval sets (AIME) skip the archive entirely — see the `## Data` section of
`../README.md` and `math_code/convert_plain_problem_answer.py` for plain
problem/answer sources.

## Per-node runtime setup (ray.sub `SETUP_COMMAND`, every allocated node)

| Script | Role |
|--------|------|
| `stage_math_code_dataset.sh` | Extract the task archive onto node-local disk (`/tmp/nemo_rl_math_code/<alias>`). |
| `preflight_math_with_code_node.sh` | Fast fail-early checks: Harbor venv, SIF, `/dev/fuse`, a sample task trial. |

Both read the `NEMO_GYM_VENV_DIR` / `MATH_CODE_*` environment contract
documented in `../README.md` ("Cluster prerequisites").

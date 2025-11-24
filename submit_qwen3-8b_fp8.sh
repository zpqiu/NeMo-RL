# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=1

# default: single submission; use -n to repeat
N_CALLS=1
while getopts "n:" opt; do
  case $opt in
    n) N_CALLS=$OPTARG;;
  esac
done

EXP_NAME="qwen3-8b-qatint4-B32-R20K-TIS2-nemorl-cuda12.9-${NUM_ACTOR_NODES}n8g"

# export EXP_NAME=qwen3-8b-fp8e2e-B32-R20K-TIS2-nemorl-cuda12.9-${NUM_ACTOR_NODES}n8g

read -r -d '' COMMAND <<EOF
export HF_HUB_OFFLINE=1
uv run --extra vllm python examples/run_grpo_math.py \
 --config examples/configs/dapo-qwen3-8b-fp8.yaml \
 loss_fn.truncated_importance_sampling_ratio=2\
 cluster.num_nodes=${NUM_ACTOR_NODES} \
 policy.model_name=Qwen/Qwen3-8B-Base 
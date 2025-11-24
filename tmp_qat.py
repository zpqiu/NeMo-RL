"""Test that vLLM streaming weight update and can save memory."""
from nemo_rl.models.policy.lm_policy import Policy
import json
import os
from copy import deepcopy
from pathlib import Path

import ray

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
# from torchao.core.config import config_to_dict
# from torchao.quantization import Int4WeightOnlyConfig
from transformers import HqqConfig
# base_config = Int4WeightOnlyConfig(group_size=32, version=1, use_hqq=True)

quant_config = HqqConfig(nbits=4, group_size=64)

model_name = "Qwen/Qwen3-0.6B"
# Define basic vLLM test config
basic_vllm_test_config: VllmConfig = {
    "backend": "vllm",
    "model_name": "./Qwen/Qwen3-0.6B-hqq-quantized",
    "tokenizer": {
        "name": model_name,
    },
    "dtype": "bfloat16",
    "max_new_tokens": 5,  # Small number of tokens for testing
    # Set temperature=1.0 to ensure consistent probability scaling when comparing vLLM and HF policy outputs.
    # Note: greedy=True is only used in tests for deterministic behavior and not used in the real training.
    # In vLLM, enabling greedy=True disables temperature scaling (temperature is overridden to None).
    # The HF policy worker does not currently support greedy=True for get_logprobs.
    # Using temperature=1.0 allows us to meaningfully test the average probability multiplicative error between the two implementations,
    # while still maintaining the deterministic behavior.
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
    "vllm_cfg": {
        "precision": "bfloat16",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 1024,
        "async_engine": False,  # Default to False for synchronous tests
        "skip_tokenizer_init": False,
        "load_format": "auto",
        "enforce_eager": "True",
    },
    "colocated": {
        "enabled": True,
        "resources": {
            "gpus_per_node": None,
            "num_nodes": None,
        },
    },
    "vllm_kwargs": {
        # "load_format": "auto",
        # "quantization": "rtn",
        "hf_overrides": {
            # "quantization_config_dict_json": json.dumps(config_to_dict(base_config)),
            "quantization_config": quant_config.to_dict()
        },
    },
}

basic_dtensor_test_config: PolicyConfig = {
    "model_name": model_name,
    "tokenizer": {
        "name": basic_vllm_test_config["tokenizer"]["name"],
    },
    # Required training parameters
    "train_global_batch_size": 1,
    "train_micro_batch_size": 1,
    "learning_rate": 5e-6,
    "logprob_batch_size": 1,
    "max_new_tokens": 16,
    "do_sample": False,
    "precision": "float32",
    "offload_optimizer_for_logprob": False,
    "optimizer": {
        "name": "torch.optim.AdamW",
        "kwargs": {
            "lr": 5e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    },
    "dtensor_cfg": {
        "enabled": True,
        "cpu_offload": False,
        "sequence_parallel": False,
        "activation_checkpointing": False,
        "tensor_parallel_size": 2,
        "context_parallel_size": 1,
        "custom_parallel_plan": None,
    },
    "dynamic_batching": {
        "enabled": True,
        "train_mb_tokens": 40,
        "logprob_mb_tokens": 40,
        "sequence_length_round": 4,
    },
    "sequence_packing": {
        "enabled": False,
    },
    "max_grad_norm": 1.0,
    "make_sequence_length_divisible_by": 1,
    "generation": deepcopy(basic_vllm_test_config),
}

def main():
    tokenizer = get_tokenizer({"name": "Qwen/Qwen3-0.6B"})

    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],  # 1 node with 2 GPU bundle
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=2,  # Use available GPUs
        name="vllm-test-cluster",
    )
    # Create separate configs for each policy
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)

    # Ensure we can get same peak memory
    # assert vllm_config["model_name"] == "Qwen/Qwen3-0.6B", (
    #     "Model name should be Qwen/Qwen3-0.6B to get expected peak memory"
    # )

    # Create policies
    print("Creating vLLM policy...")
    vllm_policy = VllmGeneration(cluster, vllm_config)
    vllm_policy.finish_generation()

    print("Creating DTensor policy...")
    dtensor_config = basic_dtensor_test_config
    lm_policy = Policy(cluster, dtensor_config, tokenizer)

    print("preparing refit info...")
    state_dict_info = lm_policy.prepare_refit_info()
    vllm_policy.prepare_refit_info(state_dict_info)

    print("refitting vllm policy...")
    # take it outside statistics to get clean peak memory during refit
    lm_policy.offload_before_refit()
    # reset peak memory stats before refit
    workers = lm_policy.worker_group.workers
    ray.get([w.reset_peak_memory_stats.remote() for w in workers])
    refit_policy_generation(
        lm_policy,
        vllm_policy,
        vllm_config["colocated"]["enabled"],
        _refit_buffer_size_gb=1.5,
    )


def llm_compressor_quant():
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM

    model_id = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype="auto"
    )

    # 使用基础的 QuantizationModifier，而不是 GPTQModifier
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W4A16",  # 指定为 INT4 权重，FP16 激活
        ignore=["lm_head"] # 跳过输出层，避免精度崩坏
    )

    # 在 oneshot 中，dataset 可以传 None（或者不传），因为 RTN 不需要数据
    oneshot(
        model=model,
        recipe=recipe,
        dataset=None, # 关键点：不需要校准数据
        output_dir="./Qwen/Qwen3-0.6B-llmcompressor-quantized"
    )


if __name__ == "__main__":
    # from transformers import AutoModelForCausalLM, TorchAoConfig, AutoTokenizer
    # # print("Step 1: Loading source model Qwen/Qwen3-0.6B on cpu...")
    # src_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype="auto", device_map="cuda:0", quantization_config=TorchAoConfig(base_config))
    # src_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # # # print("Step 2: Saving source model to Qwen/Qwen3-0.6B-quantized...")
    # src_model.save_pretrained("Qwen/Qwen3-0.6B-quantized", safe_serialization=False)
    # src_tokenizer.save_pretrained("Qwen/Qwen3-0.6B-quantized")

    # llm_compressor_quant()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        main()
    finally:
        ray.shutdown()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

SAVE = True

model_id = "Qwen/Qwen3-8B-Base"

from torchao.quantization import ModuleFqnToConfig
from torchao.quantization import Int4WeightOnlyConfig
int4_config = Int4WeightOnlyConfig(group_size=128, version=1, use_hqq=False)

# qconfig_dict = {}
# # 0...12
# for idx in range(12):
#     qconfig_dict[f"model.decoder.layers.{idx}.fc1"] = int4_config
#     qconfig_dict[f"model.decoder.layers.{idx}.fc2"] = int4_config
#     # qconfig_dict[f"model.decoder.layers.{idx}.self_attn.k_proj"] = int4_config
#     # qconfig_dict[f"model.decoder.layers.{idx}.self_attn.v_proj"] = int4_config
#     # qconfig_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"] = int4_config
#     qconfig_dict[f"model.decoder.layers.{idx}.self_attn.out_proj"] = int4_config

# quant_config = ModuleFqnToConfig(qconfig_dict)

quantization_config = TorchAoConfig(quant_type=int4_config)
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
print(quantized_model)
# processor = AutoProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Push to hub
save_to = f"jerryzh168/opt-125m-int4wo-per-module"
if SAVE:

    quantized_model.save_pretrained(save_to, safe_serialization=False)
    tokenizer.save_pretrained(save_to)

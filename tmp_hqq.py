from transformers import AutoModelForCausalLM, HqqConfig, AutoTokenizer
import torch
model_id = "Qwen/Qwen3-8B-Base"

# All linear layers will use the same quantization config
quant_config = HqqConfig(nbits=8, group_size=64)

# Load and quantize
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda", 
    quantization_config=quant_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

print(model)

save_to = "Qwen/Qwen3-8B-Base-hqq-int8-quantized"
model.save_pretrained(save_to)
tokenizer.save_pretrained(save_to)

# for k, v in model.state_dict().items():
#     print(k, v.shape, v.dtype)
    # print(v.dtype)
    # break


# from hqq.core.quantize import *
# #Quantization settings
# quant_config = BaseQuantizeConfig(nbits=4, group_size=64)

# import torch.nn as nn

# dummy_linear = nn.Linear(128, 128)

# #Replace your linear layer 
# hqq_layer = HQQLinear(dummy_linear, #torch.nn.Linear or None 
#                       quant_config=quant_config, #quantization configuration
#                       compute_dtype=torch.float16, #compute dtype
#                       device='cuda', #cuda device
#                       initialize=True, #Use False to quantize later
#                       del_orig=True #if True, delete the original layer
#                       )

# print(hqq_layer)

# for k, v in hqq_layer.state_dict().items():
#     print(k, v.shape, v.dtype)
    # break
# Cross-Tokenizer On-Policy Distillation — Research Program

## Status: ✅ Phase 1 Complete — Training Loop Working

### Completed Steps
1. ✅ Project skeleton
2. ✅ Dependencies
3-6. ✅ Token alignment module (byte-offset alignment + chunk logprob aggregation)
7-8. ✅ Cross-tokenizer loss (chunk-level KL: forward/reverse/mixed)
9-13. ✅ Algorithm module (setup + train loop integrated with Policy.train())
14-16. ✅ Config YAML + entry script + SLURM script
17. ✅ 10-step smoke test PASSED on cluster

### Key Results (Experiment 1: 10 steps, forward KL)

| Step | Loss (KL) | Chunks | Gen Length |
|------|-----------|--------|------------|
| 1 | 51.50 | 16,689 | 1,059 |
| 2 | 36.83 | 20,597 | 1,324 |
| 3 | 27.76 | 18,618 | 1,191 |
| 4 | 7.87 | 20,496 | - |
| 5 | 7.56 | 13,637 | - |
| 6 | 0.31 | 11,954 | - |
| 7 | 16.83 | 15,252 | - |
| 8 | 0.10 | 6,166 | - |
| 9 | 0.06 | 3,408 | - |
| 10 | -0.15 | 2,316 | - |

**Loss decreased from 51.5 → -0.15 over 10 steps.**

### Running Experiments
- Experiment 2: 50 steps, reverse KL, batch 32 (SLURM job 10278925)

### Architecture
- **Teacher**: Qwen/Qwen3-4B-Base (Qwen3 tokenizer, vocab=151936)
- **Student**: nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base (Nemotron tokenizer, vocab=131072)
- **Alignment**: Byte-offset alignment creates ~16K-20K chunks per batch
- **Loss**: Chunk-level KL divergence (forward/reverse/mixed)
- **Integration**: Uses NeMo RL Policy.train() with LossInputType.LOGPROB

### Key Technical Solutions
1. **Alignment tensors padded to (B, S)** to pass `check_sequence_dim` validation
2. **Flat list → torch.stack** for gradient flow through chunk aggregation  
3. **Prompt length from message log** (not `input_lengths` which is total length)
4. **Chunk index bounds filtering** (teacher logprobs has N-1 entries for N tokens)
5. **NRL_FORCE_REBUILD_VENVS** for transformers version compatibility
6. **Passthrough chat template** for base models

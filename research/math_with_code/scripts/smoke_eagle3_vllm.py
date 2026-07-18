# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""1-GPU smoke: EAGLE-3 drafter loads in vLLM and speculates at deep positions.

Gate for the math_with_code eagle3 track. Checks, in order:
1. The drafter checkpoint (LlamaForCausalLMEagle3) loads next to the
   Qwen3-30B-A3B-Instruct-2507 target under speculative_config.
2. Speculation is active on a short prompt (accepted tokens > 0 at temp=1.0).
3. Speculation stays active past the drafter's max_position_embeddings
   (SpecForge ships 2048; vLLM may silently stop speculating beyond the
   drafter's max len, which would void long multi-turn rollouts).

Usage (inside the training container, 1 GPU):
    SMOKE_DRAFT=<hf-repo-or-path> uv run python \
        research/math_with_code/scripts/smoke_eagle3_vllm.py
"""

import os

TARGET = os.environ.get("SMOKE_TARGET", "Qwen/Qwen3-30B-A3B-Instruct-2507")
DRAFT = os.environ.get(
    "SMOKE_DRAFT", "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex"
)
NUM_SPEC_TOKENS = int(os.environ.get("SMOKE_K", "3"))


def spec_counters(llm) -> dict[str, float]:
    counters: dict[str, float] = {}
    for metric in llm.get_metrics():
        if "spec_decode" in metric.name and hasattr(metric, "value"):
            counters[metric.name] = counters.get(metric.name, 0.0) + metric.value
    return counters


def report(
    tag: str, before: dict[str, float], after: dict[str, float]
) -> tuple[float, float]:
    drafts = after.get("vllm:spec_decode_num_drafts", 0.0) - before.get(
        "vllm:spec_decode_num_drafts", 0.0
    )
    accepted = after.get("vllm:spec_decode_num_accepted_tokens", 0.0) - before.get(
        "vllm:spec_decode_num_accepted_tokens", 0.0
    )
    draft_tokens = after.get("vllm:spec_decode_num_draft_tokens", 0.0) - before.get(
        "vllm:spec_decode_num_draft_tokens", 0.0
    )
    accept_len = 1.0 + (accepted / drafts) if drafts else 0.0
    accept_rate = accepted / draft_tokens if draft_tokens else 0.0
    print(
        f"[smoke:{tag}] drafts={drafts:.0f} draft_tokens={draft_tokens:.0f} "
        f"accepted={accepted:.0f} accept_len={accept_len:.3f} "
        f"accept_rate={accept_rate:.3f}"
    )
    return drafts, accept_len


def main() -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=TARGET,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        max_model_len=20480,
        enforce_eager=True,  # format-compat smoke; prod burst runs inductor
        disable_log_stats=False,  # get_metrics() asserts log_stats enabled
        speculative_config={
            "method": "eagle3",
            "model": DRAFT,
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "draft_tensor_parallel_size": 1,
        },
    )
    tokenizer = llm.get_tokenizer()
    # Match rollout sampling: temp=1.0, top_p=1.0 (acceptance is
    # temperature-sensitive; a greedy smoke would overstate it).
    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=512, logprobs=1)

    def chat(user_msg: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )

    short_prompt = chat(
        "Find the number of ordered pairs (a, b) of positive integers such "
        "that a + b = 1000 and neither a nor b has a zero digit. "
        "Reason step by step."
    )

    filler = (
        "Consider the sequence defined by the recurrence x_{n+1} = "
        "(3*x_n + 7) mod 1000 with x_0 = 42. We tabulate its values and "
        "study the orbit structure, cycle lengths, and preperiodic tails. "
    )
    long_body = filler * 120  # ~5k tokens of context, well past 2048
    long_prompt = chat(
        long_body + "\n\nNow: what is the smallest positive integer n such "
        "that n! ends in exactly 25 zeros? Reason step by step."
    )
    prompt_len = len(tokenizer(long_prompt).input_ids)
    print(f"[smoke] long prompt token length = {prompt_len} (need > 2048)")
    assert prompt_len > 2048, "long prompt too short to test deep positions"

    base = spec_counters(llm)
    out_short = llm.generate([short_prompt], params)
    mid = spec_counters(llm)
    drafts_short, _ = report("short-prompt", base, mid)

    out_long = llm.generate([long_prompt], params)
    end = spec_counters(llm)
    drafts_long, accept_len_long = report("long-prompt>2048", mid, end)

    for tag, outs in (("short", out_short), ("long", out_long)):
        text = outs[0].outputs[0].text
        n_tok = len(outs[0].outputs[0].token_ids)
        has_lp = outs[0].outputs[0].logprobs is not None
        print(f"[smoke:{tag}] generated {n_tok} tokens, logprobs={has_lp}: "
              f"{text[:160]!r}")

    assert drafts_short > 0, "no speculation on short prompt — eagle3 inactive"
    assert drafts_long > 0, (
        "speculation stopped past drafter max_position_embeddings — patch the "
        "drafter config.json (max_position_embeddings) before the val burst"
    )
    # Drafting without acceptance is worse than no speculation (pure draft
    # overhead). Seen with the pristine SpecForge ckpt: accept_len 1.012 past
    # position 2048 (its max_position_embeddings) vs 2.888 below it.
    min_long = float(os.environ.get("SMOKE_MIN_LONG_ACCEPT_LEN", "1.3"))
    assert accept_len_long >= min_long, (
        f"deep-position acceptance collapsed ({accept_len_long:.3f} < "
        f"{min_long}): drafter unusable at multi-turn context lengths"
    )
    print("[smoke] PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Debug script to visualize cross-tokenizer chunk alignment.

Prints token IDs, decoded tokens, and chunk boundaries for both teacher and
student tokenizers so you can inspect exactly how alignment works.

Usage:
    python debug_tokenizer_alignment.py \
        --teacher_model meta-llama/Llama-3.1-70B-Instruct \
        --student_model Qwen/Qwen2.5-7B-Instruct \
        --text "Hello world! 这是一个测试。"

    # Or use defaults with --text only:
    python debug_tokenizer_alignment.py --text "The quick brown fox jumps over the lazy dog."

    # Simulate on-policy student generation (original ids may differ from re-tokenization):
    python debug_tokenizer_alignment.py \
        --teacher_model meta-llama/Llama-3.1-70B-Instruct \
        --student_model Qwen/Qwen2.5-7B-Instruct \
        --text "Hello world!" \
        --simulate_original_ids

    # Test chunk merging:
    python debug_tokenizer_alignment.py \
        --teacher_model meta-llama/Llama-3.1-70B-Instruct \
        --student_model Qwen/Qwen2.5-7B-Instruct \
        --text "Hello world! This is a test." \
        --merge_min_bytes 8

    # Simulate chat template offset (as in real training):
    python debug_tokenizer_alignment.py \
        --teacher_model meta-llama/Llama-3.1-70B-Instruct \
        --student_model Qwen/Qwen2.5-7B-Instruct \
        --text "Hello world!" \
        --prompt "You are a helpful assistant." \
        --simulate_chat_template
"""

from __future__ import annotations

import argparse
import copy
import sys

import torch
from transformers import AutoTokenizer

from cross_tokenizer_distillation.token_alignment import (
    AlignmentChunk,
    AlignmentResult,
    align_tokens_by_byte_offset,
    align_tokens_by_decoded_pieces,
    align_tokens_from_original_student_ids_with_stats,
    merge_alignment_chunks,
)

# ──────────────────────────────────────────────────────────────────────────────
# Pretty-printing helpers
# ──────────────────────────────────────────────────────────────────────────────

SEPARATOR = "=" * 90
THIN_SEP = "-" * 90


def _decode_token(tokenizer, token_id: int) -> str:
    """Decode a single token id to its string representation."""
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def print_token_table(label: str, tokenizer, token_ids: list[int]) -> None:
    """Print a table: index | token_id | decoded repr."""
    print(f"\n  [{label} tokens]  (total: {len(token_ids)})")
    print(f"  {'idx':>5}  {'token_id':>10}  decoded")
    print(f"  {'-----':>5}  {'----------':>10}  -------")
    for i, tid in enumerate(token_ids):
        decoded = _decode_token(tokenizer, tid)
        print(f"  {i:>5}  {tid:>10}  {decoded!r}")


def print_chunk_detail(
    chunk_idx: int,
    chunk: AlignmentChunk,
    text: str,
    teacher_tokenizer,
    student_tokenizer,
    teacher_ids: list[int],
    student_ids: list[int],
) -> None:
    """Print one chunk with full detail."""
    text_bytes = text.encode("utf-8")
    chunk_text = text_bytes[chunk.byte_start : chunk.byte_end].decode("utf-8", errors="replace")

    print(f"\n  Chunk {chunk_idx}")
    print(f"    bytes: [{chunk.byte_start}, {chunk.byte_end})  text: {chunk_text!r}")

    # Teacher side — indices may reference a longer in-context sequence after
    # offset correction, so guard against out-of-range.
    t_ids = [teacher_ids[i] if i < len(teacher_ids) else -1 for i in chunk.teacher_token_indices]
    t_tokens = [_decode_token(teacher_tokenizer, tid) if tid >= 0 else "<OOB>" for tid in t_ids]
    print(f"    teacher indices: {chunk.teacher_token_indices}")
    print(f"    teacher ids:     {t_ids}")
    print(f"    teacher tokens:  {t_tokens}")

    # Student side
    s_ids = [student_ids[i] if i < len(student_ids) else -1 for i in chunk.student_token_indices]
    s_tokens = [_decode_token(student_tokenizer, tid) if tid >= 0 else "<OOB>" for tid in s_ids]
    print(f"    student indices: {chunk.student_token_indices}")
    print(f"    student ids:     {s_ids}")
    print(f"    student tokens:  {s_tokens}")


def print_alignment(
    title: str,
    alignment: AlignmentResult,
    teacher_tokenizer,
    student_tokenizer,
    stats: dict | None = None,
) -> None:
    """Print full alignment result."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)
    print(f"  text: {alignment.text!r}")
    print(f"  #teacher_tokens={alignment.num_teacher_tokens}  "
          f"#student_tokens={alignment.num_student_tokens}  "
          f"#chunks={alignment.num_chunks}")

    if stats:
        print(f"\n  [alignment stats]")
        for k, v in stats.items():
            if v:
                print(f"    {k}: {v}")

    print_token_table("teacher", teacher_tokenizer, alignment.teacher_token_ids)
    print_token_table("student", student_tokenizer, alignment.student_token_ids)

    print(f"\n  [chunks]  (total: {alignment.num_chunks})")
    print(THIN_SEP)
    for i, chunk in enumerate(alignment.chunks):
        print_chunk_detail(
            i, chunk, alignment.text,
            teacher_tokenizer, student_tokenizer,
            alignment.teacher_token_ids, alignment.student_token_ids,
        )
    print(THIN_SEP)


# ──────────────────────────────────────────────────────────────────────────────
# Chat template offset simulation (mirrors algorithm.py L1128-1159)
# ──────────────────────────────────────────────────────────────────────────────


def simulate_chat_template_offset(
    text: str,
    prompt: str,
    alignment: AlignmentResult,
    teacher_tokenizer,
    student_tokenizer,
) -> None:
    """Simulate the teacher offset correction that happens in training.

    In real training:
      - Alignment is computed on standalone response text
      - Teacher logprobs come from in-context tokenization (prompt + chat template + response)
      - Chat template may insert extra tokens before the response, shifting indices

    This function reproduces that logic and prints the before/after comparison.
    """
    print(f"\n{SEPARATOR}")
    print("  Chat Template Offset Simulation")
    print(SEPARATOR)

    # --- Teacher side: standalone vs in-context tokenization ---
    standalone_ids = teacher_tokenizer(text, add_special_tokens=False)["input_ids"]

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "(user turn placeholder)"},
        {"role": "assistant", "content": text},
    ]

    if hasattr(teacher_tokenizer, "apply_chat_template"):
        full_ids = teacher_tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, return_tensors="pt"
        ).squeeze(0)

        prompt_only = messages[:-1]
        prompt_ids = teacher_tokenizer.apply_chat_template(
            prompt_only, add_generation_prompt=True, return_tensors="pt"
        ).squeeze(0)
        prompt_len = prompt_ids.shape[0]
    else:
        prompt_text = prompt
        prompt_enc = teacher_tokenizer(prompt_text, add_special_tokens=True)
        gen_enc = teacher_tokenizer(text, add_special_tokens=False)
        full_ids = torch.tensor(prompt_enc["input_ids"] + gen_enc["input_ids"])
        prompt_len = len(prompt_enc["input_ids"])

    ic_ids = full_ids[prompt_len:].tolist()  # in-context response token ids

    print(f"\n  [teacher standalone tokenization]  (response text only, add_special_tokens=False)")
    print(f"  ids[:15] = {standalone_ids[:15]}")
    for i, tid in enumerate(standalone_ids[:15]):
        print(f"    [{i:>3}] {tid:>10}  {_decode_token(teacher_tokenizer, tid)!r}")

    print(f"\n  [teacher in-context tokenization]  (after chat template, response portion)")
    print(f"  prompt_len = {prompt_len},  response_len = {len(ic_ids)}")
    print(f"  ic_ids[:15] = {ic_ids[:15]}")
    for i, tid in enumerate(ic_ids[:15]):
        print(f"    [{i:>3}] {tid:>10}  {_decode_token(teacher_tokenizer, tid)!r}")

    # --- Find offset (same logic as algorithm.py) ---
    t_offset = 0
    if standalone_ids:
        match_len = min(5, len(standalone_ids))
        for o in range(len(ic_ids) - match_len + 1):
            if ic_ids[o : o + match_len] == standalone_ids[:match_len]:
                t_offset = o
                break

    if t_offset > 0:
        print(f"\n  ** OFFSET DETECTED: {t_offset} **")
        print(f"  Extra tokens inserted by chat template before response:")
        for i in range(t_offset):
            print(f"    [{i:>3}] {ic_ids[i]:>10}  {_decode_token(teacher_tokenizer, ic_ids[i])!r}")
    else:
        print(f"\n  No offset detected (standalone == in-context start)")

    # --- Apply offset to a deep copy of chunks ---
    corrected = copy.deepcopy(alignment)
    if t_offset > 0:
        for chunk in corrected.chunks:
            chunk.teacher_token_indices = [idx + t_offset for idx in chunk.teacher_token_indices]
    # Use in-context ids as the teacher id sequence for display
    corrected.teacher_token_ids = ic_ids

    print(f"\n  [chunks BEFORE offset correction]")
    print(THIN_SEP)
    for i, chunk in enumerate(alignment.chunks):
        print_chunk_detail(
            i, chunk, alignment.text,
            teacher_tokenizer, student_tokenizer,
            alignment.teacher_token_ids, alignment.student_token_ids,
        )

    print(f"\n  [chunks AFTER offset correction (offset={t_offset})]")
    print(f"  (teacher indices now reference in-context sequence, "
          f"which is what teacher logprobs are computed over)")
    print(THIN_SEP)
    for i, chunk in enumerate(corrected.chunks):
        print_chunk_detail(
            i, chunk, corrected.text,
            teacher_tokenizer, student_tokenizer,
            corrected.teacher_token_ids, corrected.student_token_ids,
        )
    print(THIN_SEP)

    # --- Student side: check if student also needs template handling ---
    print(f"\n  [student side]")
    student_standalone = student_tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(student_tokenizer, "apply_chat_template"):
        s_messages = copy.deepcopy(messages)
        s_full = student_tokenizer.apply_chat_template(
            s_messages, add_generation_prompt=False, return_tensors="pt"
        ).squeeze(0)
        s_prompt_only = s_messages[:-1]
        s_prompt_ids = student_tokenizer.apply_chat_template(
            s_prompt_only, add_generation_prompt=True, return_tensors="pt"
        ).squeeze(0)
        s_prompt_len = s_prompt_ids.shape[0]
        s_ic_ids = s_full[s_prompt_len:].tolist()

        print(f"  student standalone[:10] = {student_standalone[:10]}")
        print(f"  student in-context[:10] = {s_ic_ids[:10]}")
        if student_standalone[:5] != s_ic_ids[:5]:
            print(f"  ** NOTE: student tokenization also differs with chat template! **")
        else:
            print(f"  Student standalone and in-context match (no offset needed).")
    else:
        print(f"  Student tokenizer has no apply_chat_template; skipped.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TEXTS = [
    "Hello world!",
    "The quick brown fox jumps over the lazy dog.",
    "这是一个中英文混合的测试 with mixed English。",
    "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug cross-tokenizer chunk alignment")
    parser.add_argument("--teacher_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Teacher model name or path")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Student model name or path")
    parser.add_argument("--text", type=str, nargs="*", default=None,
                        help="Text(s) to align. If not provided, uses built-in examples.")
    parser.add_argument("--simulate_original_ids", action="store_true",
                        help="Also run align_tokens_from_original_student_ids (simulates on-policy)")
    parser.add_argument("--merge_min_bytes", type=int, default=0,
                        help="If >0, also show merged chunks with this min_bytes threshold")
    parser.add_argument("--method", choices=["byte_offset", "decoded_pieces", "both"], default="both",
                        help="Which alignment method(s) to run")
    parser.add_argument("--simulate_chat_template", action="store_true",
                        help="Simulate chat template offset correction (as done in training)")
    parser.add_argument("--prompt", type=str, default="You are a helpful assistant.",
                        help="System prompt for chat template simulation")
    args = parser.parse_args()

    texts = args.text if args.text else DEFAULT_TEXTS

    print(f"Loading teacher tokenizer: {args.teacher_model}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    print(f"Loading student tokenizer: {args.student_model}")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    print(f"Teacher vocab size: {teacher_tokenizer.vocab_size}")
    print(f"Student vocab size: {student_tokenizer.vocab_size}")

    for text in texts:
        # --- Method 1: byte offset alignment ---
        if args.method in ("byte_offset", "both"):
            alignment = align_tokens_by_byte_offset(text, teacher_tokenizer, student_tokenizer)
            print_alignment("align_tokens_by_byte_offset", alignment, teacher_tokenizer, student_tokenizer)

            if args.merge_min_bytes > 0:
                merged = merge_alignment_chunks(alignment, min_bytes=args.merge_min_bytes)
                print_alignment(
                    f"merged (min_bytes={args.merge_min_bytes})",
                    merged, teacher_tokenizer, student_tokenizer,
                )

        # --- Method 2: decoded pieces alignment ---
        if args.method in ("decoded_pieces", "both"):
            alignment_dp, piece_stats = align_tokens_by_decoded_pieces(
                text, teacher_tokenizer, student_tokenizer, return_stats=True,
            )
            print_alignment(
                "align_tokens_by_decoded_pieces", alignment_dp,
                teacher_tokenizer, student_tokenizer, stats=piece_stats,
            )

        # --- Method 3: from original student ids (on-policy simulation) ---
        if args.simulate_original_ids:
            enc = student_tokenizer(text, add_special_tokens=False)
            original_ids = enc["input_ids"]
            alignment_orig, orig_stats = align_tokens_from_original_student_ids_with_stats(
                text=text,
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
                original_student_token_ids=original_ids,
            )
            print_alignment(
                "align_tokens_from_original_student_ids",
                alignment_orig, teacher_tokenizer, student_tokenizer, stats=orig_stats,
            )

        # --- Chat template offset simulation ---
        if args.simulate_chat_template:
            # Use the byte_offset alignment as the base
            base_alignment = align_tokens_by_byte_offset(text, teacher_tokenizer, student_tokenizer)
            simulate_chat_template_offset(
                text=text,
                prompt=args.prompt,
                alignment=base_alignment,
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
            )

    print(f"\n{'=' * 90}")
    print("  Done.")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()

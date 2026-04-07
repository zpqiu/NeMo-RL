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
"""

from __future__ import annotations

import argparse
import sys

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

    # Teacher side
    t_ids = [teacher_ids[i] for i in chunk.teacher_token_indices]
    t_tokens = [_decode_token(teacher_tokenizer, tid) for tid in t_ids]
    print(f"    teacher indices: {chunk.teacher_token_indices}")
    print(f"    teacher ids:     {t_ids}")
    print(f"    teacher tokens:  {t_tokens}")

    # Student side
    s_ids = [student_ids[i] for i in chunk.student_token_indices]
    s_tokens = [_decode_token(student_tokenizer, tid) for tid in s_ids]
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

    print(f"\n{'=' * 90}")
    print("  Done.")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()

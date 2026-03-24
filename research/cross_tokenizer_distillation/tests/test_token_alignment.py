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

"""Tests for token_alignment module."""

from __future__ import annotations

import torch
import pytest

from cross_tokenizer_distillation.token_alignment import (
    AlignmentChunk,
    AlignmentResult,
    align_tokens_by_byte_offset,
    compute_chunk_logprobs,
    batch_align,
)


# ---------------------------------------------------------------------------
# Mock tokenizer — simple whitespace-based splitter for deterministic tests
# ---------------------------------------------------------------------------

class MockTokenizer:
    """A trivial tokenizer that splits on whitespace, with known byte spans."""

    def __init__(self, vocab: dict[str, int] | None = None, split_fn=None):
        self._vocab = vocab or {}
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self._split_fn = split_fn or str.split
        self._next_id = max(self._vocab.values(), default=-1) + 1

    def __call__(self, text: str, return_offsets_mapping: bool = False, add_special_tokens: bool = True):
        tokens = self._split_fn(text)
        ids = []
        offsets = []
        cursor = 0
        for tok in tokens:
            idx = text.find(tok, cursor)
            ids.append(self._vocab.get(tok, self._get_or_create_id(tok)))
            offsets.append((idx, idx + len(tok)))
            cursor = idx + len(tok)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def _get_or_create_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._id_to_token[self._next_id] = token
            self._next_id += 1
        return self._vocab[token]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_token.get(i, "?") for i in ids)


def _char_split(text: str) -> list[str]:
    """Split into individual characters (no whitespace collapsing)."""
    return list(text)


# ===========================================================================
# Step 3 tests: dataclass
# ===========================================================================

class TestDataclass:
    def test_alignment_chunk_creation(self):
        chunk = AlignmentChunk(byte_start=0, byte_end=5, teacher_token_indices=[0], student_token_indices=[0, 1])
        assert chunk.byte_start == 0
        assert chunk.byte_end == 5
        assert chunk.teacher_token_indices == [0]
        assert chunk.student_token_indices == [0, 1]

    def test_alignment_result_creation(self):
        result = AlignmentResult(text="hello world", teacher_token_ids=[1, 2], student_token_ids=[3, 4, 5])
        assert result.text == "hello world"
        assert result.num_teacher_tokens == 2
        assert result.num_student_tokens == 3
        assert result.num_chunks == 0

    def test_alignment_result_with_chunks(self):
        chunks = [
            AlignmentChunk(byte_start=0, byte_end=5, teacher_token_indices=[0], student_token_indices=[0, 1]),
            AlignmentChunk(byte_start=5, byte_end=11, teacher_token_indices=[1], student_token_indices=[2]),
        ]
        result = AlignmentResult(text="hello world", teacher_token_ids=[1, 2], student_token_ids=[3, 4, 5], chunks=chunks)
        assert result.num_chunks == 2


# ===========================================================================
# Step 4 tests: same tokenizer alignment
# ===========================================================================

class TestSameTokenizer:
    def test_same_tokenizer_identity(self):
        """When both tokenizers are identical, each token should be its own chunk."""
        tok = MockTokenizer({"hello": 0, " ": 1, "world": 2})
        result = align_tokens_by_byte_offset("hello world", tok, tok)
        assert result.num_teacher_tokens == result.num_student_tokens
        # Each chunk should have exactly 1 teacher and 1 student token
        for chunk in result.chunks:
            assert len(chunk.teacher_token_indices) >= 1
            assert len(chunk.student_token_indices) >= 1
        # All tokens should be covered
        all_teacher = sorted(sum((c.teacher_token_indices for c in result.chunks), []))
        all_student = sorted(sum((c.student_token_indices for c in result.chunks), []))
        assert all_teacher == list(range(result.num_teacher_tokens))
        assert all_student == list(range(result.num_student_tokens))

    def test_single_token(self):
        tok = MockTokenizer({"hello": 0})
        result = align_tokens_by_byte_offset("hello", tok, tok)
        assert result.num_chunks == 1
        assert result.chunks[0].teacher_token_indices == [0]
        assert result.chunks[0].student_token_indices == [0]


# ===========================================================================
# Step 5 tests: cross tokenizer alignment
# ===========================================================================

class TestCrossTokenizer:
    def test_word_vs_char_tokenizer(self):
        """Word-level teacher vs char-level student."""
        teacher = MockTokenizer({"hello": 0, " world": 1})
        student = MockTokenizer()
        # Student splits per character
        student._split_fn = _char_split

        result = align_tokens_by_byte_offset("hello world", teacher, student)

        # All teacher tokens covered
        all_teacher = sorted(sum((c.teacher_token_indices for c in result.chunks), []))
        assert all_teacher == list(range(result.num_teacher_tokens))

        # All student tokens covered
        all_student = sorted(sum((c.student_token_indices for c in result.chunks), []))
        assert all_student == list(range(result.num_student_tokens))

        # Chunks should be contiguous and non-overlapping
        for i in range(1, len(result.chunks)):
            assert result.chunks[i].byte_start >= result.chunks[i - 1].byte_end

    def test_different_granularity(self):
        """Teacher splits 'ab|cd', student splits 'a|bcd'."""
        teacher = MockTokenizer()
        teacher._split_fn = lambda text: ["ab", "cd"] if text == "abcd" else text.split()
        student = MockTokenizer()
        student._split_fn = lambda text: ["a", "bcd"] if text == "abcd" else text.split()

        result = align_tokens_by_byte_offset("abcd", teacher, student)

        # Both sides fully covered
        all_teacher = sorted(sum((c.teacher_token_indices for c in result.chunks), []))
        all_student = sorted(sum((c.student_token_indices for c in result.chunks), []))
        assert all_teacher == list(range(result.num_teacher_tokens))
        assert all_student == list(range(result.num_student_tokens))

    def test_batch_align(self):
        tok_a = MockTokenizer()
        tok_b = MockTokenizer()
        tok_b._split_fn = _char_split
        texts = ["hello world", "foo bar"]
        results = batch_align(texts, tok_a, tok_b)
        assert len(results) == 2
        for r in results:
            assert r.num_chunks > 0


# ===========================================================================
# Step 6 tests: chunk logprob aggregation
# ===========================================================================

class TestChunkLogprobs:
    def test_basic_aggregation(self):
        """Sum of token logprobs within a chunk."""
        chunks = [
            AlignmentChunk(byte_start=0, byte_end=5, teacher_token_indices=[0, 1], student_token_indices=[0]),
            AlignmentChunk(byte_start=5, byte_end=10, teacher_token_indices=[2], student_token_indices=[1, 2]),
        ]
        teacher_lps = torch.tensor([-1.0, -2.0, -3.0])
        student_lps = torch.tensor([-0.5, -1.5, -2.5])

        teacher_chunk_lps = compute_chunk_logprobs(teacher_lps, chunks, "teacher")
        student_chunk_lps = compute_chunk_logprobs(student_lps, chunks, "student")

        assert teacher_chunk_lps.shape == (2,)
        assert student_chunk_lps.shape == (2,)

        # Chunk 0 teacher: log(-1) + log(-2) = -3.0
        torch.testing.assert_close(teacher_chunk_lps[0], torch.tensor(-3.0))
        # Chunk 1 teacher: -3.0
        torch.testing.assert_close(teacher_chunk_lps[1], torch.tensor(-3.0))
        # Chunk 0 student: -0.5
        torch.testing.assert_close(student_chunk_lps[0], torch.tensor(-0.5))
        # Chunk 1 student: -1.5 + -2.5 = -4.0
        torch.testing.assert_close(student_chunk_lps[1], torch.tensor(-4.0))

    def test_empty_chunks(self):
        lps = torch.tensor([-1.0, -2.0])
        result = compute_chunk_logprobs(lps, [], "teacher")
        assert result.shape == (0,)

    def test_single_chunk(self):
        chunks = [AlignmentChunk(byte_start=0, byte_end=5, teacher_token_indices=[0], student_token_indices=[0, 1])]
        lps = torch.tensor([-1.0, -2.0])
        result = compute_chunk_logprobs(lps, chunks, "student")
        torch.testing.assert_close(result[0], torch.tensor(-3.0))

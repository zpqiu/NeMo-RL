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

"""Tests for algorithm.py — uses mock tokenizers (no nemo_rl needed)."""

from __future__ import annotations

import torch
import pytest

from cross_tokenizer_distillation.token_alignment import (
    align_tokens_by_byte_offset,
    compute_chunk_logprobs,
)
from cross_tokenizer_distillation.cross_tokenizer_loss import (
    CrossTokenizerDistillationLossFn,
)


# ---------------------------------------------------------------------------
# Mock tokenizers with different vocabularies
# ---------------------------------------------------------------------------

class MockWordTokenizer:
    """Tokenizer that splits on whitespace — simulates a word-piece model."""

    def __init__(self, name="word_tok"):
        self.name = name
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {"<pad>": 0}
        self._next_id = 1

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        tokens = text.split()
        ids = []
        offsets = []
        cursor = 0
        for tok in tokens:
            idx = text.find(tok, cursor)
            ids.append(self._get_id(tok))
            offsets.append((idx, idx + len(tok)))
            cursor = idx + len(tok)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def _get_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._next_id += 1
        return self._vocab[token]

    def decode(self, ids, skip_special_tokens=False):
        id_to_tok = {v: k for k, v in self._vocab.items()}
        return " ".join(id_to_tok.get(i, "?") for i in ids if i != 0)

    def __len__(self):
        return len(self._vocab)


class MockCharTokenizer:
    """Tokenizer that splits into individual characters."""

    def __init__(self, name="char_tok"):
        self.name = name
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {"<pad>": 0}
        self._next_id = 1

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        ids = []
        offsets = []
        for i, ch in enumerate(text):
            ids.append(self._get_id(ch))
            offsets.append((i, i + 1))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def _get_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._next_id += 1
        return self._vocab[token]

    def decode(self, ids, skip_special_tokens=False):
        id_to_tok = {v: k for k, v in self._vocab.items()}
        return "".join(id_to_tok.get(i, "?") for i in ids if i != 0)

    def __len__(self):
        return len(self._vocab)


# ===========================================================================
# Step 11: text decode + re-tokenize tests
# ===========================================================================

class TestRetokenize:
    """Test the decode → re-tokenize → align pipeline."""

    def test_basic_retokenize_and_align(self):
        """Student generates 'hello world', teacher retokenizes it."""
        teacher_tok = MockWordTokenizer("teacher")
        student_tok = MockCharTokenizer("student")

        text = "hello world"
        alignment = align_tokens_by_byte_offset(text, teacher_tok, student_tok)

        assert alignment.text == text
        assert alignment.num_teacher_tokens == 2  # "hello", "world"
        assert alignment.num_student_tokens == 11  # one per char
        assert alignment.num_chunks > 0

        # All tokens covered
        all_t = sorted(sum((c.teacher_token_indices for c in alignment.chunks), []))
        all_s = sorted(sum((c.student_token_indices for c in alignment.chunks), []))
        assert all_t == list(range(alignment.num_teacher_tokens))
        assert all_s == list(range(alignment.num_student_tokens))

    def test_empty_text(self):
        teacher_tok = MockWordTokenizer()
        student_tok = MockCharTokenizer()

        alignment = align_tokens_by_byte_offset("", teacher_tok, student_tok)
        assert alignment.num_chunks == 0

    def test_unicode_text(self):
        """Ensure alignment handles multi-byte characters."""
        teacher_tok = MockWordTokenizer()
        student_tok = MockCharTokenizer()

        text = "你好 世界"
        alignment = align_tokens_by_byte_offset(text, teacher_tok, student_tok)
        assert alignment.num_chunks > 0


# ===========================================================================
# Step 12-13: teacher logprob + end-to-end mock test
# ===========================================================================

class TestEndToEndMock:
    """Mock end-to-end: alignment → chunk logprobs → KL loss → backward."""

    def test_full_pipeline(self):
        """Simulate the complete cross-tokenizer distillation step."""
        teacher_tok = MockWordTokenizer("teacher")
        student_tok = MockCharTokenizer("student")

        # Student generates text
        text = "hello world"

        # 1) Align
        alignment = align_tokens_by_byte_offset(text, teacher_tok, student_tok)

        # 2) Simulate teacher logprobs (one per teacher token)
        n_teacher = alignment.num_teacher_tokens
        n_student = alignment.num_student_tokens
        teacher_lps = -torch.rand(n_teacher).abs() - 0.1  # negative log-probs
        student_lps = (-torch.rand(n_student).abs() - 0.1).requires_grad_(True)

        # 3) Aggregate to chunk level
        teacher_chunk_lps = compute_chunk_logprobs(teacher_lps, alignment.chunks, "teacher")
        student_chunk_lps = compute_chunk_logprobs(student_lps, alignment.chunks, "student")

        assert teacher_chunk_lps.shape[0] == alignment.num_chunks
        assert student_chunk_lps.shape[0] == alignment.num_chunks

        # 4) Compute loss
        loss_fn = CrossTokenizerDistillationLossFn({"kl_type": "forward", "mixed_kl_weight": 0.5})
        loss, metrics = loss_fn([teacher_chunk_lps], [student_chunk_lps])

        # 5) Backward
        loss.backward()
        assert student_lps.grad is not None

        print(f"  Loss: {loss.item():.4f}, Chunks: {metrics['num_chunks']}")

    def test_batch_pipeline(self):
        """Multiple samples in a batch."""
        teacher_tok = MockWordTokenizer("teacher")
        student_tok = MockCharTokenizer("student")

        texts = ["hello world", "foo bar baz"]
        all_teacher_chunk_lps = []
        all_student_chunk_lps = []

        for text in texts:
            alignment = align_tokens_by_byte_offset(text, teacher_tok, student_tok)
            t_lps = -torch.rand(alignment.num_teacher_tokens).abs() - 0.1
            s_lps = (-torch.rand(alignment.num_student_tokens).abs() - 0.1).requires_grad_(True)

            t_chunk = compute_chunk_logprobs(t_lps, alignment.chunks, "teacher")
            s_chunk = compute_chunk_logprobs(s_lps, alignment.chunks, "student")

            all_teacher_chunk_lps.append(t_chunk)
            all_student_chunk_lps.append(s_chunk)

        loss_fn = CrossTokenizerDistillationLossFn({"kl_type": "reverse", "mixed_kl_weight": 0.5})
        loss, metrics = loss_fn(all_teacher_chunk_lps, all_student_chunk_lps)
        assert metrics["num_samples"] == 2
        assert metrics["num_chunks"] > 0

    def test_mixed_kl(self):
        """Test with mixed KL."""
        teacher_tok = MockWordTokenizer()
        student_tok = MockCharTokenizer()

        text = "test case"
        alignment = align_tokens_by_byte_offset(text, teacher_tok, student_tok)
        t_lps = torch.tensor([-1.0] * alignment.num_teacher_tokens)
        s_lps = torch.tensor([-2.0] * alignment.num_student_tokens, requires_grad=True)

        t_chunk = compute_chunk_logprobs(t_lps, alignment.chunks, "teacher")
        s_chunk = compute_chunk_logprobs(s_lps, alignment.chunks, "student")

        # Compare forward, reverse, and mixed
        for kl_type in ["forward", "reverse", "mixed"]:
            loss_fn = CrossTokenizerDistillationLossFn({"kl_type": kl_type, "mixed_kl_weight": 0.5})
            loss, _ = loss_fn([t_chunk], [s_chunk.detach().requires_grad_(True)])
            assert isinstance(loss.item(), float), f"Failed for {kl_type}"

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

"""Tests for cross-tokenizer distillation pipeline — mock tokenizers only."""

from __future__ import annotations

import torch
import pytest

from cross_tokenizer_distillation.token_alignment import (
    align_tokens_by_byte_offset,
    compute_chunk_logprobs,
)
from cross_tokenizer_distillation.cross_tokenizer_loss import (
    CrossTokenizerDistillationLossFn,
    CrossTokenizerTrainLossFn,
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
# Test: text decode + re-tokenize + align
# ===========================================================================

class TestRetokenize:
    def test_basic_retokenize_and_align(self):
        teacher_tok = MockWordTokenizer("teacher")
        student_tok = MockCharTokenizer("student")

        text = "hello world"
        alignment = align_tokens_by_byte_offset(text, teacher_tok, student_tok)

        assert alignment.text == text
        assert alignment.num_teacher_tokens == 2
        assert alignment.num_student_tokens == 11
        assert alignment.num_chunks > 0

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
        teacher_tok = MockWordTokenizer()
        student_tok = MockCharTokenizer()
        text = "你好 世界"
        alignment = align_tokens_by_byte_offset(text, teacher_tok, student_tok)
        assert alignment.num_chunks > 0


# ===========================================================================
# Test: standalone loss (CrossTokenizerDistillationLossFn)
# ===========================================================================

class TestStandaloneLoss:
    def test_full_pipeline(self):
        """alignment → chunk logprobs → KL loss → backward."""
        teacher_tok = MockWordTokenizer("teacher")
        student_tok = MockCharTokenizer("student")

        text = "hello world"
        alignment = align_tokens_by_byte_offset(text, teacher_tok, student_tok)

        n_teacher = alignment.num_teacher_tokens
        n_student = alignment.num_student_tokens
        teacher_lps = -torch.rand(n_teacher).abs() - 0.1
        student_lps = (-torch.rand(n_student).abs() - 0.1).requires_grad_(True)

        teacher_chunk_lps = compute_chunk_logprobs(teacher_lps, alignment.chunks, "teacher")
        student_chunk_lps = compute_chunk_logprobs(student_lps, alignment.chunks, "student")

        assert teacher_chunk_lps.shape[0] == alignment.num_chunks
        assert student_chunk_lps.shape[0] == alignment.num_chunks

        loss_fn = CrossTokenizerDistillationLossFn({"kl_type": "forward", "mixed_kl_weight": 0.5})
        loss, metrics = loss_fn([teacher_chunk_lps], [student_chunk_lps])
        loss.backward()
        assert student_lps.grad is not None

    def test_batch_pipeline(self):
        teacher_tok = MockWordTokenizer("teacher")
        student_tok = MockCharTokenizer("student")

        texts = ["hello world", "foo bar baz"]
        all_t, all_s = [], []

        for text in texts:
            alignment = align_tokens_by_byte_offset(text, teacher_tok, student_tok)
            t_lps = -torch.rand(alignment.num_teacher_tokens).abs() - 0.1
            s_lps = (-torch.rand(alignment.num_student_tokens).abs() - 0.1).requires_grad_(True)
            all_t.append(compute_chunk_logprobs(t_lps, alignment.chunks, "teacher"))
            all_s.append(compute_chunk_logprobs(s_lps, alignment.chunks, "student"))

        loss_fn = CrossTokenizerDistillationLossFn({"kl_type": "reverse", "mixed_kl_weight": 0.5})
        loss, metrics = loss_fn(all_t, all_s)
        assert metrics["num_samples"] == 2
        assert metrics["num_chunks"] > 0


# ===========================================================================
# Test: NeMo RL-compatible loss (CrossTokenizerTrainLossFn)
# ===========================================================================

class TestTrainLoss:
    """Test the Policy.train()-compatible loss function."""

    def _make_mock_data(self):
        """Create mock data mimicking what pack_alignment_into_data produces."""
        teacher_tok = MockWordTokenizer("teacher")
        student_tok = MockCharTokenizer("student")

        text = "hello world"
        alignment = align_tokens_by_byte_offset(text, teacher_tok, student_tok)

        # Simulate student token logprobs
        n_student = alignment.num_student_tokens
        prompt_len = 3  # fake prompt

        # Build alignment data
        teacher_lps = -torch.rand(alignment.num_teacher_tokens).abs() - 0.1
        teacher_chunk_lps = compute_chunk_logprobs(teacher_lps, alignment.chunks, "teacher")
        n_chunks = alignment.num_chunks

        # Pack into (B, S) format matching algorithm.py
        total_seq_len = prompt_len + n_student + 1
        pad_dim = total_seq_len

        xalign_teacher = torch.zeros(1, pad_dim)
        xalign_teacher[0, :n_chunks] = teacher_chunk_lps
        xalign_mask = torch.zeros(1, pad_dim)
        xalign_mask[0, :n_chunks] = 1.0
        xalign_num_toks = torch.zeros(1, pad_dim, dtype=torch.long)
        xalign_num_teacher_toks = torch.zeros(1, pad_dim, dtype=torch.long)
        xalign_start = torch.zeros(1, pad_dim, dtype=torch.long)
        for c_idx, chunk in enumerate(alignment.chunks):
            n_t = len(chunk.student_token_indices)
            xalign_num_toks[0, c_idx] = n_t
            xalign_num_teacher_toks[0, c_idx] = len(chunk.teacher_token_indices)
            if n_t > 0:
                xalign_start[0, c_idx] = chunk.student_token_indices[0]

        data = {
            "input_ids": torch.randint(1, 100, (1, total_seq_len)),
            "input_lengths": torch.tensor([total_seq_len]),
            "token_mask": torch.ones(1, total_seq_len, dtype=torch.long),
            "sample_mask": torch.ones(1, dtype=torch.float32),
            "xalign_prompt_lengths": torch.tensor([prompt_len]),
            "xalign_teacher_chunk_logprobs": xalign_teacher,
            "xalign_chunk_student_start": xalign_start,
            "xalign_chunk_mask": xalign_mask,
            "xalign_num_student_toks": xalign_num_toks,
            "xalign_num_teacher_toks": xalign_num_teacher_toks,
            "xalign_teacher_terminal_eos_logprob": torch.tensor([-0.2]),
            "xalign_student_terminal_eos_token_pos": torch.tensor([total_seq_len - 1]),
            "xalign_terminal_eos_mask": torch.tensor([1.0]),
        }

        # Simulate next_token_logprobs from forward pass
        next_token_logprobs = (-torch.rand(1, total_seq_len - 1).abs() - 0.1).requires_grad_(True)

        return data, next_token_logprobs, n_chunks

    def test_train_loss_forward(self):
        data, next_token_logprobs, n_chunks = self._make_mock_data()
        loss_fn = CrossTokenizerTrainLossFn({"kl_type": "forward", "mixed_kl_weight": 0.5})
        loss, metrics = loss_fn(
            data=data,
            global_valid_seqs=torch.tensor(1),
            global_valid_toks=torch.tensor(n_chunks),
            next_token_logprobs=next_token_logprobs,
        )
        assert loss.requires_grad
        assert metrics["num_chunks"] == n_chunks

    def test_train_loss_backward(self):
        data, next_token_logprobs, _ = self._make_mock_data()
        loss_fn = CrossTokenizerTrainLossFn({"kl_type": "forward", "mixed_kl_weight": 0.5})
        loss, _ = loss_fn(
            data=data,
            global_valid_seqs=torch.tensor(1),
            global_valid_toks=torch.tensor(1),
            next_token_logprobs=next_token_logprobs,
        )
        loss.backward()
        assert next_token_logprobs.grad is not None

    def test_train_loss_terminal_eos_only(self):
        data, next_token_logprobs, _ = self._make_mock_data()
        data["xalign_chunk_mask"].zero_()
        data["xalign_num_student_toks"].zero_()
        data["xalign_num_teacher_toks"].zero_()

        loss_fn = CrossTokenizerTrainLossFn(
            {"kl_type": "forward", "mixed_kl_weight": 0.5, "terminal_eos_weight": 1.0}
        )
        loss, metrics = loss_fn(
            data=data,
            global_valid_seqs=torch.tensor(1),
            global_valid_toks=torch.tensor(1),
            next_token_logprobs=next_token_logprobs,
        )

        assert loss.requires_grad
        assert metrics["num_chunks"] == 0
        assert metrics["num_valid_terminal_eos"] == 1

    def test_train_loss_all_kl_types(self):
        for kl_type in ["forward", "reverse", "mixed"]:
            data, next_token_logprobs, _ = self._make_mock_data()
            loss_fn = CrossTokenizerTrainLossFn({"kl_type": kl_type, "mixed_kl_weight": 0.5})
            loss, metrics = loss_fn(
                data=data,
                global_valid_seqs=torch.tensor(1),
                global_valid_toks=torch.tensor(1),
                next_token_logprobs=next_token_logprobs,
            )
            assert isinstance(loss.item(), float), f"Failed for {kl_type}"

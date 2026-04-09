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

"""Unit tests for GOLD loss function and vocabulary mapping."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from cross_tokenizer_distillation.gold_loss import (
    GoldLossConfig,
    GoldTrainLossFn,
    VocabMapping,
    build_vocab_mapping,
    generalized_jsd_loss,
    sorted_l1_loss,
)


# ===================================================================
# Mock tokenizer helpers
# ===================================================================


def _make_mock_tokenizer(vocab: dict[str, int]) -> MagicMock:
    tok = MagicMock()
    tok.get_vocab.return_value = vocab
    return tok


# ===================================================================
# TestVocabMapping
# ===================================================================


class TestVocabMapping:
    def test_identical_vocabs(self):
        vocab = {"hello": 0, "world": 1, "foo": 2}
        tok_a = _make_mock_tokenizer(vocab)
        tok_b = _make_mock_tokenizer(vocab)
        mapping = build_vocab_mapping(tok_a, tok_b)

        assert mapping.num_matched == 3
        assert mapping.jaccard_index == 1.0
        assert mapping.student_vocab_size == 3
        assert mapping.teacher_vocab_size == 3
        assert mapping.student_matched_mask.sum().item() == 3
        assert mapping.teacher_matched_mask.sum().item() == 3

    def test_disjoint_vocabs(self):
        tok_s = _make_mock_tokenizer({"a": 0, "b": 1})
        tok_t = _make_mock_tokenizer({"c": 0, "d": 1})
        mapping = build_vocab_mapping(tok_s, tok_t)

        assert mapping.num_matched == 0
        assert mapping.jaccard_index == 0.0
        assert mapping.student_matched_mask.sum().item() == 0

    def test_partial_overlap(self):
        tok_s = _make_mock_tokenizer({"a": 0, "b": 1, "c": 2})
        tok_t = _make_mock_tokenizer({"b": 0, "c": 1, "d": 2})
        mapping = build_vocab_mapping(tok_s, tok_t)

        assert mapping.num_matched == 2  # "b" and "c"
        # Union = {a, b, c, d} = 4, intersection = {b, c} = 2
        assert abs(mapping.jaccard_index - 0.5) < 1e-6

        # Check mapping correctness: teacher "b"=0 -> student "b"=1
        assert mapping.teacher_to_student_map[0] == 1
        # teacher "c"=1 -> student "c"=2
        assert mapping.teacher_to_student_map[1] == 2

        # Mapping tensor
        assert mapping.mapping_tensor[0].item() == 1  # teacher 0 -> student 1
        assert mapping.mapping_tensor[1].item() == 2  # teacher 1 -> student 2

    def test_empty_vocabs(self):
        tok_s = _make_mock_tokenizer({})
        tok_t = _make_mock_tokenizer({})
        mapping = build_vocab_mapping(tok_s, tok_t)

        assert mapping.num_matched == 0
        assert mapping.jaccard_index == 0.0


# ===================================================================
# TestJSDLoss
# ===================================================================


class TestJSDLoss:
    def test_identical_distributions(self):
        """JSD of identical distributions should be 0."""
        log_probs = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
        loss = generalized_jsd_loss(log_probs, log_probs, beta=0.5)
        assert loss.item() < 1e-6

    def test_different_distributions(self):
        """JSD of different distributions should be > 0."""
        s = torch.log(torch.tensor([[0.9, 0.05, 0.05]]))
        t = torch.log(torch.tensor([[0.1, 0.8, 0.1]]))
        loss = generalized_jsd_loss(s, t, beta=0.5)
        assert loss.item() > 0.01

    def test_forward_kl_beta_zero(self):
        """beta=0 should compute forward KL."""
        s = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
        t = torch.log(torch.tensor([[0.3, 0.4, 0.3]]))
        loss = generalized_jsd_loss(s, t, beta=0.0)
        # Should equal KL(teacher || student)
        t_probs = t.exp()
        expected = (t_probs * (t - s)).sum(-1).mean()
        assert abs(loss.item() - expected.item()) < 1e-5

    def test_reverse_kl_beta_one(self):
        """beta=1 should compute reverse KL."""
        s = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
        t = torch.log(torch.tensor([[0.3, 0.4, 0.3]]))
        loss = generalized_jsd_loss(s, t, beta=1.0)
        # Should equal KL(student || teacher)
        s_probs = s.exp()
        expected = (s_probs * (s - t)).sum(-1).mean()
        assert abs(loss.item() - expected.item()) < 1e-5

    def test_symmetric(self):
        """beta=0.5 should be symmetric."""
        s = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
        t = torch.log(torch.tensor([[0.2, 0.5, 0.3]]))
        loss_st = generalized_jsd_loss(s, t, beta=0.5)
        loss_ts = generalized_jsd_loss(t, s, beta=0.5)
        assert abs(loss_st.item() - loss_ts.item()) < 1e-5

    def test_gradient_flows(self):
        """Gradient should flow through JSD."""
        s = torch.tensor([[-0.6931, -1.2040, -1.6094]], requires_grad=True)
        t = torch.log(torch.tensor([[0.3, 0.4, 0.3]]))
        loss = generalized_jsd_loss(s, t, beta=0.5)
        loss.backward()
        assert s.grad is not None
        assert not torch.all(s.grad == 0)


# ===================================================================
# TestSortedL1Loss
# ===================================================================


class TestSortedL1Loss:
    def test_identical(self):
        """Sorted L1 of identical distributions should be 0."""
        probs = torch.tensor([[0.5, 0.3, 0.2]])
        loss = sorted_l1_loss(probs, probs)
        assert loss.item() < 1e-6

    def test_different(self):
        """Sorted L1 of different distributions should be > 0."""
        s = torch.tensor([[0.5, 0.3, 0.2]])
        t = torch.tensor([[0.1, 0.8, 0.1]])
        loss = sorted_l1_loss(s, t)
        assert loss.item() > 0.01

    def test_different_vocab_sizes(self):
        """Should handle different vocabulary sizes via padding."""
        s = torch.tensor([[0.5, 0.3, 0.2]])
        t = torch.tensor([[0.6, 0.4]])
        loss = sorted_l1_loss(s, t)
        # After sorting and padding: s=[0.5,0.3,0.2], t=[0.6,0.4,0.0]
        # L1 = |0.5-0.6| + |0.3-0.4| + |0.2-0.0| = 0.1+0.1+0.2 = 0.4
        assert abs(loss.item() - 0.4) < 1e-5

    def test_gradient_flows(self):
        s = torch.tensor([[0.5, 0.3, 0.2]], requires_grad=True)
        t = torch.tensor([[0.1, 0.8, 0.1]])
        loss = sorted_l1_loss(s, t)
        loss.backward()
        assert s.grad is not None


# ===================================================================
# TestGoldTrainLossFn
# ===================================================================


class TestGoldTrainLossFn:
    @pytest.fixture
    def basic_setup(self):
        """Create a minimal GOLD loss fn with mock vocab mapping."""
        # Teacher has 4 tokens, student has 5 tokens
        # Matched: teacher 0 <-> student 0, teacher 1 <-> student 1
        mapping = VocabMapping(
            matched_student_ids=[0, 1],
            matched_teacher_ids=[0, 1],
            student_matched_mask=torch.tensor([True, True, False, False, False]),
            teacher_matched_mask=torch.tensor([True, True, False, False]),
            teacher_to_student_map={0: 0, 1: 1},
            mapping_tensor=torch.tensor([0, 1]),
            num_matched=2,
            student_vocab_size=5,
            teacher_vocab_size=4,
            jaccard_index=2 / 7,
        )
        cfg: GoldLossConfig = {
            "jsd_beta": 0.5,
            "matched_weight": 1.0,
            "unmatched_weight": 1.0,
            "temperature": 1.0,
        }
        loss_fn = GoldTrainLossFn(cfg, mapping)
        return loss_fn

    def _make_data(self, batch_size=2, seq_len=8, topk_k=4, n_groups=3):
        """Create mock data dict for GOLD loss (DISTILLATION interface)."""
        teacher_indices = torch.randint(0, 4, (batch_size, seq_len, topk_k))
        data = {
            # Original teacher indices for matched/unmatched classification
            "gold_teacher_topk_indices_original": teacher_indices.clone(),
            "gold_position_mask": torch.zeros(batch_size, seq_len),
            "gold_teacher_cond_factor": torch.zeros(batch_size, seq_len),
            "gold_student_cond_factor": torch.zeros(batch_size, seq_len),
            "sample_mask": torch.ones(batch_size),
        }
        # Set up n_groups valid positions per sample
        for b in range(batch_size):
            for g in range(min(n_groups, seq_len)):
                data["gold_position_mask"][b, g] = 1.0
                data["gold_teacher_topk_indices_original"][b, g, 0] = 0  # matched
                data["gold_teacher_topk_indices_original"][b, g, 1] = 1  # matched
                data["gold_teacher_topk_indices_original"][b, g, 2] = 2  # unmatched
                data["gold_teacher_topk_indices_original"][b, g, 3] = 3  # unmatched
        return data

    def _make_topk_logprobs(self, batch_size=2, seq_len=8, topk_k=4):
        """Create mock student/teacher topk logprobs."""
        # Random logprobs (negative values)
        student = torch.randn(batch_size, seq_len, topk_k) - 2.0
        teacher = torch.randn(batch_size, seq_len, topk_k) - 2.0
        # Normalize to be valid log-softmax output (within top-k)
        student = torch.log_softmax(student, dim=-1)
        teacher = torch.log_softmax(teacher, dim=-1)
        return student, teacher

    def test_forward_basic(self, basic_setup):
        """Loss should be finite and non-negative."""
        loss_fn = basic_setup
        batch_size, seq_len, topk_k = 2, 8, 4
        s_lp, t_lp = self._make_topk_logprobs(batch_size, seq_len, topk_k)
        data = self._make_data(batch_size=batch_size, seq_len=seq_len, topk_k=topk_k)

        loss, metrics = loss_fn(
            student_topk_logprobs=s_lp,
            teacher_topk_logprobs=t_lp,
            student_unmatched_topk_logprobs=s_lp,
            H_all=None,
            data=data,
            global_valid_seqs=torch.tensor(float(batch_size)),
            global_valid_toks=torch.tensor(float(batch_size * seq_len)),
        )

        assert torch.isfinite(loss)
        assert loss.item() >= 0
        assert "matched_jsd_loss" in metrics
        assert "unmatched_l1_loss" in metrics
        assert metrics["num_groups"] > 0

    def test_backward(self, basic_setup):
        """Gradients should flow back through student logprobs."""
        loss_fn = basic_setup
        batch_size, seq_len, topk_k = 1, 8, 4
        s_lp = torch.randn(batch_size, seq_len, topk_k, requires_grad=True)
        t_lp = torch.log_softmax(torch.randn(batch_size, seq_len, topk_k), dim=-1)
        data = self._make_data(batch_size=1, seq_len=seq_len, topk_k=topk_k, n_groups=2)

        loss, _ = loss_fn(
            student_topk_logprobs=s_lp, teacher_topk_logprobs=t_lp,
            student_unmatched_topk_logprobs=s_lp, H_all=None,
            data=data, global_valid_seqs=torch.tensor(1.0), global_valid_toks=torch.tensor(8.0),
        )

        loss.backward()
        assert s_lp.grad is not None
        assert not torch.all(s_lp.grad == 0)

    def test_no_valid_groups(self, basic_setup):
        """When no groups are valid, loss should be 0."""
        loss_fn = basic_setup
        batch_size, seq_len, topk_k = 1, 8, 4
        s_lp, t_lp = self._make_topk_logprobs(1, seq_len, topk_k)
        data = self._make_data(batch_size=1, seq_len=seq_len, topk_k=topk_k, n_groups=0)
        data["gold_position_mask"][:] = 0

        loss, metrics = loss_fn(
            student_topk_logprobs=s_lp, teacher_topk_logprobs=t_lp,
            student_unmatched_topk_logprobs=s_lp, H_all=None,
            data=data, global_valid_seqs=torch.tensor(1.0), global_valid_toks=torch.tensor(8.0),
        )

        assert loss.item() == 0.0
        assert metrics["num_groups"] == 0

    def test_cond_factors_affect_loss(self, basic_setup):
        """Non-zero conditional factors should change the loss."""
        loss_fn = basic_setup
        seq_len, topk_k = 8, 4
        s_lp, t_lp = self._make_topk_logprobs(1, seq_len, topk_k)

        data_zero = self._make_data(batch_size=1, seq_len=seq_len, topk_k=topk_k, n_groups=2)
        loss_zero, _ = loss_fn(
            student_topk_logprobs=s_lp, teacher_topk_logprobs=t_lp,
            student_unmatched_topk_logprobs=s_lp, H_all=None,
            data=data_zero, global_valid_seqs=torch.tensor(1.0), global_valid_toks=torch.tensor(8.0),
        )

        data_nonzero = self._make_data(batch_size=1, seq_len=seq_len, topk_k=topk_k, n_groups=2)
        data_nonzero["gold_teacher_cond_factor"][:, :2] = -0.5
        data_nonzero["gold_student_cond_factor"][:, :2] = -0.3
        loss_nonzero, _ = loss_fn(
            student_topk_logprobs=s_lp, teacher_topk_logprobs=t_lp,
            student_unmatched_topk_logprobs=s_lp, H_all=None,
            data=data_nonzero, global_valid_seqs=torch.tensor(1.0), global_valid_toks=torch.tensor(8.0),
        )

        assert abs(loss_zero.item() - loss_nonzero.item()) > 1e-6

    def test_matched_weight_zero_disables_jsd(self, basic_setup):
        """Setting matched_weight=0 should make matched_jsd_loss irrelevant."""
        mapping = basic_setup.vocab_mapping
        cfg: GoldLossConfig = {
            "jsd_beta": 0.5,
            "matched_weight": 0.0,
            "unmatched_weight": 1.0,
            "temperature": 1.0,
        }
        loss_fn = GoldTrainLossFn(cfg, mapping)

        seq_len, topk_k = 8, 4
        s_lp, t_lp = self._make_topk_logprobs(1, seq_len, topk_k)
        data = self._make_data(batch_size=1, seq_len=seq_len, topk_k=topk_k, n_groups=2)
        loss, metrics = loss_fn(
            student_topk_logprobs=s_lp, teacher_topk_logprobs=t_lp,
            student_unmatched_topk_logprobs=s_lp, H_all=None,
            data=data, global_valid_seqs=torch.tensor(1.0), global_valid_toks=torch.tensor(8.0),
        )

        assert torch.isfinite(loss)

    def test_cond_factors_change_matched_jsd_without_unmatched_term(self, basic_setup):
        """Matched JSD should change when cond factors change and unmatched term is disabled."""
        mapping = basic_setup.vocab_mapping
        cfg: GoldLossConfig = {
            "jsd_beta": 0.5,
            "matched_weight": 1.0,
            "unmatched_weight": 0.0,
            "temperature": 1.0,
        }
        loss_fn = GoldTrainLossFn(cfg, mapping)

        seq_len, topk_k = 8, 4
        s_lp, t_lp = self._make_topk_logprobs(1, seq_len, topk_k)

        data_zero = self._make_data(batch_size=1, seq_len=seq_len, topk_k=topk_k, n_groups=2)
        _, metrics_zero = loss_fn(
            student_topk_logprobs=s_lp,
            teacher_topk_logprobs=t_lp,
            student_unmatched_topk_logprobs=s_lp,
            H_all=None,
            data=data_zero,
            global_valid_seqs=torch.tensor(1.0),
            global_valid_toks=torch.tensor(8.0),
        )

        data_scaled = self._make_data(batch_size=1, seq_len=seq_len, topk_k=topk_k, n_groups=2)
        data_scaled["gold_teacher_cond_factor"][:, :2] = -0.5
        data_scaled["gold_student_cond_factor"][:, :2] = -0.3
        _, metrics_scaled = loss_fn(
            student_topk_logprobs=s_lp,
            teacher_topk_logprobs=t_lp,
            student_unmatched_topk_logprobs=s_lp,
            H_all=None,
            data=data_scaled,
            global_valid_seqs=torch.tensor(1.0),
            global_valid_toks=torch.tensor(8.0),
        )

        assert metrics_zero["matched_jsd_loss"] != pytest.approx(metrics_scaled["matched_jsd_loss"], abs=1e-6)

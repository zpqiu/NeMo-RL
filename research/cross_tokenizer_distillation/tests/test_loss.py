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

"""Tests for cross_tokenizer_loss module (IS-only)."""

from __future__ import annotations

import torch
import pytest

from cross_tokenizer_distillation.cross_tokenizer_loss import (
    CrossTokenizerTrainLossFn,
    CrossTokenizerDistillationLossConfig,
    _compute_is_ratio_terms,
    _compute_is_loss_from_advantage,
    _normalize_advantages,
)


def _make_loss_fn(**overrides):
    cfg: CrossTokenizerDistillationLossConfig = {
        "terminal_eos_weight": overrides.get("terminal_eos_weight", 1.0),
        "clip_epsilon": overrides.get("clip_epsilon", 0.2),
        "advantage_normalization": overrides.get("advantage_normalization", "center"),
        "negative_advantage_weight": overrides.get("negative_advantage_weight", 1.0),
    }
    return CrossTokenizerTrainLossFn(cfg)


class TestISComponents:
    """Test IS ratio and loss helper functions."""

    def test_ratio_no_change(self):
        """When current == old, ratio should be 1."""
        lp = torch.tensor(-1.0)
        ratio, clipped = _compute_is_ratio_terms(lp, lp, clip_epsilon=0.2)
        assert abs(ratio.item() - 1.0) < 1e-6
        assert abs(clipped.item() - 1.0) < 1e-6

    def test_ratio_clipping(self):
        """Large logprob change should be clipped."""
        current = torch.tensor(-0.5)
        old = torch.tensor(-2.0)
        ratio, clipped = _compute_is_ratio_terms(current, old, clip_epsilon=0.2)
        assert ratio.item() > 1.2  # unclipped > 1+eps
        assert abs(clipped.item() - 1.2) < 1e-6  # clipped to 1+eps

    def test_loss_positive_advantage(self):
        """Positive advantage with ratio=1 should give negative loss (reward)."""
        adv = torch.tensor(1.0)
        ratio = torch.tensor(1.0)
        clipped = torch.tensor(1.0)
        loss = _compute_is_loss_from_advantage(adv, ratio, clipped)
        assert loss.item() < 0  # -advantage * ratio

    def test_loss_negative_advantage_weight(self):
        """Negative advantage should be scaled by weight."""
        adv = torch.tensor(-1.0)
        ratio = torch.tensor(1.0)
        clipped = torch.tensor(1.0)
        loss_full = _compute_is_loss_from_advantage(adv, ratio, clipped, negative_advantage_weight=1.0)
        loss_scaled = _compute_is_loss_from_advantage(adv, ratio, clipped, negative_advantage_weight=0.5)
        assert abs(loss_scaled.item()) < abs(loss_full.item())

    def test_normalize_center(self):
        """Center normalization should produce zero mean."""
        advs = [torch.tensor(1.0), torch.tensor(3.0), torch.tensor(5.0)]
        normed, mean, scale = _normalize_advantages(advs, "center")
        assert abs(mean.item() - 3.0) < 1e-6
        assert abs(sum(a.item() for a in normed)) < 1e-5

    def test_normalize_standardize(self):
        """Standardize should produce zero mean and unit scale."""
        advs = [torch.tensor(1.0), torch.tensor(3.0), torch.tensor(5.0)]
        normed, mean, scale = _normalize_advantages(advs, "standardize")
        assert abs(mean.item() - 3.0) < 1e-6
        assert scale.item() > 0

    def test_normalize_empty(self):
        normed, mean, scale = _normalize_advantages([], "center")
        assert normed == []

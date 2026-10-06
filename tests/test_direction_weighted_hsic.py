"""Tests for the DIRECTION-WEIGHTED HSIC aggregation (hsic_aggregation="direction").

The pair weight of the structural HSIC term is the deterministic antisymmetric
direction gate ``d[i, j]`` of ``GatedSelfAttention`` (``d[i,j] ~ 1`` = "j is a
parent of i"), NOT the gated posterior ``z_edge * d``.  Because the coupled
Binary-Concrete parametrisation enforces ``d[i,j] + d[j,i] == 1`` per unordered
pair (``dir_bias == 0``), every pair always pays one of its two HSIC terms in
full: the optimiser can only choose WHICH node pays (orientation), never
WHETHER the pair pays -- the trivial all-zero-weight solution of the
attention-weighted variant is removed by construction.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test_dropout_selection import _make_batch, _make_forecaster_config

from causaliT.core.modules.gated_self_attention import GatedSelfAttention
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.hsic_utils import (
    hsic_attention_weighted,
    hsic_direction_weighted,
)


def _data(batch=48, n=4, seed=100):
    """X values and residuals with X_0 -> X_1 dependence."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, n, generator=g)
    x[:, 1] = 0.8 * x[:, 0] + 0.2 * torch.randn(batch, generator=g)
    residuals = torch.randn(batch, n, generator=g)
    residuals[:, 1] = 0.5 * x[:, 0] + 0.3 * torch.randn(batch, generator=g)
    return x, residuals


class TestDirectionWeightedValues:
    def test_matches_attention_weighted_with_same_weights(self):
        x, res = _data()
        d = torch.rand(4, 4, generator=torch.Generator().manual_seed(1))
        out_dir, mat_dir = hsic_direction_weighted(
            x, res, d, sigma=1.0, return_matrix=True
        )
        out_att, mat_att = hsic_attention_weighted(
            x, res, d, sigma=1.0, exclude_diagonal=True, return_matrix=True
        )
        assert torch.allclose(out_dir, out_att)
        assert torch.allclose(
            torch.nan_to_num(mat_dir), torch.nan_to_num(mat_att)
        )

    def test_diagonal_weight_is_excluded(self):
        x, res = _data()
        d = torch.full((4, 4), 0.5)
        base = hsic_direction_weighted(x, res, d, sigma=1.0)
        d_big = d.clone()
        d_big.fill_diagonal_(1.0)
        bumped = hsic_direction_weighted(x, res, d_big, sigma=1.0)
        assert torch.allclose(base, bumped), (
            "the diagonal direction weight must never enter the objective "
            "(HSIC(X_i, r_i) is irreducible)"
        )

    def test_shape_validation(self):
        x, res = _data()
        with pytest.raises(ValueError):
            hsic_direction_weighted(x, res, torch.rand(4, 3), sigma=1.0)
        with pytest.raises(ValueError):
            hsic_direction_weighted(x, res, torch.rand(4, 4, 4), sigma=1.0)

    def test_gradient_flows_to_direction(self):
        x, res = _data()
        d = torch.full((4, 4), 0.5, requires_grad=True)
        out = hsic_direction_weighted(x, res, d, sigma=1.0)
        out.backward()
        assert d.grad is not None
        assert torch.isfinite(d.grad).all()
        assert d.grad.abs().sum() > 0.0



class TestPairCoupling:
    """The coupled gate d = sigmoid(theta), theta_ji = -theta_ij, removes the
    collapse: a pair pays d_ij * H_ij + d_ji * H_ji with d_ij + d_ji = 1."""

    def test_no_zero_weight_descent_direction(self):
        # Both weights of a pair sum to 1 for every value of the logit, so
        # neither can be driven to zero without maximising the other.
        theta = torch.linspace(-6, 6, 25)
        d = torch.sigmoid(theta)
        assert torch.allclose(d + (1 - d), torch.ones_like(d))

    def test_orientation_gradient_points_to_cheaper_child(self):
        # H_ij = HSIC(X_j, res_i) (j parent of i), H_ji = HSIC(X_i, res_j).
        # dL/dtheta_ij = d(1-d) * (H_ij - H_ji): the edge orients so the
        # node with the SMALLER residual dependence is the child.
        for h_ij, h_ji, expected_sign in ((0.9, 0.1, +1.0), (0.1, 0.9, -1.0)):
            theta = torch.tensor(0.3, requires_grad=True)
            d_ij = torch.sigmoid(theta)
            d_ji = torch.sigmoid(-theta)
            loss = d_ij * h_ij + d_ji * h_ji
            loss.backward()
            assert torch.sign(theta.grad) == expected_sign

    def test_pair_cost_is_constant_when_terms_equal(self):
        # Degenerate case H_ij == H_ji: the pair cost is exactly H whatever
        # the orientation -- no orientation gradient, but also no escape.
        theta = torch.tensor(1.1, requires_grad=True)
        d_ij = torch.sigmoid(theta)
        d_ji = torch.sigmoid(-theta)
        loss = d_ij * 0.7 + d_ji * 0.7
        loss.backward()
        assert torch.allclose(loss, torch.tensor(0.7))
        assert torch.allclose(theta.grad, torch.tensor(0.0), atol=1e-7)


class TestModuleStash:
    def test_deterministic_direction_stashed_train_and_eval(self):
        torch.manual_seed(0)
        mod = GatedSelfAttention()
        q = torch.randn(2, 4, 8, requires_grad=True)
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        for training in (True, False):
            mod.train(training)
            mod.forward(q, k, v)
            d = mod.last_direction_deterministic
            assert d is not None and d.shape == (4, 4)
            assert d.requires_grad, "the HSIC weight must carry gradient"
            # Coupled gate: d_ij + d_ji == 1, diagonal exactly 1/2.
            assert torch.allclose(d + d.T, torch.ones(4, 4), atol=1e-6)
            assert torch.allclose(
                d.diagonal(), torch.full((4,), 0.5), atol=1e-6
            )

    def test_deterministic_direction_has_no_gate_noise(self):
        torch.manual_seed(0)
        mod = GatedSelfAttention()
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        mod.train()
        mod.forward(q, k, v)
        d1 = mod.last_direction_deterministic.detach().clone()
        mod.forward(q, k, v)
        d2 = mod.last_direction_deterministic.detach().clone()
        assert torch.allclose(d1, d2), (
            "the deterministic copy must not resample the Binary-Concrete noise"
        )

    def test_dir_bias_breaks_coupling(self):
        torch.manual_seed(0)
        mod = GatedSelfAttention(dir_bias=0.5)
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        mod.train()
        mod.forward(q, k, v)
        d = mod.last_direction_deterministic.detach()
        assert (d + d.T > 1.0).any(), (
            "dir_bias != 0 must relax the d_ij + d_ji == 1 coupling "
            "(the forecaster warns about this in direction mode)"
        )


def _model_direction():
    torch.manual_seed(0)
    cfg = _make_forecaster_config()
    cfg["model"]["kwargs"]["homogeneous_nodes"] = True
    cfg["training"]["hsic_aggregation"] = "direction"
    cfg["training"]["lambda_hsic"] = 1.0
    cfg["training"]["hsic_adaptive_bandwidth"] = True
    return AttentionSelectorForecaster(cfg)


class TestForecasterBranch:
    def test_config_requires_homogeneous(self):
        cfg = _make_forecaster_config()
        cfg["training"]["hsic_aggregation"] = "direction"
        with pytest.raises(ValueError, match="homogeneous_nodes"):
            AttentionSelectorForecaster(cfg)

    def test_legacy_alias_compatible_with_direction(self):
        cfg = _make_forecaster_config()
        cfg["model"]["kwargs"]["homogeneous_nodes"] = True
        cfg["training"]["hsic_aggregation"] = "direction"
        cfg["training"]["use_attention_weighted_hsic"] = True  # compatible
        AttentionSelectorForecaster(cfg)  # must not raise

    def test_step_produces_finite_hsic_and_weight_gradient(self):
        model = _model_direction()
        S, X = _make_batch(batch=16, seed=3)
        model.train()
        total_loss, _, _ = model._step((S, X), stage="train")
        assert torch.isfinite(total_loss)
        inner = model._direction_gate_module()
        d = inner.last_direction_deterministic
        assert d is not None and d.requires_grad
        # The HSIC scalar must depend on the direction weights: gradient of
        # the weighted HSIC w.r.t. the stashed gate is non-zero.
        g = torch.autograd.grad(
            model._last_hsic_reg, d, retain_graph=False, allow_unused=False
        )[0]
        assert torch.isfinite(g).all()
        assert g.abs().sum() > 0.0, (
            "no gradient through the direction weights: the orientation "
            "signal is dead"
        )
        # Pair matrix stashed for diagnostics, diagonal excluded (NaN).
        assert model._last_hsic_pair_mat is not None
        assert torch.isnan(model._last_hsic_pair_mat.diagonal()).all()

    def test_eval_step_also_uses_direction_weights(self):
        model = _model_direction()
        S, X = _make_batch(batch=16, seed=4)
        model.eval()
        total_loss, _, _ = model._step((S, X), stage="val")
        assert torch.isfinite(total_loss)
        inner = model._direction_gate_module()
        assert inner.last_direction_deterministic is not None


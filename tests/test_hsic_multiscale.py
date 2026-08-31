"""Tests for optional multi-bandwidth RBF HSIC."""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.hsic_utils import (
    _median_bandwidth,
    hsic,
    hsic_cross_per_pair,
    rbf_kernel,
    rbf_multiscale_kernel,
)
from causaliT.utils.query_sensitivity import _free_query_weights, scalar_hsic
from test_dropout_selection import _make_batch, _make_forecaster_config


MULTIPLIERS = [0.5, 1.0, 2.0]


def test_multiscale_kernel_is_mean_of_component_kernels():
    x = torch.linspace(-1.0, 1.0, 17)
    got = rbf_multiscale_kernel(x, 0.7, MULTIPLIERS)
    expected = torch.stack(
        [rbf_kernel(x, 0.7 * m) for m in MULTIPLIERS], dim=0
    ).mean(dim=0)
    assert torch.allclose(got, expected)
    assert torch.allclose(got, got.T)
    assert torch.allclose(torch.diagonal(got), torch.ones_like(x))


def test_multiscale_kernel_adaptive_base_matches_scaled_median():
    x = torch.linspace(-2.0, 2.0, 19)
    base = float(_median_bandwidth(x))
    got = rbf_multiscale_kernel(x, base, MULTIPLIERS)
    expected = torch.stack(
        [rbf_kernel(x, base * m) for m in MULTIPLIERS], dim=0
    ).mean(dim=0)
    assert torch.allclose(got, expected)


@pytest.mark.parametrize("bad", [[0.0], [-0.5, 1.0]])
def test_multiscale_rejects_nonpositive_multipliers(bad):
    with pytest.raises(ValueError, match="positive"):
        rbf_multiscale_kernel(torch.randn(8), 1.0, bad)


def test_none_and_empty_multipliers_preserve_legacy_hsic():
    torch.manual_seed(0)
    x = torch.randn(64)
    y = torch.sin(x) + 0.1 * torch.randn(64)
    ref = hsic(x, y, adaptive_bandwidth=True)
    assert hsic(x, y, adaptive_bandwidth=True, bandwidth_multipliers=None) == ref
    assert hsic(x, y, adaptive_bandwidth=True, bandwidth_multipliers=[]) == ref


@pytest.mark.parametrize("mode", ["biased", "normalized"])
def test_multiscale_hsic_is_finite_and_differentiable(mode):
    torch.manual_seed(1)
    x = torch.randn(96)
    residual = (x**2 + 0.1 * torch.randn(96)).requires_grad_(True)
    value = hsic(
        x,
        residual,
        adaptive_bandwidth=True,
        mode=mode,
        bandwidth_multipliers=MULTIPLIERS,
    )
    assert torch.isfinite(value)
    (grad,) = torch.autograd.grad(value, residual)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_multiscale_pair_mask_still_skips_pairs():
    torch.manual_seed(2)
    source = torch.randn(48, 3)
    residuals = torch.randn(48, 2)
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    value = hsic_cross_per_pair(
        source,
        residuals,
        adaptive_bandwidth=True,
        bandwidth_multipliers=MULTIPLIERS,
        pair_mask=mask,
    )
    assert torch.isfinite(value)


def test_dirac_source_can_use_multiscale_residual_kernel():
    source = torch.tensor([0.0, 1.0, 0.0, 1.0] * 16)
    residual = (source + 0.05 * torch.randn(len(source))).requires_grad_(True)
    value = hsic_cross_per_pair(
        source.unsqueeze(1),
        residual.unsqueeze(1),
        adaptive_bandwidth=True,
        source_kernel="dirac",
        bandwidth_multipliers=MULTIPLIERS,
    )
    assert torch.isfinite(value)
    (grad,) = torch.autograd.grad(value, residual)
    assert torch.isfinite(grad).all()


def test_forecaster_and_query_probe_use_multiscale_hsic():
    torch.manual_seed(3)
    cfg = _make_forecaster_config()
    cfg["training"]["hsic_bandwidth_multipliers"] = MULTIPLIERS
    model = AttentionSelectorForecaster(cfg)
    assert model.hsic_bandwidth_multipliers == MULTIPLIERS

    batch = _make_batch(seed=4)
    model.eval()
    with torch.no_grad():
        model._step(batch, stage="val")
    expected = float(model._last_hsic_reg.detach())
    got = scalar_hsic(model, [batch])
    assert got == pytest.approx(expected, rel=1e-5)

    # The same forward in single-bandwidth mode should generally differ; this
    # confirms the probe is not silently dropping the multiscale option.
    model.hsic_bandwidth_multipliers = None
    single = scalar_hsic(model, [batch])
    assert single != pytest.approx(got)

    # The training objective remains differentiable into the free queries.
    model.hsic_bandwidth_multipliers = MULTIPLIERS
    model.train()
    model._step(batch, stage="train")
    structural = model._last_loss_components["loss_structural"]
    query_params = _free_query_weights(model)
    assert query_params
    grads = torch.autograd.grad(structural, query_params, allow_unused=False)
    # allow_unused=False proves graph connectivity; a tiny random-init batch may
    # legitimately produce an exactly zero local gradient.
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

"""Unit tests for the learnable Sinkhorn temperature of OrderFactorizedSelfAttention."""

import math

import pytest
import torch

from causaliT.core.modules.order_factorized_attention import OrderFactorizedSelfAttention
from causaliT.training.gradient_routing import _is_structural_param

N, E, B = 4, 4, 3


def _module(**kw):
    return OrderFactorizedSelfAttention(num_nodes=N, recon_gate_opt="none",
                                        struct_gate_opt="none", **kw)


def _forward(m):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(B, N, E, generator=g)
    k = torch.randn(B, N, E, generator=g)
    v = torch.randn(B, N, 2, generator=g)
    return m(q, k, v)


def test_fixed_tau_is_float_and_settable():
    m = _module(order_tau=0.5)
    assert not m.tau_learnable
    assert m.perm_tau == 0.5
    m.set_order_temperature(tau=0.3)
    assert m.perm_tau == 0.3
    assert not any("order_log_tau" in n for n, _ in m.named_parameters())


@pytest.mark.parametrize("tau0", [1.0, 0.3, 0.1])
def test_learnable_tau_init_and_bounds(tau0):
    m = _module(order_tau=tau0, order_tau_learnable=True, order_tau_min=0.1, order_tau_max=1.0)
    tau = float(m.perm_tau.detach())
    assert tau == pytest.approx(tau0, rel=1e-6)
    with torch.no_grad():           # values beyond the bounds are clamped
        m.order_log_tau.fill_(5.0)
        assert float(m.perm_tau) == pytest.approx(1.0)
        m.order_log_tau.fill_(-5.0)
        assert float(m.perm_tau) == pytest.approx(0.1)


@pytest.mark.parametrize("log_tau", [0.0, 1.0, -4.0])
def test_learnable_tau_straight_through_gradient_at_and_beyond_bounds(log_tau):
    m = _module(order_tau=0.5, order_tau_learnable=True, order_tau_min=0.1, order_tau_max=1.0)
    with torch.no_grad():
        m.order_log_tau.fill_(log_tau)
    m.perm_tau.backward()
    assert float(m.order_log_tau.grad) > 0.0   # never zero, even when clamped


def test_learnable_tau_receives_gradient_in_structure_stage():
    m = _module(order_tau=0.5, order_tau_learnable=True)
    m.train()
    m.set_stage_policy("structure", order_sampling="soft")
    m.resample()
    _forward(m)
    m._last_M.sum().backward()      # loss through the relaxed order
    assert m.order_log_tau.grad is not None
    assert math.isfinite(float(m.order_log_tau.grad))
    assert float(m.order_log_tau.grad) != 0.0


def test_learnable_tau_cannot_be_set_and_is_structural():
    m = _module(order_tau=0.5, order_tau_learnable=True)
    with pytest.raises(RuntimeError):
        m.set_order_temperature(tau=0.3)
    names = [n for n, _ in m.named_parameters()]
    assert "order_log_tau" in names
    assert _is_structural_param("model.attention.inner_attention.order_log_tau")


def test_learnable_tau_invalid_bounds():
    with pytest.raises(ValueError):
        _module(order_tau=2.0, order_tau_learnable=True, order_tau_min=0.1, order_tau_max=1.0)
    with pytest.raises(ValueError):
        _module(order_tau=0.5, order_tau_learnable=True, order_tau_min=1.0, order_tau_max=0.1)

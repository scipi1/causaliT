"""BKD pair exclusion in the anti-adjacency (log-ratio) order objective.

Calls ``AttentionSelectorForecaster._anti_adjacency_hsic`` on a lightweight
stub (no model build): the order module only needs ``_last_M`` / ``_last_pi``.
"""

from types import SimpleNamespace

import torch

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.training.structural_losses import AntiAdjacencyConfig

N, B = 4, 64


def _stub(M):
    oa = SimpleNamespace(_last_M=M, _last_pi=torch.ones(N, N))
    return SimpleNamespace(
        model=SimpleNamespace(order_attention=lambda: oa),
        _hsic_bw_frozen_sigmas=None,
        hsic_sigma=1.0,
        hsic_adaptive_bandwidth=True,
        hsic_mode="hsic",
        nhsic_epsilon=1e-6,
        hsic_kernel_source="gaussian",
        hsic_bandwidth_multipliers=None,
        hsic_null_stats=None,
        anti_adjacency=AntiAdjacencyConfig(lam=1.0, eps=0.1),
    )


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, generator=g)
    x[:, 1] = x[:, 1] + torch.tanh(x[:, 0])
    x[:, 2] = x[:, 2] + x[:, 1] ** 2
    r = x + 0.3 * torch.randn(B, N, generator=g)
    return x, r


def _call(keep, seed=0):
    M = torch.full((N, N), 0.5, requires_grad=True)
    self = _stub(M)
    x, r = _data(seed)
    loss = AttentionSelectorForecaster._anti_adjacency_hsic(
        self, None, None, causal=(x, r), keep=keep)
    return loss, M, self


def test_keep_none_equals_all_kept():
    l0, _, _ = _call(None)
    l1, _, _ = _call(torch.ones(N))
    assert torch.allclose(l0, l1)


def test_dropped_node_pairs_get_no_gradient():
    keep = torch.tensor([1.0, 1.0, 0.0, 1.0])
    loss, M, _ = _call(keep)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(M.grad[2, :] == 0) and torch.all(M.grad[:, 2] == 0)
    assert M.grad[0, 1] != 0


def test_dropped_node_row_stat_is_zero():
    keep = torch.tensor([1.0, 0.0, 1.0, 1.0])
    _, _, self = _call(keep)
    assert self._last_anti_adjacency_stats["rows"][1] == 0


def test_fewer_than_two_kept_gives_exact_zero():
    keep = torch.tensor([0.0, 0.0, 1.0, 0.0])
    loss, M, _ = _call(keep)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.all(M.grad == 0)

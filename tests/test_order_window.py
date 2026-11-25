"""Banded (top-k) Sinkhorn of OrderFactorizedSelfAttention."""

import pytest
import torch

from causaliT.core.modules.order_factorized_attention import OrderFactorizedSelfAttention

N, E, B = 10, 4, 3


def _qk(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, N, E, generator=g).expand(B, N, E).clone()
    k = torch.randn(1, N, E, generator=g).expand(B, N, E).clone()
    return q, k, torch.randn(B, N, 2, generator=g)


def _mod(window=None, tau=5.0):
    torch.manual_seed(0)
    m = OrderFactorizedSelfAttention(num_nodes=N, order_r_init="random",
                                     order_tau=tau, order_sinkhorn_window=window)
    m.order_scores.data = torch.linspace(1.0, -1.0, N)[torch.randperm(N)]
    m.train()
    m.set_stage_policy("structure", gate_sampling="hard", order_sampling="hard_st")
    return m


def _run(m, seed=1):
    m.resample(torch.Generator().manual_seed(seed))
    torch.manual_seed(123)
    out, _, _ = m(*_qk())
    return out


def test_none_and_full_window_match_dense():
    out0 = _run(_mod(None)).detach()
    M0 = _mod(None)
    _run(M0)
    for w in (N - 1, N + 5):
        m = _mod(w)
        out = _run(m).detach()
        torch.testing.assert_close(out, out0)
        torch.testing.assert_close(m._last_M, M0._last_M)


def test_far_pairs_are_exact_and_sinkhorn_converges():
    w = 2
    m = _mod(w, tau=50.0)                    # very flat dense Sinkhorn
    _run(m)
    M, Mh = m._last_M.detach(), m.last_M_actual_hard
    rank = torch.empty(N, dtype=torch.long)
    # rank from the hard order: number of predecessors
    rank[:] = Mh.sum(0).long()
    far = (rank.view(N, 1) - rank.view(1, N)).abs() > 2 * w
    # Exact up to the Sinkhorn residual (20 iterations, ~1e-5).
    torch.testing.assert_close(M[far], Mh[far], atol=1e-4, rtol=0)
    assert m.last_sinkhorn_err < 1e-3
    adjacent = (rank.view(N, 1) - rank.view(1, N)).abs() == 1
    assert (M[adjacent] - 0.5).abs().max() < 0.45     # still soft in-band


def test_far_pairs_get_no_weight_gradient():
    w = 1
    m = _mod(w, tau=50.0)
    _run(m)
    M = m._last_M
    rank = m.last_M_actual_hard.sum(0).long()
    i, j = int((rank == 0).nonzero()[0]), int((rank == N - 1).nonzero()[0])
    g, = torch.autograd.grad(M[i, j], m.order_scores, allow_unused=True)
    # Dense reference: the same far pair gets a real gradient.
    d = _mod(None, tau=50.0)
    _run(d)
    gd, = torch.autograd.grad(d._last_M[i, j], d.order_scores)
    assert g is None or g.abs().max() < 1e-3 * gd.abs().max()


def test_invalid_window():
    with pytest.raises(ValueError):
        _mod(0)

"""Graph supports, per-pair permutation null, ratio clipping and A -> G_S gating."""
from types import SimpleNamespace

import pytest
import torch

from causaliT.training.forecasters.attention_selector_forecaster import AttentionSelectorForecaster
from causaliT.training.forecaster_utils.order_path_training import graph_supports
from causaliT.training.structural_losses import AntiAdjacencyConfig
from causaliT.utils.hsic_utils import hsic_cross_per_pair, hsic_permutation_null_matrix


def test_closed_form_null_matches_permutations():
    torch.manual_seed(0)
    n = 40
    x = torch.randn(n, 3)
    r = torch.stack([x[:, 0] ** 2 + 0.3 * torch.randn(n), torch.randn(n)], dim=1)
    nu = hsic_permutation_null_matrix(x, r, adaptive_bandwidth=True)
    gen = torch.Generator().manual_seed(1)
    reps = []
    for _ in range(3000):
        perm = torch.randperm(n, generator=gen)
        reps.append(hsic_cross_per_pair(x, r[perm], adaptive_bandwidth=True,
                                        return_matrix=True)[1])
    mc = torch.stack(reps).mean(0)
    torch.testing.assert_close(nu, mc, rtol=0.05, atol=0.0)


def test_graph_supports_closure_and_cycle():
    # 0 -> 1 -> 2, 3 isolated.
    e = torch.zeros(4, 4)
    e[0, 1] = e[1, 2] = 1.0
    U, sel, cyc = graph_supports(e)
    assert not cyc
    assert U[0, 2] == 0 and U[2, 0] == 0          # connected through 1
    assert U[0, 3] == 1 and U[3, 2] == 1
    assert sel[0, 1] == sel[1, 0] == 1 and sel[0, 2] == 0
    e[2, 0] = 1.0
    assert graph_supports(e)[2]


def _fm(oa, **overrides):
    aa = dict(mode='legacy_anticausal', eps=0.1, lam=0.7, terms='both')
    for key in list(overrides):
        if key.startswith('anti_adjacency_'):
            aa[key[len('anti_adjacency_'):]] = overrides.pop(key)
    if 'anticausal_selector_grad' in overrides:
        aa['selector_grad'] = overrides.pop('anticausal_selector_grad')
    base = dict(
        model=SimpleNamespace(training=True, order_attention=lambda: oa, modules=lambda: [oa]),
        separate_structure_forward=False, config={},
        _hsic_bw_frozen_sigmas=None, hsic_sigma=1.0, hsic_adaptive_bandwidth=True,
        hsic_mode='biased', nhsic_epsilon=0.01, hsic_kernel_source='rbf',
        hsic_bandwidth_multipliers=None,
        anti_adjacency=AntiAdjacencyConfig(**aa),
        order_path_surgery=True, order_anticausal_surgery=True)
    base.update(overrides)
    return SimpleNamespace(**base)


def _legacy_loss(M, H_c, H_a, lam=0.7, eps=0.1):
    Cd = M * H_c.T
    Ad = M * H_a
    C = (Cd + Cd.T).clamp_min(0)
    A = (Ad + Ad.T).clamp_min(0)
    lr = torch.log(eps + C) - lam * torch.log(eps + A)
    triu = torch.triu(torch.ones_like(M), 1).bool()
    return lr[triu].mean()



@pytest.mark.parametrize('support', ['order', 'graph'])
def test_full_dag_reduces_to_legacy(support):
    """Without G_S the hard edges are the full DAG of the order: graph == order == legacy."""
    torch.manual_seed(3)
    N, n = 4, 24
    perm = torch.randperm(N)
    full = torch.zeros(N, N)
    for a in range(N):
        for b in range(a + 1, N):
            full[perm[a], perm[b]] = 1.0
    logits = torch.randn(N, N, requires_grad=True)
    M = torch.sigmoid(logits) * (1 - torch.eye(N))
    oa = SimpleNamespace(_last_M=M, _last_pi=torch.ones(N, N) - torch.eye(N),
                         last_M_actual_hard=full, anticausal_detach_direction=False)
    target, rc = torch.randn(n, N), torch.randn(n, N)
    ra = 0.5 * rc + 0.3 * target
    fm = _fm(oa, anti_adjacency_support=support,
             _order_residuals=lambda *a: (target, ra))
    loss = AttentionSelectorForecaster._anti_adjacency_hsic(fm, None, None, causal=(target, rc))
    kw = dict(adaptive_bandwidth=True, return_matrix=True)
    H_c = hsic_cross_per_pair(target, rc, **kw)[1]
    H_a = hsic_cross_per_pair(target, ra, **kw)[1]
    torch.testing.assert_close(loss, _legacy_loss(M, H_c, H_a))
    g = torch.autograd.grad(fm._c_weight_loss + fm._a_weight_loss, logits)[0]
    assert g.abs().sum() > 0


def test_graph_support_restricts_a_and_adds_unconnected_c():
    torch.manual_seed(4)
    N, n = 4, 24
    edges = torch.zeros(N, N)
    edges[0, 1] = 1.0                               # only 0 -> 1 selected
    M = torch.full((N, N), 0.5) * (1 - torch.eye(N))
    oa = SimpleNamespace(_last_M=M, _last_pi=torch.ones(N, N) - torch.eye(N),
                         last_M_actual_hard=edges, anticausal_detach_direction=False)
    target, rc = torch.randn(n, N), torch.randn(n, N)
    fm = _fm(oa, anti_adjacency_support='graph', anti_adjacency_null='per_pair',
             anti_adjacency_ratio_floor=True, anti_adjacency_ratio_cap=5.0,
             order_path_surgery=False, _order_residuals=lambda *a: (target, rc))
    AttentionSelectorForecaster._anti_adjacency_hsic(fm, None, None, causal=(target, rc))
    st = fm._last_anti_adjacency_stats
    assert st['a_pairs'] == 1.0
    assert st['c_pairs'] == 6.0
    assert st['unconnected_pairs'] == 5.0
    assert st['A'] <= 5.0 and st['C'] >= 1.0


def test_anticausal_detach_flag_is_set_and_reset():
    N, n = 3, 16
    seen = []
    oa = SimpleNamespace(_last_M=torch.full((N, N), 0.5), _last_pi=torch.ones(N, N),
                         last_M_actual_hard=torch.zeros(N, N), anticausal_detach_direction=False)
    target = torch.randn(n, N)

    def reverse(*a):
        seen.append(oa.anticausal_detach_direction)
        return target, torch.randn(n, N)
    fm = _fm(oa, anticausal_selector_grad='none', order_path_surgery=False,
             _order_residuals=reverse)
    AttentionSelectorForecaster._anti_adjacency_hsic(fm, None, None, causal=(target, torch.randn(n, N)))
    assert seen == [True] and oa.anticausal_detach_direction is False


@pytest.mark.parametrize('detach', [True, False])
def test_directional_gate_gradient_from_anticausal_pass(detach):
    from causaliT.core.modules.order_factorized_attention import OrderFactorizedSelfAttention
    torch.manual_seed(5)
    oa = OrderFactorizedSelfAttention(num_nodes=4, order_r_init='random',
                                      recon_gate_opt='none', struct_gate_opt='directional_free')
    oa.train()
    oa.set_stage_policy('structure', gate_sampling='hard', order_sampling='hard_st')
    oa.set_bkd_phase_active(False)
    q, v = torch.randn(2, 4, 4), torch.randn(2, 4, 2)
    oa.resample(torch.Generator().manual_seed(8))
    oa.orientation = 'anticausal'
    oa.anticausal_detach_direction = detach
    out = oa(q, q, v)
    out = out[0] if isinstance(out, tuple) else out
    out.square().sum().backward()
    g_dir = oa.struct_gate_logits.grad
    assert oa.order_scores.grad is not None and oa.order_scores.grad.abs().sum() > 0
    if detach:
        assert g_dir is None or g_dir.abs().sum() == 0
    else:
        assert g_dir is not None and g_dir.abs().sum() > 0

"""Fast checks of the production causal-reference backward and decomposition."""
from types import SimpleNamespace

import pytest
import torch

from causaliT.training.forecasters.attention_selector_forecaster import AttentionSelectorForecaster
from causaliT.training.order_path_training import protect_weight


@pytest.mark.parametrize('aux', [2., -2.])
def test_scalar_reference(aux):
    reference = torch.tensor(-1.)
    kept = protect_weight(reference, torch.tensor(aux))
    torch.testing.assert_close(kept, torch.tensor(0. if aux > 0 else aux))


@pytest.mark.parametrize('enabled', [False, True])
def test_production_backward(enabled):
    scores = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
    rho = torch.nn.Parameter(torch.tensor(0.7))
    other = torch.nn.Parameter(torch.tensor(0.3))
    cw = scores[0] + rho
    cr = -2 * scores[0] + 3 * scores[1] - 2 * rho
    aw = -scores[1] + rho
    ar = -scores[1] + rho
    path = cw + cr + (aw + ar if enabled else 0)
    loss = 0.7 * path + other.square() + 0.2 * scores.square().sum()
    oa = SimpleNamespace(order_scores=scores, order_tau_rho=rho)
    fm = SimpleNamespace(model=SimpleNamespace(order_attention=lambda: oa),
                         order_path_surgery=True, order_anticausal_surgery=enabled,
                         _c_weight_loss=cw, _c_residual_loss=cr,
                         _a_weight_loss=aw, _a_residual_loss=ar, _c_path_loss=path,
                         log=lambda *a, **kw: None,
                         manual_backward=lambda value: value.backward())
    fm._causal_reference_backward = lambda value: AttentionSelectorForecaster._causal_reference_backward(fm, value)
    gcw = torch.tensor([1., 0.])
    gcr = torch.tensor([-2., 3.])
    reference = gcw + protect_weight(gcw, gcr)
    expected = reference
    if enabled:
        expected = expected + protect_weight(reference, torch.tensor([0., -2.]))
    AttentionSelectorForecaster._structural_backward(fm, loss)
    torch.testing.assert_close(scores.grad, 0.7 * expected + 0.4 * scores.detach())
    torch.testing.assert_close(rho.grad, torch.tensor(-0.7))
    torch.testing.assert_close(other.grad, torch.tensor(0.6))


def test_four_path_decomposition():
    torch.manual_seed(4)
    score = torch.randn(4, 4, requires_grad=True)
    rho = torch.tensor(0.3, requires_grad=True)
    mask = torch.sigmoid(score / torch.nn.functional.softplus(rho))
    hc = (mask @ mask.T).square()
    ha = (mask.T @ mask).square()
    def term(m, h):
        directed = m * h.T
        return torch.log(0.1 + directed + directed.T).mean()
    full = term(mask, hc) - 0.8 * term(mask, ha.T)
    paths = [term(mask, hc.detach()), term(mask.detach(), hc),
             -0.8 * term(mask, ha.detach().T), -0.8 * term(mask.detach(), ha.T)]
    components = [torch.autograd.grad(p, (score, rho), retain_graph=True) for p in paths]
    total = torch.autograd.grad(full, (score, rho))
    for i in range(2):
        torch.testing.assert_close(sum(g[i] for g in components), total[i])


def test_reverse_forward_reuses_noise_and_transposes_conditioning():
    from causaliT.core.modules.order_factorized_attention import OrderFactorizedSelfAttention
    torch.manual_seed(5)
    model = OrderFactorizedSelfAttention(num_nodes=4, order_r_init='random',
                                         recon_gate_opt='none', struct_gate_opt='none')
    model.train()
    model.set_stage_policy('structure', gate_sampling='hard', order_sampling='hard_st')
    model.set_bkd_phase_active(False)
    q = torch.randn(2, 4, 4)
    v = torch.randn(2, 4, 2)
    model.resample(torch.Generator().manual_seed(8))
    model(q, q, v)
    causal = model.last_applied_A.clone()
    mask = model._last_M.detach().clone()
    model.orientation = 'anticausal'
    model(q, q, v)
    torch.testing.assert_close(model.last_applied_A, causal.transpose(-1, -2))
    torch.testing.assert_close(model._last_M, mask)


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('terms', ['both', 'causal'])
def test_anticausal_loss_owns_policy_and_logs_signed_terms(training, terms):
    """Precomputed causal residuals must not leave A under ambient recon policy."""
    from causaliT.core.modules.order_factorized_attention import OrderFactorizedSelfAttention
    torch.manual_seed(12)
    oa = OrderFactorizedSelfAttention(num_nodes=4, batch_key_dropout=0.5)
    oa.train(training)
    oa.set_stage_policy('recon', gate_sampling='soft', order_sampling='none')
    oa._last_M = torch.sigmoid(torch.randn(4, 4, requires_grad=True))
    oa._last_pi = torch.ones(4, 4)
    target, residual = torch.randn(12, 4), torch.randn(12, 4)
    model = SimpleNamespace(training=training, order_attention=lambda: oa, modules=lambda: [oa])
    calls = []
    def reverse(S, X, overrides, orientation):
        calls.append(orientation)
        assert oa.stage == 'structure'
        assert oa.order_sampling == ('hard_st' if training else 'hard')
        assert not oa._bkd_phase_active
        return target, residual * 0.5 + target * 0.3
    fm = SimpleNamespace(
        model=model, separate_structure_forward=True,
        config={'adaptive_training': {'structure': {'order_sampling': 'hard_st'}}},
        _order_residuals=reverse, detach_order_mask=False,
        _hsic_bw_frozen_sigmas=None, hsic_sigma=1.0, hsic_adaptive_bandwidth=True,
        hsic_mode='biased', nhsic_epsilon=0.01, hsic_kernel_source='rbf',
        hsic_bandwidth_multipliers=None, anti_adjacency_detach_c_residuals=False,
        anti_adjacency_detach_a_residuals=False, anti_adjacency_mode='legacy_anticausal',
        anti_adjacency_eps=0.1, anti_adjacency_lambda=0.7,
        anti_adjacency_terms=terms, order_path_surgery=False)
    loss = AttentionSelectorForecaster._anti_adjacency_hsic(
        fm, None, None, causal=(target, residual))
    stats = fm._last_anti_adjacency_stats
    assert calls == ['anticausal']
    assert oa.stage == 'recon' and oa.order_sampling == 'none'
    assert oa.gate_mode == 'soft' and oa._bkd_phase_active
    torch.testing.assert_close(loss.detach(), torch.tensor(stats['causal_log_term'] + stats['anticausal_log_term']))
    if terms == 'causal':
        assert stats['anticausal_log_term'] == 0.0
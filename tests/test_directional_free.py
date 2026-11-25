"""Complementary structural selector and joint edge ST sampling."""
import pytest
import torch

from causaliT.core.modules.order_factorized_attention import OrderFactorizedSelfAttention
from causaliT.training.stage_policy import resolve_gate_opts


def make(recon='none'):
    return OrderFactorizedSelfAttention(num_nodes=4, struct_gate_opt='directional_free',
                                       recon_gate_opt=recon, order_tau_learnable=True)


def inputs():
    g = torch.Generator().manual_seed(42)
    return tuple(torch.randn(2, 4, 4, generator=g, requires_grad=True) for _ in range(3))


def test_configuration_and_complementarity():
    assert resolve_gate_opts({'adaptive_training': {'structure': {
        'gate_opt': 'directional_free'}}})['struct_gate_opt'] == 'directional_free'
    m = make()
    assert m.struct_gate_logits.numel() == 6
    off = 1 - torch.eye(4)
    torch.testing.assert_close(m.directional_gate(), off * .5)
    with torch.no_grad():
        m.struct_gate_logits.copy_(torch.arange(6.) - 3)
    d = m.directional_gate()
    torch.testing.assert_close(d + d.T, off)
    grad = torch.autograd.grad(d[0, 1], m.struct_gate_logits, retain_graph=True)[0]
    reverse = torch.autograd.grad(d[1, 0], m.struct_gate_logits)[0]
    torch.testing.assert_close(grad, -reverse)


def test_transition_endpoints_and_gradient_scaling():
    m = make()
    off = 1 - torch.eye(4)
    m.selector_transition_m = 1.0
    torch.testing.assert_close(m.effective_directional_gate(), off)
    grad = torch.autograd.grad(m.effective_directional_gate()[0, 1], m.struct_gate_logits)[0]
    assert grad.abs().sum() == 0
    m.selector_transition_m = .5
    torch.testing.assert_close(m.effective_directional_gate(), off * .75)
    grad = torch.autograd.grad(m.effective_directional_gate()[0, 1], m.struct_gate_logits)[0]
    assert grad[0] == .125
    m.selector_transition_m = 0.0
    torch.testing.assert_close(m.effective_directional_gate(), m.directional_gate())


@pytest.mark.parametrize('recon', ['none', 'gated', 'softmax', 'softmax_prenorm'])
def test_joint_forward_and_gradients(recon):
    m = make(recon).train()
    m.set_stage_policy('structure', gate_sampling='hard', order_sampling='hard_st')
    m.resample(torch.Generator().manual_seed(7))
    q, k, v = inputs()
    out, _, aux = m(q, k, v)
    a = m.last_applied_A[0]
    assert torch.all(a.diag() == 0)
    assert torch.all(a * a.T == 0)
    assert aux['l0_penalty'] == 0
    torch.testing.assert_close(m._last_pi, 1 - torch.eye(4))
    out.square().sum().backward()
    for p in (m.struct_gate_logits, m.order_scores, m.order_log_tau):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


def test_sampler_probabilities_and_reuse():
    prob = torch.tensor([[0., .2], [.3, 0.]])
    draws = [make()._sample_pair_edges(prob, torch.full_like(prob, u))[0]
             for u in (.1, .35, .9)]
    assert [d.sum().item() for d in draws] == [1, 1, 0]
    assert draws[0][0, 1] == 1 and draws[1][1, 0] == 1
    m = make().train()
    m.set_stage_policy('structure', order_sampling='hard_st')
    m.resample(torch.Generator().manual_seed(4))
    x = inputs()
    first = m(*x)[0]
    torch.testing.assert_close(first, m(*x)[0], rtol=0, atol=0)


def test_reconstruction_independent_of_direction():
    m = make('gated').train()
    m.set_stage_policy('recon', order_sampling='none')
    m.resample(torch.Generator().manual_seed(4))
    x = inputs()
    first = m(*x)[0]
    with torch.no_grad():
        m.struct_gate_logits.fill_(8)
    second = m(*x)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.autograd.grad(second.sum(), m.struct_gate_logits,
                               allow_unused=True)[0] is None


def test_checkpoint_and_rejected_shared_mode():
    m = make()
    restored = make()
    restored.load_state_dict(m.state_dict(), strict=True)
    with pytest.raises(ValueError, match='independent'):
        OrderFactorizedSelfAttention(num_nodes=4, struct_gate_opt='directional_free',
                                     shared_att_selector=True)


def test_eval_hard_order_and_anticausal_sample():
    m = make().eval()
    m.set_stage_policy('structure', order_sampling='hard')
    m(*inputs())
    expected = (m.deterministic_order_mask() * m.directional_gate()).T
    torch.testing.assert_close(m.last_applied_A[0], expected)
    m.train()
    m.set_stage_policy('structure', order_sampling='hard_st')
    m.resample(torch.Generator().manual_seed(4))
    m(*inputs())
    causal = m.last_applied_A.clone()
    m.orientation = 'anticausal'
    m(*inputs())
    torch.testing.assert_close(m.last_applied_A, causal.transpose(-1, -2))
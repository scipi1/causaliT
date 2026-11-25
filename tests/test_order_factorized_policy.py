"""Tests for the stage policy and the competitive (softmax) G_R of
OrderFactorizedSelfAttention (adaptive_training.recon.gate_opt="softmax")."""

import pytest
import torch

from causaliT.core.modules.order_factorized_attention import OrderFactorizedSelfAttention
from causaliT.training.adaptive_utils.stage_policy import parse_stage_policy, resolve_gate_opts

N, E, B = 4, 4, 3


def _qk(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, N, E, generator=g).expand(B, N, E).clone().requires_grad_(True)
    k = torch.randn(1, N, E, generator=g).expand(B, N, E).clone()
    v = torch.randn(B, N, 2, generator=g)
    return q, k, v


def _mod(**kw):
    torch.manual_seed(0)
    return OrderFactorizedSelfAttention(num_nodes=N, order_r_init="random", **kw)


def _softmax_ref(S, avail):
    """Plain masked softmax with -inf (reference)."""
    w = torch.softmax(S.masked_fill(avail == 0, float("-inf")), dim=-1)
    return torch.nan_to_num(w, nan=0.0)


def _applied(m, q, k, v, seed=3):
    m.resample(torch.Generator().manual_seed(seed))
    out, _, _ = m(q, k, v)
    return out, m.last_applied_A[0]


# ---------------------------------------------------------------------------
# _masked_softmax
# ---------------------------------------------------------------------------
def test_masked_softmax_matches_minus_inf_and_log_prior():
    torch.manual_seed(1)
    S = torch.randn(N, N)
    avail = (torch.rand(N, N) > 0.3).float()
    avail[0] = 0.0                                          # fully masked row
    W = OrderFactorizedSelfAttention._masked_softmax(S, avail)
    assert torch.allclose(W, _softmax_ref(S, avail), atol=1e-6)
    assert torch.all(W[0] == 0)                             # zero row, no NaN
    prior = torch.rand(N, N)                                # soft prior == log-prior
    Wp = OrderFactorizedSelfAttention._masked_softmax(S, avail, prior)
    assert torch.allclose(Wp, _softmax_ref(S + prior.log(), avail), atol=1e-6)


def test_masked_softmax_finite_gradient_at_zero_prior():
    S = torch.randn(N, N)
    prior = torch.rand(N, N)
    prior[1, 2] = 0.0
    prior.requires_grad_(True)
    W = OrderFactorizedSelfAttention._masked_softmax(S, torch.ones(N, N), prior)
    (W * torch.randn(N, N)).sum().backward()
    assert torch.isfinite(prior.grad).all()
    assert prior.grad[1, 2] == 0                           # zero-prior boundary convention


# ---------------------------------------------------------------------------
# Applied weight of gate_opt="softmax_prenorm" (normalize, then order)
# ---------------------------------------------------------------------------
def test_prenorm_soft_is_order_times_softmax():
    q, k, v = _qk()
    m = _mod(recon_gate_opt="softmax_prenorm")
    m.train()
    m.set_stage_policy("structure", order_sampling="soft")
    out, A = _applied(m, q, k, v)
    S = m._existence_score(q, k).detach()
    Mt = m._last_M.transpose(0, 1).detach()
    ref = Mt * _softmax_ref(S, 1.0 - torch.eye(N))
    assert torch.allclose(A, ref, atol=1e-5)
    assert torch.all(A.sum(-1) <= 1.0 + 1e-6)                # no renormalization
    out.sum().backward()
    assert m.order_scores.grad is not None and m.order_scores.grad.abs().sum() > 0


def test_prenorm_hard_order_first_node_gets_zero_and_rows_shrink():
    q, k, v = _qk()
    m = _mod(recon_gate_opt="softmax_prenorm")
    m.train()
    m.set_stage_policy("recon", order_sampling="none")
    _, A_full = _applied(m, q, k, v)
    m.set_stage_policy("recon", order_sampling="hard")
    _, A_ord = _applied(m, q, k, v)
    n_par = (A_ord > 0).sum(-1)
    assert (n_par == 0).sum() == 1                            # first-placed node
    surv = A_ord > 0
    assert torch.allclose(A_ord[surv], A_full[surv], atol=1e-6)   # values unchanged
    assert torch.all(A_ord.sum(-1) <= A_full.sum(-1) + 1e-6)


def test_prenorm_scale_gradient_not_zero():
    """Plain softmax is invariant to scaling a row of the order (zero gradient
    along it); prenorm is not."""
    S = torch.randn(N, N)
    avail = 1.0 - torch.eye(N)
    m_row = (torch.rand(N, N) * 0.5 + 0.1).requires_grad_(True)
    tgt = torch.randn(N, N)
    W = OrderFactorizedSelfAttention._masked_softmax(S, avail, m_row)
    g, = torch.autograd.grad((W * tgt).sum(), m_row)
    assert torch.allclose((g * m_row).sum(-1), torch.zeros(N), atol=1e-5)
    W2 = m_row * OrderFactorizedSelfAttention._masked_softmax(S, avail)
    g2, = torch.autograd.grad((W2 * tgt).sum(), m_row)
    assert (g2 * m_row).sum(-1).abs().max() > 1e-3


# ---------------------------------------------------------------------------
# Applied weight of gate_opt="softmax"
# ---------------------------------------------------------------------------
def test_softmax_rows_renormalize_and_preserve_ratios_under_hard_order():
    q, k, v = _qk()
    m = _mod(recon_gate_opt="softmax")
    m.train()
    m.set_stage_policy("recon", order_sampling="none")
    _, A_full = _applied(m, q, k, v)
    m.set_stage_policy("recon", order_sampling="hard")
    _, A_ord = _applied(m, q, k, v)
    assert torch.allclose(A_full.sum(-1), torch.ones(N), atol=1e-6)
    for i in range(N):
        surv = A_ord[i] > 0
        if surv.sum() == 0:
            assert torch.all(A_ord[i] == 0)                 # root: zero row
            continue
        assert torch.isclose(A_ord[i].sum(), torch.tensor(1.0), atol=1e-6)
        r_full = A_full[i, surv] / A_full[i, surv].sum()   # ratios preserved
        assert torch.allclose(A_ord[i, surv], r_full, atol=1e-6)
    assert torch.all(torch.diagonal(A_ord) == 0)


def test_softmax_bkd_renormalizes():
    q, k, v = _qk()
    m = _mod(recon_gate_opt="softmax", batch_key_dropout=0.5,
             batch_key_dropout_deterministic=True)
    m.train()
    m.set_stage_policy("recon", order_sampling="none")
    torch.manual_seed(7)
    _, A = _applied(m, q, k, v)
    keep = m.last_bkd_keep
    assert keep is not None and keep.sum() < N
    assert torch.all(A[:, keep == 0] == 0)
    rows = A.sum(-1)
    assert torch.all(((rows - 1.0).abs() < 1e-6) | (rows == 0))


def test_softmax_soft_order_is_log_prior_with_order_gradient():
    q, k, v = _qk()
    m = _mod(recon_gate_opt="softmax")
    m.train()
    m.set_stage_policy("structure", order_sampling="soft")
    out, A = _applied(m, q, k, v)
    S = m._existence_score(q, k).detach()
    Mt = m._last_M.transpose(0, 1).detach()
    ref = _softmax_ref(S + Mt.clamp_min(1e-30).log(), 1.0 - torch.eye(N))
    assert torch.allclose(A, ref, atol=1e-5)
    out.sum().backward()
    assert m.order_scores.grad is not None and m.order_scores.grad.abs().sum() > 0
    assert q.grad is None or q.grad.abs().sum() == 0        # scores owned by recon


def test_softmax_recon_forward_trains_scores_not_order():
    q, k, v = _qk()
    m = _mod(recon_gate_opt="softmax")
    m.train()
    m.set_stage_policy("recon", order_sampling="soft")
    out, _ = _applied(m, q, k, v)
    out.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert m.order_scores.grad is None or m.order_scores.grad.abs().sum() == 0


def test_softmax_hard_st_forward_hard_backward_soft():
    q, k, v = _qk()
    m = _mod(recon_gate_opt="softmax")
    m.train()
    m.set_stage_policy("structure", order_sampling="hard")
    _, A_h = _applied(m, q, k, v)
    m.set_stage_policy("structure", order_sampling="hard_st")
    out, A_st = _applied(m, q, k, v)
    assert torch.allclose(A_st, A_h, atol=1e-6)             # hard forward
    out.sum().backward()
    assert m.order_scores.grad is not None and m.order_scores.grad.abs().sum() > 0


def test_softmax_struct_gate_multiplies_without_renormalization():
    q, k, v = _qk()
    m = _mod(recon_gate_opt="softmax", struct_gate_opt="gated")
    ref = _mod(recon_gate_opt="softmax")
    for x in (m, ref):
        x.eval()
        x.set_stage_policy("structure", order_sampling="none")
    with torch.no_grad():
        m.struct_gate_logits[0, 1] = -50.0                  # close pair (0, 1)
        m(q, k, v)
        ref(q, k, v)
    A, W = m.last_applied_A[0], ref.last_applied_A[0]
    assert A[0, 1] == 0 and A[1, 0] == 0
    assert A[0].sum() < W[0].sum() - 1e-6                   # mass removed
    for x in (m, ref):                                      # recon ignores G_S
        x.set_stage_policy("recon", order_sampling="none")
    with torch.no_grad():
        assert torch.allclose(m(q, k, v)[0], ref(q, k, v)[0])




# ---------------------------------------------------------------------------
# Edge = G_S x order (or M without G_S); G_R is mixing only
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("recon_opt", ["gated", "softmax"])
def test_edge_is_order_only_without_g_s(recon_opt):
    """Without G_S the edge is the full DAG of the order: the posterior is M
    and independent of q/k; L0 is zero."""
    q, k, v = _qk()
    m = _mod(recon_gate_opt=recon_opt)
    m.train()
    m.set_stage_policy("structure", order_sampling="soft")
    m.resample(torch.Generator().manual_seed(3))
    _, post, aux = m(q, k, v)
    assert torch.allclose(post[0], m._last_M.transpose(0, 1), atol=1e-6)
    assert torch.equal(m._last_pi, 1.0 - torch.eye(N))
    assert float(aux["l0_penalty"]) == 0.0
    # Edge independent of the reconstruction descriptors.
    q2, _, _ = _qk(seed=9)
    m.resample(torch.Generator().manual_seed(3))
    _, post2, _ = m(q2, k, v)
    assert torch.allclose(post[0], post2[0], atol=1e-6)
    # No q/k gradient through the posterior.
    post.sum().backward()
    assert q.grad is None or q.grad.abs().sum() == 0


@pytest.mark.parametrize("recon_opt", ["gated", "softmax"])
def test_edge_is_pi_s_times_order_with_g_s(recon_opt):
    q, k, v = _qk()
    m = _mod(recon_gate_opt=recon_opt, struct_gate_opt="gated")
    m.train()
    m.set_stage_policy("structure", order_sampling="soft")
    m.resample(torch.Generator().manual_seed(4))
    _, post, aux = m(q, k, v)
    pi_S = m.structural_pi(q, k)
    assert torch.allclose(post[0], (pi_S * m._last_M).transpose(0, 1), atol=1e-6)
    assert torch.allclose(m._last_pi, pi_S)
    exp_l0 = torch.triu(pi_S, diagonal=1).sum()
    assert torch.isclose(aux["l0_penalty"], exp_l0)


def test_l0_on_shared_g():
    q, k, v = _qk()
    m = _mod(shared_att_selector=True, struct_gate_opt="gated")
    m.train()
    m.resample(torch.Generator().manual_seed(5))
    _, _, aux = m(q, k, v)
    pi = torch.sigmoid(m._struct_logits() - m._l0_offset) * (1.0 - torch.eye(N))
    assert torch.isclose(aux["l0_penalty"], torch.triu(pi, diagonal=1).sum())


def test_softmax_recon_init_rejected():
    with pytest.raises(ValueError, match="softmax"):
        _mod(recon_gate_opt="softmax", struct_gate_opt="gated",
             struct_att_selector_init="recon")


def test_softmax_l0_is_zero_and_shared_rejected():
    q, k, v = _qk()
    m = _mod(recon_gate_opt="softmax")
    m.train()
    m.resample(torch.Generator().manual_seed(0))
    _, _, aux = m(q, k, v)
    assert float(aux["l0_penalty"]) == 0.0
    with pytest.raises(ValueError, match="softmax"):
        _mod(recon_gate_opt="softmax", struct_gate_opt="gated", shared_att_selector=True)



# ---------------------------------------------------------------------------
# Stage-policy parsing / validation
# ---------------------------------------------------------------------------
def test_parse_defaults_and_overrides():
    pol = parse_stage_policy({})
    assert pol["recon"]["gate_opt"] == "gated" and pol["structure"]["gate_opt"] == "none"
    assert pol["phases"]["warmup"]["order_sampling"] == "none"
    pol = parse_stage_policy({
        "recon": {"gate_opt": "softmax", "order_sampling": "soft"},
        "warmup": {"order_sampling": "none"},
        "structure": {"gate_opt": "gated", "order_sampling": "hard_st", "max_epochs": 3}})
    assert pol["phases"]["warmup"]["order_sampling"] == "none"
    assert pol["phases"]["reconstruct"]["order_sampling"] == "soft"
    assert pol["phases"]["final_reconstruct"]["order_sampling"] == "soft"
    assert pol["structure"]["order_sampling"] == "hard_st"
    assert resolve_gate_opts({"adaptive_training": {"recon": {"gate_opt": "softmax"}}}) == \
        {"recon_gate_opt": "softmax", "struct_gate_opt": "none"}


@pytest.mark.parametrize("ad", [
    {"recon": {"gate_opt": "bogus"}},
    {"structure": {"gate_opt": "softmax"}},
    {"recon": {"order_sampling": "hard_st"}},
    {"structure": {"order_sampling": "oriented"}},
    {"recon": {"unknown": 1}},
    {"reconstruct": {"gate_opt": "gated"}},
    {"recon_gates": "hard"},
    {"structure": {"struct_gates": "hard"}},
    {"recon": {"oriented": {"p_anticausal": 2.0}}},
])
def test_parse_rejects(ad):
    with pytest.raises(ValueError):
        parse_stage_policy(ad)


def test_set_stage_policy_validation():
    m = _mod()
    with pytest.raises(ValueError):
        m.set_stage_policy("bogus")
    with pytest.raises(ValueError):
        m.set_stage_policy("recon", order_sampling="hard_st")
    with pytest.raises(ValueError):
        m.set_stage_policy("structure", gate_sampling="relaxed")


def test_stage_switch_does_not_inherit_incompatible_order():
    m = _mod()
    m.set_stage_policy("recon", order_sampling="oriented")
    m.set_stage_policy("structure")
    assert m.order_sampling == "soft"
    with pytest.raises(ValueError):
        m.set_stage_policy("recon", order_sampling="hard_st")
    assert (m.stage, m.order_sampling) == ("structure", "soft")


def test_extreme_masked_scores_and_zero_rows_backward():
    scores = torch.tensor([[10000., -10000.], [2., 3.]], requires_grad=True)
    prior = torch.tensor([[0., 1.], [0., 0.]], requires_grad=True)
    weights = OrderFactorizedSelfAttention._masked_softmax(scores, torch.ones_like(scores), prior)
    assert torch.equal(weights, torch.tensor([[0., 1.], [0., 0.]]))
    weights.sum().backward()
    assert torch.isfinite(scores.grad).all() and torch.isfinite(prior.grad).all()


def test_shared_requires_structural_role():
    with pytest.raises(ValueError, match="structure.gate_opt"):
        _mod(shared_att_selector=True, struct_gate_opt="none")


def test_separate_gates_compose_and_l0_updates_only_structure():
    q, k, v = _qk()
    m = _mod(struct_gate_opt="gated", struct_att_selector_init_logit=0.)
    m.eval()
    m.set_stage_policy("recon", gate_sampling="soft", order_sampling="none")
    m(q, k, v)
    recon = m.last_applied_A.clone()
    m.set_stage_policy("structure", gate_sampling="soft", order_sampling="none")
    _, _, aux = m(q, k, v)
    gs = (torch.sigmoid(m._struct_logits() / m.beta) * (m.zeta - m.gamma) + m.gamma).clamp(0, 1)
    assert torch.allclose(m.last_applied_A, recon * gs.T)
    before = m.struct_gate_logits.detach().clone()
    optimizer = torch.optim.SGD([m.struct_gate_logits], lr=0.1)
    aux["l0_penalty"].backward()
    assert q.grad is None
    optimizer.step()
    assert torch.triu(m.struct_gate_logits - before, diagonal=1).sum() < 0


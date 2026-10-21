"""Unit tests for OrderFactorizedSelfAttention (docs/ideas/ORDER_FACTORIZED_VISION.md)."""

import pytest
import torch

from causaliT.core.modules.order_factorized_attention import (
    OrderFactorizedSelfAttention,
    hard_assignment,
    log_sinkhorn,
)
from causaliT.training.gradient_routing import _is_structural_param
from causaliT.utils.descendant_mask import build_hsic_pair_mask


N, E, B = 3, 4, 5


def _qk(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, N, E, generator=g).expand(B, N, E).clone().requires_grad_(True)
    k = torch.randn(1, N, E, generator=g).expand(B, N, E).clone()
    v = torch.randn(B, N, 2, generator=g)
    return q, k, v


def _module(**kw):
    torch.manual_seed(0)
    return OrderFactorizedSelfAttention(num_nodes=N, order_r_init="random", **kw)


def _is_dag(adj):
    """adj source -> target binary; acyclic iff adj is nilpotent."""
    return bool(torch.linalg.matrix_power(adj.float(), adj.shape[0]).abs().sum() == 0)


def test_sinkhorn_doubly_stochastic():
    P = log_sinkhorn(torch.randn(5, 5), n_iters=50)
    assert torch.allclose(P.sum(0), torch.ones(5), atol=1e-4)
    assert torch.allclose(P.sum(1), torch.ones(5), atol=1e-4)


def test_order_mask_orientation_and_diag():
    m = _module()
    with torch.no_grad():
        m.order_scores.copy_(torch.tensor([2.0, 0.0, -2.0]))   # order 0, 1, 2
    M = m.deterministic_order_mask()
    expected = torch.tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=M.dtype)
    assert torch.equal(M, expected)          # M[i, j] = 1 iff i before j
    assert torch.all(torch.diagonal(M) == 0)


@pytest.mark.parametrize("mode", ["relaxed", "hard"])
def test_applied_weight_is_child_parent_and_hard_is_dag(mode):
    m = _module()
    m.train()
    m.set_order_path(mode)
    q, k, v = _qk()
    for s in range(10):
        m.resample(torch.Generator().manual_seed(s))
        m(q, k, v)
        A = m.last_applied_A[0]
        assert torch.all(torch.diagonal(A) == 0)
        if mode == "hard":
            assert _is_dag(A.transpose(0, 1) > 0)
        assert _is_dag(m.score_tensor_for_sparsity.transpose(0, 1) > 0)


def test_noise_reused_within_step():
    m = _module()
    m.train()
    q, k, v = _qk()
    m.resample(torch.Generator().manual_seed(1))
    out1, _, _ = m(q, k, v)
    out2, _, _ = m(q, k, v)
    assert torch.equal(out1, out2)
    m.resample(torch.Generator().manual_seed(2))
    out3, _, _ = m(q, k, v)
    assert not torch.equal(out1, out3)


def test_gradients_relaxed_vs_hard():
    q, k, v = _qk()
    m = _module()
    m.train()
    # Hard-Concrete gates can all be exactly zero for a given draw (then no
    # gradient can flow); pick the first step with at least one open gate.
    for s in range(50):
        m.resample(torch.Generator().manual_seed(s))
        out, _, _ = m(q, k, v)
        if m.last_applied_A.abs().sum() > 0:
            break
    out.sum().backward()
    assert m.order_scores.grad is not None and m.order_scores.grad.abs().sum() > 0
    assert q.grad is not None and q.grad.abs().sum() > 0

    q2, k2, v2 = _qk()
    m2 = _module()
    m2.train()
    m2.set_order_path("hard")
    m2.resample(torch.Generator().manual_seed(0))
    out2, _, _ = m2(q2, k2, v2)
    out2.sum().backward()
    # Hard order path: the order is detached; the existence gate keeps a
    # (straight-through) gradient to the queries.
    assert m2.order_scores.grad is None or m2.order_scores.grad.abs().sum() == 0


def test_threshold_changes_exclusion_graph():
    q, k, v = _qk()
    graphs = []
    for thr in (0.01, 0.99):
        m = _module(order_existence_threshold=thr)
        m.eval()
        with torch.no_grad():
            m(q, k, v)
        graphs.append(m.score_tensor_for_sparsity.sum().item())
    assert graphs[0] > graphs[1]
    with pytest.raises(ValueError):
        _module(order_existence_threshold=1.0)


def test_sampled_exclusion_graph():
    q, k, v = _qk()
    m = _module(order_existence_threshold=0.01, order_exclusion_existence="sample")
    m.train()
    graphs = []
    for s in range(20):
        m.resample(torch.Generator().manual_seed(s))
        m(q, k, v)
        g1 = m.score_tensor_for_sparsity.clone()
        m(q, k, v)
        assert torch.equal(g1, m.score_tensor_for_sparsity)   # fixed within a step
        assert _is_dag(g1.transpose(0, 1) > 0)
        assert not g1.requires_grad
        graphs.append(g1)
    assert any(not torch.equal(graphs[0], g) for g in graphs[1:])  # varies across steps
    # Hard mode: the exclusion graph is exactly the applied DAG.
    m.set_order_path("hard")
    m.resample(torch.Generator().manual_seed(3))
    m(q, k, v)
    assert torch.equal(m.score_tensor_for_sparsity, (m.last_applied_A[0] > 0).float())
    # Eval: identical to the threshold rule.
    m.eval()
    ref = _module(order_existence_threshold=0.01)
    ref.eval()
    with torch.no_grad():
        m(q, k, v)
        ref(q, k, v)
    assert torch.equal(m.score_tensor_for_sparsity, ref.score_tensor_for_sparsity)
    with pytest.raises(ValueError):
        _module(order_exclusion_existence="bogus")


def test_full_existence_mode_is_order_dag():
    q, k, v = _qk()
    m = _module(order_existence_mode="full")
    m.train()
    m.set_order_path("hard")
    m.resample(torch.Generator().manual_seed(0))
    m(q, k, v)
    A = m.last_applied_A[0].transpose(0, 1)          # source -> target
    # Full DAG of a permutation: exactly N(N-1)/2 edges, acyclic.
    assert int(A.sum()) == N * (N - 1) // 2 and _is_dag(A > 0)
    assert torch.equal(m.score_tensor_for_sparsity.transpose(0, 1), A)
    m.set_order_path("relaxed")
    out, _, _ = m(q, k, v)
    out.sum().backward()
    assert q.grad is None or q.grad.abs().sum() == 0  # no existence gradient
    assert m.order_scores.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        _module(order_existence_mode="bogus")


@pytest.mark.parametrize("existence", ["full", "learned"])
def test_straight_through_order(existence):
    q, k, v = _qk()
    m = _module(order_existence_mode=existence, order_straight_through=True)
    m.train()
    m.set_order_path("relaxed")
    for s in range(50):
        m.resample(torch.Generator().manual_seed(s))
        out, _, _ = m(q, k, v)
        if m.last_applied_A.abs().sum() > 0:
            break
    # Forward support = the step's hard DAG (exclusion graph uses M_hard).
    A = m.last_applied_A[0].transpose(0, 1)
    assert _is_dag(A > 0)
    if existence == "full":
        assert torch.allclose(A, m.score_tensor_for_sparsity.transpose(0, 1))
    out.sum().backward()
    assert m.order_scores.grad is not None and m.order_scores.grad.abs().sum() > 0
    # Eval: identical to the non-ST module.
    ref = _module(order_existence_mode=existence)
    m.eval(); ref.eval()
    with torch.no_grad():
        o1, _, _ = m(q, k, v)
        o2, _, _ = ref(q, k, v)
    assert torch.allclose(o1, o2)


def test_collider_exclusion_sources_fully_excluded():
    # Hard DAG s1 -> c <- s2 with nodes (s1, s2, c) = (0, 1, 2), child <- parent.
    score = torch.zeros(N, N)
    score[2, 0] = 1.0
    score[2, 1] = 1.0
    mask, kept, cyclic = build_hsic_pair_mask(score, s_seq_len=0, homogeneous_nodes=True)
    assert not cyclic
    # s1 row: tested against s2 only (c is a descendant, self excluded).
    assert mask[0].tolist() == [0.0, 1.0, 0.0]
    assert mask[1].tolist() == [1.0, 0.0, 0.0]
    # c row: tested against both parents.
    assert mask[2].tolist() == [1.0, 1.0, 0.0]


def test_attached_soft_mask_has_gradient_detached_does_not():
    q, k, v = _qk()
    m = _module(order_detach_exclusion=False)
    m.train()
    m.resample(torch.Generator().manual_seed(0))
    m(q, k, v)
    desc = m.soft_descendant_scores()
    assert desc.requires_grad
    (1.0 - desc).sum().backward()
    assert m.order_scores.grad.abs().sum() > 0
    assert not m.score_tensor_for_sparsity.requires_grad


def test_sample_graphs_are_dags_and_seeded():
    q, k, _ = _qk()
    m = _module()
    a = m.sample_graphs(q, k, n_samples=20, seed=3)
    b = m.sample_graphs(q, k, n_samples=20, seed=3)
    assert torch.equal(a["graphs"], b["graphs"])
    assert all(_is_dag(g > 0) for g in a["graphs"])
    of = a["order_freq"]
    assert torch.allclose(of + of.T, 1.0 - torch.eye(N))


def test_order_scores_are_structural():
    assert _is_structural_param("model.attention.inner_attention.order_scores")


def test_model_integration_homogeneous():
    from tests.test_atsel_homogeneous import _make_model, _forward
    from causaliT.training.gradient_routing import classify_parameters

    model = _make_model(self_attention_type="OrderFactorizedSelfAttention",
                        order_existence_threshold=0.3)
    oa = model.order_attention()
    assert isinstance(oa, OrderFactorizedSelfAttention)
    assert oa.existence_threshold == 0.3
    struct, _ = classify_parameters(model)
    assert any(p is oa.order_scores for p in struct)

    model.train()
    model.resample()
    pred = _forward(model)[0]
    pred.sum().backward()
    assert oa.order_scores.grad is not None
    assert model.get_score_tensor_for_sparsity().shape == (oa.num_nodes, oa.num_nodes)
    assert model.set_order_path("hard")
    _forward(model)

    with pytest.raises(ValueError):
        _make_model(homogeneous_nodes=False,
                    self_attention_type="OrderFactorizedSelfAttention")


def test_other_types_unaffected():
    from tests.test_atsel_homogeneous import _make_model
    model = _make_model()
    assert model.order_attention() is None
    assert model.set_order_path("hard") is False
    model.resample()   # no-op


def test_hard_assignment_is_permutation():
    P = hard_assignment(torch.randn(6, 6))
    assert torch.equal(P.sum(0), torch.ones(6)) and torch.equal(P.sum(1), torch.ones(6))

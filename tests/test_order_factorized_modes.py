"""Unit tests for the OrderFactorizedSelfAttention modes (orientation, gate
mode, order path, BKD) and the variable_selection_objective /
anti_adjacency_lambda / recon_gates / use_order_in_recon options."""

import types

import pytest
import torch

from causaliT.core.modules.order_factorized_attention import OrderFactorizedSelfAttention
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)


N, E, B = 3, 4, 5


def _qk(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, N, E, generator=g).expand(B, N, E).clone().requires_grad_(True)
    k = torch.randn(1, N, E, generator=g).expand(B, N, E).clone()
    v = torch.randn(B, N, 2, generator=g)
    return q, k, v


def _mod(**kw):
    torch.manual_seed(0)
    return OrderFactorizedSelfAttention(num_nodes=N, order_r_init="random", **kw)


@pytest.mark.parametrize("mode", ["relaxed", "hard"])
def test_anticausal_is_transpose(mode):
    m = _mod()
    m.train()
    m.set_order_path(mode)
    q, k, v = _qk()
    m.resample(torch.Generator().manual_seed(1))
    m(q, k, v)
    A_c = m.last_applied_A[0].clone()
    post_c = m._last_M.clone()
    m.set_orientation("anticausal")
    m(q, k, v)
    A_a = m.last_applied_A[0]
    # Same pairs, opposite orientation; stored order stays the causal one.
    assert torch.allclose(A_a, A_c.transpose(0, 1))
    assert torch.allclose(m._last_M, post_c)
    with pytest.raises(ValueError):
        m.set_orientation("sideways")


@pytest.mark.parametrize("path", ["hard", "none"])
def test_hard_gates_are_binary_with_straight_through_gradient(path):
    m = _mod()
    m.train()
    m.set_order_path(path)
    m.set_gate_mode("hard")
    q, k, v = _qk()
    for s in range(50):
        m.resample(torch.Generator().manual_seed(s))
        out, _, _ = m(q, k, v)
        if m.last_applied_A.abs().sum() > 0:
            break
    A = m.last_applied_A[0]
    assert torch.all((A == 0) | (A == 1))                       # exact 0/1
    out.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0       # ST gradient
    assert m.order_scores.grad is None or m.order_scores.grad.abs().sum() == 0


def test_soft_gates_are_relaxed():
    m = _mod()
    m.train()
    m.set_order_path("none")
    m.set_gate_mode("soft")
    q, k, v = _qk()
    vals = []
    for s in range(30):
        m.resample(torch.Generator().manual_seed(s))
        m(q, k, v)
        vals.append(m.last_applied_A[0])
    vals = torch.cat([x.flatten() for x in vals])
    assert ((vals > 0) & (vals < 1)).any()
    with pytest.raises(ValueError):
        m.set_gate_mode("relaxed")


def test_no_order_in_recon_is_symmetric():
    """order_path none: each node sees its selected neighbours both ways."""
    m = _mod()
    m.train()
    m.set_order_path("none")
    q, k, v = _qk()
    m.resample(torch.Generator().manual_seed(0))
    m(q, k, v)
    A = m.last_applied_A[0]
    assert torch.equal(A, A.transpose(0, 1))
    assert torch.all(torch.diagonal(A) == 0)
    # Exclusion graph still uses the order (DAG).
    S = m.score_tensor_for_sparsity
    assert torch.all(S * S.transpose(0, 1) == 0)


def test_sampled_exclusion_equals_applied_dag_hard_path():
    m = _mod(order_exclusion_existence="sample")
    m.train()
    m.set_order_path("hard")
    m.set_gate_mode("hard")
    q, k, v = _qk()
    m.resample(torch.Generator().manual_seed(3))
    m(q, k, v)
    assert torch.equal(m.score_tensor_for_sparsity, (m.last_applied_A[0] > 0).float())


def test_bkd_drops_keys_without_gradient():
    m = _mod(batch_key_dropout=0.5, batch_key_dropout_deterministic=True)
    m.train()
    m.set_order_path("none")
    m.set_gate_mode("hard")
    q, k, v = _qk()
    m.resample(torch.Generator().manual_seed(1))
    torch.manual_seed(7)
    out, _, _ = m(q, k, v)
    keep = m.last_bkd_keep
    assert keep is not None and keep.sum() < N
    dropped = keep == 0
    assert torch.all(m.last_applied_A[0][:, dropped] == 0)
    # No gradient reaches the dropped keys' values.
    v2 = v.clone().requires_grad_(True)
    m.resample(torch.Generator().manual_seed(1))
    torch.manual_seed(7)
    out, _, _ = m(q, k, v2)
    out.sum().backward()
    assert torch.all(v2.grad[:, dropped] == 0)
    # Phase switch off: no mask, clock still advances.
    step = int(m._bkd_step)
    m.set_bkd_phase_active(False)
    m(q, k, v)
    assert m.last_bkd_keep is None and int(m._bkd_step) == step + 1


def _toy_forecaster(scores):
    """Bare namespace driving _dual_logratio_hsic with a linear oracle head.

    Node 0 -> node 1 (additive noise, nonlinear).  The 'model' regresses
    each node on the applied adjacency through a fitted nonparametric map:
    causal x1 | x0 is exact (residual = noise), anti-causal x0 | x1 is the
    reverse regression (residual dependent on x1).
    """
    torch.manual_seed(0)
    n = 400
    x0 = torch.randn(n)
    e = 0.3 * torch.randn(n)
    x1 = torch.tanh(2.0 * x0) + e
    X = torch.stack([x0, x1], dim=1)

    oa = OrderFactorizedSelfAttention(num_nodes=2, toggle_on_recon_att_selector=False,
                                order_noise_scale=0.0, order_tau=0.1)
    with torch.no_grad():
        oa.order_scores.copy_(torch.tensor(scores))
    oa.train()
    oa.resample(torch.Generator().manual_seed(0))

    def _knn(target, inp):
        # 1-D Nadaraya-Watson regression (bandwidth 0.2), differentiable in w.
        d = (inp[:, None] - inp[None, :]) ** 2
        K = torch.exp(-d / (2 * 0.2 ** 2))
        return (K @ target) / K.sum(1)

    model = types.SimpleNamespace()
    state = {"orient": "causal"}
    model.set_order_orientation = lambda o: state.__setitem__("orient", o)
    model.order_attention = lambda: oa

    def _lean_pred(S, Xin, overrides=None):
        q = torch.zeros(1, 2, 2)
        oa.set_orientation(state["orient"])
        oa(q, q, torch.zeros(1, 2, 1))
        A = oa.last_applied_A[0]            # [child, parent], relaxed weights
        oa.set_orientation("causal")
        p1 = A[1, 0] * _knn(x1, x0) + (1 - A[1, 0]) * x1.mean()
        p0 = A[0, 1] * _knn(x0, x1) + (1 - A[0, 1]) * x0.mean()
        return torch.stack([p0, p1], dim=1)

    f = types.SimpleNamespace(
        model=model, val_idx=0, _hsic_bw_frozen_sigmas=None,
        hsic_sigma=1.0, hsic_adaptive_bandwidth=True, hsic_mode="biased",
        nhsic_epsilon=0.01, hsic_kernel_source="rbf",
        hsic_bandwidth_multipliers=None, hsic_null_stats=None,
        anti_adjacency_eps=0.1, anti_adjacency_lambda=1.0, _last_anti_adjacency_stats=None)
    f._lean_pred = _lean_pred
    f._order_residuals = types.MethodType(
        AttentionSelectorForecaster._order_residuals, f)
    S = torch.zeros(n, 0, 1)
    return f, S, X.unsqueeze(-1), oa


def test_dual_logratio_prefers_true_order_and_has_order_gradient():
    vals = {}
    for tag, scores in (("true", [1.0, -1.0]), ("rev", [-1.0, 1.0])):
        f, S, X, oa = _toy_forecaster(scores)
        loss = AttentionSelectorForecaster._anti_adjacency_hsic(f, S, X)
        vals[tag] = float(loss)
        if tag == "true":
            loss.backward()
            assert oa.order_scores.grad is not None
            assert oa.order_scores.grad.abs().sum() > 0
        assert f._last_anti_adjacency_stats["A"] >= 0.0
    assert vals["true"] < vals["rev"]


# ---------------------------------------------------------------------------
# Forecaster integration (routing, recon-trained selector, unrolled gradient)
# ---------------------------------------------------------------------------
def _dual_forecaster(model_kwargs=None, **training):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_bilevel_probe import _model, _qbatches

    cfg = _model().config
    kw = cfg["model"]["kwargs"]
    kw["homogeneous_nodes"] = True
    kw["self_attention_type"] = "OrderFactorizedSelfAttention"
    kw["free_query_embedding"] = True
    kw.update(model_kwargs or {})
    cfg["training"].update(dict(
        use_gradient_routing=True, structural_grad="hsic_unrolled",
        unrolled={"inner_lr": 1e-3, "fd_epsilon": 1e-2},
        hsic_aggregation="plain", hsic_exclude_descendants=False,
        anti_adjacency_lambda=1.0, hsic_adaptive_bandwidth=True))
    cfg["training"].update(training)
    torch.manual_seed(0)
    m = AttentionSelectorForecaster(cfg)
    m.log = lambda *a, **k: None
    return m, _qbatches(1, batch=32)[0]


def test_forecaster_routing_and_steps():
    m, batch = _dual_forecaster(lambda_l0_selector=1e-2)
    oa = m.model.order_attention()
    assert oa is not None
    sids = {id(p) for p in m._structural_params}
    rids = {id(p) for p in m._reconstruction_params}
    assert id(oa.order_scores) in sids
    q = [p for n, p in m.model.named_parameters() if "query_embed" in n]
    assert q and all(id(p) in rids and id(p) not in sids for p in q)

    m.train()
    # Reconstruct phase: no order, hard ST gates -> selector gradient.
    m.model.set_order_path("none")
    m.model.set_gate_mode("hard")
    m.model.resample()
    m._step(batch=batch, stage="train")
    loss_r = m._last_loss_components["loss_recon"]
    loss_r.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in q)
    for p in m.parameters():
        p.grad = None

    # Structure phase: unrolled anti-adjacency objective -> order gradient.
    m.model.set_order_path("relaxed")
    m.model.set_gate_mode("soft")
    m.model.resample()
    m._step(batch=batch, stage="train")
    assert m._last_anti_adjacency_stats is not None
    m._bi_level_step(batch)
    g = oa.order_scores.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert oa.orientation == "causal"


def test_forecaster_config_guards():
    with pytest.raises(ValueError, match="hsic_exclude_descendants"):
        _dual_forecaster(hsic_exclude_descendants=True)
    for old in ("dual_selector_routing", "order_selection"):
        with pytest.raises(ValueError, match="toggle_on_recon_att_selector"):
            _dual_forecaster(**{old: "recon"})
    # Removed keys: equivalent legacy values load (old checkpoints), the
    # others fail with the replacement.
    with pytest.raises(ValueError, match="toggle_on_recon_att_selector"):
        _dual_forecaster(variable_selection_objective="struct")
    _dual_forecaster(variable_selection_objective="recon")
    m, _ = _dual_forecaster(model_kwargs={"order_existence_mode": "full"})
    assert not m.model.order_attention().recon_selector_on
    with pytest.raises(ValueError, match="toggle_on_recon_att_selector"):
        _dual_forecaster(model_kwargs={"order_existence_mode": "bogus"})
    with pytest.raises(ValueError, match="anti_adjacency_lambda"):
        _dual_forecaster(hsic_aggregation="dual_logratio")
    with pytest.raises(ValueError, match="anti_adjacency_eps"):
        _dual_forecaster(order_logratio_eps=0.1)


@pytest.mark.parametrize("struct_sel", [False, True])
@pytest.mark.parametrize("gates", ["hard", "soft"])
@pytest.mark.parametrize("use_order", [False, True])
def test_recon_forward_combinations(struct_sel, gates, use_order):
    """Every structural-selector x recon_gates x use_order_in_recon
    combination runs a reconstruction step and a structure step; G_R gets a
    recon gradient in both gate modes (straight-through for hard)."""
    m, batch = _dual_forecaster(
        model_kwargs={"toggle_on_struct_att_selector": struct_sel},
        anti_adjacency_lambda=0.0)
    oa = m.model.order_attention()
    q = [p for n, p in m.model.named_parameters() if "query_embed" in n]
    sids = {id(p) for p in m._structural_params}
    assert q and not any(id(p) in sids for p in q)
    m.train()
    m.model.set_order_path("hard" if use_order else "none")
    m.model.set_gate_mode(gates)
    m.model.resample()
    m._step(batch=batch, stage="train")
    m._last_loss_components["loss_recon"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in q)
    for p in m.parameters():
        p.grad = None
    m.model.set_order_path("relaxed")
    m.model.set_gate_mode("soft")
    m.model.resample()
    m._step(batch=batch, stage="train")
    assert m._last_anti_adjacency_stats is None
    m._bi_level_step(batch)
    g = oa.order_scores.grad
    assert g is not None and torch.isfinite(g).all()
    if struct_sel:
        gs = oa.struct_gate_logits.grad
        assert gs is not None and torch.isfinite(gs).all()


# ---------------------------------------------------------------------------
# Two selectors: each forward applies exactly ONE of them
# ---------------------------------------------------------------------------
def test_struct_selector_off_matches_default_module():
    """toggle_on_struct_att_selector=False is the original module: same
    parameters, same noise stream, same outputs on every path."""
    q, k, v = _qk()
    a = _mod()
    b = _mod(toggle_on_struct_att_selector=False)
    assert b.struct_gate_logits is None
    a.train(); b.train()
    for path in ("relaxed", "hard", "none"):
        for m in (a, b):
            m.set_order_path(path)
            m.resample(torch.Generator().manual_seed(3))
        oa_, _, _ = a(q, k, v)
        ob_, _, _ = b(q, k, v)
        assert torch.equal(oa_, ob_)


def test_struct_selector_is_used_only_in_structure_forwards():
    q, k, v = _qk()
    m = _mod(toggle_on_struct_att_selector=True)
    ref = _mod()                       # same seed -> same order scores
    m.train(); ref.train()
    # Reconstruction forward: identical to the module without G_S (G_R only).
    for path in ("hard", "none"):
        for x in (m, ref):
            x.set_order_path(path)
            x.resample(torch.Generator().manual_seed(5))
        assert torch.equal(m(q, k, v)[0], ref(q, k, v)[0])
    # Structure forward: G_S only -> the queries (G_R) get no gradient,
    # the structural logits and the order do.
    # (Open Hard-Concrete gates are clamped at 1 for many noise draws, so
    # accumulate over a few steps.)
    m.set_order_path("relaxed")
    for s in range(20):
        m.resample(torch.Generator().manual_seed(s))
        out, post, _ = m(q, k, v)
        out.sum().backward()
    assert q.grad is None or q.grad.abs().sum() == 0
    assert m.struct_gate_logits.grad.abs().sum() > 0
    assert m.order_scores.grad.abs().sum() > 0
    # Learned graph = pi_S (symmetric existence) x order.
    pi_S = m.structural_pi(q, k)
    assert torch.allclose(m._last_pi, pi_S)
    assert torch.allclose(post[0], (pi_S * m._last_M).transpose(0, 1))


def test_struct_selector_paired_across_orientations():
    q, k, v = _qk()
    m = _mod(toggle_on_struct_att_selector=True)
    m.train()
    m.set_order_path("relaxed")
    m.resample(torch.Generator().manual_seed(2))
    m(q, k, v)
    A_c = m.last_applied_A[0].clone()
    m.set_orientation("anticausal")
    m(q, k, v)
    assert torch.allclose(m.last_applied_A[0], A_c.transpose(0, 1))


def test_struct_selector_routing():
    m, _ = _dual_forecaster(model_kwargs={"toggle_on_struct_att_selector": True})
    oa = m.model.order_attention()
    sids = {id(p) for p in m._structural_params}
    rids = {id(p) for p in m._reconstruction_params}
    assert id(oa.struct_gate_logits) in sids and id(oa.struct_gate_logits) not in rids
    q = [p for n, p in m.model.named_parameters() if "query_embed" in n]
    assert q and all(id(p) in rids and id(p) not in sids for p in q)



# ---------------------------------------------------------------------------
# struct_att_selector_init="recon": one-off G_S <- G_R copy
# ---------------------------------------------------------------------------
def test_recon_init_copies_once_and_matches_g_r():
    q, k, v = _qk()
    m = _mod(toggle_on_struct_att_selector=True, struct_att_selector_init="recon")
    ref = _mod()                                   # G_S off, same order scores
    m.eval(); ref.eval()
    m.set_order_path("relaxed"); ref.set_order_path("relaxed")
    assert m.request_struct_selector_sync()
    with torch.no_grad():
        o1, p1, _ = m(q, k, v)
        o2, p2, _ = ref(q, k, v)
    # pi_S = pi_R and the structure forward equals the G_R-only module.
    assert m.struct_selector_is_synced
    assert torch.allclose(m.structural_pi(q, k), ref._last_pi, atol=1e-6)
    assert torch.allclose(o1, o2, atol=1e-6) and torch.allclose(p1, p2, atol=1e-6)
    # No second copy: a later request is a no-op, the logits keep evolving.
    with torch.no_grad():
        m.struct_gate_logits.add_(1.0)
    snap = m.struct_gate_logits.detach().clone()
    assert not m.request_struct_selector_sync()
    m(q, k, v)
    assert torch.equal(m.struct_gate_logits, snap)
    # Reload: the synced flag survives the state dict.
    m2 = _mod(toggle_on_struct_att_selector=True, struct_att_selector_init="recon")
    m2.load_state_dict(m.state_dict())
    assert m2.struct_selector_is_synced and not m2.request_struct_selector_sync()


def test_recon_init_guards_and_constant_unchanged():
    with pytest.raises(ValueError, match="struct_att_selector_init"):
        _mod(toggle_on_struct_att_selector=True, struct_att_selector_init="bogus")
    with pytest.raises(ValueError, match="struct_att_selector_init"):
        _mod(struct_att_selector_init="recon")                 # G_S off
    with pytest.raises(ValueError, match="struct_att_selector_init"):
        _mod(toggle_on_struct_att_selector=True, toggle_on_recon_att_selector=False,
             struct_att_selector_init="recon")
    m = _mod(toggle_on_struct_att_selector=True)            # constant init
    assert not m.request_struct_selector_sync()
    assert "struct_selector_synced" not in m.state_dict()


def test_trainer_requests_sync_on_structure_only():
    from causaliT.training.adaptive_trainer import PhaseController as _T
    t = _T.__new__(_T)
    t.warmup_cfg, t.recon_cfg, t.final_cfg = {}, {}, {}
    t._recon_gates_default, t._use_order_in_recon_default = "hard", False
    t._bkd_managed = False
    m, _ = _dual_forecaster(model_kwargs={"toggle_on_struct_att_selector": True,
                                          "struct_att_selector_init": "recon"})
    oa = m.model.order_attention()
    t._apply_recon_forward_cfg(m, "warmup")
    assert not oa._struct_sync_pending
    t._apply_recon_forward_cfg(m, "structure")
    assert oa._struct_sync_pending


def test_trainer_applies_recon_forward_and_ladder_phases():
    from causaliT.training.adaptive_trainer import PhaseController as _T
    t = _T.__new__(_T)
    t.warmup_cfg, t.recon_cfg, t.final_cfg = {}, {"recon_gates": "soft"}, {}
    t._recon_gates_default, t._use_order_in_recon_default = "hard", True
    t._bkd_managed = False
    m, _ = _dual_forecaster(anti_adjacency_lambda=0.0)
    oa = m.model.order_attention()
    t._apply_recon_forward_cfg(m, "warmup")
    assert (oa.order_path, oa.gate_mode) == ("hard", "hard")
    t._apply_recon_forward_cfg(m, "reconstruct")
    assert (oa.order_path, oa.gate_mode) == ("hard", "soft")
    t._apply_recon_forward_cfg(m, "structure")
    assert (oa.order_path, oa.gate_mode) == ("relaxed", "soft")


def _unused_plain_hsic():
    m, batch = _dual_forecaster(anti_adjacency_lambda=0.0)
    oa = m.model.order_attention()
    m.train()
    m.model.set_order_path("relaxed")
    m.model.resample()
    m._step(batch=batch, stage="train")
    assert m._last_anti_adjacency_stats is None
    m._bi_level_step(batch)
    g = oa.order_scores.grad
    assert g is not None and torch.isfinite(g).all()

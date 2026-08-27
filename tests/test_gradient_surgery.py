"""Tests for causaliT.training.gradient_surgery (per-block PCGrad).

Run with:  pytest tests/test_gradient_surgery.py -v

Covers:
  * Projection math: an anti-aligned target gradient loses exactly its
    component along the HSIC reference; the orthogonal part is preserved.
  * Aligned / orthogonal / zero gradients are left untouched.
  * Per-block isolation: conflict in one block never touches another.
  * Metrics: cos_pre / cos_post / frac_projected per target.
  * Summed post-surgery grads equal the plain backward of the summed loss
    when no conflict exists (equivalence of the no-conflict path).
  * Forecaster integration: config validation (surgery requires gradient
    routing) and the stashed per-term loss tensors used by the hook.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training.gradient_surgery import pcgrad_reconcile
from causaliT.training.interference_utils import build_interference_blocks
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from tests.test_atsel_reg_safeguard import _make_forecaster_config, _make_batch


class _TinyModel(nn.Module):
    """Same naming scheme as test_interference_utils._TinyModel."""

    def __init__(self):
        super().__init__()
        self.query_projection = nn.Linear(2, 2, bias=False)
        self.key_projection = nn.Linear(2, 2, bias=False)
        self.value_projection = nn.Linear(2, 2, bias=False)


def _grads(loss, params):
    return torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)


def _run(model, hsic_reg, tgt_reg, target_name="l0"):
    """Compute grads of both terms over all params, reconcile, return results."""
    blocks = build_interference_blocks(model)
    all_params = [p for plist in blocks.values() for p in plist]
    g_ref = _grads(hsic_reg, all_params)
    g_tgt = _grads(tgt_reg, all_params)
    projected, metrics = pcgrad_reconcile(
        g_ref, {target_name: g_tgt}, blocks, all_params
    )
    return blocks, all_params, g_ref, g_tgt, projected[target_name], metrics

# ---------------------------------------------------------------------------
# Projection math
# ---------------------------------------------------------------------------

def test_antialigned_target_loses_reference_component():
    """l0 grad exactly anti-parallel to HSIC grad -> projected to zero."""
    model = _TinyModel()
    Wq = model.query_projection.weight
    hsic_reg = (Wq * 1.0).sum()
    l0_reg = -(Wq * 2.0).sum()  # grad = -2 * ones, anti-parallel

    _, all_params, _, _, proj, metrics = _run(model, hsic_reg, l0_reg)

    i = next(i for i, p in enumerate(all_params) if p is Wq)
    assert torch.allclose(proj[i], torch.zeros_like(Wq), atol=1e-6)
    assert metrics["frac_projected_l0"] == pytest.approx(1.0)
    assert metrics["cos_pre_l0"] == pytest.approx(-1.0)
    post = metrics["cos_post_l0"]  # NaN when fully projected away (zero norm)
    assert math.isnan(post) or post > -1e-6


def test_conflicting_component_removed_orthogonal_part_preserved():
    """Target with both an anti-HSIC and an orthogonal component."""
    model = _TinyModel()
    Wq = model.query_projection.weight
    A = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    hsic_reg = (Wq * A).sum()
    l0_reg = (Wq * torch.tensor([[-1.0, 0.0], [0.0, 1.0]])).sum()

    _, all_params, _, _, proj, metrics = _run(model, hsic_reg, l0_reg)

    i = next(i for i, p in enumerate(all_params) if p is Wq)
    assert torch.allclose(
        proj[i], torch.tensor([[0.0, 0.0], [0.0, 1.0]]), atol=1e-6
    )
    assert metrics["frac_projected_l0"] == pytest.approx(1.0)


def test_aligned_target_untouched():
    model = _TinyModel()
    Wq = model.query_projection.weight
    hsic_reg = (Wq * 1.0).sum()
    l0_reg = (Wq * 2.0).sum()  # aligned

    _, all_params, _, g_tgt, proj, metrics = _run(model, hsic_reg, l0_reg)

    i = next(i for i, p in enumerate(all_params) if p is Wq)
    assert torch.equal(proj[i], g_tgt[i])
    assert metrics["frac_projected_l0"] == pytest.approx(0.0)
    assert metrics["cos_pre_l0"] == pytest.approx(1.0)
    assert metrics["cos_post_l0"] == pytest.approx(1.0)


def test_orthogonal_target_untouched():
    model = _TinyModel()
    Wq = model.query_projection.weight
    A = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    B = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    hsic_reg = (Wq * A).sum()
    l0_reg = (Wq * B).sum()

    _, all_params, _, g_tgt, proj, metrics = _run(model, hsic_reg, l0_reg)

    i = next(i for i, p in enumerate(all_params) if p is Wq)
    assert torch.equal(proj[i], g_tgt[i])
    assert metrics["frac_projected_l0"] == pytest.approx(0.0)
    assert metrics["cos_pre_l0"] == pytest.approx(0.0, abs=1e-6)

def test_zero_target_gradient_is_untouched():
    """A param whose target grad is zero must not be written by surgery."""
    model = _TinyModel()
    Wq, Wk = model.query_projection.weight, model.key_projection.weight
    # HSIC sees Q and K; L0 sees only Q (anti-aligned there).
    hsic_reg = (Wq * 1.0).sum() + (Wk * 1.0).sum()
    l0_reg = -(Wq * 1.0).sum()

    _, all_params, _, g_tgt, proj, metrics = _run(model, hsic_reg, l0_reg)

    ik = next(i for i, p in enumerate(all_params) if p is Wk)
    assert g_tgt[ik] is None
    assert proj[ik] is None  # key_projection block had no L0 signal
    assert metrics["frac_projected_l0"] == pytest.approx(1.0)


def test_per_block_isolation():
    """Conflict in query_projection must not alter key_projection's grad."""
    model = _TinyModel()
    Wq, Wk = model.query_projection.weight, model.key_projection.weight
    hsic_reg = (Wq * 1.0).sum() - (Wk * 1.0).sum()   # HSIC grad: +1 on Q, -1 on K
    l0_reg = -(Wq * 1.0).sum() - (Wk * 1.0).sum()    # L0: -1 on Q, -1 on K
    # Q block: anti-aligned (conflict).  K block: aligned.

    _, all_params, _, g_tgt, proj, metrics = _run(model, hsic_reg, l0_reg)

    iq = next(i for i, p in enumerate(all_params) if p is Wq)
    ik = next(i for i, p in enumerate(all_params) if p is Wk)
    assert torch.allclose(proj[iq], torch.zeros_like(Wq), atol=1e-6)
    assert torch.equal(proj[ik], g_tgt[ik])  # aligned block untouched
    assert metrics["frac_projected_l0"] == pytest.approx(0.5)


def test_no_valid_block_metrics_are_nan():
    """Target with zero gradient everywhere -> NaN metrics, no writes."""
    model = _TinyModel()
    Wq = model.query_projection.weight
    hsic_reg = (Wq * 1.0).sum()
    l0_reg = torch.tensor(0.0) * (Wq * 0.0).sum()  # zero grad, still on graph

    _, _, _, _, _, metrics = _run(model, hsic_reg, l0_reg)

    assert math.isnan(metrics["cos_pre_l0"])
    assert math.isnan(metrics["cos_post_l0"])
    assert math.isnan(metrics["frac_projected_l0"])

def test_no_conflict_summed_grads_match_plain_backward():
    """Equivalence: aligned grads -> surgery sum == fused backward."""
    torch.manual_seed(0)
    model = _TinyModel()
    blocks = build_interference_blocks(model)
    all_params = [p for plist in blocks.values() for p in plist]
    Wq, Wk = model.query_projection.weight, model.key_projection.weight

    x = torch.randn(3, 2)
    shared = ((Wq @ x.T) ** 2).sum()
    hsic_reg = shared + (Wk * 1.0).sum()
    l0_reg = 0.5 * shared  # exactly aligned with HSIC on Wq by construction
    rest = ((Wk @ x.T) ** 2).sum()

    # Surgery path first (all aligned here -> no projection should fire);
    # the fused backward runs last because it frees the graph.
    g_hsic = _grads(hsic_reg, all_params)
    g_l0 = _grads(l0_reg, all_params)
    g_rest = _grads(rest, all_params)
    projected, metrics = pcgrad_reconcile(g_hsic, {"l0": g_l0}, blocks, all_params)
    for i, p in enumerate(all_params):
        parts = [g for g in (g_hsic[i], g_rest[i], projected["l0"][i])
                 if g is not None]
        if parts:
            p.grad = torch.stack([g.detach() for g in parts]).sum(dim=0)

    surgery_grads = {id(p): p.grad.clone() for p in all_params if p.grad is not None}
    for p in all_params:
        p.grad = None

    # Reference: plain backward of the fused loss (frees the graph, runs last).
    (hsic_reg + l0_reg + rest).backward()
    ref = {id(p): p.grad.clone() for p in all_params if p.grad is not None}

    for i, p in enumerate(all_params):
        if id(p) in ref:
            assert torch.allclose(surgery_grads[id(p)], ref[id(p)], atol=1e-6)
    assert metrics["frac_projected_l0"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Forecaster integration
# ---------------------------------------------------------------------------

class TestForecasterIntegration:
    def test_default_off(self):
        model = AttentionSelectorForecaster(_make_forecaster_config())
        assert model.gradient_surgery is False

    def test_requires_gradient_routing(self):
        cfg = _make_forecaster_config()
        cfg["training"]["gradient_surgery"] = True
        cfg["training"]["use_gradient_routing"] = False
        with pytest.raises(ValueError, match="use_gradient_routing"):
            AttentionSelectorForecaster(cfg)

    def test_enabled_with_routing(self):
        cfg = _make_forecaster_config()
        cfg["training"]["gradient_surgery"] = True
        cfg["training"]["use_gradient_routing"] = True
        model = AttentionSelectorForecaster(cfg)
        assert model.gradient_surgery is True

    def test_step_stashes_separate_terms(self):
        """_step must expose the per-term tensors the surgery hook reads."""
        cfg = _make_forecaster_config(kappa=0.1, lambda_l0=0.0)
        model = AttentionSelectorForecaster(cfg)
        model.train()
        model._step(_make_batch(), stage="train")
        # Reference and rest always carry graph when lambda_hsic > 0.
        assert model._last_struct_hsic_term.requires_grad
        assert model._last_struct_rest.requires_grad
        # kappa > 0 -> NOTEARS term on graph; lambda_l0 == 0 -> L0 term is a
        # detached scalar zero (the hook will skip it).
        assert model._last_acyclic_reg.requires_grad
        assert not model._last_l0_reg.requires_grad

    def test_reconcile_on_live_step_graph(self):
        """Per-term autograd.grad on a real _step graph must work, and the
        graph must survive for the fused structural backward afterwards."""
        cfg = _make_forecaster_config(kappa=0.1)
        model = AttentionSelectorForecaster(cfg)
        model.train()
        model._step(_make_batch(), stage="train")

        blocks = build_interference_blocks(model.model)
        all_params = [p for plist in blocks.values() for p in plist]
        hsic_term = model._last_struct_hsic_term
        notears = model._last_acyclic_reg

        g_ref = torch.autograd.grad(
            hsic_term, all_params, retain_graph=True, allow_unused=True
        )
        g_tgt = torch.autograd.grad(
            notears, all_params, retain_graph=True, allow_unused=True
        )
        projected, metrics = pcgrad_reconcile(
            g_ref, {"notears": g_tgt}, blocks, all_params
        )
        assert "cos_pre_notears" in metrics
        assert len(projected["notears"]) == len(all_params)
        # Post-surgery cosine must never be worse (more negative) than pre.
        pre = metrics["cos_pre_notears"]
        post = metrics["cos_post_notears"]
        if not math.isnan(pre):
            assert post >= pre - 1e-6
        # Graph survived: the fused structural backward still works.
        model._last_loss_components["loss_structural"].backward()



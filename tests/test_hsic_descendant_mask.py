"""
Tests for the descendant-excluding HSIC mask.

Motivation (see ``causaliT/utils/descendant_mask.py``): under an ANM
``X_i = f_i(PA_i) + e_i`` the ideal residual ``r_i = e_i`` is NECESSARILY
dependent on the descendants of ``i`` and on ``X_i`` itself.  Averaging those
pairs into the HSIC term means the TRUE DAG is not a minimiser of the structural
loss.  These tests pin down the three properties that make the fix safe:

1. **Correctness** — the excluded set is exactly ``Desc(i) U {i}`` under the
   thresholded posterior, in both the split ``(L_X, L_S+L_X)`` and the
   homogeneous ``(N, N)`` layouts.
2. **No self-confirmation** — the mask is always DETACHED, so the model can
   never create an edge in order to delete its own penalty term, and every
   guard (warmup, collapse, wrong shape) degrades to the *unmasked* behaviour.
3. **Exact backward compatibility** — with the feature at its defaults, the HSIC
   value is bit-for-bit what it was before the feature existed.
"""

import math
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
import torch


from causaliT.utils.descendant_mask import (
    build_hsic_pair_mask,
    build_hsic_pair_mask_budgeted,
    harden_adjacency,
    soft_transitive_closure,
    transitive_closure,
)
from causaliT.utils.hsic_utils import hsic_cross_per_pair, hsic_pair_matrix


# ---------------------------------------------------------------------------
# harden_adjacency
# ---------------------------------------------------------------------------

def test_harden_adjacency_thresholds_and_orientation():
    # [i, j] = P(j -> i).  Edge 0 -> 1 is strong, everything else is weak.
    score = torch.tensor([[0.1, 0.2],
                          [0.9, 0.1]])
    hard = harden_adjacency(score, threshold=0.5)
    assert hard.dtype == torch.bool
    assert bool(hard[1, 0]) is True      # 0 -> 1 kept
    assert bool(hard[0, 1]) is False     # 1 -> 0 dropped


def test_harden_adjacency_resolves_two_cycles_by_argmax():
    """A two-cycle would inflate the closure to "everything", so it is resolved.

    ``GatedSelfAttention`` guarantees ``d_ij + d_ji == 1``, so both directions
    crossing the threshold means the EXISTENCE gate alone did it — keeping the
    stronger direction is the informative choice.
    """
    score = torch.tensor([[0.0, 0.6],
                          [0.8, 0.0]])
    hard = harden_adjacency(score, threshold=0.5, resolve_two_cycles=True)
    assert bool(hard[1, 0]) is True       # stronger direction survives
    assert bool(hard[0, 1]) is False

    # Exact ties carry no orientation information -> drop BOTH (conservative).
    tie = torch.tensor([[0.0, 0.7],
                        [0.7, 0.0]])
    hard_tie = harden_adjacency(tie, threshold=0.5, resolve_two_cycles=True)
    assert not bool(hard_tie.any())

    # Opting out keeps the cycle (used to verify the is_cyclic diagnostic).
    kept = harden_adjacency(score, threshold=0.5, resolve_two_cycles=False)
    assert bool(kept[0, 1]) and bool(kept[1, 0])


# ---------------------------------------------------------------------------
# transitive_closure
# ---------------------------------------------------------------------------

def _chain_adjacency(n: int) -> torch.Tensor:
    """Chain 0 -> 1 -> ... -> (n-1) as a child <- parent boolean matrix."""
    adj = torch.zeros((n, n), dtype=torch.bool)
    for i in range(1, n):
        adj[i, i - 1] = True
    return adj


def test_transitive_closure_full_chain():
    """On a chain, Desc(0) must be every downstream node, not just the child."""
    desc = transitive_closure(_chain_adjacency(4), hops=None)
    # desc[i, j] = True iff j is a descendant of i
    assert desc[0].tolist() == [False, True, True, True]
    assert desc[2].tolist() == [False, False, False, True]
    assert desc[3].tolist() == [False, False, False, False]
    assert not bool(desc.diagonal().any())        # acyclic


def test_transitive_closure_hops_1_is_direct_children_only():
    desc = transitive_closure(_chain_adjacency(4), hops=1)
    assert desc[0].tolist() == [False, True, False, False]


def test_transitive_closure_hops_2_reaches_grandchildren_only():
    desc = transitive_closure(_chain_adjacency(5), hops=2)
    assert desc[0].tolist() == [False, True, True, False, False]


def test_transitive_closure_cycle_saturates_and_marks_diagonal():
    # 0 -> 1 -> 0
    adj = torch.tensor([[False, True],
                        [True, False]])
    desc = transitive_closure(adj, hops=None)
    assert bool(desc.all())                        # saturates
    assert bool(desc.diagonal().all())             # cycle => self-reachable


def test_transitive_closure_rejects_non_square():
    with pytest.raises(ValueError):
        transitive_closure(torch.zeros((2, 3), dtype=torch.bool))


# ---------------------------------------------------------------------------
# build_hsic_pair_mask — split mode
# ---------------------------------------------------------------------------

def test_build_pair_mask_split_excludes_descendants_and_self():
    """Split layout: X0 -> X1 -> X2, S columns must never be masked."""
    L_S, L_X = 2, 3
    score = torch.zeros((L_X, L_S + L_X))
    score[1, L_S + 0] = 0.9      # X0 -> X1
    score[2, L_S + 1] = 0.9      # X1 -> X2

    mask, kept_frac, is_cyclic = build_hsic_pair_mask(
        score_tensor=score, s_seq_len=L_S, homogeneous_nodes=False,
        threshold=0.5, hops=None, exclude_self=True, excluded_weight=0.0,
    )

    assert mask.shape == score.shape
    assert is_cyclic is False

    # S columns are exogenous: they can never be descendants of an X node.
    assert bool((mask[:, :L_S] == 1.0).all())

    # Row 0 (residual of X0): X0 itself + descendants X1, X2 excluded.
    assert mask[0, L_S + 0].item() == 0.0
    assert mask[0, L_S + 1].item() == 0.0
    assert mask[0, L_S + 2].item() == 0.0
    # Row 2 (residual of X2, a sink): only itself excluded — its ANCESTORS are
    # legitimate independence targets and must be kept.
    assert mask[2, L_S + 0].item() == 1.0
    assert mask[2, L_S + 1].item() == 1.0
    assert mask[2, L_S + 2].item() == 0.0

    # kept_frac counts pairs, so 9 kept out of 15.
    n_excluded = 3 + 2 + 1
    assert kept_frac == pytest.approx((L_X * (L_S + L_X) - n_excluded) / (L_X * (L_S + L_X)))


def test_build_pair_mask_soft_weight_keeps_a_gradient_trickle():
    """``excluded_weight > 0`` down-weights instead of dropping."""
    score = torch.zeros((2, 3))
    score[1, 1 + 0] = 0.9       # X0 -> X1  (s_seq_len = 1)
    mask, kept_frac, _ = build_hsic_pair_mask(
        score_tensor=score, s_seq_len=1, homogeneous_nodes=False,
        excluded_weight=0.25,
    )
    assert mask.min().item() == pytest.approx(0.25)
    # kept_frac is purely combinatorial: it must NOT depend on the weight, so it
    # means the same thing in the hard and soft variants.
    mask_hard, kept_hard, _ = build_hsic_pair_mask(
        score_tensor=score, s_seq_len=1, homogeneous_nodes=False,
        excluded_weight=0.0,
    )
    assert kept_frac == pytest.approx(kept_hard)


def test_build_pair_mask_exclude_self_false_keeps_diagonal():
    score = torch.zeros((2, 3))
    mask, _, _ = build_hsic_pair_mask(
        score_tensor=score, s_seq_len=1, homogeneous_nodes=False,
        exclude_self=False,
    )
    assert bool((mask == 1.0).all())     # empty graph + no self-exclusion = no-op


def test_build_pair_mask_split_shape_mismatch_raises():
    with pytest.raises(ValueError):
        build_hsic_pair_mask(
            score_tensor=torch.zeros((3, 4)), s_seq_len=2,
            homogeneous_nodes=False,
        )


# ---------------------------------------------------------------------------
# build_hsic_pair_mask — homogeneous mode
# ---------------------------------------------------------------------------

def test_build_pair_mask_homogeneous_square():
    """Homogeneous layout: every node is both a query and a key."""
    score = torch.zeros((3, 3))
    score[1, 0] = 0.9            # 0 -> 1
    score[2, 1] = 0.9            # 1 -> 2
    mask, _, is_cyclic = build_hsic_pair_mask(
        score_tensor=score, s_seq_len=0, homogeneous_nodes=True,
    )
    assert is_cyclic is False
    assert mask[0].tolist() == [0.0, 0.0, 0.0]      # self + both descendants
    assert mask[2].tolist() == [1.0, 1.0, 0.0]      # sink: only self


def test_build_pair_mask_homogeneous_requires_square():
    with pytest.raises(ValueError):
        build_hsic_pair_mask(
            score_tensor=torch.zeros((2, 5)), s_seq_len=0,
            homogeneous_nodes=True,
        )


def test_build_pair_mask_is_always_detached():
    """The mask must never carry gradient.

    This is THE safety property of the feature: if the mask were differentiable,
    the optimiser could reduce the loss by creating edges (thereby deleting HSIC
    terms) rather than by finding true parents — a self-confirming shortcut.
    """
    score = torch.zeros((2, 3), requires_grad=True)
    with torch.enable_grad():
        mask, _, _ = build_hsic_pair_mask(
            score_tensor=score, s_seq_len=1, homogeneous_nodes=False,
        )
    assert mask.requires_grad is False
    assert mask.grad_fn is None


# ---------------------------------------------------------------------------
# hsic_utils integration
# ---------------------------------------------------------------------------

def test_hsic_pair_matrix_without_mask_is_unchanged():
    """Regression: the default path must be EXACTLY the pre-feature behaviour."""
    torch.manual_seed(0)
    s = torch.randn(48, 3)
    res = torch.randn(48, 2)
    base = hsic_pair_matrix(source_values=s, residuals=res)
    masked = hsic_pair_matrix(source_values=s, residuals=res,
                              pair_mask=torch.ones(2, 3))
    assert torch.allclose(base, masked, equal_nan=True)
    assert torch.allclose(
        hsic_cross_per_pair(s, res),
        hsic_cross_per_pair(s, res, pair_mask=torch.ones(2, 3)),
    )


def test_hsic_pair_matrix_skips_zero_weight_pairs():
    """Zero-weight pairs are NaN placeholders: no kernel work is done for them."""
    torch.manual_seed(0)
    s = torch.randn(32, 3)
    res = torch.randn(32, 2)
    mask = torch.ones(2, 3)
    mask[0, 1] = 0.0
    out = hsic_pair_matrix(source_values=s, residuals=res, pair_mask=mask)
    assert math.isnan(float(out[0, 1]))
    assert not math.isnan(float(out[0, 0]))


def test_hsic_cross_per_pair_masked_mean_equals_manual_weighted_mean():
    torch.manual_seed(0)
    s = torch.randn(32, 3)
    res = torch.randn(32, 2)
    mask = torch.ones(2, 3)
    mask[0, 1] = 0.0
    mask[1, 2] = 0.5

    full = hsic_pair_matrix(source_values=s, residuals=res)
    expected = ((full * mask).sum() - full[0, 1] * mask[0, 1]) / mask.sum()
    got = hsic_cross_per_pair(s, res, pair_mask=mask)
    assert got.item() == pytest.approx(expected.item(), rel=1e-5)


def test_hsic_cross_per_pair_all_excluded_returns_zero():
    """Fully-masked step must not produce NaN and kill the run."""
    torch.manual_seed(0)
    s = torch.randn(16, 2)
    res = torch.randn(16, 2)
    out = hsic_cross_per_pair(s, res, pair_mask=torch.zeros(2, 2))
    assert out.item() == 0.0
    assert torch.isfinite(out)


def test_hsic_pair_matrix_rejects_wrong_mask_shape():
    s = torch.randn(8, 3)
    res = torch.randn(8, 2)
    with pytest.raises(ValueError):
        hsic_pair_matrix(source_values=s, residuals=res, pair_mask=torch.ones(3, 2))


def test_masked_hsic_keeps_gradient_to_residuals():
    """Masking must not sever the structural gradient path."""
    torch.manual_seed(0)
    s = torch.randn(24, 2)
    res = torch.randn(24, 2, requires_grad=True)
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    out = hsic_cross_per_pair(s, res, pair_mask=mask)
    out.backward()
    assert res.grad is not None
    assert torch.isfinite(res.grad).all()
    assert float(res.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Forecaster guards (exercised on the unbound method with a light stub, so the
# guard logic is tested without building a full model)
# ---------------------------------------------------------------------------

def _mask_stub(**overrides) -> Any:
    """Duck-typed stand-in for the forecaster / LightningModule.

    Typed ``Any`` on purpose: the code under test only reads/writes plain
    attributes, so a stub avoids building a full model (and keeps the type
    checker from demanding a real ``LightningModule``).
    """
    base = dict(
        hsic_exclude_descendants=True,
        _descendant_mask_supported=True,
        current_epoch=10,
        hsic_descendant_warmup_epochs=0,
        _descendant_warmup_anchor=0,

        hsic_descendant_ema=0.0,
        _descendant_ema_score=None,
        S_seq_len=1,
        homogeneous_nodes=False,
        hsic_descendant_threshold=0.5,
        hsic_descendant_hops=None,
        hsic_descendant_exclude_self=True,
        hsic_descendant_weight=0.0,
        hsic_descendant_min_kept_frac=0.25,
        hsic_descendant_degrade_first_order=True,
        # Duck-typed LightningModule surface needed by PhaseController
        # (_apply_open_gate_cfg / _apply_dense_gate_cfg iterate submodules).
        modules=lambda: iter([]),
        log=lambda *a, **k: None,
        hsic_descendant_mode="threshold",
        hsic_descendant_budget_frac=0.25,
        hsic_descendant_per_row=True,
        hsic_descendant_tnorm="min",
        hsic_descendant_confident_score=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _call_builder(stub, score):
    from causaliT.training.forecasters.attention_selector_forecaster import (
        AttentionSelectorForecaster,
    )
    return AttentionSelectorForecaster._build_hsic_descendant_mask(stub, score)


def _sparse_score():
    score = torch.zeros((3, 4))       # s_seq_len=1 -> N=4
    score[1, 1] = 0.9                 # X0 -> X1
    return score


def test_forecaster_mask_disabled_by_flag():
    mask, kept, cyc, _ = _call_builder(_mask_stub(hsic_exclude_descendants=False),
                                       _sparse_score())
    assert mask is None and kept == 1.0 and cyc is False


def test_forecaster_mask_disabled_when_unsupported_attention():
    """No DIRECTED posterior (e.g. non-GatedSelfAttention) => feature off."""
    mask, kept, _, _ = _call_builder(_mask_stub(_descendant_mask_supported=False),
                                     _sparse_score())
    assert mask is None and kept == 1.0


def test_forecaster_mask_respects_warmup():
    """Early on the graph is ~random; masking then would delete parent signal.

    With the default anchor 0 the countdown runs from the start of the run —
    the semantics used by the plain / staged trainers.
    """
    stub = _mask_stub(current_epoch=3, hsic_descendant_warmup_epochs=10)
    assert _call_builder(stub, _sparse_score())[0] is None
    stub.current_epoch = 10
    assert _call_builder(stub, _sparse_score())[0] is not None


def test_forecaster_warmup_is_counted_from_the_anchor():
    """A non-zero anchor makes the warmup PHASE-relative, not run-relative.

    This is the whole point of the anchor: under the adaptive trainer
    ``current_epoch`` is global, so a 10-epoch guard would already be spent by
    the time the (long) reconstruct warmup hands over to structure — the mask
    would go live on a still-random adjacency.
    """
    # Global epoch 105, structure phase started at 100, warmup 10 -> 5 elapsed.
    stub = _mask_stub(current_epoch=105, hsic_descendant_warmup_epochs=10,
                      _descendant_warmup_anchor=100)
    assert _call_builder(stub, _sparse_score())[0] is None
    # Without the anchor the SAME epoch would already be masking.
    stub_legacy = _mask_stub(current_epoch=105, hsic_descendant_warmup_epochs=10)
    assert _call_builder(stub_legacy, _sparse_score())[0] is not None
    # ...and the mask arrives exactly ``warmup`` epochs after the anchor.
    stub.current_epoch = 110
    assert _call_builder(stub, _sparse_score())[0] is not None


def test_forecaster_warmup_anchor_none_means_already_served():
    """``anchor is None`` skips the delay (later structure phases)."""
    stub = _mask_stub(current_epoch=0, hsic_descendant_warmup_epochs=1000,
                      _descendant_warmup_anchor=None)
    assert _call_builder(stub, _sparse_score())[0] is not None


def test_forecaster_mask_skips_missing_or_multihead_score():
    assert _call_builder(_mask_stub(), None)[0] is None
    assert _call_builder(_mask_stub(), torch.zeros((2, 3, 4)))[0] is None


def test_forecaster_mask_collapse_guard_falls_back_to_unmasked():
    """A dense graph would swallow every pair and silently kill the gradient."""
    dense = torch.full((3, 4), 0.9)
    mask, kept, _, first = _call_builder(
        _mask_stub(hsic_descendant_min_kept_frac=0.9), dense)
    assert mask is None            # fell back (even the first-order mask starved)
    assert first is True           # ...after degrading to hops=1
    assert kept < 0.9              # ...but the diagnostic is still reported


def test_forecaster_mask_shape_error_falls_back_instead_of_raising():
    """A shape inconsistency must never break training over a diagnostic mask."""
    bad = torch.zeros((3, 9))      # inconsistent with S_seq_len=1
    mask, kept, _, _ = _call_builder(_mask_stub(), bad)
    assert mask is None and kept == 1.0


def test_forecaster_mask_ema_smooths_the_adjacency():
    """EMA stops the mask thrashing when posteriors sit near the threshold."""
    stub = _mask_stub(hsic_descendant_ema=0.9)
    score = _sparse_score()
    _call_builder(stub, score)
    assert stub._descendant_ema_score is not None
    # First call seeds the EMA with the raw score...
    assert torch.allclose(stub._descendant_ema_score, score)
    # ...then a spike is damped rather than immediately trusted.
    spike = torch.zeros((3, 4))
    spike[2, 2] = 1.0
    _call_builder(stub, spike)
    assert float(stub._descendant_ema_score[2, 2]) < 0.5


def test_forecaster_mask_is_detached_end_to_end():
    score = _sparse_score().requires_grad_(True)
    with torch.enable_grad():
        mask, _, _, _ = _call_builder(_mask_stub(), score)
    assert mask is not None
    assert mask.requires_grad is False


# ---------------------------------------------------------------------------
# adaptive_trainer per-phase overrides
# ---------------------------------------------------------------------------

@pytest.fixture
def save_dir():
    """Scratch directory for ``PhaseController`` (it mkdirs stage_checkpoints/).

    Deliberately NOT pytest's ``tmp_path``: that fixture scans the shared
    ``%TEMP%/pytest-of-<user>`` root, which can raise PermissionError on locked
    Windows temp folders.  ``mkdtemp`` needs no such scan.
    """
    path = tempfile.mkdtemp(prefix="causalit_desc_mask_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _phase_controller(tmp_path, structure=None, reconstruct=None):
    from causaliT.training.adaptive_trainer import PhaseController

    config = {
        "adaptive_training": {
            "structure": structure or {},
            "reconstruct": reconstruct or {},
        },
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    return PhaseController(
        config=config, data_dir=str(tmp_path), save_dir=str(tmp_path),
        cluster=True,
    )


def test_adaptive_phase_overrides_are_noop_when_unspecified(save_dir):
    """Omitting the keys must leave the ``training:``-level values untouched."""
    ctrl = _phase_controller(save_dir)
    module = _mask_stub(hsic_exclude_descendants=False, hsic_descendant_hops=2)
    ctrl._apply_descendant_mask_cfg(module, ctrl.struct_cfg)
    assert module.hsic_exclude_descendants is False
    assert module.hsic_descendant_hops == 2


def test_adaptive_phase_overrides_applied_and_typed(save_dir):
    ctrl = _phase_controller(save_dir, structure={
        "hsic_exclude_descendants": True,
        "hsic_descendant_hops": 1,
        "hsic_descendant_weight": 0.5,
    })
    module = _mask_stub(hsic_exclude_descendants=False,
                        hsic_descendant_hops=None,
                        hsic_descendant_weight=0.0)
    ctrl._apply_descendant_mask_cfg(module, ctrl.struct_cfg)
    assert module.hsic_exclude_descendants is True
    assert module.hsic_descendant_hops == 1
    assert module.hsic_descendant_weight == pytest.approx(0.5)


def test_adaptive_phase_override_allows_null_hops(save_dir):
    """``hops: null`` means "full transitive closure", not "unset"."""
    ctrl = _phase_controller(save_dir, structure={"hsic_descendant_hops": None})
    module = _mask_stub(hsic_descendant_hops=1)
    ctrl._apply_descendant_mask_cfg(module, ctrl.struct_cfg)
    assert module.hsic_descendant_hops is None


def test_adaptive_phase_switch_resets_the_ema(save_dir):
    """A phase must not inherit an adjacency smoothed under other settings."""
    ctrl = _phase_controller(save_dir, structure={"hsic_exclude_descendants": True})
    module = _mask_stub(_descendant_ema_score=torch.ones(3, 4))
    ctrl._apply_descendant_mask_cfg(module, ctrl.struct_cfg)
    assert module._descendant_ema_score is None


def test_adaptive_phase_override_ignored_for_unsupported_module(save_dir):
    """A forecaster without the feature must not gain stray attributes."""
    ctrl = _phase_controller(save_dir, structure={"hsic_exclude_descendants": True})
    module: Any = SimpleNamespace()  # no hsic_exclude_descendants attribute
    ctrl._apply_descendant_mask_cfg(module, ctrl.struct_cfg)
    assert not hasattr(module, "hsic_exclude_descendants")


# ---------------------------------------------------------------------------
# adaptive_trainer warmup anchoring (first structure phase only)
# ---------------------------------------------------------------------------

def _trainer_stub(epoch: int) -> Any:
    return SimpleNamespace(current_epoch=epoch, optimizers=[])


def test_warmup_anchor_set_on_first_structure_phase(save_dir):
    """The warmup must be charged to structural epochs, not to the run.

    The reconstruct warmup easily consumes 100+ global epochs, which would
    silently expire a 50-epoch guard before the first structural step.
    """
    ctrl = _phase_controller(save_dir, structure={
        "hsic_exclude_descendants": True,
        "hsic_descendant_warmup_epochs": 25,
    })
    module = _mask_stub(_descendant_warmup_anchor=0)
    ctrl._struct_phase_count = 1
    ctrl._apply_descendant_warmup_anchor(_trainer_stub(180), module)
    assert module._descendant_warmup_anchor == 180


def test_warmup_anchor_served_once_on_later_structure_phases(save_dir):
    """Second and later structure phases must not re-pay the delay."""
    ctrl = _phase_controller(save_dir)
    module = _mask_stub(_descendant_warmup_anchor=180)
    ctrl._struct_phase_count = 2
    ctrl._apply_descendant_warmup_anchor(_trainer_stub(400), module)
    assert module._descendant_warmup_anchor is None


def test_warmup_anchor_skipped_for_unsupported_module(save_dir):
    ctrl = _phase_controller(save_dir)
    module: Any = SimpleNamespace()
    ctrl._struct_phase_count = 1
    ctrl._apply_descendant_warmup_anchor(_trainer_stub(10), module)
    assert not hasattr(module, "_descendant_warmup_anchor")


def _routing_module(**overrides) -> Any:
    """Stub exposing the gradient-routing param groups ``_apply_phase`` needs."""
    stub = _mask_stub(**overrides)
    stub._structural_params = [torch.zeros(1, requires_grad=True)]
    stub._reconstruction_params = [torch.zeros(1, requires_grad=True)]
    return stub


def test_apply_phase_anchors_first_structure_phase_then_serves_it(save_dir):
    """End-to-end through ``_apply_phase``: anchor on phase 1, ``None`` after."""
    ctrl = _phase_controller(save_dir, structure={
        "hsic_exclude_descendants": True,
        "hsic_descendant_warmup_epochs": 20,
    })
    module = _routing_module(_descendant_warmup_anchor=0)

    # Reconstruct phase must NOT touch the anchor (no structural epochs pass).
    ctrl._apply_phase(_trainer_stub(0), module, "reconstruct")
    assert module._descendant_warmup_anchor == 0
    assert ctrl._struct_phase_count == 0

    # First structure phase: warmup counted from here.
    ctrl._apply_phase(_trainer_stub(150), module, "structure")
    assert ctrl._struct_phase_count == 1
    assert module._descendant_warmup_anchor == 150
    # ...and the forecaster guard agrees: still masked-off mid-warmup.
    module.current_epoch = 160
    assert _call_builder(module, _sparse_score())[0] is None
    module.current_epoch = 170
    assert _call_builder(module, _sparse_score())[0] is not None

    # Back to reconstruct, then the SECOND structure phase: already served.
    ctrl._apply_phase(_trainer_stub(200), module, "reconstruct")
    ctrl._apply_phase(_trainer_stub(260), module, "structure")
    assert ctrl._struct_phase_count == 2
    assert module._descendant_warmup_anchor is None
    module.current_epoch = 260
    assert _call_builder(module, _sparse_score())[0] is not None


# ---------------------------------------------------------------------------
# soft_transitive_closure
# ---------------------------------------------------------------------------

def test_soft_transitive_closure_chain_is_monotone():
    # Chain 0 -> 1 -> 2 -> 3 with confident edges (pi[i, j] = P(j -> i)).
    pi = torch.zeros((4, 4))
    pi[1, 0] = 0.9
    pi[2, 1] = 0.9
    pi[3, 2] = 0.9
    desc = soft_transitive_closure(pi, tnorm="min")
    # desc[i, j] = soft score that j is a descendant of i
    assert desc[0, 1] > 0.8 and desc[0, 2] > 0.8 and desc[0, 3] > 0.8
    assert desc[3].max() == 0.0          # a sink has no descendants
    assert desc[0, 0] == 0.0             # acyclic: no self-reach


def test_soft_transitive_closure_matches_boolean_on_hard_graph():
    # On a 0/1 posterior the soft closure must equal the boolean closure.
    adj = _chain_adjacency(5)
    soft = soft_transitive_closure(adj.float(), tnorm="min")
    hard = transitive_closure(adj, hops=None)
    assert torch.equal(soft.bool(), hard)


def test_soft_transitive_closure_rejects_non_square():
    with pytest.raises(ValueError):
        soft_transitive_closure(torch.zeros((2, 3)))


# ---------------------------------------------------------------------------
# build_hsic_pair_mask_budgeted
# ---------------------------------------------------------------------------

def test_budgeted_mask_never_collapses_on_a_dense_cyclic_graph():
    """The collapse case of the threshold variant: the budgeted mask must NOT
    fall back — the cap IS the guard."""
    dense = torch.full((4, 4), 0.9)   # fully dense -> cyclic
    mask, kept_frac, is_cyclic = build_hsic_pair_mask_budgeted(
        score_tensor=dense, s_seq_len=0, homogeneous_nodes=True,
        budget_frac=0.25,
    )
    assert mask is not None
    assert is_cyclic is True
    # The cap holds: at most ~budget_frac of the pairs are excluded.
    assert kept_frac >= 1.0 - 0.25 - 1e-6


def test_budgeted_mask_excludes_top_descendant_scores_per_row():
    """The excluded pairs are the most descendant-ish of each row."""
    # Chain 0 -> 1 -> 2 (strong), so row 0 has a child AND a grandchild.
    score = torch.zeros((3, 3))
    score[1, 0] = 0.9
    score[2, 1] = 0.9
    mask, _, _ = build_hsic_pair_mask_budgeted(
        score_tensor=score, s_seq_len=0, homogeneous_nodes=True,
        budget_frac=0.34, per_row=True, exclude_self=True,
    )
    # Row 0: self (0,0) + descendants (0,1) and (0,2) excluded.
    assert mask[0, 0].item() == 0.0
    assert mask[0, 1].item() == 0.0
    assert mask[0, 2].item() == 0.0
    # Row 2 (a sink): only itself excluded — its ancestors are legitimate
    # independence targets and must be kept.
    assert mask[2, 0].item() == 1.0
    assert mask[2, 1].item() == 1.0
    assert mask[2, 2].item() == 0.0


def test_budgeted_mask_is_detached():
    score = torch.zeros((3, 3), requires_grad=True)
    with torch.enable_grad():
        mask, _, _ = build_hsic_pair_mask_budgeted(
            score_tensor=score, s_seq_len=0, homogeneous_nodes=True,
        )
    assert mask.requires_grad is False
    assert mask.grad_fn is None


def test_budgeted_mask_split_mode_pins_cross_columns():
    L_S, L_X = 2, 3
    score = torch.zeros((L_X, L_S + L_X))
    score[1, L_S + 0] = 0.9      # X0 -> X1
    score[2, L_S + 1] = 0.9      # X1 -> X2
    mask, _, _ = build_hsic_pair_mask_budgeted(
        score_tensor=score, s_seq_len=L_S, homogeneous_nodes=False,
        budget_frac=0.5,
    )
    # S columns are exogenous: they can never be descendants of an X node.
    assert bool((mask[:, :L_S] == 1.0).all())


def test_budgeted_mask_rejects_bad_budget():
    with pytest.raises(ValueError):
        build_hsic_pair_mask_budgeted(
            score_tensor=torch.zeros((3, 3)), s_seq_len=0,
            homogeneous_nodes=True, budget_frac=0.0,
        )


# ---------------------------------------------------------------------------
# Forecaster dispatch: budget mode
# ---------------------------------------------------------------------------

def test_forecaster_budget_mode_dispatches_and_never_collapses():
    """On a dense cyclic graph the threshold mode falls back; the budget mode
    must return a mask (the cap is the guard)."""
    dense = torch.full((3, 4), 0.9)
    stub = _mask_stub(hsic_descendant_mode="budget",
                      hsic_descendant_budget_frac=0.25)
    mask, kept, _, _ = _call_builder(stub, dense)
    assert mask is not None
    assert kept >= 1.0 - 0.25 - 1e-6


def test_forecaster_budget_mode_respects_warmup():
    """The warmup guard is orthogonal to the mode: it applies in budget mode
    too (the self-confirmation risk is bounded by the cap, not eliminated)."""
    stub = _mask_stub(current_epoch=3, hsic_descendant_warmup_epochs=10,
                      hsic_descendant_mode="budget")
    assert _call_builder(stub, _sparse_score())[0] is None
    stub.current_epoch = 10
    assert _call_builder(stub, _sparse_score())[0] is not None


# ---------------------------------------------------------------------------
# Confident-descendant override (budget mode)
# ---------------------------------------------------------------------------

def _chain_score():
    # Chain 0 -> 1 -> 2 with strong edges: node 0 is a confident source.
    score = torch.zeros((3, 3))
    score[1, 0] = 0.9
    score[2, 1] = 0.9
    return score


def test_confident_override_allows_empty_source_row():
    """A confidently upstream node loses ALL its terms (correct for a source).

    Budget alone (k = ceil(0.34 * 3) = 2 per row) could never empty row 0;
    the override extends the exclusion to every column clearing 0.8."""
    mask, kept_frac, _ = build_hsic_pair_mask_budgeted(
        score_tensor=_chain_score(), s_seq_len=0, homogeneous_nodes=True,
        budget_frac=0.34, per_row=True, exclude_self=True,
        confident_score=0.8,
    )
    assert bool((mask[0] == 0.0).all())           # source row: zero terms
    # Ancestors are legitimate independence targets and must be kept.
    assert mask[1, 0].item() == 1.0
    assert mask[2, 0].item() == 1.0 and mask[2, 1].item() == 1.0


def test_confident_override_default_off_keeps_cap():
    """confident_score=None reproduces the pure budget-cap behaviour."""
    dense = torch.full((4, 4), 0.9)
    mask, kept_frac, _ = build_hsic_pair_mask_budgeted(
        score_tensor=dense, s_seq_len=0, homogeneous_nodes=True,
        budget_frac=0.25, confident_score=None,
    )
    assert kept_frac >= 1.0 - 0.25 - 1e-6


def test_confident_override_bypasses_cap_on_confident_evidence():
    """On a dense confident posterior the override (not the cap) dominates."""
    dense = torch.full((4, 4), 0.9)
    mask, kept_frac, _ = build_hsic_pair_mask_budgeted(
        score_tensor=dense, s_seq_len=0, homogeneous_nodes=True,
        budget_frac=0.25, confident_score=0.8,
    )
    assert kept_frac < 1.0 - 0.25   # cap exceeded: nearly everything excluded


def test_confident_override_never_excludes_weak_or_zero_scores():
    """Uncertain pairs survive: the override requires score >= threshold."""
    score = _chain_score()
    score[1, 0] = 0.1               # weak edge: descendant score 0.1 < 0.8
    mask, _, _ = build_hsic_pair_mask_budgeted(
        score_tensor=score, s_seq_len=0, homogeneous_nodes=True,
        budget_frac=0.2, per_row=True, exclude_self=True,  # k=1: diag only
        confident_score=0.8,
    )
    # Row 0: (0,1) scores 0.1 (kept), (0,2) scores min(0.1, 0.9)=0.1 (kept).
    assert mask[0, 1].item() == 1.0 and mask[0, 2].item() == 1.0
    assert mask[0, 0].item() == 0.0   # self still excluded (budget top rank)


def test_confident_override_rejects_bad_value():
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            build_hsic_pair_mask_budgeted(
                score_tensor=torch.zeros((3, 3)), s_seq_len=0,
                homogeneous_nodes=True, confident_score=bad,
            )


def test_forecaster_budget_mode_passes_confident_score():
    """The forecaster plumbs the config key through to the mask builder."""
    stub = _mask_stub(hsic_descendant_mode="budget",
                      hsic_descendant_confident_score=0.8,
                      homogeneous_nodes=True)
    mask, kept, _, _ = _call_builder(stub, _chain_score())
    assert mask is not None
    assert bool((mask[0] == 0.0).all())


# ---------------------------------------------------------------------------
# First-order degrade (threshold mode)
# ---------------------------------------------------------------------------

def _cyclic_score():
    # Homogeneous 3-cycle 0 -> 1 -> 2 -> 0 (score[i, j] = P(j -> i)): the full
    # closure saturates (every node reaches every node), the first-order mask
    # does not.
    score = torch.zeros((3, 3))
    score[1, 0] = 0.9
    score[2, 1] = 0.9
    score[0, 2] = 0.9
    return score


def test_threshold_mode_degrades_to_first_order_on_cycle():
    """A cyclic hardened graph makes the full closure meaningless; instead of
    falling back to unmasked HSIC the mask is rebuilt with hops=1."""
    stub = _mask_stub(homogeneous_nodes=True, S_seq_len=0)
    mask, kept, cyc, first = _call_builder(stub, _cyclic_score())
    assert cyc is True and first is True
    assert mask is not None
    expected, exp_kept, _ = build_hsic_pair_mask(
        score_tensor=_cyclic_score(), s_seq_len=0, homogeneous_nodes=True,
        threshold=0.5, hops=1, exclude_self=True, excluded_weight=0.0,
    )
    assert torch.equal(mask, expected)
    assert kept == pytest.approx(exp_kept)


def test_threshold_mode_degrades_on_starvation_without_cycle():
    """Starvation alone (kept_frac < min_kept_frac) also triggers the degrade."""
    # Chain 0 -> 1 -> 2: the full closure excludes 6/9 pairs (diag + all
    # descendants), hops=1 excludes 5/9 (diag + direct children).
    score = torch.zeros((3, 3))
    score[1, 0] = 0.9
    score[2, 1] = 0.9
    stub = _mask_stub(homogeneous_nodes=True, S_seq_len=0,
                      hsic_descendant_min_kept_frac=0.4)
    mask, kept, cyc, first = _call_builder(stub, score)
    assert cyc is False and first is True
    assert mask is not None
    assert kept == pytest.approx(4.0 / 9.0)
    # Ancestors are legitimate independence targets and survive the degrade.
    assert mask[1, 0].item() == 1.0 and mask[2, 0].item() == 1.0


def test_threshold_mode_degrade_disabled_keeps_legacy_fallback():
    """hsic_descendant_degrade_first_order=False restores the old behaviour:
    a starving/cyclic closure falls straight back to unmasked HSIC."""
    stub = _mask_stub(homogeneous_nodes=True, S_seq_len=0,
                      hsic_descendant_degrade_first_order=False)
    mask, kept, cyc, first = _call_builder(stub, _cyclic_score())
    assert mask is None and kept == 0.0 and cyc is True and first is False


def test_threshold_mode_hops1_already_first_order_does_not_degrade():
    """With hops=1 configured there is nothing to degrade to: the mask is
    used as-is and the flag stays False."""
    stub = _mask_stub(homogeneous_nodes=True, S_seq_len=0,
                      hsic_descendant_hops=1)
    mask, kept, cyc, first = _call_builder(stub, _cyclic_score())
    assert mask is not None and first is False
    assert kept == pytest.approx(1.0 / 3.0)


def test_threshold_mode_first_order_still_starving_falls_back():
    """Only when even the first-order mask starves do we drop to unmasked."""
    stub = _mask_stub(homogeneous_nodes=True, S_seq_len=0,
                      hsic_descendant_min_kept_frac=0.5)
    mask, kept, cyc, first = _call_builder(stub, _cyclic_score())
    assert mask is None and cyc is True and first is True
    assert kept == pytest.approx(1.0 / 3.0)

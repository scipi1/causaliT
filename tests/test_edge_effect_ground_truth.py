"""Tests for `SCMDataset.compute_edge_effect_ground_truth`.

The point of the method is to distinguish an edge with a real average causal
effect from an edge that only MODULATES another edge's effect. Both cases are
checked here against SCMs whose answer is known in closed form, so the metric
cannot silently drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from scm_ds.scm import NodeSpec, SCMDataset


def _dataset(specs, input_labels, source_labels, name="t"):
    return SCMDataset(
        name=name,
        description="test",
        tags=["test"],
        specs=specs,
        params={},
        singles={s.name: (lambda rng, n: rng.standard_normal(n)) for s in specs},
        groups=[],
        input_labels=input_labels,
        target_labels=[],
        source_labels=source_labels,
    )


def _by_edge(report):
    return {e["edge"]: e for e in report["edges"]}


def test_linear_edge_matches_closed_form():
    """For `X2 = 2*X1`, the controlled direct effect is exactly 2*(hi - lo)."""
    specs = [
        NodeSpec("X1", [], "eps_X1"),
        NodeSpec("X2", ["X1"], "2*X1 + 0.01*eps_X2"),
    ]
    ds = _dataset(specs, ["X1", "X2"], None)
    rep = ds.compute_edge_effect_ground_truth(n_samples=4000, seed=0,
                                              include_ancestor_pairs=False)
    e = _by_edge(rep)["X1->X2"]

    expected = 2.0 * (e["do_hi"] - e["do_lo"])
    assert e["ate_direct"] == pytest.approx(expected, rel=0.02)
    assert e["ate_total"] == pytest.approx(expected, rel=0.02)
    # A pure linear channel is strong and modulates nothing.
    assert e["label"] == "strong"
    assert e["modifier"] == pytest.approx(0.0, abs=1e-6)


def test_pure_interaction_is_modifier_only_not_weak():
    """`X3 = X1*X2` with independent zero-mean parents.

    E[X3 | do(X1=v)] = v * E[X2] = 0 for every v, so BOTH edges have a zero
    average causal effect while each one flips the other's slope. They must be
    labelled `modifier_only` -- never `strong` (there is no average effect to
    detect) and never `weak` (the edge is not irrelevant).
    """
    specs = [
        NodeSpec("X1", [], "eps_X1"),
        NodeSpec("X2", [], "eps_X2"),
        NodeSpec("X3", ["X1", "X2"], "X1*X2 + 0.01*eps_X3"),
    ]
    ds = _dataset(specs, ["X1", "X2", "X3"], None)
    rep = ds.compute_edge_effect_ground_truth(n_samples=8000, seed=0,
                                              include_ancestor_pairs=False)
    edges = _by_edge(rep)

    for name, co in (("X1->X3", "X2"), ("X2->X3", "X1")):
        e = edges[name]
        assert abs(e["effect_std"]) < 0.02, f"{name} should have no average effect"
        assert e["modifier"] > 1.0, f"{name} should modulate {co}"
        assert e["label"] == "modifier_only"
        # The modulation is attributed to the right co-parent.
        assert e["modifier_per_coparent"][co] == pytest.approx(e["modifier"], rel=1e-6)


def test_effect_std_survives_symmetric_mechanisms():
    """`X2 = X1**2` has ate_total ~ 0 (symmetric endpoints) but a LARGE effect.

    This is why `effect_std` over the whole do-grid is the strength measure and a
    single hi-vs-lo contrast is not: an even mechanism cancels at the endpoints.
    """
    specs = [
        NodeSpec("X1", [], "eps_X1"),
        NodeSpec("X2", ["X1"], "X1**2 + 0.01*eps_X2"),
    ]
    ds = _dataset(specs, ["X1", "X2"], None)
    rep = ds.compute_edge_effect_ground_truth(n_samples=4000, seed=0,
                                              include_ancestor_pairs=False)
    e = _by_edge(rep)["X1->X2"]

    assert abs(e["ate_total"]) < 0.15, "endpoints of an even mechanism cancel"
    assert e["effect_std"] > 0.3, "but the edge is genuinely strong"
    assert e["label"] == "strong"


def test_ancestor_pairs_exclude_true_edges_and_cover_x_nodes():
    """Indirect pairs are reported, direct edges are not repeated there."""
    specs = [
        NodeSpec("S1", [], "eps_S1"),
        NodeSpec("X1", ["S1"], "S1 + 0.01*eps_X1"),
        NodeSpec("X2", ["X1"], "X1 + 0.01*eps_X2"),
    ]
    ds = _dataset(specs, ["X1", "X2"], ["S1"])
    rep = ds.compute_edge_effect_ground_truth(n_samples=3000, seed=0)

    pairs = {p["pair"] for p in rep["ancestor_pairs"]}
    assert "S1->X2" in pairs           # indirect: S1 -> X1 -> X2
    assert "S1->X1" not in pairs       # direct edge, already in `edges`
    assert "X1->X2" not in pairs

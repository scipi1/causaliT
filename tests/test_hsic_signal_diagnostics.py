"""Tests for local node-wise HSIC signal diagnostics."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.hsic_signal_diagnostics import (
    DiagnosticBatch,
    gradient_probe,
    node_hsic,
    node_query_views,
    query_intervention_probe,
)
from test_dropout_selection import _make_batch, _make_forecaster_config


def _model_and_batches():
    torch.manual_seed(0)
    cfg = _make_forecaster_config()
    cfg["model"]["kwargs"].update(
        {
            "struct_embedding_type": "orthogonal_fixed",
            "homogeneous_nodes": True,
            "remove_query_projection": True,
            "remove_key_projection": True,
            "normalize_query": True,
            "query_fanin_scale": 4.0,
        }
    )
    model = AttentionSelectorForecaster(cfg)
    batches = [
        DiagnosticBatch(*_make_batch(batch=16, seed=1)),
        DiagnosticBatch(*_make_batch(batch=16, seed=2)),
    ]
    return model, batches


def _gt():
    gt = np.zeros((6, 6), dtype=bool)
    gt[2, 0] = True
    gt[3, 1] = True
    gt[4, [2, 3]] = True
    gt[5, 4] = True
    return gt


def test_node_hsic_shape_and_multiscale_threading():
    model, batches = _model_and_batches()
    rows_single, total_single = node_hsic(model, batches)
    rows_multi, total_multi = node_hsic(
        model, batches, bandwidth_multipliers=[0.5, 1.0, 2.0]
    )
    assert rows_single.shape == rows_multi.shape == (6,)
    assert np.isfinite(rows_single).all() and np.isfinite(rows_multi).all()
    assert np.isfinite(total_single) and np.isfinite(total_multi)
    assert total_single != pytest.approx(total_multi)


def test_query_intervention_probe_restores_weights_and_is_deterministic():
    model, batches = _model_and_batches()
    before = [q.detach().clone() for q in node_query_views(model)]
    first = query_intervention_probe(model, batches, _gt(), n_wrong=1, seed=3)
    second = query_intervention_probe(model, batches, _gt(), n_wrong=1, seed=3)
    after = [q.detach() for q in node_query_views(model)]

    assert len(first) == 4
    for b, a in zip(before, after):
        assert torch.equal(b, a)
    for column in first.columns:
        np.testing.assert_allclose(first[column], second[column])


def test_gradient_probe_outputs_finite_parent_geometry():
    model, batches = _model_and_batches()
    out = gradient_probe(model, batches, _gt(), training_path=False, seed=5)
    assert len(out) == 4
    numeric = out.drop(columns=["training_path"]).to_numpy(dtype=float)
    assert np.isfinite(numeric).all()
    assert (out.grad_norm_mean >= 0).all()


def test_training_path_gradient_probe_is_finite():
    model, batches = _model_and_batches()
    out = gradient_probe(model, batches[:1], _gt(), training_path=True, seed=6)
    assert len(out) == 4
    numeric = out.drop(columns=["training_path", "direction_stability"]).to_numpy(dtype=float)
    assert np.isfinite(numeric).all()
    # One batch has no gradient-direction pair, so stability is undefined.
    assert out.direction_stability.isna().all()

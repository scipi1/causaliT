"""Unit tests of the central metrics module causaliT.metrics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from causaliT import metrics as M

# Model orientation: T[i, j] = 1 means j -> i.  Chain 0 -> 1 -> 2.
CHAIN = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)


def test_auroc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    s = rng.integers(0, 4, 50).astype(float)
    y = rng.integers(0, 2, 50)
    assert M.auroc(s, y) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(M.auroc(s, np.zeros(50)))


def test_standard_shd_counts_reversal_cellwise():
    rev = CHAIN.T.copy()
    out = M.standard_shd(rev, CHAIN)
    assert (out["missing"], out["extra"], out["reversed"]) == (2, 2, 2)
    assert out["shd"] == 6
    assert M.standard_shd(CHAIN, CHAIN)["shd"] == 0


def test_class_masks_partition_square_block():
    cls = M.causal_class_masks(CHAIN)
    stacked = sum(m.astype(int) for m in cls.values())
    assert np.array_equal(stacked, 1 - np.eye(3, dtype=int))
    assert cls["ancestors"][2, 0] and cls["descendants"][0, 2]


def test_attention_mass_sums_to_one():
    att = np.random.default_rng(1).random((3, 3))
    np.fill_diagonal(att, 0.0)
    m = M.attention_mass_metrics(att, CHAIN)
    total = sum(m[k] for k in ("mass_on_parents", "mass_on_ancestors",
                               "mass_on_descendants", "mass_on_others"))
    assert total == pytest.approx(1.0)


def test_order_agreement_perfect_and_reversed():
    assert M.order_agreement(np.array([3.0, 2.0, 1.0]), CHAIN) == (1.0, 1.0)
    assert M.order_agreement(np.array([1.0, 2.0, 3.0]), CHAIN) == (0.0, -1.0)


def test_mec_threshold_on_perfect_scores():
    theta, ok = M.mec_threshold(CHAIN * 0.8, CHAIN)
    assert ok and theta == pytest.approx(0.8)


def test_prediction_metrics_perfect_fit():
    y = torch.randn(4, 5, 2)
    res = SimpleNamespace(outputs=y, targets=y, metadata={})
    df = M.prediction_metrics(res, per_sample=False)
    assert list(df["feature"]) == ["f0", "f1"]
    assert np.allclose(df["MSE"], 0.0) and np.allclose(df["R2"], 1.0)


def test_r2_and_per_node_mse():
    t = torch.randn(32, 3)
    assert float(M.r2(t, t)) == pytest.approx(1.0)
    assert torch.allclose(M.per_node_mse(t + 1.0, t), torch.ones(3))


def test_ate_errors():
    out = M.ate_errors(3.0, 1.0, 1.5)
    assert out == {"model_ate": 2.0, "abs_error": 0.5, "rel_error": pytest.approx(1 / 3)}
    assert M.ate_errors(1.0, 0.0, None)["abs_error"] is None


def test_attention_entropy_uniform():
    att = torch.full((2, 3, 4), 0.25)
    assert torch.allclose(M.attention_entropy(att), torch.full((2, 3), np.log(4.0)), atol=1e-6)

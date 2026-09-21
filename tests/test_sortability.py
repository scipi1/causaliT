"""Unit tests for scm_ds.sortability and the node-wise _normalize methods."""

import numpy as np

from scm_ds.sortability import (
    order_alignment_paths,
    r2_sortability,
    var_sortability,
)


def _chain_adj(d):
    """W[i, j] = edge i -> j for a simple chain 0 -> 1 -> ... -> d-1."""
    W = np.zeros((d, d))
    for i in range(d - 1):
        W[i, i + 1] = 1.0
    return W


def test_varsortability_perfect_chain():
    rng = np.random.default_rng(0)
    n, d = 2000, 5
    X = np.zeros((n, d))
    X[:, 0] = rng.standard_normal(n)
    for i in range(1, d):
        # variance strictly increasing along the chain
        X[:, i] = X[:, i - 1] + rng.standard_normal(n)
    W = _chain_adj(d)
    assert var_sortability(X, W) == 1.0


def test_varsortability_reversed_chain():
    rng = np.random.default_rng(0)
    n, d = 2000, 5
    X = np.zeros((n, d))
    X[:, 0] = rng.standard_normal(n)
    for i in range(1, d):
        # variance strictly DEcreasing along the chain
        X[:, i] = 0.5 * X[:, i - 1] + rng.standard_normal(n) * 0.01
    W = _chain_adj(d)
    assert var_sortability(X, W) == 0.0


def test_varsortability_flat_is_half():
    rng = np.random.default_rng(0)
    # all nodes iid: every strict-ancestor pair ties -> 0.5
    X = rng.standard_normal((2000, 4))
    X[:] = X[:, :1]  # identical columns -> identical variances -> all ties
    W = _chain_adj(4)
    assert var_sortability(X, W) == 0.5


def test_order_alignment_paths_exact():
    # Chain 0 -> 1 -> 2: directed paths (0,1), (1,2), (0,2), each weight 1.
    W = _chain_adj(3)
    assert order_alignment_paths(W, np.array([1.0, 2.0, 3.0])) == 1.0
    assert order_alignment_paths(W, np.array([3.0, 2.0, 1.0])) == 0.0
    assert order_alignment_paths(W, np.array([1.0, 1.0, 1.0])) == 0.5  # ties
    # wrong on (0,1), right on (1,2) and (0,2): 2/3
    assert order_alignment_paths(W, np.array([2.0, 1.0, 3.0])) == 2.0 / 3.0


def test_r2_sortability_above_chance_on_chain():
    rng = np.random.default_rng(1)
    n, d = 5000, 6
    X = np.zeros((n, d))
    X[:, 0] = rng.standard_normal(n)
    for i in range(1, d):
        X[:, i] = X[:, i - 1] + rng.standard_normal(n) * 0.3
    W = _chain_adj(d)
    val = r2_sortability(X, W)
    assert 0.5 < val <= 1.0


def test_r2_sortability_scale_invariant():
    rng = np.random.default_rng(2)
    n, d = 3000, 4
    X = np.zeros((n, d))
    X[:, 0] = rng.standard_normal(n)
    for i in range(1, d):
        X[:, i] = 2.0 * X[:, i - 1] + rng.standard_normal(n)
    W = _chain_adj(d)
    scales = np.array([1.0, 100.0, 0.01, 7.0])
    Xs = X * scales
    assert r2_sortability(X, W) == r2_sortability(Xs, W)


def test_order_alignment_no_edges_nan():
    assert np.isnan(order_alignment_paths(np.zeros((3, 3)), np.arange(3.0)))


def _bare_scm_dataset():
    from scm_ds.scm import SCMDataset

    inst = SCMDataset.__new__(SCMDataset)
    inst.name = "unit-test"
    return inst


def test_nodewise_standardize():

    data = np.zeros((500, 3, 2))
    rng = np.random.default_rng(3)
    data[:, 0, 0] = rng.normal(10.0, 5.0, 500)     # big scale
    data[:, 1, 0] = rng.normal(-2.0, 0.01, 500)    # tiny scale
    data[:, 2, 0] = rng.uniform(-1, 1, 500)
    out, stats = _bare_scm_dataset()._normalize(data, method="standardize_node")
    assert stats["per_node"] is True and stats["method"] == "standardize_node"
    np.testing.assert_allclose(out[:, :, 0].mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(out[:, :, 0].std(axis=0), 1.0, atol=1e-2)
    assert len(stats["mean"]) == 3 and len(stats["std"]) == 3


def test_nodewise_minmax():
    from scm_ds.scm import SCMDataset

    rng = np.random.default_rng(4)
    data = np.zeros((500, 2, 2))
    data[:, 0, 0] = rng.normal(0.0, 100.0, 500)
    data[:, 1, 0] = rng.normal(50.0, 0.1, 500)
    out, stats = _bare_scm_dataset()._normalize(data, method="minmax_node")
    assert stats["per_node"] is True
    for col in range(2):
        assert out[:, col, 0].min() == 0.0
        assert out[:, col, 0].max() == 1.0


def test_global_methods_unchanged():
    from scm_ds.scm import SCMDataset

    rng = np.random.default_rng(5)
    data = np.zeros((400, 2, 2))
    data[:, 0, 0] = rng.normal(0.0, 1.0, 400)
    data[:, 1, 0] = rng.normal(0.0, 100.0, 400)
    out, stats = _bare_scm_dataset()._normalize(data, method="minmax")
    assert "per_node" not in stats
    # global minmax: pooled range [0, 1], node scales NOT flattened
    assert out[:, :, 0].min() == 0.0 and out[:, :, 0].max() == 1.0
    assert out[:, 1, 0].std() > 10 * out[:, 0, 0].std()

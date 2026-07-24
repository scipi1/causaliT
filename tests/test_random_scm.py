"""
Tests for `scm_ds.random_scm` — random SCM/DAG sampling.

Run with:
    pytest tests/test_random_scm.py -v

Covers:
- Reproducibility (same seed -> identical DAG; different seed -> different DAG)
- ER-k edge-count fidelity (total edges == round(degree * n_nodes))
- S/X structure (source count, sources have no parents, input count)
- Acyclicity (guaranteed by construction / topo-sort succeeds)
- Structural-equation family (linear vs nonlinear expressions)
- Config persisted in metadata
- End-to-end `generate_ds` writes the expected files (skipped if graphviz `dot`
  binary is unavailable)
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scm_ds.random_scm import RandomSCMConfig, sample_random_scm_dataset


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _adjacency(ds) -> np.ndarray:
    """Return the DAG adjacency (numpy, fixed node order)."""
    return ds.scm.adjacency(as_dataframe=False)


def _n_edges(ds) -> int:
    return sum(len(spec.parents) for spec in ds.specs)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def test_same_seed_identical_dag():
    cfg = RandomSCMConfig(n_nodes=25, degree=2, seed=123,
                          linearity="mixed", noise="mixed", s_x_ratio=0.3)
    ds1 = sample_random_scm_dataset(cfg)
    ds2 = sample_random_scm_dataset(cfg)

    # Same node names, parents and adjacency.
    assert [s.name for s in ds1.specs] == [s.name for s in ds2.specs]
    assert {s.name: s.parents for s in ds1.specs} == {s.name: s.parents for s in ds2.specs}
    assert np.array_equal(_adjacency(ds1), _adjacency(ds2))

    # Same baked expressions and params => identical mechanism.
    assert {s.name: s.expr for s in ds1.specs} == {s.name: s.expr for s in ds2.specs}

    # Same sampled data.
    df1 = ds1.sample(n=200, seed=7)
    df2 = ds2.sample(n=200, seed=7)
    assert np.allclose(df1.values, df2.values)


def test_different_seed_different_dag():
    ds_a = sample_random_scm_dataset(RandomSCMConfig(
        n_nodes=30, degree=2, seed=1, linearity="linear", noise="gaussian", s_x_ratio=0.3))
    ds_b = sample_random_scm_dataset(RandomSCMConfig(
        n_nodes=30, degree=2, seed=2, linearity="linear", noise="gaussian", s_x_ratio=0.3))
    # With 30 nodes and ~60 edges, identical adjacency across seeds is astronomically unlikely.
    assert not np.array_equal(_adjacency(ds_a), _adjacency(ds_b))


# --------------------------------------------------------------------------- #
# ER-k edge-count fidelity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_nodes,degree", [(20, 1), (20, 2), (30, 3)])
def test_edge_count_matches_erk(n_nodes, degree):
    cfg = RandomSCMConfig(n_nodes=n_nodes, degree=degree, seed=0, s_x_ratio=0.3)
    ds = sample_random_scm_dataset(cfg)

    target_m = round(degree * n_nodes)
    # With these settings target_m >= n_inputs (the guaranteed one-parent edges),
    # so the exact-edge sampler hits target_m exactly.
    assert _n_edges(ds) == target_m
    assert ds.meta["n_edges"] == target_m


# --------------------------------------------------------------------------- #
# S/X structure
# --------------------------------------------------------------------------- #

def test_source_count_and_no_parents():
    cfg = RandomSCMConfig(n_nodes=20, degree=2, seed=5, n_sources=6)
    ds = sample_random_scm_dataset(cfg)

    assert len(ds.source_labels) == 6
    assert len(ds.input_labels) == 20 - 6
    assert ds.target_labels == []

    parents = {s.name: s.parents for s in ds.specs}
    source_set = set(ds.source_labels)
    # Sources must be roots (no incoming edges) and expr must be pure noise.
    for s in ds.source_labels:
        assert parents[s] == []
    for spec in ds.specs:
        if spec.name in source_set:
            assert spec.expr == f"eps_{spec.name}"

    # No edge may point into a source node.
    for spec in ds.specs:
        if spec.parents:
            assert spec.name not in source_set


def test_s_x_ratio_resolution():
    cfg = RandomSCMConfig(n_nodes=10, degree=1, seed=0, s_x_ratio=0.5)
    ds = sample_random_scm_dataset(cfg)
    assert len(ds.source_labels) == 5
    assert len(ds.input_labels) == 5


def test_every_input_has_parent_by_default():
    cfg = RandomSCMConfig(n_nodes=25, degree=1, seed=3, s_x_ratio=0.4)
    ds = sample_random_scm_dataset(cfg)
    parents = {s.name: s.parents for s in ds.specs}
    for x in ds.input_labels:
        assert len(parents[x]) >= 1


# --------------------------------------------------------------------------- #
# Acyclicity
# --------------------------------------------------------------------------- #

def test_acyclic_topo_order():
    cfg = RandomSCMConfig(n_nodes=40, degree=4, seed=11, s_x_ratio=0.25)
    ds = sample_random_scm_dataset(cfg)
    # SCM construction already validates acyclicity; topo order must cover all nodes.
    order = ds.scm._topo_order()
    assert len(order) == cfg.n_nodes


# --------------------------------------------------------------------------- #
# Structural-equation family
# --------------------------------------------------------------------------- #

def test_linear_expressions_have_no_nonlinearities():
    cfg = RandomSCMConfig(n_nodes=20, degree=2, seed=0, linearity="linear")
    ds = sample_random_scm_dataset(cfg)
    for spec in ds.specs:
        # No power / trig / tanh in linear mode.
        assert "**" not in spec.expr
        assert "sin(" not in spec.expr
        assert "tanh(" not in spec.expr


def test_nonlinear_expressions_contain_nonlinearity():
    cfg = RandomSCMConfig(n_nodes=25, degree=3, seed=0, linearity="nonlinear")
    ds = sample_random_scm_dataset(cfg)
    joined = " ".join(spec.expr for spec in ds.specs)
    assert any(tok in joined for tok in ("**2", "**3", "sin(", "tanh("))


# --------------------------------------------------------------------------- #
# Metadata persistence
# --------------------------------------------------------------------------- #

def test_config_persisted_in_meta():
    cfg = RandomSCMConfig(n_nodes=15, degree=2, seed=99, linearity="mixed", noise="nongaussian")
    ds = sample_random_scm_dataset(cfg)
    assert "random_scm_config" in ds.meta
    stored = ds.meta["random_scm_config"]
    assert stored["seed"] == 99
    assert stored["n_nodes"] == 15
    assert stored["linearity"] == "mixed"
    assert stored["noise"] == "nongaussian"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_invalid_n_sources_raises():
    with pytest.raises(ValueError):
        sample_random_scm_dataset(RandomSCMConfig(n_nodes=5, degree=1, seed=0, n_sources=5))
    with pytest.raises(ValueError):
        sample_random_scm_dataset(RandomSCMConfig(n_nodes=1, degree=1, seed=0))


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(shutil.which("dot") is None,
                    reason="Graphviz 'dot' binary not available")
def test_generate_ds_end_to_end():
    cfg = RandomSCMConfig(n_nodes=12, degree=2, seed=42, s_x_ratio=0.3,
                          linearity="mixed", noise="mixed")
    ds = sample_random_scm_dataset(cfg)

    # Use a project-local temp dir to avoid system-temp permission issues.
    project_root = Path(__file__).resolve().parent.parent
    tmp_root = tempfile.mkdtemp(prefix="random_ds_", dir=str(project_root))
    save_dir = Path(tmp_root) / "random_ds"
    try:
        ds.generate_ds(
            mode="flat",
            n=500,
            save_dir=str(save_dir),
            normalize_method="minmax",
            shared_embedding=False,
        )

        expected = [
            "ds.npz",
            "dag_adj_mask.csv",
            "dec1_cross_att_mask.csv",
            "dec1_self_att_mask.csv",
            "meta.json",
            "dataset_metadata.json",
        ]
        for fname in expected:
            assert (save_dir / fname).exists(), f"missing {fname}"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

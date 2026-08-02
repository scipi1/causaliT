"""
Tests for `scm_ds.random_scm` - random ER-k SCM/DAG sampling.

Run with:
    pytest tests/test_random_scm.py -v

Covers:
- Reproducibility (same seed -> identical DAG; different seed -> different DAG)
- ER-k edge-count fidelity (total edges == round(degree * n_nodes))
- Emergent S/X structure: sources ARE the roots, no edge points into an S node,
  every X node has at least one parent, and the source count matches the ER-k
  analytic expectation (1-(1-p)^n)/p
- Label permutation hides the topological order (anti-leak)
- Acyclicity (topo-sort succeeds)
- Structural-equation family (linear vs nonlinear expressions)
- Config + graph statistics persisted in metadata
- End-to-end `generate_ds`, including `compute_ate=False` (skipped without the
  graphviz `dot` binary)
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scm_ds.random_scm import (
    RandomSCMConfig,
    _sample_dag,
    expected_er_roots,
    sample_random_scm_dataset,
)



# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _adjacency(ds) -> np.ndarray:
    """Return the DAG adjacency (numpy, fixed node order)."""
    return ds.scm.adjacency(as_dataframe=False)


def _parents(ds) -> dict:
    return {spec.name: list(spec.parents) for spec in ds.specs}


def _n_edges(ds) -> int:
    return sum(len(spec.parents) for spec in ds.specs)


def _n_roots(n_nodes: int, degree: float, seed: int) -> int:
    """Sample only the DAG (no sympy SCM compilation) and count its roots.

    Structure-only statistics are checked over many large graphs; building the
    full SCM for each would make this file minutes-slow for no extra coverage.
    """
    cfg = RandomSCMConfig(n_nodes=n_nodes, degree=degree, seed=seed)
    _, source_labels, _, _ = _sample_dag(cfg, np.random.default_rng(seed))
    return len(source_labels)



# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def test_same_seed_identical_dag():
    cfg = RandomSCMConfig(n_nodes=25, degree=2, seed=123,
                          linearity="mixed", noise="mixed")
    ds1 = sample_random_scm_dataset(cfg)
    ds2 = sample_random_scm_dataset(cfg)

    # Same node names, parents and adjacency.
    assert [s.name for s in ds1.specs] == [s.name for s in ds2.specs]
    assert _parents(ds1) == _parents(ds2)
    assert np.array_equal(_adjacency(ds1), _adjacency(ds2))

    # Same baked expressions => identical mechanism.
    assert {s.name: s.expr for s in ds1.specs} == {s.name: s.expr for s in ds2.specs}

    # Same sampled data.
    df1 = ds1.sample(n=200, seed=7)
    df2 = ds2.sample(n=200, seed=7)
    assert np.allclose(df1.values, df2.values)


def test_different_seed_different_dag():
    ds_a = sample_random_scm_dataset(RandomSCMConfig(n_nodes=30, degree=2, seed=1))
    ds_b = sample_random_scm_dataset(RandomSCMConfig(n_nodes=30, degree=2, seed=2))
    # With 30 nodes and 60 edges, identical parent sets across seeds is
    # astronomically unlikely.
    assert _parents(ds_a) != _parents(ds_b)


# --------------------------------------------------------------------------- #
# ER-k edge-count fidelity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_nodes,degree", [(20, 1), (20, 2), (30, 3), (50, 4)])
def test_edge_count_matches_erk(n_nodes, degree):
    ds = sample_random_scm_dataset(
        RandomSCMConfig(n_nodes=n_nodes, degree=degree, seed=0))

    target_m = round(degree * n_nodes)
    assert _n_edges(ds) == target_m
    assert ds.meta["n_edges"] == target_m
    assert ds.meta["graph_stats"]["n_edges"] == target_m


# --------------------------------------------------------------------------- #
# Emergent S/X structure
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("degree", [1, 2, 4])
def test_sources_are_exactly_the_roots(degree):
    """Sources have no parents; every non-source has at least one parent."""
    ds = sample_random_scm_dataset(
        RandomSCMConfig(n_nodes=60, degree=degree, seed=17))

    parents = _parents(ds)
    source_set = set(ds.source_labels)

    assert source_set, "an ER DAG always has at least one root"
    assert ds.target_labels == []
    assert len(ds.source_labels) + len(ds.input_labels) == 60

    # Sources are roots and pure noise.
    for s in ds.source_labels:
        assert parents[s] == []
    for spec in ds.specs:
        if spec.name in source_set:
            assert spec.expr == f"eps_{spec.name}"

    # No edge may point into a source (the S->X / X->X mask contract).
    for spec in ds.specs:
        if spec.parents:
            assert spec.name not in source_set

    # Every X node has at least one parent (by definition of the partition).
    for x in ds.input_labels:
        assert len(parents[x]) >= 1


@pytest.mark.parametrize("degree", [1, 2, 4])
def test_source_count_matches_er_expectation(degree):
    """
    The realised number of roots must track the ER-k analytic expectation

        E[#roots] = (1 - (1-p)^n) / p ,  p = 2k/(n-1)

    (~43% of n for ER1, ~24% for ER2, ~12% for ER4).  Averaging over 20 seeds
    tightens the sampling noise enough for a 20% relative tolerance.
    """
    n_nodes = 200
    counts = [_n_roots(n_nodes, degree, s) for s in range(20)]

    observed = float(np.mean(counts))
    expected = expected_er_roots(n_nodes, degree)

    assert expected == pytest.approx(observed, rel=0.2), (
        f"ER{degree}: observed mean {observed:.1f} roots vs analytic {expected:.1f}"
    )
    # Sanity: the fraction must grow as the degree drops.
    assert 0.0 < observed / n_nodes < 1.0


def test_source_count_scales_with_n_nodes():
    """Number of sources grows with the graph (it is not a fixed constant)."""
    small = _n_roots(50, 2, seed=4)
    large = _n_roots(400, 2, seed=4)

    assert large > 4 * small * 0.5  # roughly linear, generous band


def test_graph_stats_in_meta_are_consistent():
    ds = sample_random_scm_dataset(RandomSCMConfig(n_nodes=80, degree=2, seed=8))
    stats = ds.meta["graph_stats"]
    assert stats["n_nodes"] == 80
    assert stats["n_sources"] == len(ds.source_labels)
    assert stats["n_inputs"] == len(ds.input_labels)
    assert stats["source_fraction"] == pytest.approx(stats["n_sources"] / 80)
    assert stats["expected_er_roots"] == pytest.approx(expected_er_roots(80, 2))


# --------------------------------------------------------------------------- #
# Label permutation (anti-leak)
# --------------------------------------------------------------------------- #

def test_permute_labels_breaks_index_monotonicity():
    """
    With permutation ON, a parent's numeric index must NOT be systematically
    smaller than its child's: otherwise the variable index leaks the topological
    order and a baseline could exploit it.
    """
    ds = sample_random_scm_dataset(
        RandomSCMConfig(n_nodes=60, degree=2, seed=21, permute_labels=True))

    violations = 0
    total = 0
    for spec in ds.specs:
        if not spec.name.startswith("X"):
            continue
        child_id = int(spec.name[1:])
        for parent in spec.parents:
            if not parent.startswith("X"):
                continue
            total += 1
            if int(parent[1:]) > child_id:
                violations += 1

    assert total > 0, "expected some X->X edges at degree 2"
    # A random permutation gives ~50% "backward" indices; require a clear signal.
    assert violations > 0.2 * total


def test_permute_labels_off_is_topologically_ordered():
    """Without permutation, X->X edges always go from a lower to a higher index."""
    ds = sample_random_scm_dataset(
        RandomSCMConfig(n_nodes=40, degree=2, seed=21, permute_labels=False))

    for spec in ds.specs:
        if not spec.name.startswith("X"):
            continue
        child_id = int(spec.name[1:])
        for parent in spec.parents:
            if parent.startswith("X"):
                assert int(parent[1:]) < child_id


def test_permutation_preserves_structure():
    """Permutation is a relabelling only: same edge count, still acyclic."""
    kwargs = dict(n_nodes=50, degree=2, seed=33)
    ds_on = sample_random_scm_dataset(RandomSCMConfig(permute_labels=True, **kwargs))
    ds_off = sample_random_scm_dataset(RandomSCMConfig(permute_labels=False, **kwargs))

    assert _n_edges(ds_on) == _n_edges(ds_off) == round(2 * 50)
    assert len(ds_on.scm._topo_order()) == 50
    assert len(ds_off.scm._topo_order()) == 50
    # Same in-degree multiset (identical graph up to relabelling).
    deg_on = sorted(len(p) for p in _parents(ds_on).values())
    deg_off = sorted(len(p) for p in _parents(ds_off).values())
    assert deg_on == deg_off


# --------------------------------------------------------------------------- #
# Acyclicity
# --------------------------------------------------------------------------- #

def test_acyclic_topo_order():
    ds = sample_random_scm_dataset(RandomSCMConfig(n_nodes=40, degree=4, seed=11))
    order = ds.scm._topo_order()
    assert len(order) == 40


# --------------------------------------------------------------------------- #
# Structural-equation family
# --------------------------------------------------------------------------- #

def test_linear_expressions_have_no_nonlinearities():
    ds = sample_random_scm_dataset(
        RandomSCMConfig(n_nodes=20, degree=2, seed=0, linearity="linear"))
    for spec in ds.specs:
        assert "**" not in spec.expr
        assert "sin(" not in spec.expr
        assert "tanh(" not in spec.expr


def test_nonlinear_expressions_contain_nonlinearity():
    ds = sample_random_scm_dataset(
        RandomSCMConfig(n_nodes=25, degree=3, seed=0, linearity="nonlinear"))
    joined = " ".join(spec.expr for spec in ds.specs)
    assert any(tok in joined for tok in ("**2", "**3", "sin(", "tanh("))


# --------------------------------------------------------------------------- #
# Metadata persistence
# --------------------------------------------------------------------------- #

def test_config_persisted_in_meta():
    cfg = RandomSCMConfig(n_nodes=15, degree=2, seed=99,
                          linearity="mixed", noise="nongaussian")
    ds = sample_random_scm_dataset(cfg)
    stored = ds.meta["random_scm_config"]
    assert stored["seed"] == 99
    assert stored["n_nodes"] == 15
    assert stored["linearity"] == "mixed"
    assert stored["noise"] == "nongaussian"
    # The removed knobs must not reappear in the schema.
    assert "n_sources" not in stored
    assert "s_x_ratio" not in stored
    assert "ensure_x_has_parent" not in stored


def test_removed_config_fields_raise():
    """Old configs must fail loudly rather than being silently ignored."""
    for bad in ({"n_sources": 2}, {"s_x_ratio": 0.3}, {"ensure_x_has_parent": True}):
        with pytest.raises(TypeError):
            RandomSCMConfig(n_nodes=10, degree=2, seed=0, **bad)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_invalid_config_raises():
    with pytest.raises(ValueError):
        sample_random_scm_dataset(RandomSCMConfig(n_nodes=1, degree=1, seed=0))
    with pytest.raises(ValueError):
        sample_random_scm_dataset(RandomSCMConfig(n_nodes=10, degree=0, seed=0))


def test_degree_saturating_the_slots_is_clamped():
    """A degree so high that m > C(n,2) must clamp, not crash - but then every
    non-first node has a parent, leaving exactly one root."""
    ds = sample_random_scm_dataset(RandomSCMConfig(n_nodes=5, degree=10, seed=0))
    assert _n_edges(ds) == 5 * 4 // 2
    assert len(ds.source_labels) == 1


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(shutil.which("dot") is None,
                    reason="Graphviz 'dot' binary not available")
def test_generate_ds_end_to_end_without_ate():
    cfg = RandomSCMConfig(n_nodes=12, degree=2, seed=42,
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
            compute_ate=False,
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

        # ATE ground truth must be skipped for sampled DAGs.
        assert not (save_dir / "ate_ground_truth.json").exists()

        # Mask shapes follow the emergent S/X counts.
        meta = json.loads((save_dir / "dataset_metadata.json").read_text())
        info = meta["variable_info"]
        assert info["n_source"] == len(ds.source_labels)
        assert info["n_input"] == len(ds.input_labels)
        assert info["n_source"] + info["n_input"] == 12
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

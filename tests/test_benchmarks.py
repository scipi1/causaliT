"""
Tests for the external structure-learning benchmarks (``causaliT.benchmarks``).

Three things are worth testing here, in decreasing order of danger:

1. **Orientation.**  The papers use ``W[parent, child]``, causaliT uses
   ``A[child, parent]``.  A missing transpose produces perfectly plausible
   numbers for the *reverse* graph, so the ground-truth adjacency is pushed
   through ``adjacency_to_blocks`` and must reproduce the true masks exactly.
2. **Scoring conventions.**  Threshold comparison (``>=``), zero diagonal, the
   CPDAG's ``0.5`` for unoriented edges, and the optional exogeneity constraint.
3. **Plumbing.**  The loader's column order (sources first), and the runner
   writing the same artefacts as the model evaluation, so ``eval_seed_sweep``
   and the notebooks work unchanged.

The two MLP methods are minutes-per-fit; they are covered by an opt-in slow test
(``CAUSALIT_RUN_SLOW_BENCHMARKS=1``) and by the shared registry tests.
"""

import json
import os
from os.path import exists, join

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from causaliT.benchmarks.base import (
    METHOD_REQUIREMENTS,
    available_methods,
    default_params,
    is_available,
    merge_params,
    method_names,
    resolve_method,
)
from causaliT.benchmarks.data import load_benchmark_data
from causaliT.benchmarks.postprocess import (
    adjacency_to_blocks,
    count_edges,
    cpdag_to_scores,
    forbid_edges_into_sources,
    is_dag,
    to_canonical_adjacency,
    to_edge_scores,
)
from causaliT.benchmarks.runner import (
    BENCHMARK_RUN_FILENAME,
    eval_name_for,
    run_benchmark_method,
    run_benchmarks,
    summarize_benchmarks,
)

RUN_SLOW = os.environ.get("CAUSALIT_RUN_SLOW_BENCHMARKS", "") not in ("", "0", "false")

# =============================================================================
# Shared ground truth: N = 5 variables, L_S = 2 sources, L_X = 3 intermediates
#
#   S1 -> X1 -> X3 <- X2 <- S2
#
# Column order is the canonical one: [S1, S2, X1, X2, X3].
# =============================================================================
L_S, L_X = 2, 3
N = L_S + L_X
EDGE_WEIGHT = 1.5

#: Paper orientation: W[i, j] != 0 means i -> j.
W_TRUE = np.zeros((N, N))
W_TRUE[0, 2] = EDGE_WEIGHT      # S1 -> X1
W_TRUE[1, 3] = EDGE_WEIGHT      # S2 -> X2
W_TRUE[2, 4] = EDGE_WEIGHT      # X1 -> X3
W_TRUE[3, 4] = EDGE_WEIGHT      # X2 -> X3

#: causaliT orientation, split into blocks: rows = children, cols = parents.
TRUE_CROSS = np.array([[1.0, 0.0],      # X1 <- S1
                       [0.0, 1.0],      # X2 <- S2
                       [0.0, 0.0]])     # X3 <- (no source)
TRUE_SELF = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0]])  # X3 <- X1, X2

SOURCE_LABELS = ["S1", "S2"]
INPUT_LABELS = ["X1", "X2", "X3"]


def simulate_linear_sem(n_samples: int = 600, seed: int = 0) -> np.ndarray:
    """
    Sample ``(n_samples, N)`` data from the linear-Gaussian SEM defined by ``W_TRUE``.

    Nodes are already in topological order, so one forward pass suffices.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, N))
    for j in range(N):
        X[:, j] = X @ W_TRUE[:, j] + rng.normal(size=n_samples)
    return X


# =============================================================================
# Fixtures: a minimal on-disk dataset + experiment folder
# =============================================================================

@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    """
    Create a dataset folder in the on-disk layout the loader/evaluation expect.

    Contains ``ds.npz`` (token tensors ``s``/``x`` with the value in channel 0),
    ``dataset_metadata.json`` (variable labels) and the two ground-truth mask
    CSVs that the DAG metrics are computed against.
    """
    data_root = tmp_path_factory.mktemp("data")
    dataset_name = "scm_bench"
    folder = data_root / dataset_name
    folder.mkdir()

    X = simulate_linear_sem(n_samples=600, seed=0)
    # Token layout: (n_samples, seq_len, n_features); feature 0 = value.
    s = X[:, :L_S, None]
    x = X[:, L_S:, None]
    np.savez(folder / "ds.npz", s=s, x=x)

    metadata = {
        "variable_info": {
            "source_labels": SOURCE_LABELS,
            "input_labels": INPUT_LABELS,
            "target_labels": [],
        },
        "causal_structure": {
            "edges": [["S1", "X1"], ["S2", "X2"], ["X1", "X3"], ["X2", "X3"]],
        },
    }
    (folder / "dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    pd.DataFrame(TRUE_CROSS.astype(int), index=INPUT_LABELS, columns=SOURCE_LABELS) \
        .to_csv(folder / "dec1_cross_att_mask.csv")
    pd.DataFrame(TRUE_SELF.astype(int), index=INPUT_LABELS, columns=INPUT_LABELS) \
        .to_csv(folder / "dec1_self_att_mask.csv")

    return str(data_root), dataset_name


@pytest.fixture
def experiment_dir(tmp_path, dataset_dir):
    """An experiment folder with the minimal config the runner reads."""
    data_root, dataset_name = dataset_dir
    exp = tmp_path / "bench_exp"
    exp.mkdir()
    # ``standardize: false`` is deliberate here (it is *true* in production): on
    # raw data from its own model class NOTEARS recovers this SEM exactly, which
    # turns the end-to-end assertions into a strict zero-SHD check instead of a
    # vague "some numbers were written".  See
    # ``test_standardization_removes_the_varsortability_shortcut``.
    config = {
        "data": {"dataset": dataset_name, "data_root": data_root,
                 "S_seq_len": L_S, "X_seq_len": L_X},
        "evaluation": {"dag_threshold": 0.5},
        "benchmark": {"methods": ["notears_linear"], "seeds": [0],
                      "standardize": False},
    }
    OmegaConf.save(OmegaConf.create(config), exp / "config_bench.yaml")
    return str(exp)


# =============================================================================
# Orientation - the failure mode that silently scores the reverse graph
# =============================================================================

class TestOrientation:
    """``W[parent, child]`` -> ``A[child, parent]`` must be a plain transpose."""

    def test_transpose(self):
        W = np.array([[0.0, 2.0], [0.0, 0.0]])   # 0 -> 1
        A = to_canonical_adjacency(W)
        assert A[1, 0] == 2.0                    # child 1 has parent 0
        assert A[0, 1] == 0.0

    def test_ground_truth_recovers_true_blocks(self):
        """Feeding the true W through the pipeline must return the true masks."""
        blocks = adjacency_to_blocks(W_TRUE, L_S=L_S, L_X=L_X, w_threshold=0.3)
        np.testing.assert_allclose(blocks["cross"], TRUE_CROSS)
        np.testing.assert_allclose(blocks["self"], TRUE_SELF)

    def test_reversed_graph_is_not_the_true_graph(self):
        """Guards the test above: the transposed input must NOT match."""
        blocks = adjacency_to_blocks(W_TRUE.T, L_S=L_S, L_X=L_X, w_threshold=0.3)
        assert not np.allclose(blocks["cross"], TRUE_CROSS)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            adjacency_to_blocks(np.zeros((4, 4)), L_S=L_S, L_X=L_X)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            to_canonical_adjacency(np.zeros((2, 3)))


# =============================================================================
# Scoring conventions
# =============================================================================

class TestEdgeScores:
    """``to_edge_scores`` maps weights to probabilities in [0, 1]."""

    def test_binary_mode_uses_inclusive_threshold(self):
        A = np.array([[0.0, 0.3], [0.29, 0.0]])
        scores = to_edge_scores(A, w_threshold=0.3, score_mode="binary")
        assert scores[0, 1] == 1.0    # exactly at the threshold -> kept
        assert scores[1, 0] == 0.0    # just below -> dropped

    def test_scaled_mode_normalises_by_the_peak(self):
        A = np.array([[0.0, 2.0], [1.0, 0.0]])
        scores = to_edge_scores(A, w_threshold=0.3, score_mode="scaled")
        assert scores[0, 1] == pytest.approx(1.0)
        assert scores[1, 0] == pytest.approx(0.5)

    def test_negative_weights_count_as_edges(self):
        scores = to_edge_scores(np.array([[0.0, -2.0], [0.0, 0.0]]))
        assert scores[0, 1] == 1.0

    def test_diagonal_is_zeroed(self):
        A = np.ones((3, 3))
        assert np.all(np.diag(to_edge_scores(A)) == 0.0)

    def test_is_binary_passes_values_through(self):
        A = np.array([[0.0, 0.5], [0.0, 0.0]])
        scores = to_edge_scores(A, w_threshold=0.9, score_mode="binary", is_binary=True)
        assert scores[0, 1] == 0.5     # untouched despite being below w_threshold

    def test_unknown_score_mode_raises(self):
        with pytest.raises(ValueError):
            to_edge_scores(np.zeros((2, 2)), score_mode="softmax")


class TestCpdagScores:
    """PC's CPDAG: 1.0 oriented, 0.5 unoriented."""

    def test_directed_and_undirected(self):
        directed = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        undirected = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        scores = cpdag_to_scores(directed, undirected)
        assert scores[0, 1] == 1.0
        assert scores[0, 2] == 0.5 and scores[2, 0] == 0.5
        assert scores[1, 2] == 0.0

    def test_directed_wins_over_undirected(self):
        directed = np.array([[0, 1], [0, 0]])
        undirected = np.array([[0, 1], [1, 0]])
        scores = cpdag_to_scores(directed, undirected)
        assert scores[0, 1] == 1.0

    def test_undirected_edges_are_present_at_the_default_threshold(self):
        """0.5 must survive the ``>=`` binarisation used by the metrics."""
        scores = cpdag_to_scores(np.zeros((2, 2)), np.array([[0, 1], [1, 0]]))
        assert count_edges(scores, threshold=0.5) == 2


class TestConstraintsAndDiagnostics:
    """Background knowledge and the reported sanity flags."""

    def test_forbid_into_sources_zeroes_source_rows(self):
        scores = np.ones((N, N))
        out = forbid_edges_into_sources(scores, L_S=L_S)
        assert np.all(out[:L_S, :] == 0.0)
        assert np.all(out[L_S:, :] == 1.0)

    def test_forbid_into_sources_is_off_by_default(self):
        W = np.zeros((N, N))
        W[2, 0] = 1.0                       # X1 -> S1, impossible but predicted
        blocks = adjacency_to_blocks(W, L_S=L_S, L_X=L_X)
        forbidden = adjacency_to_blocks(W, L_S=L_S, L_X=L_X, forbid_into_sources=True)
        # The offending edge lives outside the X-child blocks in both cases, so
        # the blocks agree; what matters is that the flag is accepted and pure.
        np.testing.assert_allclose(blocks["cross"], forbidden["cross"])

    def test_count_edges_and_is_dag(self):
        scores = to_edge_scores(to_canonical_adjacency(W_TRUE))
        assert count_edges(scores) == 4
        assert is_dag(scores) is True

    def test_cycle_is_detected(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]])   # 0 <-> 1
        assert is_dag(A) is False


# =============================================================================
# Registry
# =============================================================================

class TestRegistry:
    """Method registry: names, availability probes, paper defaults."""

    def test_expected_methods_are_registered(self):
        assert set(method_names()) == {
            "notears_linear", "notears_mlp", "dagma_linear", "dagma_mlp", "pc",
        }

    def test_unknown_method_raises(self):
        with pytest.raises(KeyError):
            resolve_method("golem")

    def test_available_methods_covers_every_name(self):
        assert set(available_methods()) == set(method_names())

    def test_notears_linear_is_always_available(self):
        """It only needs the vendored code + scipy, so it must never be optional."""
        assert METHOD_REQUIREMENTS["notears_linear"] is None
        assert is_available("notears_linear")

    @pytest.mark.parametrize("method", ["notears_mlp", "dagma_mlp"])
    def test_nonlinear_variants_use_the_paper_mlp_width(self, method):
        """Both papers report a single hidden layer of 10 units for every N."""
        if not is_available(method):
            pytest.skip(f"{method} unavailable")
        assert default_params(method)["hidden_units"] == 10

    def test_merge_params_overrides_defaults(self):
        merged = merge_params("notears_linear", {"lambda1": 0.5})
        assert merged["lambda1"] == 0.5
        assert merged["loss_type"] == default_params("notears_linear")["loss_type"]


# =============================================================================
# Data loading
# =============================================================================

class TestDataLoading:
    """The loader owns the sources-first column ordering."""

    def test_columns_are_sources_then_intermediates(self, dataset_dir):
        data_root, dataset_name = dataset_dir
        data = load_benchmark_data(data_root, dataset_name, standardize=False)
        assert data.L_S == L_S and data.L_X == L_X
        assert data.n_nodes == N
        assert data.labels == SOURCE_LABELS + INPUT_LABELS

    def test_labels_fall_back_without_metadata(self, dataset_dir):
        data_root, dataset_name = dataset_dir
        data = load_benchmark_data(data_root, dataset_name, metadata={})
        assert data.labels == ["S1", "S2", "X1", "X2", "X3"]

    def test_standardization_is_per_column(self, dataset_dir):
        data_root, dataset_name = dataset_dir
        data = load_benchmark_data(data_root, dataset_name, standardize=True)
        assert data.standardized
        np.testing.assert_allclose(data.X.mean(axis=0), np.zeros(N), atol=1e-10)
        np.testing.assert_allclose(data.X.std(axis=0), np.ones(N), atol=1e-10)

    def test_max_samples_caps_rows(self, dataset_dir):
        data_root, dataset_name = dataset_dir
        data = load_benchmark_data(data_root, dataset_name, max_samples=50)
        assert data.n_samples == 50

    def test_missing_dataset_raises(self, dataset_dir):
        data_root, _ = dataset_dir
        with pytest.raises(FileNotFoundError):
            load_benchmark_data(data_root, "does_not_exist")


# =============================================================================
# Methods
# =============================================================================

FAST_METHODS = ["notears_linear", "dagma_linear", "pc"]
SLOW_METHODS = ["notears_mlp", "dagma_mlp"]


def _fit(method: str, X: np.ndarray):
    if not is_available(method):
        pytest.skip(f"{method} unavailable ({METHOD_REQUIREMENTS.get(method)} missing)")
    return resolve_method(method)(X, seed=0)


class TestMethods:
    """Every method returns an (N, N) paper-orientation matrix."""

    @pytest.mark.parametrize("method", FAST_METHODS)
    def test_output_contract(self, method):
        X = simulate_linear_sem(n_samples=300, seed=1)
        result = _fit(method, X)
        assert result.W.shape == (N, N)
        assert np.all(np.isfinite(result.W))
        assert result.seconds >= 0.0
        assert "implementation" in result.extra

    @pytest.mark.parametrize("method", ["notears_linear", "dagma_linear"])
    def test_linear_methods_recover_the_chain(self, method):
        """
        On raw data from their own model class recovery must be exact.

        Note this is the *easy* regime: the simulated SEM is varsortable, i.e.
        marginal variance grows along the topological order, which the continuous
        methods exploit (Reisach et al., 2021).  See
        ``test_standardization_removes_the_varsortability_shortcut``.
        """
        X = simulate_linear_sem(n_samples=1000, seed=2)
        result = _fit(method, X)
        blocks = adjacency_to_blocks(result.W, L_S=L_S, L_X=L_X, w_threshold=0.3)
        np.testing.assert_allclose(blocks["cross"], TRUE_CROSS)
        np.testing.assert_allclose(blocks["self"], TRUE_SELF)

    def test_standardization_removes_the_varsortability_shortcut(self):
        """
        Standardised data is much harder for NOTEARS - by design of the protocol.

        The benchmark runner standardises by default (``benchmark.standardize``)
        precisely so the reported numbers do not credit the methods for reading
        the topological order off the marginal variances, which is an artefact of
        simulated SEMs rather than causal signal.  This test pins the behaviour so
        that nobody "fixes" a benchmark's weak score by silently dropping
        standardisation: exact recovery on raw data, degraded on standardised.
        """
        X = simulate_linear_sem(n_samples=1000, seed=5)
        Z = (X - X.mean(axis=0)) / X.std(axis=0)

        raw_blocks = adjacency_to_blocks(_fit("notears_linear", X).W, L_S=L_S, L_X=L_X)
        std_blocks = adjacency_to_blocks(_fit("notears_linear", Z).W, L_S=L_S, L_X=L_X)

        np.testing.assert_allclose(raw_blocks["cross"], TRUE_CROSS)
        assert not (
            np.allclose(std_blocks["cross"], TRUE_CROSS)
            and np.allclose(std_blocks["self"], TRUE_SELF)
        ), "standardised recovery was exact - varsortability may have leaked back in"

    def test_pc_returns_a_cpdag_with_the_right_skeleton(self):
        """PC identifies the skeleton; orientations may stay ambiguous (0.5)."""
        X = simulate_linear_sem(n_samples=1000, seed=3)
        result = _fit("pc", X)
        assert result.is_binary
        scores = to_edge_scores(to_canonical_adjacency(result.W), is_binary=True)
        skeleton = (scores > 0) | (scores.T > 0)
        true_full = np.zeros((N, N))
        true_full[L_S:, :L_S] = TRUE_CROSS
        true_full[L_S:, L_S:] = TRUE_SELF
        true_skeleton = (true_full > 0) | (true_full.T > 0)
        np.testing.assert_array_equal(skeleton, true_skeleton)

    @pytest.mark.skipif(not RUN_SLOW, reason="set CAUSALIT_RUN_SLOW_BENCHMARKS=1")
    @pytest.mark.parametrize("method", SLOW_METHODS)
    def test_nonlinear_methods_output_contract(self, method):
        X = simulate_linear_sem(n_samples=200, seed=4)
        result = _fit(method, X)
        assert result.W.shape == (N, N)
        assert np.all(np.isfinite(result.W))


# =============================================================================
# Runner: same artefacts as the model evaluation
# =============================================================================

class TestRunner:
    """``run_benchmark_method`` must emit the standard eval folder."""

    def test_writes_the_standard_artefacts(self, experiment_dir, dataset_dir):
        data_root, dataset_name = dataset_dir
        metrics = run_benchmark_method(
            experiment=experiment_dir,
            method="notears_linear",
            datadir_path=data_root,
            dataset_name=dataset_name,
            seeds=[0],
            standardize=False,
            verbose=False,
        )
        files = join(experiment_dir, "eval", eval_name_for("notears_linear"), "files")
        for name in ("dag_metrics.json", "learned_dag_edges.json",
                     "attention_labels.json", BENCHMARK_RUN_FILENAME):
            assert exists(join(files, name)), f"missing {name}"

        # Perfect recovery on its own model class -> zero SHD in both blocks.
        # Key names are the model evaluation's, which is the whole point of
        # routing benchmarks through ``write_dag_report``.
        assert metrics["standard_shd_cross"]["mean"] == 0
        assert metrics["standard_shd_self"]["mean"] == 0
        assert metrics["soft_hamming_cross"]["mean"] == 0.0
        assert metrics["mec_membership_rate"] == 1.0
        assert metrics["benchmark"]["method"] == "notears_linear"

    def test_seeds_become_folds(self, experiment_dir, dataset_dir):
        data_root, dataset_name = dataset_dir
        run_benchmark_method(
            experiment=experiment_dir,
            method="notears_linear",
            datadir_path=data_root,
            dataset_name=dataset_name,
            seeds=[0, 1],
            standardize=False,
            verbose=False,
        )
        path = join(experiment_dir, "eval", eval_name_for("notears_linear"),
                    "files", "learned_dag_edges.json")
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        per_fold = payload["blocks"]["cross"]["learned_per_fold"]
        assert set(per_fold) == {"seed_0", "seed_1"}

    def test_raw_record_allows_offline_rescoring(self, experiment_dir, dataset_dir):
        data_root, dataset_name = dataset_dir
        run_benchmark_method(
            experiment=experiment_dir,
            method="notears_linear",
            datadir_path=data_root,
            dataset_name=dataset_name,
            seeds=[0],
            standardize=False,
            verbose=False,
        )
        path = join(experiment_dir, "eval", eval_name_for("notears_linear"),
                    "files", BENCHMARK_RUN_FILENAME)
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
        W = np.array(record["per_seed"]["seed_0"]["W_paper_orientation"])
        assert W.shape == (N, N)
        assert record["L_S"] == L_S and record["L_X"] == L_X
        assert record["labels"] == SOURCE_LABELS + INPUT_LABELS
        assert record["standardized"] is False
        # Re-scoring the stored W offline reproduces the true blocks.
        blocks = adjacency_to_blocks(W, L_S=L_S, L_X=L_X,
                                     w_threshold=record["scoring"]["w_threshold"])
        np.testing.assert_allclose(blocks["cross"], TRUE_CROSS)

    def test_dimension_mismatch_raises(self, experiment_dir, dataset_dir):
        """A dataset whose variable count contradicts L_S/L_X must fail loudly."""
        data_root, dataset_name = dataset_dir
        with pytest.raises(ValueError):
            run_benchmark_method(
                experiment=experiment_dir,
                method="notears_linear",
                datadir_path=data_root,
                dataset_name=dataset_name,
                metadata={"variable_info": {"source_labels": ["S1"],
                                            "input_labels": ["X1", "X2"]}},
                seeds=[0],
                verbose=False,
            )

    def test_run_benchmarks_reads_the_config_section(self, experiment_dir):
        results = run_benchmarks(experiment_dir, verbose=False)
        assert set(results) == {"notears_linear"}
        assert "error" not in results["notears_linear"]

    def test_failed_method_does_not_abort_the_others(self, experiment_dir):
        results = run_benchmarks(
            experiment_dir,
            methods=["notears_linear"],
            overrides={"split": "nonexistent_split"},
            verbose=False,
        )
        assert "error" in results["notears_linear"]

    def test_unknown_method_in_config_raises(self, experiment_dir):
        with pytest.raises(ValueError):
            run_benchmarks(experiment_dir, methods=["nonexistent"], verbose=False)

    def test_summarize_produces_one_row_per_method(self):
        """The summary must read the artefacts' own key names (``standard_shd_*``)."""
        results = {
            "notears_linear": {
                "standard_shd_cross": {"mean": 0.0},
                "standard_shd_self": {"mean": 1.0},
                "soft_hamming_cross": {"mean": 0.1}, "soft_hamming_self": {"mean": 0.2},
                "mec_distance": {"mean": 0.25},
                "benchmark": {"seconds": {"seed_0": 1.0, "seed_1": 3.0}},
            },
            "dagma_mlp": {"error": "boom"},
        }
        rows = {row["method"]: row for row in summarize_benchmarks(results)}
        assert rows["notears_linear"]["shd_cross_mean"] == 0.0
        assert rows["notears_linear"]["shd_self_mean"] == 1.0
        assert rows["notears_linear"]["mec_distance_mean"] == 0.25
        assert rows["notears_linear"]["seconds_mean"] == pytest.approx(2.0)
        assert rows["dagma_mlp"]["error"] == "boom"

    def test_summary_columns_are_populated_from_a_real_run(
        self, experiment_dir, dataset_dir
    ):
        """Guards against the summary silently reading a key nobody writes."""
        results = run_benchmarks(experiment_dir, verbose=False)
        row = summarize_benchmarks(results)[0]
        assert row["shd_cross_mean"] == 0
        assert row["shd_self_mean"] == 0
        assert row["soft_hamming_cross_mean"] is not None
        assert row["seconds_mean"] is not None


# =============================================================================
# Sweep integration
# =============================================================================

class TestSweepIntegration:
    """A DAG sweep must be able to run the baselines like any trainer."""

    def test_benchmark_trainer_is_registered(self):
        from causaliT.euler_sweep.euler_sweep.opt_train_sweep import resolve_trainer

        fn, module, attr = resolve_trainer("benchmark")
        assert attr == "benchmark_function_for_sweep"
        assert module == "causaliT.euler_sweep.euler_sweep.cli"
        assert callable(fn)

    def test_sweep_wrapper_returns_one_row_per_method(self, experiment_dir, dataset_dir):
        """
        The wrapper reads the run folder's own config (staged by the sweep), so the
        ``config`` argument is only there to match the trainer signature.
        """
        from causaliT.euler_sweep.euler_sweep.cli import benchmark_function_for_sweep

        data_root, _ = dataset_dir
        df = benchmark_function_for_sweep(
            config=OmegaConf.create({}),  # type: ignore[arg-type]
            save_dir=experiment_dir,
            data_dir=data_root,
            cluster=False,
        )
        assert list(df["method"]) == ["notears_linear"]
        assert df["shd_cross_mean"].iloc[0] == 0

    def test_sweep_wrapper_persists_the_summary_csv(self, experiment_dir, dataset_dir):
        """
        The sweep DISCARDS the trainer's return value, so the summary must also
        land on disk - otherwise a run folder can only be summarised by walking
        every eval subfolder.
        """
        from causaliT.euler_sweep.euler_sweep.cli import benchmark_function_for_sweep

        data_root, _ = dataset_dir
        df = benchmark_function_for_sweep(
            config=OmegaConf.create({}),  # type: ignore[arg-type]
            save_dir=experiment_dir,
            data_dir=data_root,
            cluster=False,
        )

        csv_path = join(experiment_dir, "benchmark_summary.csv")
        assert exists(csv_path)
        pd.testing.assert_frame_equal(pd.read_csv(csv_path), df)



"""
Tests for the MAIN benchmark study plumbing: normalization guard, seed policy,
raw-output persistence, status files and the global collector.

Reuses the on-disk fixtures of ``test_benchmarks.py`` (5-node linear SEM,
fast ``var_sort``).  Run with ``--basetemp=.pytest_tmp`` on Windows.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from causaliT.benchmarks.base import (
    METHOD_DEFAULT_N_SEEDS,
    BenchmarkResult,
    default_seeds,
    method_names,
)
from causaliT.benchmarks.collect import collect
from causaliT.benchmarks.runner import (
    BENCHMARK_RUN_FILENAME,
    STATUS_FILENAME,
    check_normalization,
    dataset_normalization,
    eval_name_for,
    load_result_npz,
    run_benchmarks,
    save_result_npz,
)
from tests.test_benchmarks import dataset_dir, experiment_dir  # noqa: F401


def _files(exp: str, method: str = "var_sort") -> Path:
    return Path(exp) / "eval" / eval_name_for(method) / "files"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_seed_policy_covers_registry():
    assert set(METHOD_DEFAULT_N_SEEDS) == set(method_names())
    assert default_seeds("dagma_mlp") == [0, 1, 2]
    assert default_seeds("daguerreo") == [0, 1, 2]
    assert default_seeds("pc") == [0]


def test_seeds_omitted_uses_registry_default(experiment_dir):  # noqa: F811
    cfg_path = Path(experiment_dir) / "config_bench.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.benchmark.seeds = None
    OmegaConf.save(cfg, cfg_path)
    run_benchmarks(experiment_dir, verbose=False)
    assert _json(_files(experiment_dir) / BENCHMARK_RUN_FILENAME)["seeds"] == [0]
    assert _json(_files(experiment_dir) / STATUS_FILENAME)["status"] == "ok"


def test_check_normalization_raises_on_double_zscore():
    with pytest.raises(ValueError, match="standardize"):
        check_normalization("standardize_node", True)
    check_normalization("standardize_node", False)
    check_normalization(None, True)
    check_normalization("minmax", True)


def test_runner_fails_loudly_and_records_normalization(experiment_dir, dataset_dir):  # noqa: F811
    data_root, dataset = dataset_dir
    norm_file = Path(data_root) / dataset / "normalization.json"
    norm_file.write_text(json.dumps({"input": {"method": "standardize_node"}}),
                         encoding="utf-8")
    try:
        assert dataset_normalization(data_root, dataset) == "standardize_node"
        res = run_benchmarks(experiment_dir, overrides={"standardize": True}, verbose=False)
        assert "error" in res["var_sort"]
        assert _json(_files(experiment_dir) / STATUS_FILENAME)["status"] == "error"

        res = run_benchmarks(experiment_dir, verbose=False)   # standardize: false
        expected = {"at_generation": "standardize_node", "loader_standardize": False}
        assert res["var_sort"]["normalization"] == expected
        record = _json(_files(experiment_dir) / BENCHMARK_RUN_FILENAME)
        assert record["normalization"] == expected
    finally:
        norm_file.unlink()


def test_pc_rcit_runs_and_is_seed_reproducible():
    """PC with the nonlinear RCIT test (MAIN setting) is reproducible per seed."""
    from causaliT.benchmarks.methods.pc import fit
    from tests.test_benchmarks import simulate_nonlinear_sem

    X = simulate_nonlinear_sem(n_samples=300, seed=0)
    a = fit(X, indep_test="rcit", seed=0)
    b = fit(X, indep_test="rcit", seed=0)
    assert a.output_type == "cpdag"
    np.testing.assert_array_equal(a.W, b.W)
    assert a.params["indep_test"] == "rcit"


def test_direct_lingam_recovers_linear_nongaussian_dag():
    """Identifiable regime: linear SEM, Laplace noise, z-scored (no var shortcut)."""
    from causaliT.benchmarks.methods.direct_lingam import fit
    from tests.test_benchmarks import N, W_TRUE

    rng = np.random.default_rng(3)
    X = np.zeros((2000, N))
    for j in range(N):
        X[:, j] = X @ W_TRUE[:, j] + rng.laplace(size=2000)
    X = (X - X.mean(0)) / X.std(0)
    res = fit(X, seed=0)
    assert res.output_type == "weighted" and sorted(res.order) == list(range(N))
    np.testing.assert_array_equal(np.abs(res.W) >= 0.3, W_TRUE != 0)


def test_result_npz_roundtrip(tmp_path):
    samples = np.random.default_rng(0).integers(0, 2, size=(4, 3, 3))
    res = BenchmarkResult(W=np.zeros((3, 3)), seconds=1.5, order=[2, 0, 1],
                          graph_samples=samples)
    path = tmp_path / "r.npz"
    save_result_npz(str(path), res)
    back = load_result_npz(str(path))
    assert back.output_type == "samples"
    assert back.order == [2, 0, 1]
    np.testing.assert_allclose(back.edge_probs, res.edge_probs)
    assert back.seconds == pytest.approx(1.5)


def test_collector_rescoring_and_missing(tmp_path, experiment_dir, dataset_dir):  # noqa: F811
    """Fake one arm with one finished run; the other expected run is 'missing'."""
    data_root, dataset = dataset_dir
    run_benchmarks(experiment_dir, verbose=False)

    # Cluster naming: $SCRATCH/<exp_id>_<jobid>.  An older, empty resubmission
    # of the same arm must be ignored (else every run is reported twice).
    older = tmp_path / "MAIN" / "var_sort_ER1_nlin_gauss_zscore_100"
    older.mkdir(parents=True)
    arm = tmp_path / "MAIN" / "var_sort_ER1_nlin_gauss_zscore_200"
    group = arm / "groups" / "n_nodes_5"
    run = group / "sweeper" / "runs" / "combinations" / "arm_n_nodes_5_seed_0"
    shutil.copytree(Path(experiment_dir) / "eval", run / "eval")
    shutil.copytree(Path(data_root) / dataset, group / "datasets" / dataset)
    recipe = {"random_scm_config": {"n_nodes": 5, "degree": 1, "linearity": "nonlinear",
                                    "noise": "gaussian", "seed": 0},
              "generation": {"n_samples": 600, "normalize_method": "standardize_node"}}
    (group / "datasets" / dataset / "dag_recipe.json").write_text(
        json.dumps(recipe), encoding="utf-8")
    OmegaConf.save(OmegaConf.create({
        "group_axes": {"n_nodes": [5]}, "dag_seeds": [0, 1],
        "dag": {"degree": 1, "linearity": "nonlinear", "noise": "gaussian",
                "n_samples": 600, "normalize_method": "standardize_node"},
    }), arm / "dagsweep.yaml")
    OmegaConf.save(OmegaConf.create({"benchmark": {"methods": ["var_sort"]}}),
                   arm / "config_benchmark.yaml")
    shutil.copy(arm / "dagsweep.yaml", older / "dagsweep.yaml")
    shutil.copy(arm / "config_benchmark.yaml", older / "config_benchmark.yaml")

    df = collect(str(tmp_path / "MAIN"), out_path=str(tmp_path / "out.csv"))
    assert (tmp_path / "out.csv").exists()
    shd = df[(df.status == "ok") & (df.metric == "shd")]
    assert len(shd) == 1 and shd.value.iloc[0] == 0.0     # var_sort is exact here
    row = shd.iloc[0]
    assert row["degree"] == 1 and row["noise"] == "gaussian"
    assert row["normalization"] == "standardize_node"
    assert list(df[df.status == "missing"].dag_seed) == [1]
    assert set(df.arm) == {"var_sort_ER1_nlin_gauss_zscore"}
    assert set(df.job_id) == {200}

"""
Unit tests for the DAG-aggregation features of ``eval_seed_sweep``.

These tests build a synthetic seed-sweep tree on disk (no training,
no torch, no real experiments) and verify that:

1. The DAG metric flattening helper picks up the right scalars from
   ``dag_metrics.json`` (soft Hamming, standard SHD, MEC, zeroness,
   skeleton/v-structure recall/precision).

2. ``eval_seed_sweep`` aggregates DAG metrics across seeds, emitting
   ``dag_summary.{csv,json}`` with correct mean/std and per-seed values,
   and the new ``aggregate_dag.{csv,json}`` with per-edge mean/std/min/max
   plus the true mask.

3. Seeds without DAG artifacts (i.e. baselines) are tolerated and
   simply skipped from the DAG outputs without breaking the ATE/exp pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causaliT.evaluation.eval_funs.eval_seed_sweep import (
    _extract_dag_metrics_per_seed,
    _aggregate_learned_dag_across_seeds,
    eval_seed_sweep,
)


# ---------------------------------------------------------------------------
# Tree builders
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def _make_dag_metrics(sh_cross: float, sh_self: float,
                      shd_cross: int, shd_self: int,
                      mec_dist: float, in_mec: bool,
                      skel_recall: float, skel_prec: float,
                      v_recall: float, v_prec: float,
                      zc: float, zs: float) -> dict:
    """Synthetic dag_metrics.json payload mirroring eval_attention_scores."""
    return {
        "dataset": "synthetic_scm",
        "architecture": "SingleCausalForecaster",
        "soft_hamming_cross": {
            "best": sh_cross, "mean": sh_cross, "worst": sh_cross,
            "std": 0.0, "per_fold": {"k_0": sh_cross},
        },
        "soft_hamming_self": {
            "best": sh_self, "mean": sh_self, "worst": sh_self,
            "std": 0.0, "per_fold": {"k_0": sh_self},
        },
        "standard_shd_cross": {
            "best": shd_cross, "mean": float(shd_cross), "worst": shd_cross,
            "std": 0.0, "per_fold": {"k_0": shd_cross},
        },
        "standard_shd_self": {
            "best": shd_self, "mean": float(shd_self), "worst": shd_self,
            "std": 0.0, "per_fold": {"k_0": shd_self},
        },
        "zeroness_cross": {
            "mean_nonedge": 0.1, "max_nonedge": 0.2,
            "mean_edge": 0.1 + zc, "min_edge": 0.05 + zc,
            "contrast": zc,
        },
        "zeroness_self": {
            "mean_nonedge": 0.1, "max_nonedge": 0.2,
            "mean_edge": 0.1 + zs, "min_edge": 0.05 + zs,
            "contrast": zs,
        },
        "mec_distance": {
            "best": mec_dist, "mean": mec_dist, "worst": mec_dist,
            "std": 0.0,
            "per_fold": {
                "k_0": {
                    "mec_distance": mec_dist,
                    "in_mec": in_mec,
                    "skeleton_recall": skel_recall,
                    "skeleton_precision": skel_prec,
                    "v_structure_recall": v_recall,
                    "v_structure_precision": v_prec,
                }
            },
        },
        "mec_membership_rate": 1.0 if in_mec else 0.0,
        "n_true_v_structures": 2,
    }


def _make_learned_edges(seed_offset: float) -> dict:
    """Synthetic learned_dag_edges.json payload (1 fold, 2 blocks, 2x2)."""
    # cross block: rows = X targets, cols = S sources
    learned_cross = [
        [0.10 + seed_offset, 0.80 + seed_offset],
        [0.15 + seed_offset, 0.05 + seed_offset],
    ]
    true_cross = [[0, 1], [0, 0]]
    # self block: rows/cols = X targets
    learned_self = [
        [0.05 + seed_offset, 0.20 + seed_offset],
        [0.90 + seed_offset, 0.05 + seed_offset],
    ]
    true_self = [[0, 0], [1, 0]]

    return {
        "dataset": "synthetic_scm",
        "architecture": "SingleCausalForecaster",
        "blocks": {
            "dec_cross": {
                "att_key": "dec_cross",
                "mask_type": "dec_cross",
                "source": "attention",
                "n_rows": 2, "n_cols": 2,
                "row_labels": ["X1", "X2"],
                "col_labels": ["S1", "S2"],
                "true": true_cross,
                "learned_mean": learned_cross,
                "learned_std": [[0.0, 0.0], [0.0, 0.0]],
                "learned_per_fold": {"k_0": learned_cross},
            },
            "dec_self": {
                "att_key": "dec_self",
                "mask_type": "dec_self",
                "source": "attention",
                "n_rows": 2, "n_cols": 2,
                "row_labels": ["X1", "X2"],
                "col_labels": ["X1", "X2"],
                "true": true_self,
                "learned_mean": learned_self,
                "learned_std": [[0.0, 0.0], [0.0, 0.0]],
                "learned_per_fold": {"k_0": learned_self},
            },
        },
    }


def _make_seed_run(combinations_dir: Path, seed: int, *,
                   include_dag: bool, seed_offset: float) -> None:
    """Build a single seed run directory (ATE + optional DAG artifacts)."""
    run = combinations_dir / f"sweep_combo_seed_{seed}"
    eval_dir = run / "eval"
    # ATE artifact (always present)
    _write_json(
        eval_dir / "eval_ate_mc" / "files" / "ate_metrics_mc.json",
        {
            "per_intervention_variable": [
                {
                    "intervention": "S2=1.0",
                    "variable": "X1",
                    "true_ate": 0.5,
                    "model_ate_mean": 0.5 + 0.01 * seed,
                    "abs_error_mean": 0.01 * seed,
                },
            ]
        },
    )
    # kfold summary
    _write_json(
        run / "kfold_summary.json",
        {
            "statistics": {
                "test_x_mae": {"mean": 0.10 + 0.001 * seed},
                "val_x_mae":  {"mean": 0.09 + 0.001 * seed},
                "val_loss_x": {"mean": 0.05},
            }
        },
    )

    if include_dag:
        eval_att = eval_dir / "eval_attention_scores" / "files"
        # Vary metrics per seed so mean/std are non-trivial
        _write_json(
            eval_att / "dag_metrics.json",
            _make_dag_metrics(
                sh_cross=0.10 + 0.01 * seed,
                sh_self=0.20 + 0.01 * seed,
                shd_cross=1, shd_self=2,
                mec_dist=0.30 + 0.01 * seed,
                in_mec=(seed % 2 == 0),
                skel_recall=0.8, skel_prec=0.9,
                v_recall=0.5, v_prec=1.0,
                zc=0.50 + 0.01 * seed,
                zs=0.40 + 0.01 * seed,
            ),
        )
        _write_json(
            eval_att / "learned_dag_edges.json",
            _make_learned_edges(seed_offset=seed_offset),
        )


# ---------------------------------------------------------------------------
# Unit-level tests
# ---------------------------------------------------------------------------

def test_extract_dag_metrics_picks_canonical_columns():
    payload = _make_dag_metrics(
        sh_cross=0.1, sh_self=0.2,
        shd_cross=1, shd_self=2,
        mec_dist=0.3, in_mec=True,
        skel_recall=0.8, skel_prec=0.9,
        v_recall=0.5, v_prec=1.0,
        zc=0.55, zs=0.45,
    )
    out = _extract_dag_metrics_per_seed(payload)

    assert out["soft_hamming_cross"] == pytest.approx(0.1)
    assert out["soft_hamming_self"] == pytest.approx(0.2)
    assert out["soft_hamming_total"] == pytest.approx(0.3)
    assert out["standard_shd_cross"] == pytest.approx(1.0)
    assert out["standard_shd_self"] == pytest.approx(2.0)
    assert out["standard_shd_total"] == pytest.approx(3.0)
    assert out["zeroness_cross_contrast"] == pytest.approx(0.55)
    assert out["zeroness_self_contrast"] == pytest.approx(0.45)
    assert out["mec_distance"] == pytest.approx(0.3)
    assert out["mec_membership_rate"] == pytest.approx(1.0)
    assert out["skeleton_recall"] == pytest.approx(0.8)
    assert out["skeleton_precision"] == pytest.approx(0.9)
    assert out["v_structure_recall"] == pytest.approx(0.5)
    assert out["v_structure_precision"] == pytest.approx(1.0)


def test_extract_dag_metrics_tolerates_missing_fields():
    # Only soft_hamming_cross present → other fields silently absent
    payload = {
        "soft_hamming_cross": {"mean": 0.42},
    }
    out = _extract_dag_metrics_per_seed(payload)
    assert out == {"soft_hamming_cross": 0.42}


def test_aggregate_learned_dag_writes_expected_files(tmp_path: Path):
    out_dir = tmp_path / "files"
    out_dir.mkdir()

    seed_to_edges = {
        0: _make_learned_edges(seed_offset=0.00),
        1: _make_learned_edges(seed_offset=0.05),
        2: _make_learned_edges(seed_offset=-0.05),
    }
    _aggregate_learned_dag_across_seeds(
        seed_to_edges=seed_to_edges,
        eval_path_files=str(out_dir),
        sweep_experiment=str(tmp_path),
    )

    json_path = out_dir / "aggregate_dag.json"
    csv_path = out_dir / "aggregate_dag.csv"
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["n_seeds"] == 3
    assert payload["seeds"] == [0, 1, 2]
    assert "dec_cross" in payload["blocks"]
    assert "dec_self" in payload["blocks"]

    block = payload["blocks"]["dec_cross"]
    # Mean across seed_offset {0, 0.05, -0.05} should match the seed=0 matrix
    expected_mean_cross = np.array([[0.10, 0.80], [0.15, 0.05]])
    np.testing.assert_allclose(block["mean"], expected_mean_cross, atol=1e-9)
    # std should be > 0 since seeds differ
    assert np.max(np.array(block["std"])) > 0
    # min/max bracket the mean
    assert np.all(np.array(block["min"]) <= np.array(block["mean"]) + 1e-9)
    assert np.all(np.array(block["max"]) >= np.array(block["mean"]) - 1e-9)
    # true mask round-trip
    assert block["true"] == [[0, 1], [0, 0]]
    assert block["row_labels"] == ["X1", "X2"]
    assert block["col_labels"] == ["S1", "S2"]
    assert sorted(block["per_seed"].keys()) == ["0", "1", "2"]

    df_csv = pd.read_csv(csv_path)
    # 2 blocks × 2x2 edges = 8 rows
    assert len(df_csv) == 8
    assert set(df_csv["block"].unique()) == {"dec_cross", "dec_self"}
    needed_cols = {"block", "row", "col", "row_label", "col_label",
                   "true", "mean", "std", "min", "max", "n_seeds"}
    assert needed_cols.issubset(set(df_csv.columns))
    assert (df_csv["n_seeds"] == 3).all()


# ---------------------------------------------------------------------------
# End-to-end test on a synthetic sweep tree
# ---------------------------------------------------------------------------

def test_eval_seed_sweep_full_pipeline(tmp_path: Path):
    sweep = tmp_path / "sweep_synthetic"
    combos = sweep / "sweeper" / "runs" / "combinations"
    combos.mkdir(parents=True)

    # 3 causal-model seeds (with DAG outputs) + 1 baseline seed (no DAG)
    for s in (0, 1, 2):
        _make_seed_run(combos, seed=s, include_dag=True, seed_offset=0.01 * s)
    _make_seed_run(combos, seed=99, include_dag=False, seed_offset=0.0)

    # Run
    df_ate, df_exp = eval_seed_sweep(str(sweep))

    files_dir = sweep / "eval" / "eval_seed_sweep" / "files"
    assert (files_dir / "ate_summary.csv").exists()
    assert (files_dir / "experiment_summary.csv").exists()
    assert (files_dir / "dag_summary.csv").exists()
    assert (files_dir / "dag_summary.json").exists()
    assert (files_dir / "aggregate_dag.csv").exists()
    assert (files_dir / "aggregate_dag.json").exists()

    # ATE: 4 seeds contribute
    assert (df_ate["n_seeds"] == 4).all()

    # Experiment summary contains DAG mean/std columns now
    cols = df_exp.columns.tolist()
    for required in ["soft_hamming_cross_mean", "soft_hamming_cross_std",
                     "mec_distance_mean", "mec_membership_rate_mean",
                     "skeleton_recall_mean", "v_structure_precision_mean"]:
        assert required in cols, f"missing column {required}"

    # dag_summary.json: n_seeds=3 (the baseline seed didn't contribute)
    dag_json = json.loads((files_dir / "dag_summary.json").read_text())
    assert dag_json["n_seeds"] == 3
    assert dag_json["seeds"] == [0, 1, 2]
    sh_cross = dag_json["metrics"]["soft_hamming_cross"]
    assert sh_cross["n_seeds"] == 3
    # Seed values: 0.10, 0.11, 0.12 → mean ≈ 0.11
    assert sh_cross["mean"] == pytest.approx(0.11, abs=1e-9)
    # per-seed dict keyed by str(seed)
    assert set(sh_cross["per_seed"].keys()) == {"0", "1", "2"}
    assert sh_cross["per_seed"]["1"] == pytest.approx(0.11)

    # dag_summary.csv: long format, one row per metric
    df_dag_summary = pd.read_csv(files_dir / "dag_summary.csv")
    assert "metric" in df_dag_summary.columns
    assert "soft_hamming_cross" in df_dag_summary["metric"].values

    # aggregate_dag.json: 3 seeds, 2 blocks, with true masks
    agg = json.loads((files_dir / "aggregate_dag.json").read_text())
    assert agg["n_seeds"] == 3
    assert agg["seeds"] == [0, 1, 2]
    assert set(agg["blocks"].keys()) == {"dec_cross", "dec_self"}
    cross = agg["blocks"]["dec_cross"]
    assert cross["true"] == [[0, 1], [0, 0]]
    assert len(cross["mean"]) == 2 and len(cross["mean"][0]) == 2


def test_eval_seed_sweep_baseline_only_skips_dag(tmp_path: Path):
    """When no seed has DAG artifacts (pure baseline sweep), the
    DAG/aggregate-DAG outputs are simply not emitted; the ATE/exp pipeline
    still completes successfully."""
    sweep = tmp_path / "baseline_sweep"
    combos = sweep / "sweeper" / "runs" / "combinations"
    combos.mkdir(parents=True)

    for s in (0, 1):
        _make_seed_run(combos, seed=s, include_dag=False, seed_offset=0.0)

    df_ate, df_exp = eval_seed_sweep(str(sweep))

    files_dir = sweep / "eval" / "eval_seed_sweep" / "files"
    assert (files_dir / "ate_summary.csv").exists()
    assert (files_dir / "experiment_summary.csv").exists()
    assert not (files_dir / "dag_summary.csv").exists()
    assert not (files_dir / "dag_summary.json").exists()
    assert not (files_dir / "aggregate_dag.csv").exists()
    assert not (files_dir / "aggregate_dag.json").exists()

"""
Benchmark runner: fit an external structure learner and emit causaliT artefacts.

The runner is the benchmark counterpart of ``eval_attention_scores``.  Both end
in the *same* call to ``write_dag_report``, so a benchmark run produces byte-for-
byte compatible ``dag_metrics.json`` / ``learned_dag_edges.json`` files and can
be aggregated by the existing notebooks and by ``eval_seed_sweep`` without any
special-casing::

    model:     checkpoint -> attention -> query_dag_blocks -\
                                                             >-- write_dag_report
    benchmark: design matrix -> W -> adjacency_to_blocks ----/

**Seeds play the role of folds.**  causaliT trains ``k`` folds per experiment and
reports best/mean/worst across them.  The deterministic benchmarks have no folds,
so the runner refits each method once per seed and stores the results under
``seed_<i>`` keys, which ``write_dag_report`` consumes exactly like ``fold_<i>``.
Linear NOTEARS / DAGMA / PC are deterministic given the data, so their seeds
differ only if the data does; the MLP variants differ through initialisation.
That is intentional: the reported spread then reflects the true variability of
each method rather than a pretend one.

Outputs, under ``<experiment>/eval/eval_benchmark_<method>/``:

``files/dag_metrics.json``        metrics, identical schema to the models
``files/learned_dag_edges.json``  per-seed blocks + ground truth + labels
``files/attention_labels.json``   block descriptions
``files/benchmark_run.json``      raw ``W`` per seed, hyperparameters, timings,
                                  package versions - everything needed to
                                  re-score offline at another threshold
"""

import json
import platform
import sys
import time
from os import listdir, makedirs
from os.path import join
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from omegaconf import OmegaConf

from causaliT.benchmarks.base import (
    METHOD_DESCRIPTIONS,
    merge_params,
    method_names,
    resolve_method,
)
from causaliT.benchmarks.data import load_benchmark_data
from causaliT.benchmarks.postprocess import (
    adjacency_to_blocks,
    count_edges,
    is_dag,
    to_canonical_adjacency,
    to_edge_scores,
)
from causaliT.evaluation.eval_funs.helpers.datadir import resolve_datadir
from causaliT.evaluation.eval_funs.helpers.eval_dag_report import (
    resolve_dag_dims,
    write_dag_report,
)
from causaliT.evaluation.eval_funs.helpers.eval_dag_scores import make_json_serializable
from causaliT.evaluation.eval_funs.helpers.eval_utils import load_dataset_metadata

#: Filename of the raw-fit record inside the eval folder.
BENCHMARK_RUN_FILENAME = "benchmark_run.json"

#: Eval-name prefix; the method name is appended (``eval_benchmark_dagma_mlp``).
EVAL_NAME_PREFIX = "eval_benchmark"

#: Defaults for the ``benchmark`` config section.
DEFAULT_BENCHMARK_CONFIG: Dict[str, Any] = {
    "methods": ["notears_linear", "dagma_linear", "pc"],
    "seeds": [0],
    "split": "train",
    "standardize": True,
    "max_samples": None,
    "w_threshold": 0.3,
    "score_mode": "binary",
    "forbid_into_sources": False,
    "params": {},
}


def eval_name_for(method: str) -> str:
    """Eval folder name used for *method* (``eval_benchmark_notears_mlp``)."""
    return f"{EVAL_NAME_PREFIX}_{method}"


def _load_experiment_config(experiment: str):
    """Load the single ``config*.yaml`` of an experiment folder."""
    config_files = [
        f for f in listdir(experiment)
        if f.startswith("config") and f.endswith(".yaml")
    ]
    if not config_files:
        raise ValueError(f"No config file found in {experiment}")
    return OmegaConf.load(join(experiment, config_files[0]))


def _benchmark_settings(config, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resolve the ``benchmark`` config section over the defaults.

    Precedence: :data:`DEFAULT_BENCHMARK_CONFIG` < ``config.benchmark`` <
    *overrides* (CLI / sweep arguments).
    """
    settings = dict(DEFAULT_BENCHMARK_CONFIG)

    # The section may arrive as a DictConfig (config file), a plain dict (tests,
    # programmatic use) or be absent; anything else is not a settings mapping and
    # is ignored rather than crashing the run.
    section: Any = config.get("benchmark", {}) if config is not None else {}
    try:
        section = OmegaConf.to_container(section, resolve=True) if section else {}
    except Exception:  # noqa: BLE001 - already a plain container
        pass
    if not isinstance(section, dict):
        section = {}

    for key, value in section.items():
        settings[str(key)] = value
    for key, value in (overrides or {}).items():
        if value is not None:
            settings[key] = value

    if isinstance(settings["methods"], str):
        settings["methods"] = [settings["methods"]]
    if isinstance(settings["seeds"], int):
        settings["seeds"] = [settings["seeds"]]

    unknown = [m for m in settings["methods"] if m not in method_names()]
    if unknown:
        raise ValueError(
            f"Unknown benchmark method(s) {unknown}. Available: {', '.join(method_names())}."
        )
    return settings


def _package_versions(method: str) -> Dict[str, str]:
    """Version strings of the packages that determine a method's result."""
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import numpy as _np

        versions["numpy"] = _np.__version__
    except Exception:  # pragma: no cover
        pass
    if method.startswith("dagma"):
        try:
            import dagma

            versions["dagma"] = getattr(dagma, "__version__", "unknown")
        except Exception:  # pragma: no cover
            pass
    if method == "pc":
        try:
            import causallearn

            versions["causallearn"] = getattr(causallearn, "__version__", "unknown")
        except Exception:  # pragma: no cover
            pass
    if method.endswith("mlp"):
        try:
            import torch

            versions["torch"] = torch.__version__
        except Exception:  # pragma: no cover
            pass
    return versions


def run_benchmark_method(
    experiment: str,
    method: str,
    datadir_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    seeds: Sequence[int] = (0,),
    split: str = "train",
    standardize: bool = True,
    max_samples: Optional[int] = None,
    w_threshold: float = 0.3,
    score_mode: str = "binary",
    forbid_into_sources: bool = False,
    params: Optional[Dict[str, Any]] = None,
    config: Any = None,
    metadata: Optional[dict] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fit one benchmark method on one dataset and write the causaliT DAG report.

    Args:
        experiment: Experiment folder that receives ``eval/eval_benchmark_<method>/``.
            Its config supplies the dataset name and DAG dimensions unless those
            are passed explicitly.
        method: Registered method name (see ``base.method_names``).
        datadir_path: Data root; resolved from the config when omitted.
        dataset_name: Dataset folder; read from ``config.data.dataset`` when omitted.
        seeds: One fit per seed, stored as ``seed_<i>`` (the fold analogue).
        split: Which split to fit on (``train`` by default, matching training).
        standardize: Z-score the columns before fitting.
        max_samples: Optional cap on the number of rows.
        w_threshold: Magnitude threshold applied to ``|W|`` when scoring.
        score_mode: ``binary`` or ``scaled`` (see ``postprocess.to_edge_scores``).
        forbid_into_sources: Zero edges into source variables (background
            knowledge; off by default).
        params: Hyperparameter overrides merged over the method's paper defaults.
        config: Pre-loaded experiment config (loaded from *experiment* if None).
        metadata: Pre-loaded dataset metadata.
        verbose: Print progress.

    Returns:
        The ``dag_metrics`` dict from ``write_dag_report``, augmented with
        ``benchmark`` bookkeeping (method, seconds per seed, edge counts).

    Raises:
        ValueError: no dataset could be determined, or the method is unknown.
        ImportError: the method's optional dependency is missing.
    """
    if config is None:
        config = _load_experiment_config(experiment)

    if dataset_name is None:
        dataset_name = config.get("data", {}).get("dataset")
    if not dataset_name:
        raise ValueError("No dataset specified in experiment config.")

    if datadir_path is None:
        datadir_path = resolve_datadir(config=config, experiment=experiment)

    if metadata is None:
        metadata = load_dataset_metadata(datadir_path, dataset_name)
    if not metadata:
        raise ValueError(f"Dataset metadata not found for '{dataset_name}'.")

    L_S, L_X, dims_origin = resolve_dag_dims(
        config=config,
        metadata=metadata,
        datadir_path=datadir_path,
        dataset_name=dataset_name,
    )

    data = load_benchmark_data(
        datadir_path=datadir_path,
        dataset_name=dataset_name,
        split=split,
        standardize=standardize,
        max_samples=max_samples,
        metadata=metadata,
    )
    if verbose:
        print(f"  [benchmark] {data.summary()}")

    if data.n_nodes != L_S + L_X:
        raise ValueError(
            f"Dataset '{dataset_name}' provides {data.n_nodes} variables but the "
            f"DAG dimensions say L_S + L_X = {L_S} + {L_X} = {L_S + L_X} "
            f"(dims from {dims_origin})."
        )

    fit_fn = resolve_method(method)
    merged_params = merge_params(method, params)

    per_fold_blocks: Dict[str, Dict[str, np.ndarray]] = {}
    per_seed_record: Dict[str, Any] = {}

    for seed in seeds:
        key = f"seed_{int(seed)}"
        if verbose:
            print(f"  [benchmark] fitting {method} ({key}) ...")

        result = fit_fn(data.X, **{**merged_params, "seed": int(seed)})

        blocks = adjacency_to_blocks(
            result.W,
            L_S=L_S,
            L_X=L_X,
            w_threshold=w_threshold,
            score_mode=score_mode,
            is_binary=result.is_binary,
            forbid_into_sources=forbid_into_sources,
            verbose=verbose,
        )
        per_fold_blocks[key] = blocks

        scores = to_edge_scores(
            to_canonical_adjacency(result.W),
            w_threshold=w_threshold,
            score_mode=score_mode,
            is_binary=result.is_binary,
        )
        per_seed_record[key] = {
            "W_paper_orientation": np.asarray(result.W, dtype=float).tolist(),
            "seconds": result.seconds,
            "n_edges": count_edges(scores),
            "is_dag": is_dag(scores),
            "extra": result.extra,
        }
        if verbose:
            print(
                f"  [benchmark] {key}: {per_seed_record[key]['n_edges']} edges, "
                f"{result.seconds:.2f}s, blocks={sorted(blocks)}"
            )

    # ------------------------------------------------------------------
    # Shared report (same function the model evaluation calls)
    # ------------------------------------------------------------------
    eval_name = eval_name_for(method)
    dag_metrics = write_dag_report(
        experiment=experiment,
        per_fold_blocks=per_fold_blocks,
        datadir_path=datadir_path,
        dataset_name=dataset_name,
        architecture=f"benchmark:{method}",
        L_S=L_S,
        L_X=L_X,
        metadata=metadata,
        dag_threshold=config.get("evaluation", {}).get("dag_threshold", 0.5),
        dims_origin=dims_origin,
        source="benchmark",
        eval_name=eval_name,
        description=(
            f"Benchmark '{method}' ({METHOD_DESCRIPTIONS.get(method, '')}) fitted "
            f"on the {split} split with fixed paper hyperparameters; "
            f"{len(list(seeds))} seed(s)."
        ),
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Raw-fit record: allows re-scoring at any threshold without refitting
    # ------------------------------------------------------------------
    files_dir = join(experiment, "eval", eval_name, "files")
    makedirs(files_dir, exist_ok=True)
    record = {
        "method": method,
        "description": METHOD_DESCRIPTIONS.get(method, ""),
        "dataset": dataset_name,
        "datadir_path": datadir_path,
        "split": split,
        "n_samples": data.n_samples,
        "n_nodes": data.n_nodes,
        "L_S": L_S,
        "L_X": L_X,
        "labels": data.labels,
        "standardized": data.standardized,
        "n_dropped_samples": data.n_dropped,
        "max_samples": max_samples,
        "seeds": [int(s) for s in seeds],
        "params": merged_params,
        "scoring": {
            "w_threshold": w_threshold,
            "score_mode": score_mode,
            "forbid_into_sources": forbid_into_sources,
        },
        "orientation": "W[i, j] != 0 means i -> j (paper convention)",
        "versions": _package_versions(method),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_seed": per_seed_record,
    }
    with open(join(files_dir, BENCHMARK_RUN_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(make_json_serializable(record), fh, indent=2)

    dag_metrics["benchmark"] = {
        "method": method,
        "seeds": [int(s) for s in seeds],
        "seconds": {k: v["seconds"] for k, v in per_seed_record.items()},
        "n_edges": {k: v["n_edges"] for k, v in per_seed_record.items()},
        "params": merged_params,
    }
    if verbose:
        print(f"  [benchmark] wrote {join(files_dir, BENCHMARK_RUN_FILENAME)}")
    return dag_metrics


def run_benchmarks(
    experiment: str,
    methods: Optional[Sequence[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    datadir_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run every configured benchmark method on an experiment's dataset.

    Reads the ``benchmark`` section of the experiment config (see
    :data:`DEFAULT_BENCHMARK_CONFIG`) and runs one method after the other.  A
    method that raises - typically a missing optional dependency - is reported
    and skipped so that one bad install does not lose the other results.

    Args:
        experiment: Experiment folder.
        methods: Explicit method list, overriding the config.
        overrides: Other setting overrides (``seeds``, ``w_threshold``, ...).
        datadir_path: Explicit data root, for datasets that do not live in the
            default location (a DAG sweep passes its group-local ``datasets/``).
        verbose: Print progress.

    Returns:
        ``{method: dag_metrics}``; failed methods map to ``{"error": str}``.
    """
    config = _load_experiment_config(experiment)
    settings = _benchmark_settings(config, {**(overrides or {}), "methods": methods})

    results: Dict[str, Dict[str, Any]] = {}
    for method in settings["methods"]:
        if verbose:
            print(f"\n[benchmark] === {method} ===")
        try:
            results[method] = run_benchmark_method(
                experiment=experiment,
                method=method,
                datadir_path=datadir_path,
                seeds=settings["seeds"],
                split=settings["split"],
                standardize=settings["standardize"],
                max_samples=settings["max_samples"],
                w_threshold=settings["w_threshold"],
                score_mode=settings["score_mode"],
                forbid_into_sources=settings["forbid_into_sources"],
                params=(settings["params"] or {}).get(method, {}),
                config=config,
                verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001 - one method must not kill the rest
            print(f"[benchmark] {method} FAILED: {exc}")
            results[method] = {"error": str(exc)}
    return results


def summarize_benchmarks(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten ``run_benchmarks`` output into rows for a DataFrame / CSV.

    One row per method with the headline metrics (mean SHD and soft Hamming per
    block, MEC distance, fit time), mirroring the columns the model sweeps report.
    The SHD key in ``dag_metrics.json`` is ``standard_shd_<block>`` - the same name
    the model evaluation writes - and is exposed here under the shorter column
    ``shd_<block>_mean`` used by the comparison tables.
    """
    rows: List[Dict[str, Any]] = []
    for method, metrics in results.items():
        if "error" in metrics:
            rows.append({"method": method, "error": metrics["error"]})
            continue
        row: Dict[str, Any] = {"method": method}
        for block in ("cross", "self"):
            soft = metrics.get(f"soft_hamming_{block}") or {}
            shd = metrics.get(f"standard_shd_{block}") or {}
            row[f"soft_hamming_{block}_mean"] = soft.get("mean")
            row[f"shd_{block}_mean"] = shd.get("mean")
        mec = metrics.get("mec_distance") or {}
        row["mec_distance_mean"] = mec.get("mean") if isinstance(mec, dict) else None
        row["mec_membership_rate"] = metrics.get("mec_membership_rate")
        seconds = (metrics.get("benchmark") or {}).get("seconds") or {}
        row["seconds_mean"] = float(np.mean(list(seconds.values()))) if seconds else None
        rows.append(row)
    return rows


__all__ = [
    "BENCHMARK_RUN_FILENAME",
    "DEFAULT_BENCHMARK_CONFIG",
    "EVAL_NAME_PREFIX",
    "eval_name_for",
    "run_benchmark_method",
    "run_benchmarks",
    "summarize_benchmarks",
]

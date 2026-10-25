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
Linear DAGMA / PC / the order-based methods are (nearly) deterministic given
the data; the MLP and learned-order variants differ through initialisation.
That is intentional: the reported spread then reflects the true variability of
each method rather than a pretend one.

Outputs, under ``<experiment>/eval/eval_benchmark_<method>/``:

``files/dag_metrics.json``        metrics, identical schema to the models,
                                  plus ``structure`` (full-graph suite from
                                  ``metrics.py``) and ``data_diagnostics``
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
    METHOD_REQUIREMENTS,
    default_seeds,
    merge_params,
    method_names,
    resolve_method,
)
from causaliT.benchmarks.data import load_benchmark_data
from causaliT.benchmarks.metrics import (
    aggregate,
    compute_structure_metrics,
    data_diagnostics,
)
from causaliT.benchmarks.postprocess import (
    adjacency_to_blocks,
    count_edges,
    is_dag,
    to_canonical_adjacency,
    to_edge_scores,
)
from causaliT.evaluation.eval_funs.helpers.datadir import resolve_datadir
from causaliT.evaluation.eval_funs.helpers.eval_dag_report import (
    DAG_METRICS_FILENAME,
    resolve_dag_dims,
    write_dag_report,
)
from causaliT.evaluation.eval_funs.helpers.eval_dag_scores import make_json_serializable
from causaliT.evaluation.eval_funs.helpers.eval_utils import load_dataset_metadata

#: Filename of the raw-fit record inside the eval folder.
BENCHMARK_RUN_FILENAME = "benchmark_run.json"

#: Eval-name prefix; the method name is appended (``eval_benchmark_dagma_mlp``).
EVAL_NAME_PREFIX = "eval_benchmark"

#: Full-graph metrics (``dag_metrics["structure"]["mean"]``) copied into the
#: summary rows as ``<key>_mean`` columns.
SUMMARY_STRUCTURE_KEYS = (
    "shd", "shd_normalized", "f1", "skeleton_f1", "sid", "sid_normalized",
    "order_divergence", "order_accuracy", "auprc", "expected_shd",
)

#: Defaults for the ``benchmark`` config section.
DEFAULT_BENCHMARK_CONFIG: Dict[str, Any] = {
    "methods": ["dagma_linear", "pc", "score"],
    # None -> per-method default from base.METHOD_DEFAULT_N_SEEDS.
    "seeds": None,
    "split": "train",
    "standardize": True,
    "max_samples": None,
    "w_threshold": 0.3,
    "score_mode": "binary",
    "forbid_into_sources": False,
    "params": {},
}


#: Raw per-seed output file, re-scored by the global collector (collect.py).
RESULT_NPZ_TEMPLATE = "benchmark_result_{key}.npz"

#: Status record written for every attempted method (ok / unavailable / error).
STATUS_FILENAME = "benchmark_status.json"

#: Normalization methods applied PER NODE at generation time.  Loader-side
#: z-scoring on top of them is forbidden (sweep protocol: one normalization).
PER_NODE_NORMALIZATIONS = ("standardize_node", "minmax_node")


def dataset_normalization(datadir_path: str, dataset_name: str) -> Optional[str]:
    """
    Normalization applied at generation time, or None if unknown / disabled.

    Reads ``normalization.json`` (``input.method``) and falls back to
    ``dag_recipe.json`` (``generation.normalize_method``).
    """
    folder = join(str(datadir_path), str(dataset_name))
    try:
        with open(join(folder, "normalization.json"), encoding="utf-8") as fh:
            norm = json.load(fh)
        for section in norm.values():
            if isinstance(section, dict) and section.get("method"):
                return str(section["method"])
    except (OSError, ValueError):
        pass
    try:
        with open(join(folder, "dag_recipe.json"), encoding="utf-8") as fh:
            recipe = json.load(fh)
        gen = recipe.get("generation", {}) or {}
        if gen.get("normalize") is False:
            return None
        method = gen.get("normalize_method")
        return None if method is None else str(method)
    except (OSError, ValueError):
        return None


def check_normalization(at_generation: Optional[str], standardize: bool) -> None:
    """Fail loudly when loader z-scoring is stacked on per-node normalization."""
    if standardize and at_generation in PER_NODE_NORMALIZATIONS:
        raise ValueError(
            f"benchmark.standardize=true combined with a dataset normalized at "
            f"generation ('{at_generation}'). Normalization must be applied once, "
            f"at generation: set benchmark.standardize: false."
        )


def save_result_npz(path: str, result) -> None:
    """Store the raw ``BenchmarkResult`` fields needed for offline re-scoring."""
    arrays = {
        "W": np.asarray(result.W, dtype=float),
        "output_type": np.array(result.output_type),
        "seconds": np.array(float(result.seconds)),
    }
    if result.order is not None:
        arrays["order"] = np.asarray(result.order, dtype=np.int64)
    if result.edge_probs is not None:
        arrays["edge_probs"] = np.asarray(result.edge_probs, dtype=float)
    if result.graph_samples is not None:
        arrays["graph_samples"] = (np.asarray(result.graph_samples) != 0).astype(np.int8)
    np.savez_compressed(path, **arrays)


def load_result_npz(path: str):
    """Rebuild a ``BenchmarkResult`` from :func:`save_result_npz` output."""
    from causaliT.benchmarks.base import BenchmarkResult

    with np.load(path, allow_pickle=False) as z:
        return BenchmarkResult(
            W=z["W"],
            seconds=float(z["seconds"]),
            output_type=str(z["output_type"]),
            order=z["order"].tolist() if "order" in z else None,
            edge_probs=z["edge_probs"] if "edge_probs" in z else None,
            graph_samples=z["graph_samples"] if "graph_samples" in z else None,
        )


def write_status(experiment: str, method: str, status: str, **info) -> None:
    """Write ``benchmark_status.json`` for one method of one run."""
    files_dir = join(experiment, "eval", eval_name_for(method), "files")
    makedirs(files_dir, exist_ok=True)
    payload = {"method": method, "status": status,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    with open(join(files_dir, STATUS_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(make_json_serializable(payload), fh, indent=2)


def eval_name_for(method: str) -> str:
    """Eval folder name used for *method* (``eval_benchmark_dagma_mlp``)."""
    return f"{EVAL_NAME_PREFIX}_{method}"


def _true_dag_paper(datadir_path: str, dataset_name: str) -> Optional[np.ndarray]:
    """Full binary true DAG in paper orientation, or None if unavailable."""
    from causaliT.evaluation.eval_funs.helpers.eval_utils import _load_full_true_dag

    canonical = _load_full_true_dag(datadir_path, dataset_name)  # [child, parent]
    if canonical is None:
        return None
    return (np.asarray(canonical).T != 0).astype(np.int8)


def _predicted_graph(result, w_threshold: float, L_S: int, forbid_into_sources: bool):
    """
    Binary prediction and edge scores (paper orientation) for ``metrics.py``.

    Weighted outputs are thresholded at ``|W| >= w_threshold`` (same rule as
    ``postprocess.to_edge_scores``); CPDAGs keep undirected edges in both
    directions; DAG and sample outputs are already binary.
    """
    W = np.asarray(result.W, dtype=float)
    if result.output_type == "weighted":
        pred = np.abs(W) >= float(w_threshold)
        scores = np.abs(W)
    else:
        pred = W != 0
        scores = result.edge_probs if result.edge_probs is not None else np.abs(W)
    pred = pred.astype(np.int8)
    np.fill_diagonal(pred, 0)
    if forbid_into_sources:
        pred[:, :L_S] = 0
    return pred, np.asarray(scores, dtype=float)


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
    # The method's own package (registry) plus the ones that shape its result.
    packages = {METHOD_REQUIREMENTS.get(method), "torch", "sklearn", "gadjid"}
    for package in sorted(p for p in packages if p):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - optional package, version is best effort
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

    # Normalization protocol: applied once, at generation (fail loudly otherwise).
    norm_at_generation = dataset_normalization(datadir_path, dataset_name)
    check_normalization(norm_at_generation, standardize)

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
    normalization = {
        "at_generation": norm_at_generation,
        "loader_standardize": bool(standardize),
    }

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
    per_seed_structure: Dict[str, Dict[str, Any]] = {}
    true_paper = _true_dag_paper(datadir_path, dataset_name)
    files_dir = join(experiment, "eval", eval_name_for(method), "files")
    makedirs(files_dir, exist_ok=True)

    for seed in seeds:
        key = f"seed_{int(seed)}"
        if verbose:
            print(f"  [benchmark] fitting {method} ({key}) ...")

        result = fit_fn(data.X, **{**merged_params, "seed": int(seed)})
        # Raw output first: survives even if scoring/reporting fails later.
        save_result_npz(join(files_dir, RESULT_NPZ_TEMPLATE.format(key=key)), result)

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
            "output_type": result.output_type,
            "order": result.order,
            "edge_probs": None if result.edge_probs is None
            else np.asarray(result.edge_probs).tolist(),
            "extra": result.extra,
        }
        if true_paper is not None and true_paper.shape == np.shape(result.W):
            pred, edge_scores = _predicted_graph(result, w_threshold, L_S, forbid_into_sources)
            per_seed_structure[key] = compute_structure_metrics(
                true_paper,
                pred,
                scores=edge_scores,
                probabilistic=result.output_type == "samples",
                order=result.order,
                graph_samples=result.graph_samples,
            )
            per_seed_structure[key]["seconds"] = result.seconds
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
        "normalization": normalization,
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

    # Full-graph metric suite (metrics.py), added next to the block metrics.
    # Data diagnostics use the RAW data: standardising hides varsortability.
    dag_metrics["normalization"] = normalization
    if not per_seed_structure:
        with open(join(files_dir, DAG_METRICS_FILENAME), "w", encoding="utf-8") as fh:
            json.dump(make_json_serializable(dag_metrics), fh, indent=2)
    if per_seed_structure:
        dag_metrics["structure"] = {
            "per_seed": per_seed_structure,
            **aggregate(per_seed_structure),
        }
        raw = data if not data.standardized else load_benchmark_data(
            datadir_path=datadir_path, dataset_name=dataset_name, split=split,
            standardize=False, max_samples=max_samples, metadata=metadata,
        )
        dag_metrics["data_diagnostics"] = data_diagnostics(raw.X, true_paper)
        with open(join(files_dir, DAG_METRICS_FILENAME), "w", encoding="utf-8") as fh:
            json.dump(make_json_serializable(dag_metrics), fh, indent=2)

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
        # Seed policy: explicit list wins, else the registry default.
        seeds = settings["seeds"] if settings["seeds"] is not None else default_seeds(method)
        # max_samples: scalar (all methods) or {method: cap}.
        cap = settings["max_samples"]
        if isinstance(cap, dict):
            cap = cap.get(method)
        write_status(experiment, method, "started", seeds=list(seeds), max_samples=cap)
        t0 = time.time()
        try:
            results[method] = run_benchmark_method(
                experiment=experiment,
                method=method,
                datadir_path=datadir_path,
                seeds=seeds,
                split=settings["split"],
                standardize=settings["standardize"],
                max_samples=cap,
                w_threshold=settings["w_threshold"],
                score_mode=settings["score_mode"],
                forbid_into_sources=settings["forbid_into_sources"],
                params=(settings["params"] or {}).get(method, {}),
                config=config,
                verbose=verbose,
            )
            write_status(experiment, method, "ok", seeds=list(seeds), max_samples=cap,
                         seconds_total=time.time() - t0)
        except ImportError as exc:
            print(f"[benchmark] {method} UNAVAILABLE: {exc}")
            results[method] = {"error": str(exc)}
            write_status(experiment, method, "unavailable", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - one method must not kill the rest
            print(f"[benchmark] {method} FAILED: {exc}")
            results[method] = {"error": str(exc)}
            write_status(experiment, method, "error", error=str(exc),
                         seconds_total=time.time() - t0)
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
        structure_mean = (metrics.get("structure") or {}).get("mean") or {}
        for key in SUMMARY_STRUCTURE_KEYS:
            row[f"{key}_mean"] = structure_mean.get(key)
        seconds = (metrics.get("benchmark") or {}).get("seconds") or {}
        row["seconds_mean"] = float(np.mean(list(seconds.values()))) if seconds else None
        rows.append(row)
    return rows


__all__ = [
    "BENCHMARK_RUN_FILENAME",
    "DEFAULT_BENCHMARK_CONFIG",
    "EVAL_NAME_PREFIX",
    "RESULT_NPZ_TEMPLATE",
    "STATUS_FILENAME",
    "check_normalization",
    "dataset_normalization",
    "eval_name_for",
    "load_result_npz",
    "save_result_npz",
    "run_benchmark_method",
    "run_benchmarks",
    "summarize_benchmarks",
]

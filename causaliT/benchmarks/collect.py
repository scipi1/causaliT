"""
Global results collector for benchmark sweeps.

Walks benchmark arms (folders holding a ``dagsweep.yaml``), reloads the raw
per-seed outputs (``benchmark_result_seed_<i>.npz``), re-scores them with
``metrics.compute_structure_metrics`` and writes ONE long table::

    arm, method, graph_type, degree, linearity, noise, normalization,
    n_nodes, n_samples, dag_seed, init_seed, status, metric, value, ...

Every number is computed here, by the same code, so model and baseline rows
are comparable by construction.  Expected runs (``group_axes.n_nodes`` x
``dag_seeds`` x ``benchmark.methods``) that left no trace, or only a
``started`` status, are reported as ``missing`` (typically SLURM walltime or
OOM) with null values; they are never dropped silently.

Layout (see ``euler_sweep.opt_train_sweep``)::

    <arm>/groups/<group>/datasets/<dataset>/   true DAG csv + dag_recipe.json
    <arm>/groups/<group>/sweeper/runs/combinations/<run>/eval/
        eval_benchmark_<method>/files/{benchmark_run.json,
            benchmark_status.json, benchmark_result_seed_<i>.npz}
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from causaliT.benchmarks.metrics import compute_structure_metrics
from causaliT.benchmarks.runner import (
    BENCHMARK_RUN_FILENAME,
    EVAL_NAME_PREFIX,
    STATUS_FILENAME,
    _predicted_graph,
    _true_dag_paper,
    load_result_npz,
)

#: Columns describing the grid cell of a row.
GRID_COLUMNS = ("graph_type", "degree", "linearity", "noise", "normalization",
                "n_nodes", "n_samples", "dag_seed", "resample_seed")

NPZ_PREFIX = "benchmark_result_seed_"

#: Scratch folder ``<arm>_<slurm job id>`` written by the cluster script.
_JOB_SUFFIX = re.compile(r"^(.*\D)_(\d+)$")


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _grid_from_dag_block(dag: Dict[str, Any]) -> Dict[str, Any]:
    """Grid settings from a ``dag:`` block (or a recipe's merged fields)."""
    norm = dag.get("normalize_method") if dag.get("normalize", True) is not False else None
    return {
        "graph_type": str(dag.get("graph_type", "er")).upper(),
        "degree": dag.get("degree"),
        "linearity": dag.get("linearity"),
        "noise": dag.get("noise"),
        "normalization": norm,
        "n_nodes": dag.get("n_nodes"),
        "n_samples": dag.get("n_samples"),
        "dag_seed": dag.get("seed"),
        "resample_seed": None,
    }


def _grid_from_recipe(recipe: Optional[dict]) -> Dict[str, Any]:
    """Grid settings of one dataset from its ``dag_recipe.json``."""
    if not recipe:
        return {}
    # Generation first, SCM config last: both carry a ``seed`` and the DAG seed
    # (random_scm_config.seed) must win over the sampling seed.
    merged = {**(recipe.get("generation") or {}), **(recipe.get("random_scm_config") or {})}
    grid = _grid_from_dag_block(merged)
    # Same SCM, new noise draw (stability study); None outside resampling.
    grid["resample_seed"] = recipe.get("resample_seed")
    return grid


def _score_files(files_dir: Path, datasets_dir: Path, record: dict) -> List[Dict[str, Any]]:
    """Re-score every saved seed of one (run, method); one dict per seed."""
    dataset = record.get("dataset")
    true_paper = _true_dag_paper(str(datasets_dir), dataset) if dataset else None
    scoring = record.get("scoring") or {}
    out = []
    for npz in sorted(files_dir.glob(f"{NPZ_PREFIX}*.npz")):
        result = load_result_npz(str(npz))
        metrics: Dict[str, Any] = {}
        status = "ok"
        if true_paper is None or true_paper.shape != np.shape(result.W):
            status = "no_ground_truth"
        else:
            pred, scores = _predicted_graph(
                result, float(scoring.get("w_threshold", 0.3)),
                int(record.get("L_S", 0)),
                bool(scoring.get("forbid_into_sources", False)),
            )
            metrics = compute_structure_metrics(
                true_paper, pred, scores=scores,
                probabilistic=result.output_type == "samples",
                order=result.order, graph_samples=result.graph_samples,
            )
        metrics["seconds"] = result.seconds
        out.append({"init_seed": int(npz.stem[len(NPZ_PREFIX):]), "status": status,
                    "output_type": result.output_type, "metrics": metrics})
    return out


def _rows_for_method(arm: str, files_dir: Path) -> List[Dict[str, Any]]:
    """Long rows for one ``eval_benchmark_<method>/files`` folder."""
    method = files_dir.parent.name[len(EVAL_NAME_PREFIX) + 1:]
    run_dir = files_dir.parent.parent.parent      # files -> eval_x -> eval -> run
    group_dir = run_dir.parent.parent.parent.parent  # combinations/runs/sweeper/group
    datasets_dir = group_dir / "datasets"

    record = _read_json(files_dir / BENCHMARK_RUN_FILENAME) or {}
    status_rec = _read_json(files_dir / STATUS_FILENAME) or {}
    recipe = _read_json(datasets_dir / str(record.get("dataset")) / "dag_recipe.json")
    grid = _grid_from_recipe(recipe)
    norm = record.get("normalization") or {}
    base = {
        "arm": arm, "method": method, "run": run_dir.name,
        **{k: grid.get(k) for k in GRID_COLUMNS},
        "loader_standardize": norm.get("loader_standardize"),
        "w_threshold": (record.get("scoring") or {}).get("w_threshold"),
        "max_samples": record.get("max_samples", status_rec.get("max_samples")),
        "versions": json.dumps(record.get("versions", {}), sort_keys=True),
    }
    if norm.get("at_generation") is not None:
        base["normalization"] = norm["at_generation"]

    seeds = _score_files(files_dir, datasets_dir, record) if record else []
    if not seeds:
        status = status_rec.get("status", "missing")
        if status in ("started", "ok"):
            status = "missing"  # killed mid-run (walltime / OOM) or outputs lost
        return [{**base, "init_seed": None, "status": status, "output_type": None,
                 "error": status_rec.get("error"), "metric": None, "value": None}]

    rows = []
    for s in seeds:
        for metric, value in s["metrics"].items():
            # Non-scalar or textual entries (e.g. sid_note) are not metrics;
            # is_dag / sid=NaN already carry that information.
            if isinstance(value, (list, dict, tuple, str)):
                continue
            rows.append({**base, "init_seed": s["init_seed"], "status": s["status"],
                         "output_type": s["output_type"], "error": None,
                         "metric": metric,
                         "value": None if value is None else float(value)})
    return rows


def split_arm_folder(name: str):
    """
    ``<arm>_<jobid>`` -> (arm, jobid); a name without a numeric suffix -> (name, None).

    The cluster script stages every submission into ``$SCRATCH/<exp_id>_<jobid>``.
    Arm names never end in digits (they end in ``_zscore``), so the split is safe.
    """
    match = _JOB_SUFFIX.match(name)
    if match:
        return match.group(1), int(match.group(2))
    return name, None


def collect_arm(arm_dir: str) -> List[Dict[str, Any]]:
    """Long-format rows for one arm (a folder with ``dagsweep.yaml``)."""
    arm_path = Path(arm_dir)
    arm, job_id = split_arm_folder(arm_path.name)
    spec = OmegaConf.to_container(OmegaConf.load(arm_path / "dagsweep.yaml"), resolve=True)
    configs = sorted(arm_path.glob("config*.yaml"))
    cfg_raw = OmegaConf.load(configs[0]) if configs else None
    # resolve=False: stochastic arms use ``seeds: ${training.seed}``, which has
    # no target outside a staged run; _expected_init_seeds reads model_seeds
    # from the spec first, so the interpolation is never needed.
    cfg = OmegaConf.to_container(cfg_raw, resolve=False) if cfg_raw is not None else {}
    methods = list((cfg.get("benchmark") or {}).get("methods") or [])

    rows: List[Dict[str, Any]] = []
    pattern = f"groups/*/sweeper/runs/combinations/*/eval/{EVAL_NAME_PREFIX}_*/files"
    for files_dir in sorted(arm_path.glob(pattern)):
        rows.extend(_rows_for_method(arm, files_dir))

    # Expected-but-absent runs: no folder at all (job never started / lost).
    # The expectation comes from the SWEEP's own plan (build_groups + run_plan),
    # so multi-axis arms (degree, scm bundles) and resample/model seeds are all
    # covered; rows carry each group's grid (bUNDLE dag overrides included).
    dag_block = spec.get("dag") or {}
    method_seeds = _expected_init_seeds(spec, cfg, methods)
    seen = {(r["n_nodes"], r["dag_seed"], r.get("resample_seed"),
             r.get("init_seed"), r["method"]) for r in rows}
    for grid, dag_seed, resample_seed in _expected_datasets(spec):
        for method in methods:
            for init_seed in method_seeds[method]:
                if (grid.get("n_nodes"), dag_seed, resample_seed,
                        init_seed, method) in seen:
                    continue
                rows.append({**grid, "arm": arm, "method": method, "run": None,
                             "dag_seed": dag_seed, "resample_seed": resample_seed,
                             "init_seed": init_seed, "status": "missing",
                             "output_type": None, "error": None,
                             "metric": None, "value": None})
    for row in rows:
        row["job_id"] = job_id
    return rows


def _expected_datasets(spec: Dict[str, Any]) -> List[Tuple[Dict[str, Any], int, Optional[int]]]:
    """
    ``(grid, dag_seed, resample_seed)`` for every dataset the sweep should
    produce, expanded with the sweep's own ``build_groups`` / ``run_plan``.

    Falls back to the flat ``n_nodes x dag_seeds x resample_seeds`` product for
    legacy specs the current sweep code cannot expand.
    """
    try:
        from causaliT.euler_sweep.euler_sweep.opt_train_sweep import (
            build_groups,
            run_plan,
        )
        groups = build_groups(spec)
    except Exception:
        groups = None

    if groups is None:
        dag_block = spec.get("dag") or {}
        template = _grid_from_dag_block(dag_block)
        n_nodes_list = (spec.get("group_axes") or {}).get("n_nodes") or [dag_block.get("n_nodes")]
        dag_seeds = spec.get("dag_seeds") or spec.get("seeds") or []
        resample_seeds = spec.get("resample_seeds") or [None]
        return [({**template, "n_nodes": n}, d, r)
                for n in n_nodes_list for d in dag_seeds for r in resample_seeds]

    dag_block = spec.get("dag") or {}
    out: List[Tuple[Dict[str, Any], int, Optional[int]]] = []
    for group in groups:
        # Grid cell = spec dag block + plain axes (n_nodes, degree, ...)
        # + the bundles' dag fields (linearity, noise, ...).
        axes = {k: v for k, v in group["axes"].items()
                if k not in group.get("bundle_axes", ())}
        merged = {**dag_block, **axes, **(group.get("dag_overrides") or {})}
        grid = _grid_from_dag_block(merged)
        datasets = {(d, r) for d, r, _m, _k, _s in run_plan(group)}
        out.extend((grid, d, r) for d, r in sorted(
            datasets, key=lambda t: (t[0], -1 if t[1] is None else t[1])))
    return out


def _expected_init_seeds(spec: Dict[str, Any], cfg: Dict[str, Any],
                         methods: List[str]) -> Dict[str, List[int]]:
    """
    Init seeds each (dataset, method) pair should have produced.

    With ``model_seeds`` every model seed is its own run task, so the init
    seeds ARE the model seeds.  Otherwise the init seeds of a run folder are
    the config's ``benchmark.seeds`` (registry default when omitted).
    """
    from causaliT.benchmarks.base import default_seeds

    model_seeds = spec.get("model_seeds")
    cfg_seeds = ((cfg.get("benchmark") or {}).get("seeds"))
    out: Dict[str, List[int]] = {}
    for method in methods:
        if model_seeds:
            out[method] = [int(s) for s in model_seeds]
        elif isinstance(cfg_seeds, int):
            out[method] = [cfg_seeds]
        elif cfg_seeds:
            out[method] = [int(s) for s in cfg_seeds]
        else:
            out[method] = default_seeds(method)
    return out


def find_arms(root: str) -> List[str]:
    """Arm folders under *root* (or *root* itself) holding a ``dagsweep.yaml``."""
    root_path = Path(root)
    if (root_path / "dagsweep.yaml").exists():
        return [str(root_path)]
    return sorted(str(p.parent) for p in root_path.glob("*/dagsweep.yaml"))


def collect(root: str, out_path: Optional[str] = None) -> pd.DataFrame:
    """
    Collect every arm under *root* into one long DataFrame.

    Args:
        root: Study folder (e.g. ``experiments/8_BENCHMARKS/MAIN``) or one arm.
        out_path: Optional output. ``.parquet`` needs pyarrow or fastparquet;
            when neither is installed a ``.csv`` is written instead.

    Returns:
        The long table (one row per arm x run x init seed x metric).
    """
    # A resubmitted arm leaves several <arm>_<jobid> folders: keep the newest
    # job only, otherwise its runs would be counted twice.
    newest: Dict[str, Any] = {}
    for arm_dir in find_arms(root):
        arm, job_id = split_arm_folder(Path(arm_dir).name)
        if arm not in newest or (job_id or -1) > (newest[arm][1] or -1):
            newest[arm] = (arm_dir, job_id)
        else:
            print(f"[collect] skipping older submission {arm_dir}")
    rows: List[Dict[str, Any]] = []
    for arm_dir, _ in newest.values():
        rows.extend(collect_arm(arm_dir))
    df = pd.DataFrame(rows)
    if out_path:
        if str(out_path).endswith(".parquet"):
            try:
                df.to_parquet(out_path, index=False)
            except ImportError:
                csv_path = str(out_path)[:-len(".parquet")] + ".csv"
                print(f"[collect] no parquet engine installed; writing {csv_path}")
                df.to_csv(csv_path, index=False)
        else:
            df.to_csv(out_path, index=False)
    return df


__all__ = ["collect", "collect_arm", "find_arms", "split_arm_folder"]


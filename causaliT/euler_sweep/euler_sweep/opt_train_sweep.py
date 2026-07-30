"""
Grouped DAG sweep: optimize once per DAG *size*, then train every seed.

Motivation
----------
A naive sweep over ``n_nodes x seed`` with an Optuna study per cell explodes:
``4 sizes x 10 seeds x 10 trials = 400`` runs.  Hyper-parameters, however, are a
property of the *problem scale*, not of the particular DAG draw.  So we group:

* a **group** = one point on the group axes (e.g. ``n_nodes=50``);
* each group runs **one** Optuna study on a dedicated *optimisation DAG*
  (``opt_seed``, disjoint from the evaluation seeds - no HP leakage);
* the resulting ``best_trial.yaml`` is reused by **all** seeds of that group.

Cost becomes ``4 studies (40 trials) + 40 runs`` instead of 400 runs.

Layout::

    experiments/<exp_id>/
    ├── config_atsel.yaml            # base config
    ├── optuna_atsel.yaml            # Optuna search-space settings
    ├── dagsweep.yaml                # this module's spec
    └── groups/n_nodes_50/
        ├── config_atsel.yaml        # staged: data.dataset -> opt DAG
        ├── optuna_atsel.yaml
        ├── best_trial.yaml          # <- reused by every seed
        ├── datasets/                # <- data_dir for optuna + training
        │   ├── random_..._s1000/    # optimisation DAG
        │   └── random_..._s0/ ...   # evaluation DAGs (ds.npz pruned after use)
        └── sweeper/runs/combinations/<exp_id>_n_nodes_50_seed_0/

Both phases accept **any** trainer from :data:`TRAINER_REGISTRY`, and they are
configured independently: a cheap protocol can drive the search while the seed
sweep runs the full ``adaptive_trainer``.
"""

from __future__ import annotations

import glob
import logging
from functools import partial
from os.path import basename, exists, join
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from causaliT.euler_sweep.euler_sweep.dag_provider import (
    build_dag_config,
    dag_dataset_name,
    materialized_dag,
    split_dag_block,
)
from causaliT.euler_sweep.euler_sweep.sweeper import run_single_combination

logger = logging.getLogger(__name__)

DAGSWEEP_FILENAME_GLOB = "dagsweep*.yaml"
DEFAULT_OPT_SEED = 1000


def _as_dict(node: Any) -> Dict[str, Any]:
    """Resolve an OmegaConf node (or plain mapping) to a ``{str: Any}`` dict."""
    if node is None:
        return {}
    container = OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node
    if not isinstance(container, dict):
        return {}
    return {str(k): v for k, v in container.items()}


# =============================================================================
# Spec loading / validation
# =============================================================================

def load_dagsweep_spec(exp_dir: str) -> Any:
    """Load ``dagsweep*.yaml`` from an experiment folder."""
    matches = sorted(glob.glob(join(exp_dir, DAGSWEEP_FILENAME_GLOB)))
    if not matches:
        raise FileNotFoundError(
            f"No {DAGSWEEP_FILENAME_GLOB} found in {exp_dir}. "
            "A DAG sweep needs a spec with at least 'group_axes' and 'seeds'."
        )
    spec: Any = OmegaConf.load(matches[0])

    for key in ("group_axes", "seeds"):
        if key not in spec:
            raise ValueError(f"{basename(matches[0])} must define '{key}'.")
    if not spec.get("seeds"):
        raise ValueError("'seeds' must list at least one evaluation seed.")

    opt_seed = int((spec.get("optuna", {}) or {}).get("opt_seed", DEFAULT_OPT_SEED))
    overlap = [s for s in spec["seeds"] if int(s) == opt_seed]
    if overlap:
        # Optimising and evaluating on the same DAG draw leaks hyper-parameters
        # into the reported numbers - refuse rather than silently bias the paper.
        raise ValueError(
            f"optuna.opt_seed={opt_seed} also appears in 'seeds'. Use a "
            "dedicated optimisation seed so tuning cannot leak into results."
        )
    return spec


def build_groups(spec: Any) -> List[Dict[str, Any]]:
    """
    Expand ``group_axes`` into groups, each carrying the full seed list.

    A group is one Cartesian-product point of the group axes; seeds are *not*
    part of that product - they are members inside each group.
    """
    import itertools

    axes: Dict[str, List[Any]] = {k: list(v) for k, v in _as_dict(spec["group_axes"]).items()}
    seeds = [int(s) for s in spec["seeds"]]

    names = list(axes.keys())
    groups: List[Dict[str, Any]] = []
    for combo in itertools.product(*(axes[n] for n in names)):
        values = dict(zip(names, combo))
        groups.append({
            "axes": values,
            "name": "_".join(f"{k}_{v}" for k, v in values.items()),
            "seeds": seeds,
        })
    return groups


# =============================================================================
# Config staging
# =============================================================================

def _find_base_files(exp_dir: str) -> Tuple[str, Optional[str]]:
    """Locate the base ``config*.yaml`` and optional ``optuna*.yaml``."""
    configs = sorted(
        p for p in glob.glob(join(exp_dir, "config*.yaml"))
        if "dagsweep" not in basename(p)
    )
    if not configs:
        raise FileNotFoundError(f"No config*.yaml found in {exp_dir}")
    optunas = sorted(glob.glob(join(exp_dir, "optuna*.yaml")))
    return configs[0], (optunas[0] if optunas else None)


def _set_dotted(config: Any, dotted: str, value: Any) -> None:
    """Set ``a.b.c`` on a config, creating intermediate nodes as needed."""
    OmegaConf.update(config, dotted, value, merge=True)


def _metadata_lookup(metadata: Dict[str, Any], dotted: str) -> Any:
    node: Any = metadata
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def apply_dataset_derived(config: Any, spec: Any,
                          data_root: str, dataset_name: str) -> None:
    """
    Write dataset-derived sizes (e.g. variable counts) into the config.

    Because every group trains on a differently-sized DAG, any config field that
    must track the number of variables has to be refreshed per group.  Rather
    than guessing field names, the mapping is explicit in ``dagsweep.yaml``::

        dataset_derived:
          experiment.n_source: variable_info.source_labels
          experiment.n_input:  variable_info.input_labels

    Left-hand side = dotted config path; right-hand side = dotted path into
    ``dataset_metadata.json`` whose **length** is written.  A plain (non-list)
    metadata value is written as-is.
    """
    mapping = spec.get("dataset_derived", None)
    if not mapping:
        return

    from causaliT.evaluation.eval_funs.helpers.eval_utils import load_dataset_metadata

    metadata = load_dataset_metadata(data_root, dataset_name)
    if not metadata:
        logger.warning("No dataset_metadata.json for %s; skipping dataset_derived",
                       dataset_name)
        return

    for cfg_path, meta_path in _as_dict(mapping).items():
        value = _metadata_lookup(metadata, str(meta_path))
        if value is None:
            logger.warning("dataset_derived: '%s' not in metadata of %s",
                           meta_path, dataset_name)
            continue
        _set_dotted(config, str(cfg_path),
                    len(value) if isinstance(value, (list, tuple)) else value)


def stage_group_config(base_config_path: str, base_optuna_path: Optional[str],
                       group_dir: Path, group: Dict[str, Any], spec: Any,
                       dataset_name: str, datasets_dir: str) -> Any:
    """
    Materialize a group's config directory (as ``OptunaStudy`` expects it).

    Interpolations are deliberately **not** resolved: downstream ``update_config``
    and per-seed overrides must still be able to flow through ``${...}`` refs.
    """
    group_dir.mkdir(parents=True, exist_ok=True)

    config: Any = OmegaConf.load(base_config_path)

    # Group axes are exposed under `experiment.*` so a config can interpolate
    # them (e.g. `${experiment.n_nodes}`) exactly like in a normal sweep.
    for key, value in group["axes"].items():
        _set_dotted(config, f"experiment.{key}", value)

    _set_dotted(config, "data.dataset", dataset_name)
    # Point the whole stack at the group-local dataset store, so nothing is
    # written to the shared `data/` folder.
    _set_dotted(config, "data.data_root", datasets_dir)

    apply_dataset_derived(config, spec, datasets_dir, dataset_name)

    OmegaConf.save(config, group_dir / basename(base_config_path))
    if base_optuna_path is not None:
        OmegaConf.save(OmegaConf.load(base_optuna_path),
                       group_dir / basename(base_optuna_path))
    return config


# =============================================================================
# Optuna phase (once per group)
# =============================================================================

def load_best_params(group_dir: Path) -> Dict[str, Any]:
    """
    Read the flat ``{dotted_param: value}`` mapping from ``best_trial.yaml``.

    Returns an empty dict when the study has not produced a summary yet.
    """
    path = group_dir / "best_trial.yaml"
    if not path.exists():
        return {}
    summary: Any = OmegaConf.load(path)
    return _as_dict(summary.get("params", summary.get("best_params", {})))


def optimize_group(group_dir: Path, datasets_dir: str, spec: Any,
                   cluster: bool, force: bool = False) -> Dict[str, Any]:
    """
    Run (or reuse) the group's single Optuna study and return its best params.

    Resumable by construction: an existing ``best_trial.yaml`` short-circuits the
    whole phase, so an interrupted sweep never re-tunes a finished group.
    """
    if not force:
        cached = load_best_params(group_dir)
        if cached:
            logger.info("  Reusing existing best_trial.yaml (%d params)", len(cached))
            return cached

    from causaliT.euler_optuna.euler_optuna.cli import (
        get_metrics_for_optuna,
        sample_params_for_optuna,
    )
    from causaliT.euler_optuna.euler_optuna.optuna_opt import OptunaStudy

    opt_cfg = spec.get("optuna", {}) or {}
    base_config = OmegaConf.load(_find_base_files(str(group_dir))[0])
    train_fn = resolve_trainer(str(opt_cfg.get("trainer", "standard")))[0]

    study = OptunaStudy(
        exp_dir=Path(group_dir),
        data_dir=Path(datasets_dir),
        cluster=cluster,
        study_name=f"dagsweep_{group_dir.name}",
        manifest_tag=f"dagsweep_{group_dir.name}",
        sample_params_fn=partial(sample_params_for_optuna, config=base_config),
        train_fn=train_fn,
        get_metrics_fn=get_metrics_for_optuna,
        optimization_metric=str(opt_cfg.get("metric", "val_loss")),
        optimization_direction=str(opt_cfg.get("direction", "minimize")),
    )

    try:
        study.create()
    except Exception:
        # Study already exists (e.g. a previous partial run) -> continue it.
        logger.info("  Study exists; resuming remaining trials")
    study.resume()
    study.summary()  # writes best_trial.yaml into group_dir

    return load_best_params(group_dir)


# =============================================================================
# Trainer registry
# =============================================================================

def resolve_trainer(name: str) -> Tuple[Callable, str, str]:
    """
    Map a trainer name to ``(callable, module, attribute)``.

    The module/attribute strings are what SLURM workers need to re-import the
    trainer, so they are returned alongside the callable.
    """
    from causaliT.euler_sweep.euler_sweep import cli as sweep_cli

    registry = {
        "standard": "train_function_for_sweep",
        "staged": "staged_train_function_for_sweep",
        "anm": "anm_train_function_for_sweep",
        "adaptive": "adaptive_train_function_for_sweep",
    }
    if name not in registry:
        raise ValueError(
            f"Unknown trainer '{name}'. Available: {sorted(registry)}."
        )
    attr = registry[name]
    return getattr(sweep_cli, attr), "causaliT.euler_sweep.euler_sweep.cli", attr


# =============================================================================
# Orchestration
# =============================================================================

def run_dag_sweep(exp_dir: str, cluster: bool = False, keep_data: bool = False,
                  skip_optuna: bool = False, force_optuna: bool = False,
                  dry_run: bool = False) -> Dict[str, Any]:
    """
    Execute the full grouped DAG sweep.

    Per group: sample the optimisation DAG -> one Optuna study -> prune arrays.
    Per seed:  sample the evaluation DAG -> train with the group's best params
    -> post-training evaluations (inside the trainer) -> prune arrays.

    Args:
        exp_dir: Experiment folder containing ``config*.yaml`` and ``dagsweep*.yaml``.
        cluster: Passed through to the trainers (worker counts, etc.).
        keep_data: Keep every ``ds.npz`` instead of pruning (debugging; costly).
        skip_optuna: Reuse existing ``best_trial.yaml`` files, never tune.
        force_optuna: Re-run studies even when a summary already exists.
        dry_run: Print the plan and exit without generating or training anything.

    Returns:
        A summary dict with the per-group/per-seed outcome.
    """
    spec = load_dagsweep_spec(exp_dir)
    groups = build_groups(spec)
    base_config_path, base_optuna_path = _find_base_files(exp_dir)

    dag_block: Dict[str, Any] = _as_dict(spec.get("dag", {}))
    _, gen_kwargs = split_dag_block(dag_block)

    opt_cfg = spec.get("optuna", {}) or {}
    opt_seed = int(opt_cfg.get("opt_seed", DEFAULT_OPT_SEED))
    optuna_enabled = bool(opt_cfg.get("enabled", True)) and not skip_optuna

    train_name = str((spec.get("training", {}) or {}).get("trainer", "standard"))
    train_fn = resolve_trainer(train_name)[0]

    delete_dataset = not (keep_data or not spec.get("delete_dataset", True))

    exp_id = basename(str(exp_dir).rstrip("/\\"))
    n_runs = sum(len(g["seeds"]) for g in groups)
    logger.info("DAG sweep '%s': %d group(s), %d run(s), trainer=%s, optuna=%s",
                exp_id, len(groups), n_runs, train_name, optuna_enabled)

    if dry_run:
        for group in groups:
            print(f"[group] {group['name']}  seeds={group['seeds']}"
                  f"  opt_seed={opt_seed if optuna_enabled else '-'}")
        return {"groups": [g["name"] for g in groups], "n_runs": n_runs,
                "dry_run": True}

    results: Dict[str, Any] = {"experiment": exp_id, "groups": {}}

    for group in groups:
        group_dir = Path(exp_dir) / "groups" / group["name"]
        datasets_dir = str(group_dir / "datasets")
        Path(datasets_dir).mkdir(parents=True, exist_ok=True)
        logger.info("=" * 60)
        logger.info("Group %s", group["name"])

        group_result: Dict[str, Any] = {"seeds": {}, "best_params": {}}

        # ---- Phase 1: optimise once, on a dedicated DAG ---------------------
        best_params: Dict[str, Any] = {}
        if optuna_enabled:
            opt_dag_cfg = build_dag_config(dag_block, seed=opt_seed, **group["axes"])
            with materialized_dag(opt_dag_cfg, datasets_dir, gen_kwargs,
                                  delete_dataset=delete_dataset) as opt_dataset:
                stage_group_config(base_config_path, base_optuna_path, group_dir,
                                   group, spec, opt_dataset, datasets_dir)
                best_params = optimize_group(group_dir, datasets_dir, spec,
                                             cluster, force=force_optuna)
        else:
            best_params = load_best_params(group_dir)
            if best_params:
                logger.info("  Using %d cached best param(s)", len(best_params))

        group_result["best_params"] = best_params

        # ---- Phase 2: train every seed, reusing those params ---------------
        for seed in group["seeds"]:
            dag_cfg = build_dag_config(dag_block, seed=seed, **group["axes"])
            run_name = f"{exp_id}_{group['name']}_seed_{seed}"
            save_dir = group_dir / "sweeper" / "runs" / "combinations" / run_name

            logger.info("-" * 60)
            logger.info("Run %s", run_name)

            try:
                with materialized_dag(dag_cfg, datasets_dir, gen_kwargs,
                                  delete_dataset=delete_dataset) as dataset_name:
                    config: Any = OmegaConf.load(base_config_path)

                    for key, value in group["axes"].items():
                        _set_dotted(config, f"experiment.{key}", value)
                    _set_dotted(config, "data.dataset", dataset_name)
                    _set_dotted(config, "data.data_root", datasets_dir)
                    # The *training* seed is tied to the DAG seed so a run is
                    # fully identified by (group, seed).
                    _set_dotted(config, "training.seed", int(seed))

                    # Best params last: tuning must win over base defaults.
                    for dotted, value in best_params.items():
                        _set_dotted(config, str(dotted), value)

                    apply_dataset_derived(config, spec, datasets_dir, dataset_name)

                    run_single_combination(
                        config=config,
                        save_dir=save_dir,
                        train_fn=train_fn,
                        data_dir=Path(datasets_dir),
                        cluster=cluster,
                    )
                group_result["seeds"][seed] = {"status": "ok",
                                               "save_dir": str(save_dir)}
                logger.info("Run %s completed", run_name)
            except Exception as exc:  # keep sweeping; one bad DAG must not stop us
                logger.error("Run %s failed: %s", run_name, exc, exc_info=True)
                group_result["seeds"][seed] = {"status": "failed", "error": str(exc)}

        results["groups"][group["name"]] = group_result

    ok = sum(1 for g in results["groups"].values()
             for r in g["seeds"].values() if r["status"] == "ok")
    logger.info("=" * 60)
    logger.info("DAG sweep finished: %d/%d run(s) succeeded", ok, n_runs)
    return results


__all__ = [
    "apply_dataset_derived",
    "build_groups",
    "load_best_params",
    "load_dagsweep_spec",
    "optimize_group",
    "resolve_trainer",
    "run_dag_sweep",
    "stage_group_config",
]

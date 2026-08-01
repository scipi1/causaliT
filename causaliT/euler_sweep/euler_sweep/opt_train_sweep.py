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

Two decoupled seeds (phase 2 only)
----------------------------------
The evaluation phase distinguishes the two things a single ``seed`` used to
conflate:

* ``dag_seeds``   - the DAG *draw* (and, through ``training.data_seed``, the
  train/val/test split): one sampled dataset per value;
* ``model_seeds`` - the model *initialisation* (``training.seed``): one training
  run per value, all on the SAME sampled DAG and the SAME data split.

Holding the DAG fixed while varying the initialisation is what makes **edge
stability** measurable: repeated runs differ only in the optimisation path, so
the spread of a learned edge is attributable to the model, not to the graph.
Averaging over ``dag_seeds`` then answers the orthogonal question (how the method
behaves across graphs).  The sweep is therefore
``dag_seeds x model_seeds`` runs per group, with each DAG generated ONCE and
reused by all of its model seeds.

``model_seeds`` is optional: when omitted, each DAG is trained once with
``training.seed == dag_seed`` - exactly the legacy behaviour.  ``seeds`` remains
accepted as an alias for ``dag_seeds``.  Optuna (phase 1) is untouched: it always
runs on the single ``opt_seed`` DAG.

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
        └── sweeper/runs/combinations/
            ├── <exp_id>_n_nodes_50_seed_0/                  # no model_seeds
            └── <exp_id>_n_nodes_50_dag_0_model_7/           # with model_seeds

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

from causaliT.euler_sweep.euler_sweep.data_source import (

    DATASET_AXIS,
    DataSource,
    build_data_source,
)

from causaliT.euler_sweep.euler_sweep.search_space import (
    apply_protocol,
    build_sample_params_fn,
    derive_size_fields,
    n_keys_from_metadata,
    select_best,
    validate_dimensions,
)
from causaliT.euler_sweep.euler_sweep.sweeper import run_single_combination

logger = logging.getLogger(__name__)

#: Spec file names.  ``atesweep*.yaml`` is a pure ALIAS (identical parsing and
#: execution): a fixed-dataset ATE study reads better in a folder listing when
#: its spec is not called "dagsweep", but there is only one code path.
DAGSWEEP_FILENAME_GLOBS = ("dagsweep*.yaml", "atesweep*.yaml")
DAGSWEEP_FILENAME_GLOB = DAGSWEEP_FILENAME_GLOBS[0]  # backwards compatibility
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

def dag_seeds_of(spec: Any) -> List[int]:
    """
    Return the DAG seeds of a spec, accepting its two aliases.

    All three names denote the SAME axis - the seed that fixes the data a run
    trains on (and, through ``training.data_seed``, its train/val/test split):

    * ``dag_seeds``  - current name for a sampled-DAG sweep (the seed IS the graph);
    * ``data_seeds`` - preferred name for a FIXED-dataset sweep, where the graph
      never changes and the seed only moves the split;
    * ``seeds``      - the original name, kept so old specs keep working.

    Declaring more than one is rejected: they would silently disagree.
    """
    present = [k for k in ("dag_seeds", "data_seeds", "seeds")
               if k in spec and spec.get(k)]
    if len(present) > 1:
        raise ValueError(
            f"Define only ONE seed list, got {present}. 'dag_seeds' (sampled "
            "DAGs), 'data_seeds' (fixed datasets) and 'seeds' (legacy) are "
            "aliases of the same axis."
        )
    raw = spec.get(present[0]) if present else None
    return [int(s) for s in (raw or [])]



def model_seeds_of(spec: Any) -> Optional[List[int]]:
    """
    Return the model-initialisation seeds, or ``None`` when unspecified.

    ``None`` means "one run per DAG with ``training.seed == dag_seed``" (the
    legacy behaviour).  A list means the DAG is held fixed and trained once per
    entry, which is what edge-stability estimates are computed from.
    """
    if "model_seeds" not in spec:
        return None
    raw = spec.get("model_seeds")
    if raw is None:
        return None
    seeds = [int(s) for s in raw]
    if not seeds:
        raise ValueError(
            "'model_seeds' is present but empty. Remove the key to train each "
            "DAG once (model seed = DAG seed), or list at least one seed."
        )
    return seeds


def find_dagsweep_spec(exp_dir: str) -> str:
    """
    Locate the single sweep spec of an experiment folder.

    ``dagsweep*.yaml`` and ``atesweep*.yaml`` are ALIASES - identical parsing,
    identical execution - so a fixed-dataset study can be named after what it
    measures.  Finding several is an error rather than a precedence rule: which
    file a sweep actually ran would otherwise depend on alphabetical order.
    """
    matches: List[str] = []
    for pattern in DAGSWEEP_FILENAME_GLOBS:
        matches.extend(glob.glob(join(exp_dir, pattern)))
    matches = sorted(set(matches))

    if not matches:
        raise FileNotFoundError(
            f"No {' / '.join(DAGSWEEP_FILENAME_GLOBS)} found in {exp_dir}. "
            "A sweep needs a spec with a data source ('dag' for sampled graphs "
            "or 'datasets' for fixed SCMs) and 'dag_seeds'/'data_seeds'."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Several sweep specs in {exp_dir}: {[basename(m) for m in matches]}. "
            "Keep exactly one ('dagsweep*.yaml' and 'atesweep*.yaml' are "
            "aliases), so what the sweep ran is unambiguous."
        )
    return matches[0]


def load_dagsweep_spec(exp_dir: str) -> Any:
    """
    Load and validate the sweep spec of an experiment folder.

    Everything that can be checked without touching the disk is checked HERE, so
    a malformed spec fails in the second it is submitted instead of after hours
    of queueing: the data source (``dag`` xor ``datasets``, registry keys), the
    seed aliases, and the tuning/evaluation seed disjointness.

    ``group_axes`` is required only for a SAMPLED-DAG sweep.  A fixed-dataset
    sweep already has an axis - the dataset itself - so demanding a second one
    would force a dummy entry into every ATE spec.
    """
    path = find_dagsweep_spec(exp_dir)
    spec: Any = OmegaConf.load(path)

    # Rejects 'dag' + 'datasets', an empty/unknown/duplicated registry key, ...
    source = build_data_source(spec)

    if "group_axes" not in spec and source.group_axis_values() is None:
        raise ValueError(
            f"{basename(path)} must define 'group_axes' (a sampled-DAG sweep "
            "groups by DAG size/shape). Only a 'datasets' sweep may omit it, "
            "because the dataset is then the group axis."
        )
    if "dag_seeds" not in spec and "data_seeds" not in spec and "seeds" not in spec:
        raise ValueError(
            f"{basename(path)} must define 'dag_seeds' (sampled DAGs) or "
            "'data_seeds' (fixed datasets); 'seeds' is the legacy alias."
        )

    dag_seeds = dag_seeds_of(spec)

    if not dag_seeds:
        raise ValueError("'dag_seeds' must list at least one evaluation seed.")
    model_seeds_of(spec)  # validate early (raises on an empty list)

    opt_seed = int((spec.get("optuna", {}) or {}).get("opt_seed", DEFAULT_OPT_SEED))
    if opt_seed in dag_seeds:
        # Optimising and evaluating on the same DAG draw leaks hyper-parameters
        # into the reported numbers - refuse rather than silently bias the paper.
        # Only the DAG seed matters here: model seeds never select a dataset.
        raise ValueError(
            f"optuna.opt_seed={opt_seed} also appears in 'dag_seeds'. Use a "
            "dedicated optimisation seed so tuning cannot leak into results."
        )
    return spec


def group_name_of(values: Dict[str, Any]) -> str:
    """
    Render a group's folder name from its axis values.

    Axes are tagged (``n_nodes_50``) so a listing is self-describing.  The
    ``dataset`` axis is the exception: its value is already the alias the user
    typed, so it is emitted bare (``ds_scm1``, not ``dataset_ds_scm1``).
    """
    return "_".join(
        str(value) if key == DATASET_AXIS else f"{key}_{value}"
        for key, value in values.items()
    )


def build_groups(spec: Any, source: Optional[DataSource] = None) -> List[Dict[str, Any]]:
    """
    Expand the group axes into groups, each carrying the full seed plan.

    A group is one Cartesian-product point of the group axes; seeds are *not*
    part of that product - they are members inside each group.  Every group gets
    the same ``dag_seeds`` (one sampled DAG each, or one SHARED dataset for a
    fixed-SCM sweep) and ``model_seeds`` (initialisations trained on each of
    those datasets; ``None`` = one run with model seed == DAG seed).  ``seeds``
    is kept as a mirror of ``dag_seeds`` for backwards compatibility.

    A fixed-dataset source contributes an IMPLICIT, outermost ``dataset`` axis
    (its aliases), so one group is created per dataset - each with its own study,
    ``best_trial.yaml`` and folder - without repeating the list in ``group_axes``.

    Args:
        spec: The loaded sweep spec.
        source: Data source providing the implicit axis; built from the spec when
            omitted.
    """
    import itertools

    if source is None:
        source = build_data_source(spec)

    axes: Dict[str, List[Any]] = {}
    dataset_values = source.group_axis_values()
    if dataset_values is not None:
        axes[DATASET_AXIS] = list(dataset_values)
    for key, values in _as_dict(spec.get("group_axes")).items():
        if key == DATASET_AXIS:
            raise ValueError(
                f"'{DATASET_AXIS}' is a reserved group axis (it selects the "
                "DATA, not a hyper-parameter). Use 'datasets.names' instead."
            )
        axes[key] = list(values)

    dag_seeds = dag_seeds_of(spec)

    model_seeds = model_seeds_of(spec)

    names = list(axes.keys())
    groups: List[Dict[str, Any]] = []
    for combo in itertools.product(*(axes[n] for n in names)):
        values = dict(zip(names, combo))
        groups.append({
            "axes": values,
            "name": group_name_of(values),

            "dag_seeds": dag_seeds,
            "model_seeds": model_seeds,
            # Legacy alias: the DAG seeds are what used to be called "seeds".
            "seeds": dag_seeds,
        })
    return groups


def run_plan(group: Dict[str, Any]) -> List[Tuple[int, Optional[int], Any, str]]:
    """
    Expand a group into its ``(dag_seed, model_seed, run_key, suffix)`` runs.

    ``model_seed is None`` marks the legacy single-run-per-DAG case, where the
    model seed follows the DAG seed and the run keeps its historical
    ``seed_<dag_seed>`` name / integer result key.  With explicit
    ``model_seeds`` the DAG and the initialisation appear separately in both the
    folder name and the result key, so a stability set is trivially groupable by
    DAG.
    """
    plan: List[Tuple[int, Optional[int], Any, str]] = []
    model_seeds = group.get("model_seeds")
    for dag_seed in group["dag_seeds"]:
        if model_seeds is None:
            plan.append((dag_seed, None, dag_seed, f"seed_{dag_seed}"))
        else:
            for model_seed in model_seeds:
                plan.append((
                    dag_seed,
                    model_seed,
                    f"dag_{dag_seed}_model_{model_seed}",
                    f"dag_{dag_seed}_model_{model_seed}",
                ))
    return plan


def _group_plan_by_dag(
    plan: List[Tuple[int, Optional[int], Any, str]],
) -> List[Tuple[int, List[Tuple[Optional[int], Any, str]]]]:
    """
    Collapse a run plan into ``[(dag_seed, [(model_seed, run_key, suffix), ...])]``.

    Grouping by DAG seed is what lets the orchestrator generate each dataset once
    and prune it after the last model seed that needs it, keeping disk usage flat
    no matter how many initialisations are averaged over.
    """
    grouped: Dict[int, List[Tuple[Optional[int], Any, str]]] = {}
    order: List[int] = []
    for dag_seed, model_seed, run_key, suffix in plan:
        if dag_seed not in grouped:
            grouped[dag_seed] = []
            order.append(dag_seed)
        grouped[dag_seed].append((model_seed, run_key, suffix))
    return [(dag_seed, grouped[dag_seed]) for dag_seed in order]


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


def _write_axes(config: Any, group: Dict[str, Any]) -> None:
    """
    Expose the group axes under ``experiment.*`` so a config can interpolate
    them (e.g. ``${experiment.n_nodes}``) exactly like in a normal sweep.

    ``dataset`` is skipped: it selects the DATA (already written as
    ``data.dataset``), not a hyper-parameter.
    """
    for key, value in group["axes"].items():
        if key == DATASET_AXIS:
            continue
        _set_dotted(config, f"experiment.{key}", value)


def stage_group_config(base_config_path: str, base_optuna_path: Optional[str],
                       group_dir: Path, group: Dict[str, Any], spec: Any,
                       dataset_name: str, datasets_dir: str,
                       n_keys: Optional[int] = None,
                       search_protocol: Optional[str] = None,
                       opt_seed: Optional[int] = None) -> Any:

    """
    Materialize a group's config directory (as ``OptunaStudy`` expects it).

    The file written here is the **search** config: every Optuna trial starts
    from it and only overrides the sampled hyper-parameters.  Besides the group
    axes and the dataset pointer it therefore carries

    * the size-derived fields (batch size, fan-in scale) for THIS DAG size, and
    * the search protocol (``search_protocol``), i.e. reconstruction-only
      training - so a trial measures capacity and nothing else.

    ``opt_seed`` pins ``training.data_seed`` for the search.  It matters for a
    FIXED-dataset sweep, where every group shares one dataset and the seeds are
    split seeds: tuning must then happen on a split that no evaluation run uses.
    (For sampled DAGs the optimisation graph is already a separate dataset.)


    Because the file lands next to ``best_trial.yaml`` it is also the audit trail
    of what the search actually trained.  ``d_model`` is deliberately NOT
    validated here: the base value is a placeholder that every trial replaces
    with a draw from the adaptive width range.

    Interpolations are deliberately **not** resolved: downstream ``update_config``
    and per-seed overrides must still be able to flow through ``${...}`` refs.
    """
    group_dir.mkdir(parents=True, exist_ok=True)

    config: Any = OmegaConf.load(base_config_path)

    _write_axes(config, group)

    _set_dotted(config, "data.dataset", dataset_name)
    # Point the whole stack at the group-local dataset store, so nothing is
    # written to the shared `data/` folder.
    _set_dotted(config, "data.data_root", datasets_dir)
    if opt_seed is not None:
        _set_dotted(config, "training.data_seed", int(opt_seed))


    apply_dataset_derived(config, spec, datasets_dir, dataset_name)

    if n_keys is not None:
        derive_size_fields(config, int(n_keys), _as_dict(spec.get("size_derived")))
    if search_protocol is not None:
        apply_protocol(config, search_protocol,
                       _as_dict((spec.get("optuna", {}) or {}).get("overrides")))

    OmegaConf.save(config, group_dir / basename(base_config_path))
    if base_optuna_path is not None:
        OmegaConf.save(OmegaConf.load(base_optuna_path),
                       group_dir / basename(base_optuna_path))
    return config


def stage_run_config(base_config_path: str, group: Dict[str, Any], spec: Any,
                     dataset_name: str, datasets_dir: str, dag_seed: int,
                     model_seed: Optional[int],
                     best_params: Optional[Dict[str, Any]] = None,
                     n_keys: Optional[int] = None) -> Any:
    """
    Build the config of ONE evaluation run (phase 2).

    Single source of truth for what a run trains, shared by the sequential sweep
    and by the SLURM training worker - a parallel run must be bit-identical to
    the sequential one, so this logic may exist only once.

    Order matters:

    1. group axes -> ``experiment.*`` (so ``${experiment.n_nodes}`` interpolates);
    2. the sampled dataset + group-local ``data_root``;
    3. ``training.data_seed`` = DAG seed (graph AND train/val/test split) while
       ``training.seed`` = model seed (weight init only), so several
       initialisations share one dataset and one split;
    4. the tuned ``best_params`` LAST, so they win over the base defaults;
    5. dataset-derived and size-derived fields, recomputed with the SAME rules as
       the search (a tuned lr belongs to the batch size actually used), then
       ``validate_dimensions`` so a model that cannot represent this DAG is
       repaired instead of crashing hours later.

    Returns:
        The staged (still unresolved) config.
    """
    config: Any = OmegaConf.load(base_config_path)

    _write_axes(config, group)
    _set_dotted(config, "data.dataset", dataset_name)

    _set_dotted(config, "data.data_root", datasets_dir)
    _set_dotted(config, "training.data_seed", int(dag_seed))
    _set_dotted(config, "training.seed",
                int(dag_seed if model_seed is None else model_seed))

    for dotted, value in (best_params or {}).items():
        _set_dotted(config, str(dotted), value)

    apply_dataset_derived(config, spec, datasets_dir, dataset_name)

    if n_keys is None:
        n_keys = n_keys_from_metadata(
            datasets_dir, dataset_name, fallback=group["axes"].get("n_nodes"),
        )
    derive_size_fields(config, int(n_keys), _as_dict(spec.get("size_derived")))
    validate_dimensions(config, int(n_keys))
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


def load_search_settings(group_dir: Path) -> Dict[str, Any]:
    """
    Load the group's ``optuna*.yaml``: search space, selection rule, budget.

    The file is required as soon as tuning is enabled - the search space is the
    definition of the experiment, so guessing it would silently change what the
    benchmark measures.
    """
    matches = sorted(glob.glob(join(str(group_dir), "optuna*.yaml")))
    if not matches:
        raise FileNotFoundError(
            f"No optuna*.yaml in {group_dir}. A DAG sweep with optuna.enabled "
            "needs one next to the base config, declaring 'search_space'."
        )
    return _as_dict(OmegaConf.load(matches[0]))


def group_study_name(group_dir: Path) -> str:
    """Optuna study name of a group (stable, derived from the folder name)."""
    return f"dagsweep_{Path(group_dir).name}"


def resolve_metric_direction(spec: Any, settings: Dict[str, Any]) -> Tuple[str, str]:
    """
    Single source of truth for the optimisation metric/direction: the sweep spec.

    A conflicting ``direction`` in ``optuna*.yaml`` would make the study and the
    selection rule disagree about what "better" means, so it is rejected rather
    than merged.
    """
    opt_cfg = _as_dict(spec.get("optuna", {}))
    metric = str(opt_cfg.get("metric", "val_x_mae_mean"))
    direction = str(opt_cfg.get("direction", "minimize"))
    declared = settings.get("direction")
    if declared is not None and str(declared) != direction:
        raise ValueError(
            f"direction mismatch: dagsweep.yaml optuna.direction='{direction}' "
            f"but optuna settings say '{declared}'. Declare it once."
        )
    return metric, direction


def build_group_study(group_dir: Path, datasets_dir: str, spec: Any,
                      cluster: bool, n_keys: int) -> Tuple[Any, str, str, Dict[str, Any]]:
    """
    Construct the group's ``OptunaStudy`` (search space, objective, storage).

    Factored out of :func:`optimize_group` so that EVERY execution mode builds
    the identical study: the sequential sweep, a SLURM trial worker (which only
    asks/tells one trial) and the selection step.  Constructing this object is
    cheap - it loads the staged group config and the search settings, nothing
    else - so a worker can rebuild it instead of serialising callables.

    Two properties are load-bearing:

    * the SEARCH SPACE is a function of ``n_keys`` (the width range starts at the
      node count), so each group tunes a model that can actually represent its
      DAG;
    * parameter names are DOTTED config paths, so ``best_trial.yaml`` can be
      applied verbatim to the evaluation configs.

    Returns:
        ``(study, metric, direction, settings)``
    """
    from causaliT.euler_optuna.euler_optuna.cli import get_metrics_for_optuna
    from causaliT.euler_optuna.euler_optuna.optuna_opt import OptunaStudy

    opt_cfg = _as_dict(spec.get("optuna", {}))
    settings = load_search_settings(group_dir)
    metric, direction = resolve_metric_direction(spec, settings)

    if n_keys is None:
        raise ValueError("build_group_study needs n_keys to build the width range.")

    sample_fn = build_sample_params_fn(
        _as_dict(settings.get("search_space")), int(n_keys)
    )
    train_fn = resolve_trainer(str(opt_cfg.get("trainer", "standard")))[0]

    study_name = group_study_name(group_dir)
    study = OptunaStudy(
        exp_dir=Path(group_dir),
        data_dir=Path(datasets_dir),
        cluster=cluster,
        study_name=study_name,
        manifest_tag=study_name,
        sample_params_fn=sample_fn,
        train_fn=train_fn,
        get_metrics_fn=get_metrics_for_optuna,
        optimization_metric=metric,
        optimization_direction=direction,
    )
    return study, metric, direction, settings


def finalize_group_study(group_dir: Path, datasets_dir: str, spec: Any,
                         cluster: bool, n_keys: int) -> Dict[str, Any]:
    """
    Choose the winning trial of a finished study and write ``best_trial.yaml``.

    The winner is chosen by :func:`search_space.select_best`, which by default
    takes the SMALLEST model within a tolerance of the best metric instead of the
    plain argmin (reconstruction error is monotone in capacity, so argmin would
    always pick the largest width in the range).

    Returns:
        The flat ``{dotted_param: value}`` mapping that phase 2 must apply.
    """
    import optuna
    import yaml

    study, metric, direction, settings = build_group_study(
        group_dir, datasets_dir, spec, cluster, n_keys
    )
    loaded = optuna.load_study(study_name=study.study_name, storage=study.storage)
    chosen = select_best(loaded, _as_dict(settings.get("selection")), direction)

    summary = {
        "trial_number": chosen["trial_number"],
        "optimization_metric": metric,
        "optimization_value": chosen["optimization_value"],
        "n_keys": int(n_keys),
        "selection": {k: chosen[k] for k in ("mode", "tol") if k in chosen},
        "capacity": chosen.get("capacity"),
        "config_path": chosen.get("config_path"),
        "params": chosen["params"],
        "metrics": chosen.get("metrics", {}),
        "raw_best": chosen.get("raw_best"),
        "curve": chosen.get("curve", []),
    }
    with open(Path(group_dir) / "best_trial.yaml", "w") as fh:
        yaml.dump(summary, fh, default_flow_style=False, sort_keys=False)

    logger.info("  Selected trial %s (%s=%.6g, capacity=%.0f) of %d complete trial(s)",
                chosen["trial_number"], metric, chosen["optimization_value"],
                chosen.get("capacity", 0.0), len(chosen.get("curve", [])))
    for dotted, value in chosen["params"].items():
        logger.info("    %s = %s", dotted, value)

    return dict(chosen["params"])


def optimize_group(group_dir: Path, datasets_dir: str, spec: Any,
                   cluster: bool, n_keys: Optional[int] = None,
                   force: bool = False) -> Dict[str, Any]:
    """
    Run (or reuse) the group's single Optuna study and return the chosen params.

    Sequential driver of the two primitives above: build the study, run its
    trials in-process, then select the winner.

    Resumable by construction: an existing ``best_trial.yaml`` short-circuits the
    whole phase, so an interrupted sweep never re-tunes a finished group.
    ``force=True`` deletes the study database so the search really starts over.
    """
    if not force:
        cached = load_best_params(group_dir)
        if cached:
            logger.info("  Reusing existing best_trial.yaml (%d params)", len(cached))
            return cached

    if n_keys is None:
        raise ValueError("optimize_group needs n_keys to build the width range.")

    study, _metric, _direction, _settings = build_group_study(
        group_dir, datasets_dir, spec, cluster, int(n_keys)
    )

    if force and exists(study.study_file_path):
        # Without this, `resume` would see max_trials already reached and the
        # "forced" re-tune would silently be a no-op.
        logger.info("  --force_optuna: discarding the previous study database")
        Path(study.study_file_path).unlink()

    try:
        study.create()
    except Exception:
        # Study already exists (e.g. a previous partial run) -> continue it.
        logger.info("  Study exists; resuming remaining trials")
    study.resume()

    return finalize_group_study(group_dir, datasets_dir, spec, cluster, int(n_keys))



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
        # Not a trainer: fits the external baselines (NOTEARS / DAGMA / PC) on
        # the generated datasets, so a DAG sweep can produce baseline curves
        # with exactly the same graphs, splits and reporting as the model runs.
        "benchmark": "benchmark_function_for_sweep",
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

    Per group:     sample the optimisation DAG -> one Optuna study -> prune arrays.
    Per dag_seed:  sample the evaluation DAG once, then train it with EVERY
                   ``model_seed`` (same graph, same split, different weight
                   init) -> post-training evaluations (inside the trainer) ->
                   prune arrays after the last model seed.

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
    # One source for the whole sweep: it decides what a group's data IS
    # (sampled graph vs fixed SCM) and when the heavy arrays may be pruned.
    source = build_data_source(spec)
    groups = build_groups(spec, source)
    base_config_path, base_optuna_path = _find_base_files(exp_dir)

    opt_cfg = spec.get("optuna", {}) or {}

    opt_seed = int(opt_cfg.get("opt_seed", DEFAULT_OPT_SEED))
    optuna_enabled = bool(opt_cfg.get("enabled", True)) and not skip_optuna

    train_name = str((spec.get("training", {}) or {}).get("trainer", "standard"))
    train_fn = resolve_trainer(train_name)[0]

    delete_dataset = not (keep_data or not spec.get("delete_dataset", True))

    exp_id = basename(str(exp_dir).rstrip("/\\"))
    plans = {g["name"]: run_plan(g) for g in groups}
    n_runs = sum(len(p) for p in plans.values())
    model_seeds = groups[0]["model_seeds"] if groups else None
    logger.info("DAG sweep '%s': %d group(s), %d run(s), trainer=%s, optuna=%s, "
                "model_seeds=%s",
                exp_id, len(groups), n_runs, train_name, optuna_enabled,
                model_seeds if model_seeds is not None else "= dag_seeds")

    if dry_run:
        for group in groups:
            # A fixed dataset has no DAG to draw, so the same seed list means
            # "split seed" there; print the label the spec actually used.
            seed_label = ("data_seeds" if source.group_axis_values() is not None
                          else "dag_seeds")
            print(f"[group] {group['name']}  {seed_label}={group['dag_seeds']}"

                  f"  model_seeds="
                  f"{group['model_seeds'] if group['model_seeds'] is not None else '= dag_seed'}"
                  f"  runs={len(plans[group['name']])}"
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

        # The scope owns the group's data: for sampled DAGs it is a thin
        # per-seed wrapper, for a fixed SCM it generates once here and prunes
        # once at exit (regenerating per seed would dominate the runtime).
        with source.group_scope(group, datasets_dir, delete_dataset) as scope:

            # ---- Phase 1: optimise once, on a dedicated dataset -------------
            best_params: Dict[str, Any] = {}
            if optuna_enabled:
                with scope.dataset(opt_seed) as opt_dataset:
                    # The node count comes from the generated dataset, not from
                    # the group axis: the width range must be built on it.
                    n_keys = n_keys_from_metadata(
                        datasets_dir, opt_dataset,
                        fallback=group["axes"].get("n_nodes"),
                    )
                    logger.info("  Optimisation dataset %s (n_keys=%d)",
                                opt_dataset, n_keys)
                    stage_group_config(
                        base_config_path, base_optuna_path, group_dir, group, spec,
                        opt_dataset, datasets_dir, n_keys=n_keys,
                        search_protocol=str(opt_cfg.get("protocol", "reconstruction")),
                        opt_seed=opt_seed,
                    )
                    best_params = optimize_group(group_dir, datasets_dir, spec,
                                                 cluster, n_keys=n_keys,
                                                 force=force_optuna)
            else:
                best_params = load_best_params(group_dir)
                if best_params:
                    logger.info("  Using %d cached best param(s)", len(best_params))
                else:
                    # Explicit fallback (--skip_optuna / optuna.enabled: false):
                    # the base config is trained AS IS.  Loud, because an untuned
                    # run looks exactly like a tuned one in the output folder.
                    logger.warning(
                        "  No best_trial.yaml in %s -> training the BASE config "
                        "UNTUNED for every seed of this group", group_dir,
                    )

            group_result["best_params"] = best_params

            # ---- Phase 2: train every seed, reusing those params ------------
            # Runs are grouped by data seed so each dataset is materialized ONCE
            # and all of its model seeds train on the very same arrays (identical
            # graph AND identical split - the premise of an edge-stability
            # estimate).
            for dag_seed, dag_runs in _group_plan_by_dag(plans[group["name"]]):
              try:
                with scope.dataset(dag_seed) as dataset_name:
                    for model_seed, run_key, suffix in dag_runs:

                        run_name = f"{exp_id}_{group['name']}_{suffix}"
                        save_dir = (group_dir / "sweeper" / "runs"
                                    / "combinations" / run_name)

                        logger.info("-" * 60)
                        logger.info("Run %s", run_name)

                        try:
                            # Staging lives in stage_run_config so the SLURM
                            # training worker trains exactly the same config.
                            config = stage_run_config(
                                base_config_path, group, spec, dataset_name,
                                datasets_dir, dag_seed, model_seed, best_params,
                            )

                            run_single_combination(
                                config=config,
                                save_dir=save_dir,

                                train_fn=train_fn,
                                data_dir=Path(datasets_dir),
                                cluster=cluster,
                            )
                            group_result["seeds"][run_key] = {
                                "status": "ok",
                                "save_dir": str(save_dir),
                                "dag_seed": int(dag_seed),
                                "model_seed": int(
                                    dag_seed if model_seed is None else model_seed
                                ),
                                "dataset": dataset_name,
                            }
                            logger.info("Run %s completed", run_name)
                        except Exception as exc:
                            # One diverging init must not cost the other seeds.
                            logger.error("Run %s failed: %s", run_name, exc,
                                         exc_info=True)
                            group_result["seeds"][run_key] = {
                                "status": "failed",
                                "error": str(exc),
                                "dag_seed": int(dag_seed),
                            }
              except Exception as exc:
                # Dataset generation / pruning failed: no run of this seed ran.
                logger.error("Data seed %s failed: %s", dag_seed, exc, exc_info=True)
                for model_seed, run_key, _suffix in dag_runs:
                    group_result["seeds"].setdefault(run_key, {
                        "status": "failed",
                        "error": str(exc),
                        "dag_seed": int(dag_seed),
                    })


        results["groups"][group["name"]] = group_result

    ok = sum(1 for g in results["groups"].values()
             for r in g["seeds"].values() if r["status"] == "ok")
    logger.info("=" * 60)
    logger.info("DAG sweep finished: %d/%d run(s) succeeded", ok, n_runs)
    return results


__all__ = [
    "apply_dataset_derived",
    "build_group_study",
    "build_groups",
    "dag_seeds_of",
    "finalize_group_study",
    "group_study_name",
    "load_best_params",
    "load_dagsweep_spec",
    "load_search_settings",
    "model_seeds_of",
    "optimize_group",
    "resolve_metric_direction",
    "resolve_trainer",
    "run_dag_sweep",
    "run_plan",
    "stage_group_config",
    "stage_run_config",
]



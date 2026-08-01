"""
Dataset sources for the grouped sweep (``dagsweep``).

Why this module exists
----------------------
``opt_train_sweep.run_dag_sweep`` implements a scheme that is entirely about
*hyper-parameters and seeds*: tune ONCE per group, then train every seed of that
group with the winning parameters.  Nothing in that scheme requires the data to
be a randomly sampled DAG - it only needs, per group, a dataset folder name that
the trainers can consume.

Two study families need exactly the same scheme with different data:

* **structure benchmarks** - a NEW graph per seed, sampled from
  ``RandomSCMConfig`` (``SampledDagSource``, the historical behaviour);
* **ATE studies** - a FIXED, hand-written SCM from ``scm_ds.datasets``, where the
  seeds must vary only the split and the initialisation, never the graph
  (``FixedDatasetSource``).

Isolating "where does the data come from" behind one small interface keeps the
whole orchestration (Optuna per group, ``best_trial.yaml`` reuse, SLURM chain,
progress rollup, pruning) shared and unduplicated.

Pruning contract (the load-bearing difference between the two sources)
----------------------------------------------------------------------
``ds*.npz`` must exist for the WHOLE run: post-training evaluations
(``eval_attention_scores``, ``eval_ate_mc``) do a forward pass over the test
split from inside ``trainer()``.  It may be deleted only afterwards.

* ``SampledDagSource``: one dataset per DAG seed -> pruned right after the last
  run of that DAG (``per_use_pruning = True``), which keeps peak disk flat over a
  sweep of hundreds of graphs.
* ``FixedDatasetSource``: ONE dataset shared by every seed of the group ->
  pruning after each seed would force a full regeneration (50k samples + a
  Monte-Carlo ATE ground truth) per seed.  It is therefore generated once at
  group entry and pruned once at group exit.

Both sources write their data INSIDE the experiment folder
(``groups/<group>/datasets/``) and leave a recipe next to it, so a sweep is
self-contained and exactly reproducible without shipping any array:

* sampled DAGs  -> ``dag_recipe.json``  (see :mod:`dag_provider`)
* fixed SCMs    -> ``scm_recipe.json``  (registry key + generation options)
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from os.path import exists, join
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from causaliT.euler_sweep.euler_sweep.dag_provider import (
    build_dag_config,
    dag_dataset_name,
    ensure_dag_dataset,
    has_arrays,
    is_materialized,
    prune_dag_arrays,
    split_dag_block,
)

logger = logging.getLogger(__name__)

#: Recipe written next to a fixed (registry) dataset.
SCM_RECIPE_FILENAME = "scm_recipe.json"

#: Reserved group axis holding the registry key of a fixed dataset.  Reserved
#: because it selects the DATA, not a model hyper-parameter, so it must never be
#: written into ``experiment.*`` where a config could interpolate it.
DATASET_AXIS = "dataset"

DEFAULT_FIXED_N = 50_000
DEFAULT_FIXED_MODE = "flat"
DEFAULT_FIXED_NORMALIZE_METHOD = "minmax"
DEFAULT_FIXED_SEED = 42


# =============================================================================
# Group scope: what a group's data looks like while the group is running
# =============================================================================

class GroupScope:
    """
    Per-group dataset handle handed to the sweep body.

    ``dataset(seed)`` is a context manager yielding the dataset folder NAME to
    train on.  Whether that name depends on the seed, and when the heavy arrays
    are pruned, is entirely the source's business.
    """

    def __init__(self, source: "DataSource", group: Dict[str, Any],
                 datasets_dir: str, delete_dataset: bool):
        self.source = source
        self.group = group
        self.datasets_dir = datasets_dir
        self.delete_dataset = delete_dataset

    @contextmanager
    def dataset(self, seed: int) -> Iterator[str]:
        """Yield a ready-to-train dataset name for ``seed``."""
        name = self.source.ensure(self.group, self.datasets_dir, int(seed))
        try:
            yield name
        finally:
            if self.delete_dataset and self.source.per_use_pruning:
                prune_dag_arrays(join(self.datasets_dir, name))


# =============================================================================
# Interface
# =============================================================================

class DataSource:
    """
    Where a group's training data comes from.

    Implementations must be IDEMPOTENT (``ensure`` may be called again after a
    walltime kill) and DETERMINISTIC (``dataset_name`` must be computable without
    touching the disk, because the parallel driver builds the job plan before any
    dataset exists).
    """

    #: True when each seed gets its OWN dataset (prune after each use).
    per_use_pruning: bool = True

    #: Human-readable tag for logs / the sweep summary.
    kind: str = "abstract"

    def dataset_name(self, group: Dict[str, Any], seed: int) -> str:
        """Folder name of the dataset used by ``seed`` (no I/O)."""
        raise NotImplementedError

    def ensure(self, group: Dict[str, Any], datasets_dir: str, seed: int) -> str:
        """Materialize (or reuse) that dataset and return its folder name."""
        raise NotImplementedError

    def group_axis_values(self) -> Optional[List[Any]]:
        """
        Values of the implicit group axis this source contributes, if any.

        ``FixedDatasetSource`` returns its registry keys so that one group is
        created per dataset; ``SampledDagSource`` returns ``None`` (its groups
        come exclusively from ``group_axes``).
        """
        return None

    @contextmanager
    def group_scope(self, group: Dict[str, Any], datasets_dir: str,
                    delete_dataset: bool) -> Iterator[GroupScope]:
        """Enter a group: default scope does nothing beyond per-use handling."""
        Path(datasets_dir).mkdir(parents=True, exist_ok=True)
        yield GroupScope(self, group, datasets_dir, delete_dataset)


# =============================================================================
# Sampled DAGs (structure benchmarks) - the historical behaviour
# =============================================================================

class SampledDagSource(DataSource):
    """
    One randomly sampled DAG per seed, from the spec's ``dag:`` block.

    Thin wrapper over :mod:`dag_provider`: the seed IS the graph, so every seed
    gets its own dataset folder and its own ``dag_recipe.json``.
    """

    per_use_pruning = True
    kind = "sampled_dag"

    def __init__(self, dag_block: Dict[str, Any]):
        self.dag_block = dict(dag_block or {})
        # Validates the block early (typo protection) and splits the two concerns.
        _, self.gen_kwargs = split_dag_block(self.dag_block)

    def _config(self, group: Dict[str, Any], seed: int):
        return build_dag_config(self.dag_block, seed=int(seed), **group["axes"])

    def dataset_name(self, group: Dict[str, Any], seed: int) -> str:
        return dag_dataset_name(self._config(group, seed))

    def ensure(self, group: Dict[str, Any], datasets_dir: str, seed: int) -> str:
        return ensure_dag_dataset(self._config(group, seed), datasets_dir,
                                  self.gen_kwargs)


# =============================================================================
# Fixed SCM datasets (ATE studies)
# =============================================================================

def write_scm_recipe(dataset_dir: str, registry_name: str,
                     generation: Dict[str, Any]) -> None:
    """Persist the registry key + generation options next to the data."""
    payload = {
        "registry_name": registry_name,
        "generation": dict(generation),
        "created_by": "causaliT.euler_sweep.data_source",
    }
    with open(join(dataset_dir, SCM_RECIPE_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def read_scm_recipe(dataset_dir: str) -> Optional[Dict[str, Any]]:
    """Load an ``scm_recipe.json``, or ``None`` when absent."""
    path = join(str(dataset_dir), SCM_RECIPE_FILENAME)
    if not exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def generate_fixed_dataset(registry_name: str, data_root: str,
                           generation: Optional[Dict[str, Any]] = None,
                           folder_name: Optional[str] = None,
                           force: bool = False, verbose: bool = True) -> str:
    """
    Materialize a registry SCM into ``data_root/<folder_name>/``.

    Idempotent: a folder that already holds both the light artefacts and the
    heavy arrays is reused; a PRUNED folder is regenerated from the same
    ``(registry_name, generation)``, which is deterministic because ``seed`` is
    part of ``generation``.

    Args:
        registry_name: Key of ``scm_ds.datasets.DATASET_REGISTRY``.
        data_root: Folder that will contain the dataset folder.
        generation: ``generate_ds`` options (``n``, ``mode``, ``normalize_method``,
            ``seed``, ``shared_embedding``, ``test_split_method``, ...).
        folder_name: Dataset folder name; defaults to ``registry_name``.
        force: Regenerate even when the dataset is complete.

    Returns:
        The dataset folder name (to be used as ``config.data.dataset``).
    """
    from scm_ds.datasets import get_dataset

    name = folder_name or registry_name
    dataset_dir = join(str(data_root), name)

    if not force and is_materialized(dataset_dir) and has_arrays(dataset_dir):
        if verbose:
            print(f"  [data_source] reusing {name}")
        return name
    if not force and is_materialized(dataset_dir) and verbose:
        print(f"  [data_source] {name} was pruned - regenerating arrays")

    options: Dict[str, Any] = dict(generation or {})
    n = int(options.pop("n", options.pop("n_samples", DEFAULT_FIXED_N)))
    mode = options.pop("mode", DEFAULT_FIXED_MODE)
    options.setdefault("normalize_method", DEFAULT_FIXED_NORMALIZE_METHOD)
    # The content seed is FIXED (not a sweep seed): the point of an ATE study is
    # that every seed of the sweep sees the very same data.
    options.setdefault("seed", DEFAULT_FIXED_SEED)

    if verbose:
        print(f"  [data_source] generating {name} from '{registry_name}' "
              f"(n={n}, mode={mode})")

    dataset = get_dataset(registry_name)
    dataset.generate_ds(mode=mode, n=n, save_dir=dataset_dir, **options)

    write_scm_recipe(dataset_dir, registry_name,
                     {"n": n, "mode": mode, **options})
    return name


def regenerate_from_scm_recipe(dataset_dir: str, verbose: bool = True) -> str:
    """
    Restore a pruned fixed dataset from its ``scm_recipe.json``.

    Counterpart of ``dag_provider.regenerate_from_recipe`` for registry datasets;
    use it before re-running a forward-pass evaluation on an archived sweep.
    """
    recipe = read_scm_recipe(dataset_dir)
    if recipe is None:
        raise FileNotFoundError(
            f"No {SCM_RECIPE_FILENAME} in {dataset_dir}; cannot regenerate. "
            "This dataset was not produced by a fixed-dataset sweep."
        )
    path = Path(str(dataset_dir))
    return generate_fixed_dataset(
        registry_name=str(recipe["registry_name"]),
        data_root=str(path.parent),
        generation=dict(recipe.get("generation") or {}),
        folder_name=path.name,
        force=True,
        verbose=verbose,
    )


class FixedDatasetSource(DataSource):
    """
    One hard-coded SCM per group, generated inside the experiment folder.

    The dataset does NOT depend on the sweep seeds: ``dag_seeds`` become SPLIT
    seeds (``training.data_seed``) and ``model_seeds`` stay initialisation seeds,
    which is exactly the decomposition an ATE study needs (same data, varying
    split and initialisation).

    Data is written to ``groups/<group>/datasets/<registry_key>/`` and never to
    the shared ``data/`` folder: an experiment stays self-contained, and the
    ``scm_recipe.json`` makes it regenerable from this repository alone.
    """

    per_use_pruning = False
    kind = "fixed_dataset"

    def __init__(self, names: List[str], generation: Optional[Dict[str, Any]] = None):
        if not names:
            raise ValueError(
                "datasets.names is empty: a fixed-dataset sweep needs at least "
                "one key of scm_ds.datasets.DATASET_REGISTRY."
            )
        self.names = [str(n) for n in names]
        self.generation = dict(generation or {})
        self._validate_names()

    def _validate_names(self) -> None:
        """Fail at spec-load time, not after hours of queueing."""
        from scm_ds.datasets import DATASET_REGISTRY

        unknown = [n for n in self.names if n not in DATASET_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown dataset key(s) {unknown} in datasets.names. "
                f"Available: {sorted(DATASET_REGISTRY)}."
            )

        # The key names the group folder, so a repeat would make two arms share
        # one folder and be aggregated as if they were one.
        if len(set(self.names)) != len(self.names):
            raise ValueError(
                f"Repeated key in datasets.names: {self.names}. Each key names "
                "a group folder, so list it once."
            )


    def group_axis_values(self) -> Optional[List[Any]]:
        return list(self.names)

    def _registry_name(self, group: Dict[str, Any]) -> str:
        """The dataset of a group: its ``dataset`` axis, or the single entry."""
        value = group.get("axes", {}).get(DATASET_AXIS)
        if value is not None:
            return str(value)
        if len(self.names) == 1:
            return self.names[0]
        raise ValueError(
            "Group has no 'dataset' axis but several datasets are declared; "
            "the group expansion must add one group per dataset."
        )

    def dataset_name(self, group: Dict[str, Any], seed: int) -> str:
        # Seed-independent on purpose: all seeds of a group share the data.
        return self._registry_name(group)

    def ensure(self, group: Dict[str, Any], datasets_dir: str, seed: int) -> str:
        return generate_fixed_dataset(self._registry_name(group), datasets_dir,
                                      self.generation)

    @contextmanager
    def group_scope(self, group: Dict[str, Any], datasets_dir: str,
                    delete_dataset: bool) -> Iterator[GroupScope]:
        """
        Generate the group's dataset ONCE, prune it once at the end.

        Per-seed pruning would regenerate 50k samples and a Monte-Carlo ATE
        ground truth for every seed - the dominant cost of the whole sweep.
        """
        Path(datasets_dir).mkdir(parents=True, exist_ok=True)
        name = self.ensure(group, datasets_dir, 0)
        try:
            yield GroupScope(self, group, datasets_dir, delete_dataset)
        finally:
            if delete_dataset:
                prune_dag_arrays(join(datasets_dir, name))


# =============================================================================
# Factory
# =============================================================================

def build_data_source(spec: Any) -> DataSource:
    """
    Build the data source declared by a sweep spec.

    A spec declares EXACTLY ONE of:

    * ``dag:``      -> :class:`SampledDagSource` (a new graph per seed);
    * ``datasets:`` -> :class:`FixedDatasetSource` (hard-coded SCMs).

    Declaring both is rejected: the two answer the same question ("what data does
    this sweep train on") and a silent precedence rule would make a spec's meaning
    depend on the reader's memory.
    """
    from omegaconf import OmegaConf

    def _as_dict(node: Any) -> Dict[str, Any]:
        if node is None:
            return {}
        container = (OmegaConf.to_container(node, resolve=True)
                     if OmegaConf.is_config(node) else node)
        return {str(k): v for k, v in container.items()} if isinstance(container, dict) else {}

    has_dag = "dag" in spec and spec.get("dag") is not None
    datasets_block = _as_dict(spec.get("datasets")) if "datasets" in spec else {}
    has_datasets = bool(datasets_block)

    if has_dag and has_datasets:
        raise ValueError(
            "The sweep spec declares BOTH 'dag' (sampled graphs) and 'datasets' "
            "(fixed SCMs). Keep exactly one: 'dag' for structure benchmarks "
            "(a new graph per seed), 'datasets' for fixed-SCM studies (ATE)."
        )
    if not has_dag and not has_datasets:
        raise ValueError(
            "The sweep spec declares no data source. Add either a 'dag' block "
            "(sampled DAGs) or a 'datasets' block with 'names' (fixed SCMs from "
            "scm_ds.datasets.DATASET_REGISTRY)."
        )

    if has_datasets:
        names = datasets_block.get("names")
        if isinstance(names, str):
            names = [names]
        return FixedDatasetSource(
            names=list(names or []),
            generation=_as_dict(datasets_block.get("generate")),
        )

    return SampledDagSource(_as_dict(spec.get("dag")))


__all__ = [
    "DATASET_AXIS",
    "SCM_RECIPE_FILENAME",
    "DataSource",
    "FixedDatasetSource",
    "GroupScope",
    "SampledDagSource",
    "build_data_source",
    "generate_fixed_dataset",
    "read_scm_recipe",
    "regenerate_from_scm_recipe",
    "write_scm_recipe",
]



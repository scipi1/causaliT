"""
DAG dataset provider for grouped DAG sweeps.

Responsibilities
----------------
1. **Deterministic naming** - map ``(dag_cfg, seed)`` to a stable dataset folder
   name, so the same DAG is always found at the same place.
2. **Idempotent materialization** - ``ensure_dag_dataset`` generates the dataset
   only when it is missing; if the heavy arrays were pruned it regenerates them
   (deterministically, from the stored recipe) without touching anything else.
3. **Ephemeral storage** - ``prune_dag_arrays`` deletes *only* the heavy sample
   arrays (``ds*.npz``), keeping every light artefact that downstream evaluation
   needs (``dataset_metadata.json``, attention masks, ``ate_ground_truth.json``,
   ``normalization.json``, variable maps, ``meta.json``).

Why this split matters
----------------------
``eval_attention_scores`` and ``eval_ate_mc`` run a **forward pass over the test
split** (via ``predict_test_from_ckpt`` / ``create_predictor``), so ``ds.npz``
*must* exist while ``trainer()`` runs - post-training evaluations included.
It may be deleted only after the trainer returns.  Everything that
``eval_seed_sweep`` consumes is light JSON and is therefore always retained.

Each dataset folder also gets a ``dag_recipe.json`` holding the full sampling
recipe (``RandomSCMConfig`` fields + ``n_samples`` + ``normalize_method`` +
generation seed), which makes any pruned dataset exactly regenerable::

    from causaliT.euler_sweep.euler_sweep.dag_provider import regenerate_from_recipe
    regenerate_from_recipe("experiments/.../datasets/random_n50_k2_mixed_mixed_s0")
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from contextlib import contextmanager
from glob import glob
from os import remove
from os.path import basename, exists, join
from pathlib import Path
from typing import Any, Dict, Optional, Union

from scm_ds.random_scm import RandomSCMConfig, sample_random_scm_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECIPE_FILENAME = "dag_recipe.json"

#: Sentinel file proving a dataset was fully generated at least once.
#: (``ds.npz`` cannot serve as the sentinel because it is pruned.)
SENTINEL_FILENAME = "dataset_metadata.json"

#: Heavy artefacts removed by :func:`prune_dag_arrays`.
HEAVY_GLOBS = ("ds.npz", "ds_train.npz", "ds_test.npz")

DEFAULT_N_SAMPLES = 20_000
DEFAULT_NORMALIZE_METHOD = "minmax"
DEFAULT_MODE = "flat"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _random_scm_field_names() -> set:
    """Field names accepted by ``RandomSCMConfig`` (robust to schema changes)."""
    return {f.name for f in dataclasses.fields(RandomSCMConfig)}


def split_dag_block(dag_block: Dict[str, Any]) -> tuple:
    """
    Split a ``dag:`` config block into SCM-sampling fields and generation kwargs.

    The ``dag:`` block in ``dagsweep.yaml`` mixes two concerns:

    * fields of ``RandomSCMConfig``  (``n_nodes``, ``degree``, ``linearity``, ...)
    * dataset-generation options     (``n_samples``, ``normalize_method``, ``mode``)

    Splitting them here keeps the YAML flat and user-friendly while staying
    robust if ``RandomSCMConfig`` gains or loses fields.

    Returns:
        ``(scm_fields, gen_kwargs)``

    Raises:
        ValueError: on keys that belong to neither group (typo protection - a
            silently ignored ``n_node`` would invalidate a whole sweep).
    """
    scm_names = _random_scm_field_names()
    gen_names = {"n_samples", "normalize_method", "mode", "normalize", "compute_ate"}


    scm_fields: Dict[str, Any] = {}
    gen_kwargs: Dict[str, Any] = {}
    unknown = []

    for key, value in (dag_block or {}).items():
        if key in scm_names:
            scm_fields[key] = value
        elif key in gen_names:
            gen_kwargs[key] = value
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(
            f"Unknown key(s) in the 'dag' config block: {sorted(unknown)}. "
            f"Valid RandomSCMConfig fields: {sorted(scm_names)}. "
            f"Valid generation options: {sorted(gen_names)}."
        )

    return scm_fields, gen_kwargs


def build_dag_config(dag_block: Dict[str, Any], seed: int, **overrides) -> RandomSCMConfig:
    """
    Build a ``RandomSCMConfig`` from a ``dag:`` block for one specific seed.

    Args:
        dag_block: The ``dag:`` mapping from ``dagsweep.yaml``.
        seed: Sampling seed (this is what distinguishes DAGs within a group).
        **overrides: Group-axis values, e.g. ``n_nodes=50``.
    """
    scm_fields, _ = split_dag_block(dag_block)
    scm_fields.update(overrides)
    scm_fields["seed"] = int(seed)
    # ``name`` is derived deterministically; never honour a user-supplied one,
    # otherwise every seed of a group would collide in the same folder.
    scm_fields.pop("name", None)
    return RandomSCMConfig(**scm_fields)


def dag_dataset_name(cfg: RandomSCMConfig) -> str:
    """
    Deterministic folder name for a sampled DAG.

    Delegates to ``scm_ds.random_scm``'s own default-naming so a dataset
    generated by the sweeper is indistinguishable from one generated by hand.
    """
    if getattr(cfg, "name", None):
        return str(cfg.name)
    try:
        from scm_ds.random_scm import _default_name  # type: ignore

        return _default_name(cfg)
    except Exception:
        # Defensive fallback: keep every identity-bearing field in the name.
        return (
            f"random_n{cfg.n_nodes}_k{cfg.degree}"
            f"_{getattr(cfg, 'linearity', 'na')}_{getattr(cfg, 'noise', 'na')}"
            f"_s{cfg.seed}"
        )


# ---------------------------------------------------------------------------
# Recipe persistence
# ---------------------------------------------------------------------------

def write_recipe(dataset_dir: Union[str, Path], cfg: RandomSCMConfig,
                 gen_kwargs: Dict[str, Any]) -> None:
    """Persist the exact sampling + generation recipe next to the dataset."""
    payload = {
        "random_scm_config": dataclasses.asdict(cfg),
        "generation": dict(gen_kwargs),
        "created_by": "causaliT.euler_sweep.dag_provider",
    }
    with open(join(str(dataset_dir), RECIPE_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def read_recipe(dataset_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load a ``dag_recipe.json``, or ``None`` when absent."""
    path = join(str(dataset_dir), RECIPE_FILENAME)
    if not exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Generation / pruning
# ---------------------------------------------------------------------------

def _filtered_generate_kwargs(gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only ``generate_ds`` kwargs that the installed ``SCMDataset`` accepts.

    Guards against version drift between ``scm_ds`` and this module.
    """
    from scm_ds.scm import SCMDataset

    accepted = set(inspect.signature(SCMDataset.generate_ds).parameters)
    return {k: v for k, v in gen_kwargs.items() if k in accepted}


def has_arrays(dataset_dir: Union[str, Path]) -> bool:
    """True when at least one heavy sample array is present."""
    return any(exists(join(str(dataset_dir), name)) for name in HEAVY_GLOBS)


def is_materialized(dataset_dir: Union[str, Path]) -> bool:
    """True when the dataset was fully generated at least once (light artefacts present)."""
    return exists(join(str(dataset_dir), SENTINEL_FILENAME))


def prune_dag_arrays(dataset_dir: Union[str, Path], verbose: bool = True) -> int:
    """
    Delete only the heavy sample arrays, keeping every light artefact.

    Retained on purpose (needed by evaluation / reproducibility):
    ``dataset_metadata.json``, ``*_att_mask.csv`` / ``dag_adj_mask.csv``,
    ``ate_ground_truth.json``, ``normalization.json``, ``*_vars_map.json``,
    ``*_feat_map.json``, ``meta.json``, ``graph.pdf``, ``dag_recipe.json``.

    Returns:
        Number of bytes reclaimed.
    """
    dataset_dir = str(dataset_dir)
    freed = 0
    for pattern in HEAVY_GLOBS:
        for path in glob(join(dataset_dir, pattern)):
            try:
                freed += Path(path).stat().st_size
                remove(path)
            except OSError as exc:  # pragma: no cover - platform dependent
                print(f"  [dag_provider] could not remove {path}: {exc}")
    if verbose and freed:
        print(f"  [dag_provider] pruned arrays in {basename(dataset_dir)} "
              f"({freed / 1e6:.1f} MB reclaimed)")
    return freed


def generate_dag_dataset(cfg: RandomSCMConfig, data_root: Union[str, Path],
                         gen_kwargs: Optional[Dict[str, Any]] = None,
                         verbose: bool = True) -> str:
    """
    Sample a DAG and materialize the full dataset under ``data_root/<name>/``.

    Returns:
        The dataset folder name (to be used as ``config.data.dataset``).
    """
    gen_kwargs = dict(gen_kwargs or {})
    name = dag_dataset_name(cfg)
    dataset_dir = join(str(data_root), name)

    n_samples = int(gen_kwargs.pop("n_samples", DEFAULT_N_SAMPLES))
    mode = gen_kwargs.pop("mode", DEFAULT_MODE)
    gen_kwargs.setdefault("normalize_method", DEFAULT_NORMALIZE_METHOD)
    # Sampled DAGs are scored on structure recovery (SHD), not on ATE: skip the two
    # 50k-sample Monte Carlo passes unless explicitly requested from the config.
    gen_kwargs.setdefault("compute_ate", False)

    # Reuse the DAG seed for sampling so the whole dataset is a pure function
    # of (dag_cfg, seed) - no hidden state, fully regenerable.
    gen_kwargs.setdefault("seed", int(cfg.seed))

    if verbose:
        print(f"  [dag_provider] generating {name} (n={n_samples}, mode={mode})")

    dataset = sample_random_scm_dataset(cfg)
    dataset.generate_ds(
        mode=mode,
        n=n_samples,
        save_dir=dataset_dir,
        **_filtered_generate_kwargs(gen_kwargs),
    )

    write_recipe(dataset_dir, cfg, {"n_samples": n_samples, "mode": mode, **gen_kwargs})
    return name


def ensure_dag_dataset(cfg: RandomSCMConfig, data_root: Union[str, Path],
                       gen_kwargs: Optional[Dict[str, Any]] = None,
                       force: bool = False, verbose: bool = True) -> str:
    """
    Idempotently make a DAG dataset available (arrays included).

    Three cases:

    * fully present (light artefacts **and** arrays) -> reuse as-is;
    * previously pruned (light artefacts, no arrays) -> regenerate
      deterministically from ``(cfg, gen_kwargs)``;
    * absent -> generate from scratch.

    Returns:
        The dataset folder name.
    """
    name = dag_dataset_name(cfg)
    dataset_dir = join(str(data_root), name)

    if not force and is_materialized(dataset_dir) and has_arrays(dataset_dir):
        if verbose:
            print(f"  [dag_provider] reusing {name}")
        return name

    if not force and is_materialized(dataset_dir):
        if verbose:
            print(f"  [dag_provider] {name} was pruned - regenerating arrays")

    return generate_dag_dataset(cfg, data_root, gen_kwargs, verbose=verbose)


def regenerate_from_recipe(dataset_dir: Union[str, Path], verbose: bool = True) -> str:
    """
    Restore a pruned dataset from its ``dag_recipe.json``.

    Use this before re-running a forward-pass evaluation (``eval_attention_scores``,
    ``eval_ate_mc``) on a run whose arrays were pruned.
    """
    dataset_dir = str(dataset_dir)
    recipe = read_recipe(dataset_dir)
    if recipe is None:
        raise FileNotFoundError(
            f"No {RECIPE_FILENAME} in {dataset_dir}; cannot regenerate. "
            "This dataset was not produced by the DAG sweeper."
        )

    scm_fields = dict(recipe.get("random_scm_config") or {})
    # Drop the persisted name so the folder name is re-derived identically.
    scm_fields.pop("name", None)
    known = _random_scm_field_names()
    cfg = RandomSCMConfig(**{k: v for k, v in scm_fields.items() if k in known})

    return generate_dag_dataset(
        cfg,
        data_root=str(Path(dataset_dir).parent),
        gen_kwargs=dict(recipe.get("generation") or {}),
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Scoped materialization
# ---------------------------------------------------------------------------

@contextmanager
def materialized_dag(cfg: RandomSCMConfig, data_root: Union[str, Path],
                     gen_kwargs: Optional[Dict[str, Any]] = None,
                     delete_dataset: bool = True, verbose: bool = True):
    """
    Context manager yielding a ready-to-train dataset name, pruned on exit.

    Pruning happens **after** the body completes, which is exactly what the
    evaluation stack requires: post-training evaluations run inside
    ``trainer()`` and still need ``ds.npz``.

    Pruning also runs when the body raises, so a crashed run cannot leave a
    40 MB array behind - the light artefacts and the recipe survive either way.

    Example::

        with materialized_dag(cfg, data_root, gen, delete_dataset=True) as name:
            config.data.dataset = name
            trainer(config, data_dir=data_root, save_dir=run_dir)
    """
    name = ensure_dag_dataset(cfg, data_root, gen_kwargs, verbose=verbose)
    dataset_dir = join(str(data_root), name)
    try:
        yield name
    finally:
        if delete_dataset:
            prune_dag_arrays(dataset_dir, verbose=verbose)
        elif verbose:
            print(f"  [dag_provider] keeping arrays for {name} (delete_dataset=False)")


__all__ = [
    "RECIPE_FILENAME",
    "build_dag_config",
    "dag_dataset_name",
    "ensure_dag_dataset",
    "generate_dag_dataset",
    "has_arrays",
    "is_materialized",
    "materialized_dag",
    "prune_dag_arrays",
    "read_recipe",
    "regenerate_from_recipe",
    "split_dag_block",
    "write_recipe",
]

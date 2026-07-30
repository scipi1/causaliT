"""
Data-root resolution for evaluation functions.

Historically every evaluation function hardcoded ``join(root_path, "data")``,
which silently pinned all evaluation to the single shared ``data/`` folder.
DAG sweeps break that assumption: each group keeps its sampled datasets inside
the experiment folder (``groups/<group>/datasets/``) so that nothing pollutes
``data/`` and the heavy arrays can be pruned per run.

:func:`resolve_datadir` centralises the lookup with an explicit precedence and
stays backward compatible - when nothing else is available it still returns
``<repo>/data``.
"""

from __future__ import annotations

import glob
from os.path import abspath, dirname, exists, isabs, join
from typing import Any, Optional

from omegaconf import OmegaConf


def _repo_data_dir() -> str:
    """The legacy default: ``<repo>/data``."""
    try:
        from causaliT.paths import DATA_DIR

        return str(DATA_DIR)
    except Exception:  # pragma: no cover - fallback for odd installs
        here = dirname(dirname(dirname(dirname(abspath(__file__)))))
        return join(here, "data")


def _config_data_root(config: Any, experiment: Optional[str]) -> Optional[str]:
    """Read ``data.data_root`` from a config, resolved against the run folder."""
    if config is None:
        return None
    try:
        data_root = config.get("data", {}).get("data_root", None)
    except Exception:
        return None
    if not data_root:
        return None

    data_root = str(data_root)
    if not isabs(data_root) and experiment:
        # Relative roots are interpreted w.r.t. the run folder, which keeps a
        # run directory relocatable (e.g. copied off a cluster scratch disk).
        candidate = abspath(join(str(experiment), data_root))
        if exists(candidate):
            return candidate
    return data_root


def _load_run_config(experiment: str) -> Optional[Any]:
    """Load the ``config*.yaml`` saved inside a run folder, if any."""
    matches = sorted(glob.glob(join(str(experiment), "config*.yaml")))
    if not matches:
        return None
    try:
        return OmegaConf.load(matches[0])
    except Exception:
        return None


def resolve_datadir(config: Any = None, experiment: Optional[str] = None,
                    explicit: Optional[str] = None) -> str:
    """
    Resolve the dataset root for an evaluation.

    Precedence (first hit wins):

    1. ``explicit`` - an argument passed by the caller (e.g. the ``datadir_path``
       that ``trainer()`` threads into its post-training evaluations);
    2. ``config.data.data_root`` - written by the DAG sweeper so a run knows
       where its own datasets live;
    3. the run's saved ``config*.yaml`` (same key) when no config was passed -
       this makes re-running an evaluation from a notebook "just work";
    4. ``<repo>/data`` - the historical default.

    Args:
        config: Optional (Omega)config of the run.
        experiment: Optional path to the run folder (used for relative roots
            and for the config fallback).
        explicit: Optional caller-supplied path that overrides everything.

    Returns:
        An absolute-or-config-provided dataset root path.
    """
    if explicit:
        return str(explicit)

    from_config = _config_data_root(config, experiment)
    if from_config:
        return from_config

    if config is None and experiment:
        from_run = _config_data_root(_load_run_config(experiment), experiment)
        if from_run:
            return from_run

    return _repo_data_dir()


__all__ = ["resolve_datadir"]

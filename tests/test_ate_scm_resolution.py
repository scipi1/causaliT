"""
Regression tests for SCM resolution in the ATE evaluation.

Bug (experiments/7_PUBLISH/ATE/*): the sweepers name dataset folders with the
registry keys ("ds_scm3_continuous"), while eval_interventions kept a *hardcoded
copy* of the registry using the old un-prefixed names ("scm3_continuous").
Nothing matched, so eval_ate_mc printed a warning, returned an EMPTY DataFrame,
and the evaluation wrapper happily reported "[OK] eval_interventions: success".

These tests pin both halves of the fix:
1. resolution works for registry keys, legacy aliases and recipe-backed folders;
2. an unresolvable SCM raises instead of returning an empty frame.
"""

import importlib
import json

import pytest
from omegaconf import OmegaConf

from scm_ds.datasets import DATASET_REGISTRY

# Grab the MODULE, not the function: the package __init__ re-exports the
# `eval_interventions` *function* under the module's own name, which shadows it
# for both `import ... as` and `from ... import`.
ei = importlib.import_module("causaliT.evaluation.eval_funs.eval_interventions")




# ---------------------------------------------------------------------------
# 1. Resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["ds_scm1", "ds_scm1_continuous", "ds_scm2_continuous", "ds_scm3_continuous"],
)
def test_registry_keys_resolve(name):
    """The names used by the sweepers (== registry keys) must resolve."""
    assert ei.get_scm_for_dataset(name) is DATASET_REGISTRY[name]


@pytest.mark.parametrize(
    "legacy,expected",
    [
        ("scm1", "ds_scm1"),
        ("scm3_continuous", "ds_scm3_continuous"),
    ],
)
def test_legacy_unprefixed_names_still_resolve(legacy, expected):
    """Old un-prefixed names remain supported (older runs / notebooks)."""
    assert ei.get_scm_for_dataset(legacy) is DATASET_REGISTRY[expected]


def test_scm_recipe_fallback(tmp_path):
    """A folder name that is not a registry key resolves via scm_recipe.json."""
    datadir = tmp_path / "data"
    ds_dir = datadir / "some_aliased_folder"
    ds_dir.mkdir(parents=True)
    (ds_dir / "scm_recipe.json").write_text(
        json.dumps({"registry_name": "ds_scm1_continuous",
                    "generation": {"n": 10, "mode": "flat"}}),
        encoding="utf-8",
    )

    scm = ei.get_scm_for_dataset("some_aliased_folder", datadir_path=str(datadir))
    assert scm is DATASET_REGISTRY["ds_scm1_continuous"]


def test_unresolvable_dataset_raises(tmp_path):
    """No registry key and no recipe -> explicit, informative ValueError."""
    with pytest.raises(ValueError) as excinfo:
        ei.get_scm_for_dataset("ds_does_not_exist", datadir_path=str(tmp_path))

    msg = str(excinfo.value)
    assert "ds_does_not_exist" in msg
    assert "scm_recipe.json" in msg
    # The available keys must be the real registry ones, not a stale copy.
    assert "ds_scm3_continuous" in msg


# ---------------------------------------------------------------------------
# 2. Failures must be loud
# ---------------------------------------------------------------------------

def test_eval_ate_mc_raises_on_missing_scm(tmp_path, monkeypatch):
    """
    eval_ate_mc must propagate the unresolvable-SCM error.

    Previously it caught the ValueError and returned an empty DataFrame, which
    the evaluation wrapper recorded as a success.
    """
    experiment = tmp_path / "run"
    experiment.mkdir()
    OmegaConf.save(
        OmegaConf.create({"data": {"dataset": "ds_totally_unknown"}}),
        str(experiment / "config_atsel.yaml"),
    )

    datadir = tmp_path / "data"
    (datadir / "ds_totally_unknown").mkdir(parents=True)

    # Bypass the datadir/metadata lookups: only SCM resolution is under test.
    monkeypatch.setattr(ei, "resolve_datadir", lambda **kwargs: str(datadir))
    monkeypatch.setattr(
        ei, "load_dataset_metadata",
        lambda *args, **kwargs: {"variable_info": {"source_labels": [],
                                                   "input_labels": []}},
    )

    with pytest.raises(ValueError, match="ds_totally_unknown"):
        ei.eval_ate_mc(str(experiment))


def test_eval_interventions_alias_propagates(tmp_path, monkeypatch):
    """The eval_interventions alias must not swallow the error either."""
    experiment = tmp_path / "run"
    experiment.mkdir()
    OmegaConf.save(
        OmegaConf.create({"data": {"dataset": "ds_totally_unknown"}}),
        str(experiment / "config_atsel.yaml"),
    )

    monkeypatch.setattr(ei, "resolve_datadir", lambda **kwargs: str(tmp_path / "data"))
    monkeypatch.setattr(
        ei, "load_dataset_metadata",
        lambda *args, **kwargs: {"variable_info": {"source_labels": [],
                                                   "input_labels": []}},
    )

    with pytest.raises(ValueError):
        ei.eval_interventions(str(experiment))

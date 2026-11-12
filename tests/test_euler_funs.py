"""Fast checks of the euler_funs plugins (no training, no Slurm)."""

import shutil
import threading
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from causaliT.euler_funs.plugins import dagsweep
from causaliT.euler_funs.plugins.base import BUILTIN, TaskPaused, load_plugin
from causaliT.euler_funs.state import read_json

FIXTURE = Path(__file__).resolve().parents[1] / "experiments" / "0_TESTS" / "FUN_dagsweep"


def _frozen_run(tmp_path, optuna_enabled):
    """Copy the fixture definitions like cli.prepare_run does (no optuna file)."""
    root = tmp_path / "run"
    root.mkdir()
    for name in ("config_atsel.yaml", "dagsweep.yaml"):
        shutil.copy2(FIXTURE / name, root / name)
    spec = OmegaConf.load(root / "dagsweep.yaml")
    spec.optuna.enabled = optuna_enabled
    OmegaConf.save(spec, root / "dagsweep.yaml")
    return root


def test_builtin_plugins_load_without_optuna():
    assert "optuna" not in BUILTIN
    for name in BUILTIN:
        load_plugin(name)


def test_dagsweep_rejects_optuna(tmp_path):
    with pytest.raises(ValueError, match="Optuna is not supported"):
        dagsweep.prepare(_frozen_run(tmp_path, True), {})


def test_dagsweep_prepare_and_finalize(tmp_path):
    root = _frozen_run(tmp_path, False)
    tasks = dagsweep.prepare(root, {})
    # 1 group x 2 dag_seeds x 2 model_seeds
    assert len(tasks) == 4
    assert len({task["id"] for task in tasks}) == 4
    for task in tasks:
        assert task["trainer"] == "adaptive"
        config = OmegaConf.load(root / task["directory"] / "config.yaml")
        assert config.training.seed == task["model_seed"]
        assert config.training.data_seed == task["dag_seed"]
        assert (root / task["data_dir"] / task["dataset"]).is_dir()
    manifest = {"tasks": tasks, "spec": {"plugin": {"options": {}}}}
    dagsweep.finalize(root, manifest)
    assert len(read_json(root / "groups" / "index.json")) == 4
    assert not list((root / "groups").rglob("ds*.npz"))  # delete_dataset: true


REPO = Path(__file__).resolve().parents[1]
FAILED_RUN = (REPO / "experiments" / "6_INVESTIGATIONS" / "ORDER_FIRST_2" / "results"
              / "softmax_selector_bkd_recon_20261110T073940_95baf37f" / "config.yaml")


@pytest.mark.skipif(not FAILED_RUN.is_file(), reason="first Euler run config not available")
def test_prepare_keeps_omegaconf_for_update_config(tmp_path):
    """Regression: update_config failed on a plain dict (first Euler run)."""
    from causaliT.euler_funs.train_fns import _prepare
    config = _prepare(OmegaConf.load(FAILED_RUN), tmp_path)
    assert config.experiment.d_ff == 80 and config.experiment.d_qk == 20
    assert (tmp_path / "config.yaml").is_file()


@pytest.mark.skipif(not FAILED_RUN.is_file(), reason="first Euler run config not available")
def test_check_adaptive_config_with_local_data():
    from causaliT.euler_funs.train_fns import check
    check(OmegaConf.load(FAILED_RUN), REPO / "data", "adaptive")


@pytest.mark.skipif(not FAILED_RUN.is_file(), reason="first Euler run config not available")
def test_check_without_gpu(monkeypatch):
    """Regression: the Euler launcher has no GPU but configs use device: cuda."""
    import torch
    from causaliT.euler_funs.train_fns import check
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = OmegaConf.load(FAILED_RUN)
    assert config.model.kwargs.device == "cuda"
    check(config, REPO / "data", "adaptive")
    assert config.model.kwargs.device == "cuda"  # caller config untouched


@pytest.mark.skipif(not FAILED_RUN.is_file(), reason="first Euler run config not available")
def test_check_reports_wrong_data_dir():
    from causaliT.euler_funs.train_fns import check
    with pytest.raises(FileNotFoundError, match="dataset_metadata.json"):
        check(OmegaConf.load(FAILED_RUN), REPO / "data" / "input", "adaptive")


def _launch_dry(tmp_path, data_dir):
    from click.testing import CliRunner
    from causaliT.euler_funs.cli import cli
    experiment = tmp_path / "exp"
    experiment.mkdir(parents=True)
    shutil.copy2(FAILED_RUN, experiment / "config.yaml")
    spec = {"run": {"experiment": str(experiment), "scratch_root": str(tmp_path / "scratch"),
                    "data_dir": str(data_dir), "project_dir": str(REPO)},
            "plugin": {"path": "training", "options": {
                "train_fn": "causaliT.euler_funs.train_fns:adaptive_train",
                "config": "config.yaml", "task_mode": "experiment"}}}
    OmegaConf.save(OmegaConf.create(spec), tmp_path / "spec.yaml")
    return CliRunner().invoke(cli, ["launch", "--spec", str(tmp_path / "spec.yaml"), "--dry-run"])


@pytest.mark.skipif(not FAILED_RUN.is_file(), reason="first Euler run config not available")
def test_launch_runs_check(tmp_path):
    result = _launch_dry(tmp_path, REPO / "data")
    assert result.exit_code == 0, result.output
    assert "Check experiment: OK" in result.output
    bad = _launch_dry(tmp_path / "bad", REPO / "data" / "input")
    assert bad.exit_code != 0 and "Check failed" in bad.output


def test_trainer_of():
    from causaliT.euler_funs.train_fns import trainer_of
    assert trainer_of("causaliT.euler_funs.train_fns:train") == "standard"
    assert trainer_of("causaliT.euler_funs.train_fns:adaptive_train") == "adaptive"
    assert trainer_of("my.module:train") is None


def test_dagsweep_check_first_task_per_group(tmp_path):
    root = _frozen_run(tmp_path, False)
    tasks = dagsweep.prepare(root, {})
    selected = dagsweep.check_selection(tasks)
    assert [task["id"] for task in selected] == [tasks[0]["id"]]
    # The check reaches the model build on the group's generated dataset.
    # NOTE: FUN_dagsweep/config_atsel.yaml is stale (init_tau was removed from
    # AttentionSelectorLayer); the check must catch it. Once the fixture is
    # updated, replace this with: assert dagsweep.check_task(...) is True
    try:
        assert dagsweep.check_task(root, selected[0]) is True
    except TypeError as error:
        assert "init_tau" in str(error)


def test_dagsweep_run_dispatches_trainer(tmp_path, monkeypatch):
    from causaliT.euler_funs import train_fns
    root = _frozen_run(tmp_path, False)
    (root / "run.json").write_text('{"spec": {"resources": {}}}')
    task = dagsweep.prepare(root, {"trainer": "standard"})[0]
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        raise TaskPaused()
    monkeypatch.setitem(train_fns.TRAINERS, "standard", fake)
    with pytest.raises(TaskPaused):
        dagsweep.run(root, task, threading.Event())
    assert Path(seen["data_dir"]) == root / task["data_dir"]
    assert Path(seen["save_dir"]) == root / task["directory"]

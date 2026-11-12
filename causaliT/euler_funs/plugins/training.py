"""Plain training of one frozen config; optionally one task per fold.

options:
  train_fn: "module:function"   required, see call_train for the signature
  config: config.yaml           frozen config file, relative to the run
  task_mode: experiment | fold  fold needs folds_key
  folds_key: training.k_fold    dotted key holding the number of folds
"""

from pathlib import Path

from omegaconf import OmegaConf

from ..state import read_json
from .base import load_object


def call_train(root, save_dir, config, stop_event, folds=None):
    """Invoke the project train function with a stable keyword signature.

    train_fn(config, save_dir, data_dir, stop_event, folds, resources) -> results
    It must resume from its own checkpoints in save_dir and raise TaskPaused
    if it stops before completion.
    """
    manifest = read_json(Path(root) / "run.json")
    train_fn = load_object(manifest["spec"]["plugin"]["options"]["train_fn"])
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return train_fn(config=config, save_dir=str(save_dir), data_dir=manifest["spec"]["run"]["data_dir"],
                    stop_event=stop_event, folds=folds, resources=manifest["spec"]["resources"])


def prepare(root, options):
    if "train_fn" not in options:
        raise ValueError("plugin.options.train_fn is required")
    mode = options.get("task_mode", "experiment")
    if mode == "experiment":
        return [{"id": "experiment", "folds": None}]
    if mode != "fold":
        raise ValueError("task_mode must be experiment or fold")
    config = OmegaConf.load(Path(root) / options.get("config", "config.yaml"))
    count = int(OmegaConf.select(config, options["folds_key"]))
    return [{"id": f"fold_{fold}", "folds": [fold]} for fold in range(count)]


def check_config(root, config):
    """Pre-flight for the causaliT train_fns; custom train_fns are not checked."""
    from ..train_fns import check, trainer_of
    spec = read_json(Path(root) / "run.json")["spec"]
    trainer = trainer_of(spec["plugin"]["options"]["train_fn"])
    if trainer is None:
        return False
    check(config, spec["run"]["data_dir"], trainer)
    return True


def check_task(root, task):
    options = read_json(Path(root) / "run.json")["spec"]["plugin"]["options"]
    return check_config(root, OmegaConf.load(Path(root) / options.get("config", "config.yaml")))


def run(root, task, stop_event):
    options = read_json(Path(root) / "run.json")["spec"]["plugin"]["options"]
    config = OmegaConf.load(Path(root) / options.get("config", "config.yaml"))
    call_train(root, root, config, stop_event, task["folds"])

"""Parameter sweep: every combination is one task with its own frozen config.

options:
  train_fn: "module:function"        see training.call_train
  config: config.yaml                base config, relative to the run
  sweep: sweeper/sweep.yaml          {category: {param: [values, ...]}}
  sweep_mode: combination | independent
Layout (same as before):
  sweeper/runs/combinations/combo_<p1>_<v1>_<p2>_<v2>/config.yaml
  sweeper/runs/sweeps/sweep_<p>/sweep_<p>_<v>/config.yaml
"""

import itertools
from pathlib import Path

from omegaconf import OmegaConf

from .training import call_train, check_config


def generate(config, sweep, mode):
    """Yield (relative directory, {(category, param): value})."""
    entries = [(category, param, list(values))
               for category in sweep for param, values in sweep[category].items()]
    for category, param, _ in entries:
        if OmegaConf.select(config, f"{category}.{param}") is None:
            raise KeyError(f"{category}.{param} not found in base config")
    if mode == "independent":
        for category, param, values in entries:
            for value in values:
                yield (Path("sweeper/runs/sweeps") / f"sweep_{param}" / f"sweep_{param}_{value}",
                       {(category, param): value})
    elif mode == "combination":
        for values in itertools.product(*(entry[2] for entry in entries)):
            name = "combo_" + "_".join(f"{p}_{v}" for (_, p, _), v in zip(entries, values))
            yield (Path("sweeper/runs/combinations") / name,
                   {(c, p): v for (c, p, _), v in zip(entries, values)})
    else:
        raise ValueError("sweep_mode must be combination or independent")


def prepare(root, options):
    root = Path(root)
    config = OmegaConf.load(root / options.get("config", "config.yaml"))
    sweep = OmegaConf.to_container(OmegaConf.load(root / options.get("sweep", "sweeper/sweep.yaml")))
    tasks, seen = [], set()
    for directory, params in generate(config, sweep, options.get("sweep_mode", "combination")):
        if any(char in directory.name for char in "/\\") or directory in seen:
            continue  # duplicates arise when a value equals the baseline
        seen.add(directory)
        resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        for (category, param), value in params.items():
            OmegaConf.update(resolved, f"{category}.{param}", value)
        (root / directory).mkdir(parents=True)
        OmegaConf.save(resolved, root / directory / "config.yaml")
        tasks.append({"id": f"combination_{len(tasks)}", "directory": directory.as_posix()})
    return tasks


def check_task(root, task):
    directory = Path(root) / task["directory"]
    return check_config(root, OmegaConf.load(directory / "config.yaml"))


def run(root, task, stop_event):
    directory = Path(root) / task["directory"]
    call_train(root, directory, OmegaConf.load(directory / "config.yaml"), stop_event)


def finalize(root, manifest):
    """Write a sweep index mapping task ids to folders."""
    index = {task["id"]: task["directory"] for task in manifest["tasks"]}
    OmegaConf.save(OmegaConf.create(index), Path(root) / "sweeper" / "index.yaml")

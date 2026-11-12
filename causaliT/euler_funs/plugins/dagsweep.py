"""DAG sweep without Optuna: one task per (group, dag_seed, resample, model_seed).

options:
  trainer: standard | adaptive   default: dagsweep*.yaml training.trainer
  keep_data: false               true keeps the dataset arrays at finalize
The frozen run must contain the base config*.yaml and one dagsweep*.yaml
(or atesweep*.yaml). Data are generated inside the run at prepare time:
  groups/<group>/datasets/...
  groups/<group>/sweeper/runs/combinations/<run_name>/config.yaml
Runs train the base config only; an enabled optuna block is rejected.
"""

from pathlib import Path
import re

from omegaconf import OmegaConf

from ..state import read_json, write_json

TRAINERS = ("standard", "adaptive")


def _sweep_modules():
    # Reused setup code of the legacy sweep (Optuna-free functions only).
    from causaliT.euler_sweep.euler_sweep import dag_provider, data_source, opt_train_sweep
    return dag_provider, data_source, opt_train_sweep


def prepare(root, options):
    root = Path(root)
    _, data_source, ots = _sweep_modules()
    spec = ots.load_dagsweep_spec(str(root))
    optuna = ots._as_dict(spec.get("optuna", {}))
    if optuna and bool(optuna.get("enabled", True)):
        raise ValueError("dagsweep: Optuna is not supported; remove the optuna block "
                         "or set optuna.enabled: false")
    trainer = str(options.get("trainer")
                  or ots._as_dict(spec.get("training", {})).get("trainer", "standard"))
    if trainer not in TRAINERS:
        raise ValueError(f"dagsweep: trainer must be one of {TRAINERS}, got {trainer}")
    base_config, _ = ots._find_base_files(str(root))
    if trainer == "adaptive" and not OmegaConf.select(
            OmegaConf.load(base_config), "training.use_gradient_routing", default=False):
        raise ValueError("dagsweep: trainer 'adaptive' needs training.use_gradient_routing=true")

    source = data_source.build_data_source(spec)
    tasks, datasets = [], {}
    for group in ots.build_groups(spec, source):
        group_dir = root / "groups" / group["name"]
        datasets_dir = group_dir / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        group_spec = {"axes": group["axes"],
                      "dag_overrides": group.get("dag_overrides") or {},
                      "config_overrides": group.get("config_overrides") or {},
                      "bundle_axes": group.get("bundle_axes") or []}
        for dag_seed, resample_seed, model_seed, _, suffix in ots.run_plan(group):
            name = source.ensure(group_spec, str(datasets_dir), int(dag_seed),
                                 resample_seed=resample_seed)
            datasets.setdefault(datasets_dir.relative_to(root).as_posix(), set()).add(name)
            config = ots.stage_run_config(base_config, group, spec, name, str(datasets_dir),
                                          dag_seed, model_seed)
            run_name = f"{group['name']}_{suffix}"
            directory = group_dir / "sweeper" / "runs" / "combinations" / run_name
            directory.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, directory / "config.yaml")
            tasks.append({"id": re.sub(r"[^A-Za-z0-9_.-]", "-", run_name), "group": group["name"],
                          "directory": directory.relative_to(root).as_posix(),
                          "data_dir": datasets_dir.relative_to(root).as_posix(),
                          "trainer": trainer, "dataset": name, "dag_seed": int(dag_seed),
                          "resample_seed": resample_seed, "model_seed": model_seed})
    write_json(root / "groups" / "datasets.json",
               {key: sorted(value) for key, value in datasets.items()})
    return tasks


def check_task(root, task):
    from ..train_fns import check
    root = Path(root)
    check(OmegaConf.load(root / task["directory"] / "config.yaml"),
          str(root / task["data_dir"]), task["trainer"])
    return True


def check_selection(tasks):
    """First task of each group: groups differ in size, seeds do not."""
    first = {}
    for task in tasks:
        first.setdefault(task["group"], task)
    return list(first.values())


def run(root, task, stop_event):
    from ..train_fns import TRAINERS as FUNCTIONS
    root = Path(root)
    directory = root / task["directory"]
    # Trainer chosen per task; data_dir is the group's own dataset folder
    # (not run.data_dir as in plugins.training.call_train).
    FUNCTIONS[task["trainer"]](config=OmegaConf.load(directory / "config.yaml"),
                               save_dir=str(directory), data_dir=str(root / task["data_dir"]),
                               stop_event=stop_event, folds=None,
                               resources=read_json(root / "run.json")["spec"]["resources"])


def finalize(root, manifest):
    """Write the run index and prune dataset arrays (recipes are kept)."""
    root = Path(root)
    index = {task["id"]: {k: task[k] for k in ("group", "directory", "dataset", "dag_seed",
                                                "resample_seed", "model_seed", "trainer")}
             for task in manifest["tasks"]}
    write_json(root / "groups" / "index.json", index)
    if manifest["spec"]["plugin"]["options"].get("keep_data"):
        return
    dag_provider, _, ots = _sweep_modules()
    if not bool(ots.load_dagsweep_spec(str(root)).get("delete_dataset", True)):
        return
    for datasets_dir, names in read_json(root / "groups" / "datasets.json").items():
        for name in names:
            dag_provider.prune_dag_arrays(root / datasets_dir / name, verbose=False)

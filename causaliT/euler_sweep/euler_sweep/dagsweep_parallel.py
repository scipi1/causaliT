"""
Parallel (SLURM) execution of the grouped DAG sweep.

Why a chain of jobs
-------------------
``opt_train_sweep.run_dag_sweep`` is intrinsically two-phase: phase 2 must train
with the hyper-parameters phase 1 selected, so the seed sweep cannot start before
the study of its group is finished.  Inside each phase, however, the work is
embarrassingly parallel: trials do not talk to each other (they synchronise
through the study database) and neither do runs.

This module therefore turns one sweep into FIVE SLURM jobs chained by
``--dependency``, with a single global barrier between the phases (the simplest
correct option: every trial of every group is done before any training starts)::

    prep (CPU)                      generate all DAGs, stage group configs,
      |                             create the study DBs, write the plan
      v
    trials[0..T-1%C] (1 GPU each)   one array task = ONE Optuna trial
      |            (afterany)
      v
    select (CPU)                    per group: select_best -> best_trial.yaml
      |            (afterok)
      v
    train[0..R-1%C] (1 GPU each)    one array task = ONE (dag_seed, model_seed) run
      |            (afterany)
      v
    cleanup (CPU)                   prune ds*.npz, roll up the progress report

Design decisions that make this safe
------------------------------------
* **Datasets are generated only in ``prep``.**  Two array tasks that share a DAG
  (several model seeds, or several trials) would otherwise race on the same
  folder.  As a consequence pruning moves to ``cleanup``: peak disk holds every
  dataset of the sweep, which is the price of running the runs concurrently.
* **``select`` never aborts the chain.**  It finalizes each group independently;
  a group whose study produced no COMPLETE trial is recorded as failed and its
  runs are skipped, instead of raising and cancelling the whole train array
  (which depends on it with ``afterok``).
* **One progress file per item.**  Array tasks never write the same file, so no
  locking is needed and nothing is lost when the walltime kills a task.  The
  rollup ``dagsweep_progress.json`` (planned vs reached) is rebuilt from those
  files by ``prep``/``select``/``cleanup`` and by ``cli dagsweep-status``.
* **The plan is static.**  Array sizes must be known at submit time, so the
  driver derives them from the spec alone (no dataset needed): ``T = n_groups *
  n_trials``, ``R = n_groups * |dag_seeds| * |model_seeds|``.  ``prep`` only adds
  the resolved dataset names / ``n_keys`` (``prepared.json``).
* **Config staging is shared** with the sequential path
  (``opt_train_sweep.stage_run_config``), so a parallel run trains exactly what a
  sequential run would.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from datetime import datetime
from glob import glob
from os.path import basename, exists, join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf

from causaliT.euler_sweep.euler_sweep.dag_provider import (
    build_dag_config,
    ensure_dag_dataset,
    prune_dag_arrays,
    split_dag_block,
)
from causaliT.euler_sweep.euler_sweep.opt_train_sweep import (
    DEFAULT_OPT_SEED,
    _as_dict,
    _find_base_files,
    build_group_study,
    build_groups,
    finalize_group_study,
    load_best_params,
    load_dagsweep_spec,
    resolve_trainer,
    run_plan,
    stage_group_config,
    stage_run_config,
)
from causaliT.euler_sweep.euler_sweep.search_space import n_keys_from_metadata
from causaliT.euler_sweep.euler_sweep.sweeper import run_single_combination

logger = logging.getLogger(__name__)

# Everything this module writes lives in ONE folder, so a sweep can be inspected
# (or deleted) without touching the results.
STATE_DIRNAME = "dagsweep"
PLAN_FILENAME = "plan.json"
PREPARED_FILENAME = "prepared.json"
PROGRESS_FILENAME = "dagsweep_progress.json"

WORKER_MODULE = "causaliT.euler_sweep.euler_sweep.dagsweep_worker"


# =============================================================================
# State-folder helpers
# =============================================================================

def state_dir(exp_dir: str) -> Path:
    return Path(exp_dir) / STATE_DIRNAME


def plan_path(exp_dir: str) -> Path:
    return state_dir(exp_dir) / PLAN_FILENAME


def prepared_path(exp_dir: str) -> Path:
    return state_dir(exp_dir) / PREPARED_FILENAME


def progress_dir(exp_dir: str) -> Path:
    return state_dir(exp_dir) / "progress"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def load_plan(exp_dir: str) -> Dict[str, Any]:
    """Load the static plan written by the driver."""
    path = plan_path(exp_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No {PLAN_FILENAME} in {state_dir(exp_dir)}. Submit the sweep with "
            "`cli dagsweep --exp_id ... --cluster` first."
        )
    return _read_json(path)


def load_prepared(exp_dir: str) -> Dict[str, Any]:
    """Load the dataset/n_keys resolution written by the ``prepare`` stage."""
    path = prepared_path(exp_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No {PREPARED_FILENAME} in {state_dir(exp_dir)}: the prepare stage "
            "did not finish, so no dataset is guaranteed to exist."
        )
    return _read_json(path)


def record_progress(exp_dir: str, item_id: str, payload: Dict[str, Any]) -> None:
    """
    Write ONE item's state to its own file (never shared -> no lock, no races).

    ``item_id`` is stable across attempts (``trial_007``, ``run_<group>_<key>``,
    ``stage_prep``), so a re-submission overwrites the previous attempt instead of
    appending noise.
    """
    payload = dict(payload)
    payload.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    _write_json(progress_dir(exp_dir) / f"{item_id}.json", payload)


def read_progress_items(exp_dir: str) -> Dict[str, Dict[str, Any]]:
    """Read every per-item progress file, keyed by item id."""
    items: Dict[str, Dict[str, Any]] = {}
    for path in sorted(glob(join(str(progress_dir(exp_dir)), "*.json"))):
        try:
            items[basename(path)[:-len(".json")]] = _read_json(Path(path))
        except Exception:  # a task killed mid-write must not break the report
            continue
    return items


def run_item_id(group_name: str, run_key: Any) -> str:
    return f"run_{group_name}_{run_key}"


def trial_item_id(task_id: int) -> str:
    return f"trial_{int(task_id):04d}"


# =============================================================================
# Static plan
# =============================================================================

def _n_trials_of(exp_dir: str, base_optuna_path: Optional[str]) -> int:
    """Trial budget per group, read from the base ``optuna*.yaml``."""
    if base_optuna_path is None:
        return 0
    settings = _as_dict(OmegaConf.load(base_optuna_path))
    return int(settings.get("n_trials", 0))


def build_static_plan(exp_dir: str, home_exp_dir: str, cluster: bool = True,
                      keep_data: bool = False, skip_optuna: bool = False,
                      force_optuna: bool = False,
                      slurm_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Derive the whole job plan from the spec alone - no dataset required.

    This is what makes a single ``sbatch`` chain possible: the array sizes must be
    known before any DAG exists, so groups, runs and trial slots are expanded
    here, while ``prepare`` only resolves dataset names and ``n_keys``.

    Returns:
        The plan dict (also written to ``<exp_dir>/dagsweep/plan.json``).
    """
    spec = load_dagsweep_spec(exp_dir)
    groups = build_groups(spec)
    base_config_path, base_optuna_path = _find_base_files(exp_dir)

    opt_cfg = _as_dict(spec.get("optuna", {}))
    optuna_enabled = bool(opt_cfg.get("enabled", True)) and not skip_optuna
    n_trials = _n_trials_of(exp_dir, base_optuna_path) if optuna_enabled else 0
    if optuna_enabled and n_trials <= 0:
        raise ValueError(
            "optuna.enabled is true but the base optuna*.yaml declares no "
            "n_trials, so the trial array would be empty."
        )

    trainer_name = str(_as_dict(spec.get("training", {})).get("trainer", "standard"))
    resolve_trainer(trainer_name)  # fail now, not inside 40 array tasks

    exp_id = basename(str(home_exp_dir).rstrip("/\\"))

    plan_groups: List[Dict[str, Any]] = []
    trial_slots: List[Dict[str, Any]] = []
    train_slots: List[Dict[str, Any]] = []

    for group in groups:
        group_dir = Path(exp_dir) / "groups" / group["name"]
        datasets_dir = str(group_dir / "datasets")

        runs: List[Dict[str, Any]] = []
        for dag_seed, model_seed, run_key, suffix in run_plan(group):
            run_name = f"{exp_id}_{group['name']}_{suffix}"
            runs.append({
                "index": len(runs),
                "run_key": str(run_key),
                "dag_seed": int(dag_seed),
                "model_seed": None if model_seed is None else int(model_seed),
                "run_name": run_name,
                "save_dir": str(group_dir / "sweeper" / "runs" / "combinations" / run_name),
            })

        plan_groups.append({
            "name": group["name"],
            "axes": group["axes"],
            "dag_seeds": group["dag_seeds"],
            "model_seeds": group["model_seeds"],
            "group_dir": str(group_dir),
            "datasets_dir": datasets_dir,
            "n_trials": n_trials,
            "runs": runs,
        })

        for slot in range(n_trials):
            trial_slots.append({"task_id": len(trial_slots),
                                "group": group["name"], "slot": slot})
        for run in runs:
            train_slots.append({"task_id": len(train_slots),
                                "group": group["name"], "run_index": run["index"]})

    plan = {
        "experiment": exp_id,
        "exp_dir": str(exp_dir),
        "home_exp_dir": str(home_exp_dir),
        "base_config": base_config_path,
        "base_optuna": base_optuna_path,
        "cluster": bool(cluster),
        "keep_data": bool(keep_data),
        # `delete_dataset` is honoured in the CLEANUP stage only (concurrent runs
        # share datasets, so nothing may be pruned while the array is alive).
        "delete_dataset": bool(spec.get("delete_dataset", True)) and not keep_data,
        "trainer": trainer_name,
        "optuna": {
            "enabled": optuna_enabled,
            "declared_enabled": bool(opt_cfg.get("enabled", True)),
            "skip_optuna": bool(skip_optuna),
            "force_optuna": bool(force_optuna),
            "opt_seed": int(opt_cfg.get("opt_seed", DEFAULT_OPT_SEED)),
            "protocol": str(opt_cfg.get("protocol", "reconstruction")),
            "metric": str(opt_cfg.get("metric", "val_x_mae_mean")),
            "direction": str(opt_cfg.get("direction", "minimize")),
            "n_trials_per_group": n_trials,
        },
        "groups": plan_groups,
        "trial_slots": trial_slots,
        "train_slots": train_slots,
        "n_trial_tasks": len(trial_slots),
        "n_train_tasks": len(train_slots),
        "slurm": dict(slurm_params or {}),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(plan_path(exp_dir), plan)
    return plan


def _group_of(plan: Dict[str, Any], name: str) -> Dict[str, Any]:
    for group in plan["groups"]:
        if group["name"] == name:
            return group
    raise KeyError(f"Group '{name}' is not in the plan.")


# =============================================================================
# Progress rollup (planned vs reached)
# =============================================================================

def rebuild_progress(exp_dir: str) -> Dict[str, Any]:
    """
    Aggregate the per-item files into ``dagsweep_progress.json``.

    Answers the "walltime hit - what was planned and what was reached?" question
    without parsing any log: every group reports its trial and run counts plus
    whether it ended up tuned.
    """
    plan = load_plan(exp_dir)
    items = read_progress_items(exp_dir)

    groups: Dict[str, Any] = {}
    for group in plan["groups"]:
        trials = [v for k, v in items.items()
                  if k.startswith("trial_") and v.get("group") == group["name"]]
        runs = {}
        for run in group["runs"]:
            state = items.get(run_item_id(group["name"], run["run_key"]))
            runs[run["run_key"]] = {
                "state": (state or {}).get("state", "pending"),
                "dag_seed": run["dag_seed"],
                "model_seed": run["model_seed"],
                "save_dir": run["save_dir"],
                "error": (state or {}).get("error"),
            }
        group_state = items.get(f"group_{group['name']}", {})
        groups[group["name"]] = {
            "planned_trials": group["n_trials"],
            "trials": {
                "ok": sum(1 for t in trials if t.get("state") == "ok"),
                "failed": sum(1 for t in trials if t.get("state") == "failed"),
                "skipped": sum(1 for t in trials if t.get("state") == "skipped"),
            },
            "tuned": group_state.get("tuned"),
            "n_tuned_params": group_state.get("n_tuned_params"),
            "select_error": group_state.get("error"),
            "planned_runs": len(group["runs"]),
            "runs": {
                "ok": sum(1 for r in runs.values() if r["state"] == "ok"),
                "failed": sum(1 for r in runs.values() if r["state"] == "failed"),
                "running": sum(1 for r in runs.values() if r["state"] == "running"),
                "pending": sum(1 for r in runs.values() if r["state"] == "pending"),
            },
            "run_details": runs,
        }

    rollup = {
        "experiment": plan["experiment"],
        "exp_dir": plan["exp_dir"],
        "updated": datetime.now().isoformat(timespec="seconds"),
        "stages": {k: v for k, v in items.items() if k.startswith("stage_")},
        "planned": {"trials": plan["n_trial_tasks"], "runs": plan["n_train_tasks"]},
        "reached": {
            "trials": sum(g["trials"]["ok"] for g in groups.values()),
            "runs": sum(g["runs"]["ok"] for g in groups.values()),
        },
        "groups": groups,
    }
    _write_json(state_dir(exp_dir) / PROGRESS_FILENAME, rollup)
    return rollup


def format_progress(rollup: Dict[str, Any]) -> str:
    """Render the rollup as a compact planned-vs-reached table."""
    lines = [
        "=" * 68,
        f"DAG SWEEP STATUS - {rollup['experiment']}  ({rollup['updated']})",
        "=" * 68,
        f"trials {rollup['reached']['trials']}/{rollup['planned']['trials']} ok    "
        f"runs {rollup['reached']['runs']}/{rollup['planned']['runs']} ok",
        "-" * 68,
    ]
    for name, group in rollup["groups"].items():
        tuned = {True: "yes", False: "no", None: "?"}[group.get("tuned")]
        lines.append(
            f"{name}\n"
            f"  trials  {group['trials']['ok']}/{group['planned_trials']} ok, "
            f"{group['trials']['failed']} failed     tuned: {tuned}"
            + (f" ({group['n_tuned_params']} param(s))"
               if group.get("n_tuned_params") else "")
        )
        lines.append(
            f"  runs    {group['runs']['ok']}/{group['planned_runs']} ok, "
            f"{group['runs']['failed']} failed, {group['runs']['pending']} pending"
        )
        if group.get("select_error"):
            lines.append(f"  select ERROR: {group['select_error']}")
        for key, run in group["run_details"].items():
            if run["state"] == "failed":
                lines.append(f"    [failed] {key}: {run.get('error')}")
    lines.append("=" * 68)
    for stage, payload in rollup.get("stages", {}).items():
        lines.append(f"{stage}: {payload.get('state')} ({payload.get('timestamp')})")
    lines.append("=" * 68)
    return "\n".join(lines)


# =============================================================================
# Stage: PREPARE (CPU) - all dataset generation happens here
# =============================================================================

def prepare_stage(exp_dir: str) -> Dict[str, Any]:
    """
    Generate every DAG of the sweep, stage the group configs, create the studies.

    Centralising generation is what removes the races: no array task ever creates
    a dataset, so several trials (one DAG) or several model seeds (one DAG) can
    read the same folder concurrently.  ``ensure_dag_dataset`` is idempotent, so
    re-running this stage after a walltime kill only fills in what is missing.
    """
    plan = load_plan(exp_dir)
    record_progress(exp_dir, "stage_prep", {"state": "running"})

    try:
        spec = load_dagsweep_spec(exp_dir)
        base_config_path, base_optuna_path = _find_base_files(exp_dir)
        dag_block = _as_dict(spec.get("dag", {}))
        _, gen_kwargs = split_dag_block(dag_block)
        optuna_enabled = bool(plan["optuna"]["enabled"])
        opt_seed = int(plan["optuna"]["opt_seed"])

        prepared: Dict[str, Any] = {"groups": {}}

        for group in plan["groups"]:
            group_dir = Path(group["group_dir"])
            datasets_dir = group["datasets_dir"]
            Path(datasets_dir).mkdir(parents=True, exist_ok=True)
            logger.info("[prep] group %s", group["name"])

            entry: Dict[str, Any] = {"datasets": {}, "opt_dataset": None}

            # --- optimisation DAG + staged search config + study DB -----------
            if optuna_enabled:
                opt_cfg = build_dag_config(dag_block, seed=opt_seed, **group["axes"])
                opt_dataset = ensure_dag_dataset(opt_cfg, datasets_dir, gen_kwargs)
                n_keys = n_keys_from_metadata(
                    datasets_dir, opt_dataset,
                    fallback=group["axes"].get("n_nodes"),
                )
                entry["opt_dataset"] = opt_dataset
                entry["n_keys"] = int(n_keys)

                stage_group_config(
                    base_config_path, base_optuna_path, group_dir,
                    {"axes": group["axes"]}, spec, opt_dataset, datasets_dir,
                    n_keys=n_keys, search_protocol=plan["optuna"]["protocol"],
                )

                study, _m, _d, _s = build_group_study(
                    group_dir, datasets_dir, spec, plan["cluster"], int(n_keys)
                )
                if plan["optuna"]["force_optuna"] and exists(study.study_file_path):
                    # Otherwise the trial workers would see the budget already
                    # spent and the "forced" re-tune would be a no-op.
                    logger.info("[prep] force_optuna: dropping %s",
                                study.study_file_path)
                    Path(study.study_file_path).unlink()
                    best = group_dir / "best_trial.yaml"
                    if best.exists():
                        best.unlink()
                study.create()  # no-op message when it already exists

            # --- evaluation DAGs (one per dag_seed, shared by model seeds) -----
            for dag_seed in group["dag_seeds"]:
                cfg = build_dag_config(dag_block, seed=int(dag_seed), **group["axes"])
                name = ensure_dag_dataset(cfg, datasets_dir, gen_kwargs)
                entry["datasets"][str(dag_seed)] = name
                entry.setdefault("n_keys", int(n_keys_from_metadata(
                    datasets_dir, name, fallback=group["axes"].get("n_nodes"))))

            prepared["groups"][group["name"]] = entry

        _write_json(prepared_path(exp_dir), prepared)
        record_progress(exp_dir, "stage_prep", {"state": "ok"})
        rebuild_progress(exp_dir)
        return prepared
    except Exception as exc:
        record_progress(exp_dir, "stage_prep", {"state": "failed", "error": str(exc)})
        raise


# =============================================================================
# Stage: TRIAL (one array task = one Optuna trial)
# =============================================================================

def trial_task(exp_dir: str, task_id: int) -> None:
    """
    Run ONE Optuna trial of the group this array index belongs to.

    Mirrors ``euler_optuna.optuna_worker``: the study database (sqlite with a
    60 s lock timeout) is the only synchronisation point, and the trial budget is
    re-checked here so an over-sized array exits gracefully instead of spending
    GPU hours on extra trials.
    """
    import optuna

    plan = load_plan(exp_dir)
    slots = plan["trial_slots"]
    if task_id >= len(slots):
        print(f"[trial {task_id}] no slot for this task id; nothing to do")
        return

    slot = slots[task_id]
    group = _group_of(plan, slot["group"])
    prepared = load_prepared(exp_dir)["groups"][group["name"]]
    item = trial_item_id(task_id)

    spec = load_dagsweep_spec(exp_dir)
    study_obj, metric, _direction, _settings = build_group_study(
        Path(group["group_dir"]), group["datasets_dir"], spec,
        plan["cluster"], int(prepared["n_keys"]),
    )
    study = optuna.load_study(study_name=study_obj.study_name,
                              storage=study_obj.storage)

    budget = int(group["n_trials"])
    if len(study.trials) >= budget:
        print(f"[trial {task_id}] budget reached ({len(study.trials)}/{budget}); "
              "exiting gracefully")
        record_progress(exp_dir, item, {"state": "skipped", "group": group["name"],
                                        "reason": "budget reached"})
        return

    trial = study.ask()
    record_progress(exp_dir, item, {"state": "running", "group": group["name"],
                                    "trial_number": trial.number})
    print(f"[trial {task_id}] group={group['name']} trial={trial.number} "
          f"({len(study.trials)}/{budget})")

    try:
        value = study_obj.objective(trial)
        study.tell(trial, value)
        record_progress(exp_dir, item, {"state": "ok", "group": group["name"],
                                        "trial_number": trial.number,
                                        "metric": metric, "value": value})
        print(f"[trial {task_id}] done: {metric}={value}")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        record_progress(exp_dir, item, {"state": "failed", "group": group["name"],
                                        "trial_number": trial.number,
                                        "error": str(exc)})
        raise


# =============================================================================
# Stage: SELECT (the barrier) - one best_trial.yaml per group
# =============================================================================

def select_stage(exp_dir: str) -> Dict[str, Any]:
    """
    Pick each group's winning trial and write ``best_trial.yaml``.

    Deliberately fault-isolating: the train array depends on this job with
    ``afterok``, so raising here would cancel EVERY group's runs because one
    group's study came out empty.  Failures are recorded per group instead; the
    training worker then refuses only that group's runs.
    """
    plan = load_plan(exp_dir)
    record_progress(exp_dir, "stage_select", {"state": "running"})

    spec = load_dagsweep_spec(exp_dir)
    prepared = load_prepared(exp_dir)["groups"]
    summary: Dict[str, Any] = {}

    for group in plan["groups"]:
        group_dir = Path(group["group_dir"])
        name = group["name"]

        if not plan["optuna"]["enabled"]:
            # Explicit untuned arm (--skip_optuna / optuna.enabled: false).
            params = load_best_params(group_dir)
            if not params:
                logger.warning(
                    "[select] %s has no best_trial.yaml -> its runs train the "
                    "BASE config UNTUNED", name)
            record_progress(exp_dir, f"group_{name}", {
                "state": "ok", "tuned": bool(params),
                "n_tuned_params": len(params),
                "reason": "optuna disabled" if not params else "cached best_trial.yaml",
            })
            summary[name] = params
            continue

        try:
            params = finalize_group_study(
                group_dir, group["datasets_dir"], spec, plan["cluster"],
                int(prepared[name]["n_keys"]),
            )
            record_progress(exp_dir, f"group_{name}", {
                "state": "ok", "tuned": True, "n_tuned_params": len(params),
                "params": params,
            })
            summary[name] = params
        except Exception as exc:
            logger.error("[select] group %s failed: %s", name, exc, exc_info=True)
            record_progress(exp_dir, f"group_{name}", {
                "state": "failed", "tuned": False, "error": str(exc),
            })
            summary[name] = {}

    record_progress(exp_dir, "stage_select", {"state": "ok"})
    rebuild_progress(exp_dir)
    return summary


# =============================================================================
# Stage: TRAIN (one array task = one (dag_seed, model_seed) run)
# =============================================================================

def train_task(exp_dir: str, task_id: int, force: bool = False) -> None:
    """
    Train ONE run of the seed sweep, with its group's tuned parameters.

    Two guards keep a parallel result honest:

    * an already-``ok`` run is skipped (resume after a walltime kill) unless
      ``force``;
    * if tuning was REQUESTED but the group has no ``best_trial.yaml``, the task
      fails loudly instead of quietly producing an untuned run that is
      indistinguishable from a tuned one.  The untuned fallback stays allowed
      only when it was asked for (``--skip_optuna`` / ``optuna.enabled: false``).
    """
    plan = load_plan(exp_dir)
    slots = plan["train_slots"]
    if task_id >= len(slots):
        print(f"[train {task_id}] no slot for this task id; nothing to do")
        return

    slot = slots[task_id]
    group = _group_of(plan, slot["group"])
    run = group["runs"][slot["run_index"]]
    item = run_item_id(group["name"], run["run_key"])

    previous = read_progress_items(exp_dir).get(item, {})
    if previous.get("state") == "ok" and not force:
        print(f"[train {task_id}] {run['run_name']} already ok; skipping")
        return

    group_dir = Path(group["group_dir"])
    best_params = load_best_params(group_dir)
    if not best_params and plan["optuna"]["declared_enabled"] \
            and not plan["optuna"]["skip_optuna"]:
        message = (
            f"group {group['name']}: tuning was requested but "
            f"{group_dir / 'best_trial.yaml'} is missing (the study produced no "
            "usable trial). Refusing to train an untuned run silently; pass "
            "--skip_optuna to train the base config on purpose."
        )
        record_progress(exp_dir, item, {"state": "failed", "group": group["name"],
                                        "error": message})
        raise RuntimeError(message)

    prepared = load_prepared(exp_dir)["groups"][group["name"]]
    dataset_name = prepared["datasets"][str(run["dag_seed"])]

    spec = load_dagsweep_spec(exp_dir)
    train_fn = resolve_trainer(plan["trainer"])[0]

    record_progress(exp_dir, item, {"state": "running", "group": group["name"],
                                    "run_name": run["run_name"],
                                    "dag_seed": run["dag_seed"],
                                    "model_seed": run["model_seed"],
                                    "dataset": dataset_name,
                                    "tuned": bool(best_params)})
    print(f"[train {task_id}] {run['run_name']} (dataset={dataset_name}, "
          f"{len(best_params)} tuned param(s))")

    started = time.time()
    try:
        # Same staging function as the sequential sweep -> identical config.
        config = stage_run_config(
            plan["base_config"], {"axes": group["axes"]}, spec, dataset_name,
            group["datasets_dir"], run["dag_seed"], run["model_seed"], best_params,
        )
        run_single_combination(
            config=config,
            save_dir=Path(run["save_dir"]),
            train_fn=train_fn,
            data_dir=Path(group["datasets_dir"]),
            cluster=plan["cluster"],
        )
        record_progress(exp_dir, item, {
            "state": "ok", "group": group["name"], "run_name": run["run_name"],
            "dag_seed": run["dag_seed"], "model_seed": run["model_seed"],
            "dataset": dataset_name, "save_dir": run["save_dir"],
            "tuned": bool(best_params), "duration_s": round(time.time() - started, 1),
        })
        print(f"[train {task_id}] {run['run_name']} completed")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        record_progress(exp_dir, item, {
            "state": "failed", "group": group["name"], "run_name": run["run_name"],
            "dag_seed": run["dag_seed"], "model_seed": run["model_seed"],
            "error": str(exc), "duration_s": round(time.time() - started, 1),
        })
        raise


# =============================================================================
# Stage: CLEANUP (prune + final report)
# =============================================================================

def cleanup_stage(exp_dir: str) -> Dict[str, Any]:
    """
    Prune the heavy arrays and write the final planned-vs-reached report.

    Pruning cannot happen inside a run (concurrent tasks share a dataset), so it
    is deferred to here.  Only ``ds*.npz`` goes: ``dag_recipe.json`` stays, so any
    dataset remains exactly regenerable with ``cli dagsweep-regen``.
    """
    plan = load_plan(exp_dir)
    record_progress(exp_dir, "stage_cleanup", {"state": "running"})

    freed = 0
    if plan["delete_dataset"]:
        try:
            prepared = load_prepared(exp_dir)["groups"]
        except FileNotFoundError:
            prepared = {}
        for group in plan["groups"]:
            entry = prepared.get(group["name"], {})
            names = list(entry.get("datasets", {}).values())
            if entry.get("opt_dataset"):
                names.append(entry["opt_dataset"])
            for name in names:
                try:
                    freed += prune_dag_arrays(join(group["datasets_dir"], name))
                except Exception as exc:  # pruning must never fail the sweep
                    logger.warning("[cleanup] could not prune %s: %s", name, exc)

    record_progress(exp_dir, "stage_cleanup",
                    {"state": "ok", "freed_bytes": int(freed)})
    rollup = rebuild_progress(exp_dir)
    print(format_progress(rollup))
    if freed:
        print(f"[cleanup] pruned dataset arrays ({freed / 1e6:.1f} MB reclaimed)")
    return rollup


# =============================================================================
# SLURM script generation + chained submission
# =============================================================================

# Cluster environment block, kept identical to sweeper.generate_slurm_job_array_script
# so a working cluster setup keeps working for the DAG sweep too.
_ENV_BLOCK = """
# ---------------------------------------------------------------------------
# ENVIRONMENT SETUP
# TODO: Customize these module loads for your cluster
# ---------------------------------------------------------------------------
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

VENV_PATH="{venv_path}"
source "$VENV_PATH/bin/activate"

if [[ -z "${{VIRTUAL_ENV:-}}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"
"""


def _script_header(job_name: str, log_dir: str, log_tag: str, walltime: str,
                   mem_per_cpu: str, array: Optional[str] = None,
                   gpu_mem: Optional[str] = None,
                   dependency: Optional[str] = None) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir}/{log_tag}_%A_%a.out" if array
        else f"#SBATCH --output={log_dir}/{log_tag}_%j.out",
        f"#SBATCH --error={log_dir}/{log_tag}_%A_%a.err" if array
        else f"#SBATCH --error={log_dir}/{log_tag}_%j.err",
        "#SBATCH --ntasks=1",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --mem-per-cpu={mem_per_cpu}",
    ]
    if array:
        lines.append(f"#SBATCH --array={array}")
    if gpu_mem:
        lines.append("#SBATCH --gpus=1")
        lines.append(f"#SBATCH --gres=gpumem:{gpu_mem}")
    if dependency:
        lines.append(f"#SBATCH --dependency={dependency}")
    return "\n".join(lines) + "\n"


def _write_script(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write(content)
    try:
        path.chmod(0o755)
    except Exception:
        pass  # Windows
    return str(path)


def generate_stage_scripts(exp_dir: str, plan: Dict[str, Any],
                           slurm_params: Dict[str, Any]) -> Dict[str, str]:
    """
    Write the five SLURM scripts of the chain (without submitting them).

    Dependencies are patched in at submission time, so the scripts themselves are
    reusable: any stage can be re-run by hand with ``sbatch`` after a partial
    failure (e.g. re-submit only ``train.sh``).
    """
    scripts_dir = state_dir(exp_dir) / "scripts"
    log_dir = str(state_dir(exp_dir) / "slurm_logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    exp_id = plan["experiment"]
    walltime = slurm_params.get("walltime", "4:00:00")
    mem = slurm_params.get("mem_per_cpu", "10g")
    gpu_mem = slurm_params.get("gpu_mem", "11g")
    concurrent = int(slurm_params.get("max_concurrent_jobs", 6))
    venv = slurm_params.get("venv_path", "$HOME/myenv")
    env = _ENV_BLOCK.format(venv_path=venv)

    def _body(subcommand: str, array_task: bool) -> str:
        # The worker call is emitted on ONE line on purpose: backslash line
        # continuations are fragile in sbatch scripts (a single trailing space
        # silently truncates the command).
        task_arg = " --task_id $SLURM_ARRAY_TASK_ID" if array_task else ""
        return (
            "\nset -euo pipefail\n"
            f'echo "[$(date)] dagsweep {subcommand} on $(hostname)"\n'
            f'EXP_DIR="{exp_dir}"\n'
            f"{env}\n"
            f'python -m {WORKER_MODULE} {subcommand} --exp_dir "$EXP_DIR"{task_arg}\n'
            f'\necho "[$(date)] dagsweep {subcommand} finished"\n'
        )


    scripts: Dict[str, str] = {}

    scripts["prep"] = _write_script(
        scripts_dir / "prep.sh",
        _script_header(f"dagprep_{exp_id}", log_dir, "prep", walltime, mem)
        + _body("prepare", False))

    if plan["n_trial_tasks"] > 0:
        scripts["trials"] = _write_script(
            scripts_dir / "trials.sh",
            _script_header(f"dagtrial_{exp_id}", log_dir, "trial", walltime, mem,
                           array=f"0-{plan['n_trial_tasks'] - 1}%{concurrent}",
                           gpu_mem=gpu_mem)
            + _body("trial", True))
        scripts["select"] = _write_script(
            scripts_dir / "select.sh",
            _script_header(f"dagselect_{exp_id}", log_dir, "select", walltime, mem)
            + _body("select", False))

    scripts["train"] = _write_script(
        scripts_dir / "train.sh",
        _script_header(f"dagtrain_{exp_id}", log_dir, "train", walltime, mem,
                       array=f"0-{max(plan['n_train_tasks'] - 1, 0)}%{concurrent}",
                       gpu_mem=gpu_mem)
        + _body("train", True))

    scripts["cleanup"] = _write_script(
        scripts_dir / "cleanup.sh",
        _script_header(f"dagclean_{exp_id}", log_dir, "cleanup", walltime, mem)
        + _body("cleanup", False))

    return scripts


def _sbatch(script_path: str, cwd: str, dependency: Optional[str] = None) -> str:
    """Submit one script and return its job id (raises on a failed submission)."""
    command = ["sbatch"]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(script_path)
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed for {script_path}: {result.stderr}")
    job_id = result.stdout.strip().split()[-1]
    print(f"  submitted {basename(script_path)} -> job {job_id}"
          + (f" (dependency {dependency})" if dependency else ""))
    return job_id


def submit_parallel_dag_sweep(exp_dir: str, home_exp_dir: str,
                              slurm_params: Dict[str, Any],
                              keep_data: bool = False, skip_optuna: bool = False,
                              force_optuna: bool = False,
                              submit_jobs: bool = True) -> Dict[str, Any]:
    """
    Plan the sweep, write the scripts and submit the dependency chain.

    ``submit_jobs=False`` is a dry run: the plan and all scripts are written (so
    they can be inspected or submitted by hand) but nothing is queued.

    Returns:
        ``{"plan": ..., "scripts": {...}, "job_ids": {...}}``
    """
    plan = build_static_plan(
        exp_dir, home_exp_dir, cluster=True, keep_data=keep_data,
        skip_optuna=skip_optuna, force_optuna=force_optuna,
        slurm_params=slurm_params,
    )
    scripts = generate_stage_scripts(exp_dir, plan, slurm_params)
    rebuild_progress(exp_dir)

    print("=" * 60)
    print("PARALLEL DAG SWEEP PLAN")
    print("=" * 60)
    print(f"Groups          : {len(plan['groups'])}")
    print(f"Optuna trials   : {plan['n_trial_tasks']} "
          f"({plan['optuna']['n_trials_per_group']} per group, "
          f"enabled={plan['optuna']['enabled']})")
    print(f"Training runs   : {plan['n_train_tasks']} (trainer={plan['trainer']})")
    print(f"Max concurrent  : {slurm_params.get('max_concurrent_jobs')}")
    print(f"Walltime        : {slurm_params.get('walltime')}")
    print(f"State folder    : {state_dir(exp_dir)}")
    print("=" * 60)

    if not submit_jobs:
        print("Dry run: scripts written, nothing submitted.")
        for name, path in scripts.items():
            print(f"  {name:<8} {path}")
        return {"plan": plan, "scripts": scripts, "job_ids": {}}

    job_ids: Dict[str, str] = {}
    job_ids["prep"] = _sbatch(scripts["prep"], exp_dir)

    if "trials" in scripts:
        job_ids["trials"] = _sbatch(scripts["trials"], exp_dir,
                                    dependency=f"afterok:{job_ids['prep']}")
        # afterany: ONE crashed trial must not cancel the selection (and with it
        # the whole sweep); select decides per group whether it has enough.
        job_ids["select"] = _sbatch(scripts["select"], exp_dir,
                                    dependency=f"afterany:{job_ids['trials']}")
        train_dep = f"afterok:{job_ids['select']}"
    else:
        train_dep = f"afterok:{job_ids['prep']}"

    job_ids["train"] = _sbatch(scripts["train"], exp_dir, dependency=train_dep)
    job_ids["cleanup"] = _sbatch(scripts["cleanup"], exp_dir,
                                 dependency=f"afterany:{job_ids['train']}")

    _write_json(state_dir(exp_dir) / "job_ids.json",
                {"job_ids": job_ids, "submitted": datetime.now().isoformat()})

    print("=" * 60)
    print("Submitted chain: " + " -> ".join(f"{k}({v})" for k, v in job_ids.items()))
    print(f"Monitor with : squeue -u $USER")
    print(f"Progress     : python -m causaliT.euler_sweep.euler_sweep.cli "
          f"dagsweep-status --exp_id {plan['experiment']}")
    print("=" * 60)
    return {"plan": plan, "scripts": scripts, "job_ids": job_ids}


# =============================================================================
# Scratch staging
# =============================================================================

def stage_experiment_to_scratch(home_exp_dir: str, scratch_path: str) -> str:
    """
    Copy the experiment's *inputs* to a scratch folder and return it.

    Only the light spec files are copied: results, datasets and state are written
    directly into the scratch folder by the jobs, which is exactly why heavy
    output never touches ``$HOME``.
    """
    Path(scratch_path).mkdir(parents=True, exist_ok=True)
    for pattern in ("config*.yaml", "optuna*.yaml", "dagsweep*.yaml"):
        for src in glob(join(home_exp_dir, pattern)):
            shutil.copy2(src, join(scratch_path, basename(src)))
    return scratch_path


__all__ = [
    "build_static_plan",
    "cleanup_stage",
    "format_progress",
    "generate_stage_scripts",
    "load_plan",
    "load_prepared",
    "prepare_stage",
    "rebuild_progress",
    "record_progress",
    "select_stage",
    "stage_experiment_to_scratch",
    "state_dir",
    "submit_parallel_dag_sweep",
    "train_task",
    "trial_task",
]

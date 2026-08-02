"""
Tests for the PARALLEL (SLURM) DAG sweep (``euler_sweep.dagsweep_parallel``).

The parallel path must produce exactly what the sequential one produces, only
faster.  What can silently break that equivalence is orchestration, so these
tests pin the invariants a wrong plan would violate:

* the array sizes are derivable from the SPEC ALONE (they must be known at
  submit time, before a single DAG exists);
* every trial slot and every (dag_seed, model_seed) run gets exactly one task,
  and one task maps to exactly one item;
* the SELECT barrier is fault-isolating: one group's empty study must not cancel
  the other groups' runs (the train array depends on select with ``afterok``);
* a training task refuses to run untuned when tuning WAS requested - otherwise a
  failed study would silently produce results that look tuned;
* model seeds of one DAG read the same dataset and the same data split;
* pruning happens in cleanup only (concurrent runs share a dataset);
* the generated sbatch scripts carry the right dependencies / array specs and
  contain no backslash line continuations.

Training and DAG generation are stubbed: this module is orchestration, and both
are covered by ``test_dag_sweep_grouped.py``.
"""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from causaliT.euler_sweep.euler_sweep import data_source as dsrc
from causaliT.euler_sweep.euler_sweep import dagsweep_parallel as dsp



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DAG_BLOCK = {
    "degree": 2,
    "linearity": "linear",
    "noise": "gaussian",
    "n_samples": 200,
}


def _write_experiment(tmp_path, name="exp", dag_seeds=(0, 1), model_seeds=None,
                      n_nodes=(4,), optuna_enabled=True, n_trials=3,
                      extra_spec=None):
    """Minimal experiment folder: base config (+ optuna settings) + spec."""
    exp_dir = tmp_path / name
    exp_dir.mkdir(parents=True)

    OmegaConf.save(
        OmegaConf.create({
            "experiment": {"name": "test"},
            "data": {"dataset": "PLACEHOLDER"},
            "training": {"seed": 0, "lr": 0.001},
        }),
        exp_dir / "config_test.yaml",
    )

    if optuna_enabled:
        OmegaConf.save(
            OmegaConf.create({
                "n_trials": n_trials,
                "sampler": {"name": "sobol"},
                "pruner": "none",
                "search_space": {"training.lr": {"type": "float", "low": 1e-4,
                                                 "high": 1e-2, "log": True}},
            }),
            exp_dir / "optuna_settings.yaml",
        )

    spec = {
        "group_axes": {"n_nodes": list(n_nodes)},
        "dag_seeds": list(dag_seeds),
        "dag": dict(DAG_BLOCK),
        "optuna": {"enabled": optuna_enabled, "opt_seed": 1000,
                   "metric": "val_x_mae_mean", "direction": "minimize"},
        "training": {"trainer": "standard"},
    }
    if model_seeds is not None:
        spec["model_seeds"] = list(model_seeds)
    spec.update(extra_spec or {})
    OmegaConf.save(OmegaConf.create(spec), exp_dir / "dagsweep.yaml")
    return exp_dir


def _plan(tmp_path, **kwargs):
    """Build a static plan for a freshly written experiment folder."""
    exp_dir = _write_experiment(tmp_path, **kwargs)
    plan = dsp.build_static_plan(str(exp_dir), str(exp_dir),
                                 slurm_params={"max_concurrent_jobs": 4})
    return exp_dir, plan


@pytest.fixture
def fake_datasets(monkeypatch):
    """
    Replace DAG generation with a deterministic naming stub.

    Generation is reached through the DataSource, so the stub is installed on
    ``data_source`` (the prepare stage no longer calls the provider itself).
    """
    generated = []

    def fake_ensure(cfg, datasets_dir, gen_kwargs, verbose=True):
        name = f"ds_n{cfg.n_nodes}_s{cfg.seed}"
        Path(datasets_dir, name).mkdir(parents=True, exist_ok=True)
        generated.append(name)
        return name

    monkeypatch.setattr(dsrc, "ensure_dag_dataset", fake_ensure)
    monkeypatch.setattr(dsp, "n_keys_from_metadata",
                        lambda datasets_dir, name, fallback=None: int(fallback or 4))
    return generated


# ---------------------------------------------------------------------------
# Fixed-dataset (ATE) fixtures - same scheme, data from scm_ds.datasets
# ---------------------------------------------------------------------------

FIXED_NAMES = ("ds_fake_a", "ds_fake_b")


def _write_fixed_experiment(tmp_path, name="ate", names=FIXED_NAMES,
                            data_seeds=(0,), model_seeds=(1, 2),
                            optuna_enabled=True, n_trials=2):
    """Experiment folder whose spec declares FIXED SCMs (an 'atesweep.yaml')."""
    exp_dir = tmp_path / name
    exp_dir.mkdir(parents=True)

    OmegaConf.save(
        OmegaConf.create({
            "experiment": {"name": "test"},
            "data": {"dataset": "PLACEHOLDER"},
            "training": {"seed": 0, "data_seed": 0, "lr": 0.001},
        }),
        exp_dir / "config_test.yaml",
    )
    if optuna_enabled:
        OmegaConf.save(
            OmegaConf.create({
                "n_trials": n_trials,
                "sampler": {"name": "sobol"},
                "pruner": "none",
                "search_space": {"training.lr": {"type": "float", "low": 1e-4,
                                                 "high": 1e-2, "log": True}},
            }),
            exp_dir / "optuna_settings.yaml",
        )

    OmegaConf.save(
        OmegaConf.create({
            "datasets": {"names": list(names), "generate": {"n": 100}},
            "data_seeds": list(data_seeds),
            "model_seeds": list(model_seeds),
            "optuna": {"enabled": optuna_enabled, "opt_seed": 1000,
                       "metric": "val_x_mae_mean", "direction": "minimize"},
            "training": {"trainer": "standard"},
        }),
        exp_dir / "atesweep.yaml",
    )
    return exp_dir


@pytest.fixture
def fake_registry(monkeypatch):
    """
    Register fake fixed datasets and stub their (expensive) generation.

    A real ATE dataset is 50k samples plus a Monte-Carlo ground truth; the
    orchestration only cares about the folder NAME, so the stub only writes the
    metadata the config staging reads back.

    Like the real generator it is IDEMPOTENT (an existing folder is reused), so
    ``calls`` counts actual materializations - which is what tells "generated
    once and shared" from "regenerated per seed".
    """
    import scm_ds.datasets as datasets_mod

    registry = {n: object() for n in FIXED_NAMES}
    monkeypatch.setattr(datasets_mod, "DATASET_REGISTRY", registry)

    calls = []

    def fake_generate(registry_name, data_root, generation=None, folder_name=None,
                      force=False, verbose=True):
        name = folder_name or registry_name
        folder = Path(data_root, name)
        if folder.exists() and not force:
            return name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "dataset_metadata.json").write_text(json.dumps(
            {"variable_info": {"source_labels": ["x0", "x1"],
                               "input_labels": ["x2", "x3"]}}
        ))
        calls.append((name, dict(generation or {})))
        return name

    monkeypatch.setattr(dsrc, "generate_fixed_dataset", fake_generate)
    return calls




# ---------------------------------------------------------------------------
# Static plan
# ---------------------------------------------------------------------------

def test_plan_sizes_come_from_the_spec_alone(tmp_path):
    """
    Array sizes must be known WITHOUT any dataset (sbatch needs them upfront).

    2 sizes x 3 trials = 6 trial tasks; 2 sizes x 2 DAGs x 2 inits = 8 runs.
    """
    exp_dir, plan = _plan(tmp_path, n_nodes=(4, 5), dag_seeds=(0, 1),
                          model_seeds=(7, 8), n_trials=3)

    assert plan["n_trial_tasks"] == 6
    assert plan["n_train_tasks"] == 8
    assert len(plan["groups"]) == 2
    # No DAG was generated while planning.
    assert not (exp_dir / "groups" / "n_nodes_4" / "datasets").exists()
    # The plan is on disk for the workers to read.
    assert dsp.plan_path(str(exp_dir)).exists()


def test_every_task_id_maps_to_exactly_one_item(tmp_path):
    """A duplicated slot would train the same run twice and lose another."""
    _exp_dir, plan = _plan(tmp_path, n_nodes=(4, 5), dag_seeds=(0, 1),
                           model_seeds=(7,), n_trials=2)

    assert [s["task_id"] for s in plan["trial_slots"]] == list(range(4))
    assert [s["task_id"] for s in plan["train_slots"]] == list(range(4))

    identified = {(s["group"], s["run_index"]) for s in plan["train_slots"]}
    assert len(identified) == plan["n_train_tasks"]


def test_disabled_optuna_yields_no_trial_tasks(tmp_path):
    """With no search there is nothing to parallelise in phase 1."""
    _exp_dir, plan = _plan(tmp_path, optuna_enabled=False)

    assert plan["n_trial_tasks"] == 0
    assert plan["optuna"]["enabled"] is False
    assert plan["n_train_tasks"] == 2


def test_enabled_optuna_without_trial_budget_is_rejected(tmp_path):
    """An empty trial array would silently produce untuned runs."""
    exp_dir = _write_experiment(tmp_path, n_trials=0)
    with pytest.raises(ValueError, match="n_trials"):
        dsp.build_static_plan(str(exp_dir), str(exp_dir))


def test_skip_optuna_keeps_the_declared_flag(tmp_path):
    """
    ``declared_enabled`` is what lets a worker tell "untuned on purpose" from
    "the study failed": only the former may train the base config.
    """
    exp_dir = _write_experiment(tmp_path)
    plan = dsp.build_static_plan(str(exp_dir), str(exp_dir), skip_optuna=True)

    assert plan["optuna"]["enabled"] is False
    assert plan["optuna"]["declared_enabled"] is True
    assert plan["optuna"]["skip_optuna"] is True


def test_keep_data_disables_pruning(tmp_path):
    _exp_dir, plan = _plan(tmp_path)
    assert plan["delete_dataset"] is True

    exp_dir = _write_experiment(tmp_path, name="exp2")
    plan2 = dsp.build_static_plan(str(exp_dir), str(exp_dir), keep_data=True)
    assert plan2["delete_dataset"] is False


# ---------------------------------------------------------------------------
# Prepare stage
# ---------------------------------------------------------------------------

def test_prepare_generates_one_dataset_per_dag_seed(tmp_path, fake_datasets,
                                                    monkeypatch):
    """
    Model seeds must SHARE a DAG: generating per run would both waste time and
    (worse) let two concurrent tasks race on the same folder.
    """
    monkeypatch.setattr(dsp, "stage_group_config",
                        lambda *a, **k: OmegaConf.create({}))
    monkeypatch.setattr(dsp, "build_group_study",
                        lambda *a, **k: (_FakeStudy(tmp_path), "m", "minimize", {}))

    exp_dir, _plan_obj = _plan(tmp_path, dag_seeds=(0, 1), model_seeds=(7, 8))
    prepared = dsp.prepare_stage(str(exp_dir))

    entry = prepared["groups"]["n_nodes_4"]
    assert sorted(entry["datasets"]) == ["0", "1"]        # per DAG seed, not per run
    assert entry["opt_dataset"] == "ds_n4_s1000"          # disjoint opt DAG
    assert entry["n_keys"] == 4

    rollup = dsp.rebuild_progress(str(exp_dir))
    assert rollup["stages"]["stage_prep"]["state"] == "ok"


class _FakeStudy:
    """Minimal stand-in for OptunaStudy (create/paths only)."""

    def __init__(self, tmp_path):
        self.study_name = "fake"
        self.storage = "sqlite:///:memory:"
        self.study_file_path = str(Path(tmp_path) / "study.db")
        self.created = False

    def create(self):
        self.created = True


# ---------------------------------------------------------------------------
# Select stage (the barrier)
# ---------------------------------------------------------------------------

def test_select_isolates_a_failing_group(tmp_path, monkeypatch, fake_datasets):
    """
    One empty study must not cancel the whole train array.

    ``select`` is a hard dependency of the train array (``afterok``), so it must
    record per-group failures instead of raising.
    """
    monkeypatch.setattr(dsp, "stage_group_config",
                        lambda *a, **k: OmegaConf.create({}))
    monkeypatch.setattr(dsp, "build_group_study",
                        lambda *a, **k: (_FakeStudy(tmp_path), "m", "minimize", {}))

    exp_dir, _plan_obj = _plan(tmp_path, n_nodes=(4, 5))
    dsp.prepare_stage(str(exp_dir))

    def flaky_finalize(group_dir, *a, **k):
        if "n_nodes_4" in str(group_dir):
            raise RuntimeError("no complete trial")
        return {"training.lr": 0.01}

    monkeypatch.setattr(dsp, "finalize_group_study", flaky_finalize)

    summary = dsp.select_stage(str(exp_dir))       # must NOT raise

    assert summary["n_nodes_4"] == {}
    assert summary["n_nodes_5"] == {"training.lr": 0.01}

    rollup = dsp.rebuild_progress(str(exp_dir))
    assert rollup["groups"]["n_nodes_4"]["tuned"] is False
    assert "no complete trial" in rollup["groups"]["n_nodes_4"]["select_error"]
    assert rollup["groups"]["n_nodes_5"]["tuned"] is True


# ---------------------------------------------------------------------------
# Train stage
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_train(monkeypatch):
    """Record what each training task would run."""
    calls = []

    def fake_run_single_combination(config, save_dir, train_fn, data_dir, cluster,
                                    **kwargs):
        calls.append({
            "save_dir": str(save_dir),
            "dataset": config["data"]["dataset"],
            "seed": config["training"]["seed"],
            "data_seed": config["training"].get("data_seed"),
            "lr": config["training"]["lr"],
        })

    monkeypatch.setattr(dsp, "run_single_combination", fake_run_single_combination)
    monkeypatch.setattr(dsp, "resolve_trainer",
                        lambda name: (lambda **kw: None, "mod", "attr"))
    return calls


def _prepared_experiment(tmp_path, monkeypatch, fake_datasets, best_params=None,
                         **kwargs):
    """Plan + prepare an experiment, optionally writing a best_trial.yaml."""
    monkeypatch.setattr(dsp, "stage_group_config",
                        lambda *a, **k: OmegaConf.create({}))
    monkeypatch.setattr(dsp, "build_group_study",
                        lambda *a, **k: (_FakeStudy(tmp_path), "m", "minimize", {}))

    exp_dir, plan = _plan(tmp_path, **kwargs)
    dsp.prepare_stage(str(exp_dir))

    if best_params is not None:
        for group in plan["groups"]:
            OmegaConf.save(OmegaConf.create({"params": best_params}),
                           Path(group["group_dir"]) / "best_trial.yaml")
    return exp_dir, plan


def test_model_seeds_of_one_dag_share_dataset_and_split(tmp_path, monkeypatch,
                                                       fake_datasets, stub_train):
    """
    The edge-stability contract, preserved under parallelism: two tasks that
    differ only in ``model_seed`` must read the same dataset and the same
    ``data_seed`` (the split), and both must get the tuned params.
    """
    exp_dir, plan = _prepared_experiment(
        tmp_path, monkeypatch, fake_datasets,
        best_params={"training.lr": 0.042},
        dag_seeds=(0, 1), model_seeds=(7, 8),
    )

    for task_id in range(plan["n_train_tasks"]):
        dsp.train_task(str(exp_dir), task_id)

    assert len(stub_train) == 4
    assert {c["lr"] for c in stub_train} == {0.042}      # shared tuned params

    per_dag = {}
    for call in stub_train:
        per_dag.setdefault(call["data_seed"], []).append(call)

    assert sorted(per_dag) == [0, 1]
    for calls in per_dag.values():
        assert len({c["dataset"] for c in calls}) == 1   # one DAG, one dataset
        assert sorted(c["seed"] for c in calls) == [7, 8]


def test_train_refuses_to_run_untuned_when_tuning_was_requested(
        tmp_path, monkeypatch, fake_datasets, stub_train):
    """
    A missing best_trial.yaml after a requested search means the study failed.
    Training anyway would yield results indistinguishable from tuned ones.
    """
    exp_dir, _plan_obj = _prepared_experiment(
        tmp_path, monkeypatch, fake_datasets, best_params=None)

    with pytest.raises(RuntimeError, match="untuned"):
        dsp.train_task(str(exp_dir), 0)

    assert stub_train == []
    rollup = dsp.rebuild_progress(str(exp_dir))
    assert rollup["groups"]["n_nodes_4"]["runs"]["failed"] == 1


def test_train_allows_untuned_when_optuna_is_disabled(tmp_path, monkeypatch,
                                                      fake_datasets, stub_train):
    """``optuna.enabled: false`` is an explicit "train the base config" arm."""
    exp_dir, _plan_obj = _prepared_experiment(
        tmp_path, monkeypatch, fake_datasets, optuna_enabled=False)

    dsp.train_task(str(exp_dir), 0)

    assert len(stub_train) == 1
    assert stub_train[0]["lr"] == 0.001        # base config value


def test_completed_run_is_skipped_unless_forced(tmp_path, monkeypatch,
                                                fake_datasets, stub_train):
    """Re-submitting after a walltime kill must not redo finished work."""
    exp_dir, _plan_obj = _prepared_experiment(
        tmp_path, monkeypatch, fake_datasets, best_params={"training.lr": 0.01})

    dsp.train_task(str(exp_dir), 0)
    dsp.train_task(str(exp_dir), 0)                 # resume: skipped
    assert len(stub_train) == 1

    dsp.train_task(str(exp_dir), 0, force=True)     # explicit re-run
    assert len(stub_train) == 2


def test_failed_run_is_recorded_and_reraised(tmp_path, monkeypatch, fake_datasets):
    """
    The task must fail (so SLURM marks it FAILED) AND leave a readable trace, so
    the status report can attribute the loss to a specific run.
    """
    monkeypatch.setattr(dsp, "resolve_trainer",
                        lambda name: (lambda **kw: None, "mod", "attr"))
    monkeypatch.setattr(dsp, "run_single_combination",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    exp_dir, _plan_obj = _prepared_experiment(
        tmp_path, monkeypatch, fake_datasets, best_params={"training.lr": 0.01})

    with pytest.raises(RuntimeError, match="boom"):
        dsp.train_task(str(exp_dir), 0)

    rollup = dsp.rebuild_progress(str(exp_dir))
    group = rollup["groups"]["n_nodes_4"]
    assert group["runs"]["failed"] == 1
    assert any("boom" in (r["error"] or "") for r in group["run_details"].values())


def test_out_of_range_task_id_is_a_no_op(tmp_path, monkeypatch, fake_datasets,
                                         stub_train):
    """An over-sized array (or a stale script) must not crash the phase."""
    exp_dir, plan = _prepared_experiment(
        tmp_path, monkeypatch, fake_datasets, best_params={"training.lr": 0.01})

    dsp.train_task(str(exp_dir), plan["n_train_tasks"] + 5)
    dsp.trial_task(str(exp_dir), plan["n_trial_tasks"] + 5)
    assert stub_train == []


# ---------------------------------------------------------------------------
# Cleanup + progress report
# ---------------------------------------------------------------------------

def test_cleanup_prunes_every_dataset_once(tmp_path, monkeypatch, fake_datasets):
    """
    Pruning is deferred to cleanup precisely because runs share datasets; it must
    then cover the eval DAGs AND the optimisation DAG.
    """
    pruned = []
    monkeypatch.setattr(dsp, "prune_dag_arrays",
                        lambda path: pruned.append(Path(path).name) or 1000)

    exp_dir, _plan_obj = _prepared_experiment(
        tmp_path, monkeypatch, fake_datasets, dag_seeds=(0, 1), model_seeds=(7, 8))

    dsp.cleanup_stage(str(exp_dir))

    assert sorted(pruned) == ["ds_n4_s0", "ds_n4_s1", "ds_n4_s1000"]


def test_cleanup_keeps_data_when_requested(tmp_path, monkeypatch, fake_datasets):
    pruned = []
    monkeypatch.setattr(dsp, "prune_dag_arrays", lambda path: pruned.append(path))

    monkeypatch.setattr(dsp, "stage_group_config",
                        lambda *a, **k: OmegaConf.create({}))
    monkeypatch.setattr(dsp, "build_group_study",
                        lambda *a, **k: (_FakeStudy(tmp_path), "m", "minimize", {}))

    exp_dir = _write_experiment(tmp_path)
    dsp.build_static_plan(str(exp_dir), str(exp_dir), keep_data=True)
    dsp.prepare_stage(str(exp_dir))
    dsp.cleanup_stage(str(exp_dir))

    assert pruned == []


def test_progress_reports_planned_versus_reached(tmp_path, monkeypatch,
                                                 fake_datasets, stub_train):
    """The whole point of the rollup: what was planned, what actually landed."""
    exp_dir, plan = _prepared_experiment(
        tmp_path, monkeypatch, fake_datasets, best_params={"training.lr": 0.01},
        dag_seeds=(0, 1), model_seeds=(7, 8))

    dsp.train_task(str(exp_dir), 0)          # 1 of 4 runs done

    rollup = dsp.rebuild_progress(str(exp_dir))
    assert rollup["planned"]["runs"] == 4
    assert rollup["reached"]["runs"] == 1
    group = rollup["groups"]["n_nodes_4"]
    assert group["runs"]["ok"] == 1
    assert group["runs"]["pending"] == 3

    text = dsp.format_progress(rollup)
    assert "runs 1/4 ok" in text
    # The rollup is persisted next to the plan for later inspection.
    saved = json.loads((dsp.state_dir(str(exp_dir)) / dsp.PROGRESS_FILENAME)
                       .read_text())
    assert saved["reached"]["runs"] == 1


def test_status_before_submission_raises_a_clear_error(tmp_path):
    exp_dir = _write_experiment(tmp_path)
    with pytest.raises(FileNotFoundError, match="plan.json"):
        dsp.rebuild_progress(str(exp_dir))


# ---------------------------------------------------------------------------
# SLURM script generation
# ---------------------------------------------------------------------------

def test_scripts_encode_arrays_gpus_and_no_line_continuations(tmp_path):
    """
    The scripts are the contract with SLURM: array bounds, concurrency cap and a
    GPU only where training happens.  Backslash continuations are avoided on
    purpose (a trailing space would silently truncate the worker call).
    """
    exp_dir, plan = _plan(tmp_path, dag_seeds=(0, 1), model_seeds=(7, 8),
                          n_trials=3)
    scripts = dsp.generate_stage_scripts(
        str(exp_dir), plan,
        {"max_concurrent_jobs": 2, "walltime": "1:00:00", "gpu_mem": "11g",
         "mem_per_cpu": "8g", "venv_path": "/env"},
    )

    assert set(scripts) == {"prep", "trials", "select", "train", "cleanup"}

    trials = Path(scripts["trials"]).read_text()
    assert "#SBATCH --array=0-2%2" in trials          # 1 group x 3 trials
    assert "--gres=gpumem:11g" in trials
    assert "dagsweep_worker trial" in trials
    assert "--task_id $SLURM_ARRAY_TASK_ID" in trials

    train = Path(scripts["train"]).read_text()
    assert "#SBATCH --array=0-3%2" in train           # 2 DAGs x 2 inits
    assert "--gres=gpumem:11g" in train

    prep = Path(scripts["prep"]).read_text()
    assert "--array" not in prep                      # single CPU job
    assert "--gres" not in prep
    assert "--task_id" not in prep

    for path in scripts.values():
        assert "\\\n" not in Path(path).read_text(), f"{path} uses continuations"


def test_no_trial_script_when_search_is_disabled(tmp_path):
    """Without a search there is no trial array and no barrier to insert."""
    exp_dir, plan = _plan(tmp_path, optuna_enabled=False)
    scripts = dsp.generate_stage_scripts(str(exp_dir), plan,
                                         {"max_concurrent_jobs": 2})

    assert set(scripts) == {"prep", "train", "cleanup"}


def test_submission_chain_dependencies(tmp_path, monkeypatch):
    """
    The chain is the correctness argument of the design:
    prep -> trials -> select -> train -> cleanup, with ``afterany`` where a
    single failed task must not cancel the rest.
    """
    submitted = []

    def fake_sbatch(script_path, cwd, dependency=None):
        submitted.append((Path(script_path).stem, dependency))
        return str(100 + len(submitted))

    monkeypatch.setattr(dsp, "_sbatch", fake_sbatch)

    exp_dir = _write_experiment(tmp_path, dag_seeds=(0, 1), n_trials=2)
    result = dsp.submit_parallel_dag_sweep(
        str(exp_dir), str(exp_dir),
        slurm_params={"max_concurrent_jobs": 2, "walltime": "1:00:00"},
    )

    assert [s[0] for s in submitted] == ["prep", "trials", "select", "train",
                                         "cleanup"]
    deps = dict(submitted)
    assert deps["prep"] is None
    assert deps["trials"] == "afterok:101"
    # afterany: a crashed trial must still let the selection run.
    assert deps["select"] == "afterany:102"
    assert deps["train"] == "afterok:103"
    assert deps["cleanup"] == "afterany:104"

    assert set(result["job_ids"]) == {"prep", "trials", "select", "train",
                                      "cleanup"}
    assert (dsp.state_dir(str(exp_dir)) / "job_ids.json").exists()


def test_train_depends_on_prep_when_search_is_disabled(tmp_path, monkeypatch):
    submitted = []
    monkeypatch.setattr(
        dsp, "_sbatch",
        lambda script_path, cwd, dependency=None:
            submitted.append((Path(script_path).stem, dependency))
            or str(100 + len(submitted)),
    )

    exp_dir = _write_experiment(tmp_path, optuna_enabled=False)
    dsp.submit_parallel_dag_sweep(str(exp_dir), str(exp_dir),
                                  slurm_params={"max_concurrent_jobs": 2})

    deps = dict(submitted)
    assert [s[0] for s in submitted] == ["prep", "train", "cleanup"]
    assert deps["train"] == "afterok:101"


def test_dry_run_writes_scripts_but_submits_nothing(tmp_path, monkeypatch):
    """Inspecting the generated jobs must never queue them."""
    def explode(*a, **k):
        raise AssertionError("sbatch must not be called during a dry run")

    monkeypatch.setattr(dsp, "_sbatch", explode)

    exp_dir = _write_experiment(tmp_path)
    result = dsp.submit_parallel_dag_sweep(
        str(exp_dir), str(exp_dir),
        slurm_params={"max_concurrent_jobs": 2}, submit_jobs=False,
    )

    assert result["job_ids"] == {}
    for path in result["scripts"].values():
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Scratch staging
# ---------------------------------------------------------------------------

def test_scratch_staging_copies_only_spec_files(tmp_path):
    """
    Heavy output must be born in scratch, not copied there: only the light spec
    files travel, which is what keeps $HOME small.
    """
    exp_dir = _write_experiment(tmp_path)
    (exp_dir / "results").mkdir()
    (exp_dir / "results" / "big.npz").write_bytes(b"0" * 32)

    scratch = tmp_path / "scratch_run"
    dsp.stage_experiment_to_scratch(str(exp_dir), str(scratch))

    names = sorted(p.name for p in scratch.iterdir())
    assert names == ["config_test.yaml", "dagsweep.yaml", "optuna_settings.yaml"]


def test_scratch_staging_copies_the_atesweep_alias(tmp_path, fake_registry):
    """
    ``atesweep.yaml`` is an ALIAS of ``dagsweep.yaml``; if staging forgets it the
    prep job starts in a folder with no spec at all.
    """
    exp_dir = _write_fixed_experiment(tmp_path)

    scratch = tmp_path / "scratch_ate"
    dsp.stage_experiment_to_scratch(str(exp_dir), str(scratch))

    names = sorted(p.name for p in scratch.iterdir())
    assert names == ["atesweep.yaml", "config_test.yaml", "optuna_settings.yaml"]


# ---------------------------------------------------------------------------
# Fixed-dataset (ATE) sweeps in parallel
#
# The same five-stage chain must drive a spec whose data does NOT come from a
# sampled DAG.  The failure this pins: forwarding the implicit 'dataset' group
# axis into the random-SCM config (TypeError in the prep job).
# ---------------------------------------------------------------------------

def _prepared_fixed_experiment(tmp_path, monkeypatch, best_params=None, **kwargs):
    """Plan + prepare a fixed-dataset experiment, optionally tuned."""
    monkeypatch.setattr(dsp, "build_group_study",
                        lambda *a, **k: (_FakeStudy(tmp_path), "m", "minimize", {}))

    exp_dir = _write_fixed_experiment(tmp_path, **kwargs)
    plan = dsp.build_static_plan(str(exp_dir), str(exp_dir),
                                 slurm_params={"max_concurrent_jobs": 4})
    dsp.prepare_stage(str(exp_dir))

    if best_params is not None:
        for group in plan["groups"]:
            OmegaConf.save(OmegaConf.create({"params": best_params}),
                           Path(group["group_dir"]) / "best_trial.yaml")
    return exp_dir, plan


def test_fixed_dataset_plan_groups_by_dataset(tmp_path, fake_registry):
    """
    One group per registry key, exactly as one group per node count:
    2 datasets x 2 trials = 4 trial tasks; 2 x 1 split x 2 inits = 4 runs.
    """
    exp_dir = _write_fixed_experiment(tmp_path, data_seeds=(0,),
                                      model_seeds=(1, 2), n_trials=2)
    plan = dsp.build_static_plan(str(exp_dir), str(exp_dir))

    assert plan["data_source"] == "fixed_dataset"
    assert [g["name"] for g in plan["groups"]] == list(FIXED_NAMES)
    assert plan["n_trial_tasks"] == 4
    assert plan["n_train_tasks"] == 4
    # Planning stays dataset-free (the array sizes are needed before prep runs).
    assert fake_registry == []


def test_fixed_dataset_prepare_materializes_each_scm_once(tmp_path, monkeypatch,
                                                          fake_registry):
    """
    All seeds of a fixed group share ONE dataset: generating per seed would
    re-run a 50k-sample SCM (and its MC ground truth) for nothing.
    """
    monkeypatch.setattr(dsp, "build_group_study",
                        lambda *a, **k: (_FakeStudy(tmp_path), "m", "minimize", {}))

    exp_dir = _write_fixed_experiment(tmp_path, data_seeds=(0, 1),
                                      model_seeds=(1, 2))
    dsp.build_static_plan(str(exp_dir), str(exp_dir))
    prepared = dsp.prepare_stage(str(exp_dir))

    assert prepared["source"] == "fixed_dataset"
    entry = prepared["groups"]["ds_fake_a"]
    # Both data seeds AND the search point at the same folder ...
    assert entry["datasets"] == {"0": "ds_fake_a", "1": "ds_fake_a"}
    assert entry["opt_dataset"] == "ds_fake_a"
    # ... which was written exactly once per dataset (idempotent generator).
    assert sorted(name for name, _gen in fake_registry) == ["ds_fake_a",
                                                            "ds_fake_b"]
    assert fake_registry[0][1] == {"n": 100}    # generation options forwarded


def test_fixed_dataset_search_split_is_disjoint_from_the_runs(tmp_path,
                                                              monkeypatch,
                                                              fake_registry):
    """
    With one shared dataset the SPLIT is the only separation left between tuning
    and evaluation, so the staged group config must carry ``opt_seed`` as
    ``training.data_seed``.  Without it the search would select on an evaluation
    split.
    """
    staged = {}

    def spy_stage_group_config(base_config_path, base_optuna_path, group_dir,
                               group, spec, dataset_name, datasets_dir, **kwargs):
        staged[Path(group_dir).name] = kwargs
        return OmegaConf.create({})

    monkeypatch.setattr(dsp, "stage_group_config", spy_stage_group_config)
    monkeypatch.setattr(dsp, "build_group_study",
                        lambda *a, **k: (_FakeStudy(tmp_path), "m", "minimize", {}))

    exp_dir = _write_fixed_experiment(tmp_path, data_seeds=(0,))
    dsp.build_static_plan(str(exp_dir), str(exp_dir))
    dsp.prepare_stage(str(exp_dir))

    assert staged["ds_fake_a"]["opt_seed"] == 1000      # not the run's seed 0


def test_fixed_dataset_runs_share_data_and_vary_only_the_init(tmp_path,
                                                              monkeypatch,
                                                              fake_registry,
                                                              stub_train):
    """An ATE arm: same SCM, same split, different model seeds."""
    exp_dir, plan = _prepared_fixed_experiment(
        tmp_path, monkeypatch, best_params={"training.lr": 0.02},
        data_seeds=(0,), model_seeds=(1, 2))

    for task_id in range(plan["n_train_tasks"]):
        dsp.train_task(str(exp_dir), task_id)

    per_dataset = {}
    for call in stub_train:
        per_dataset.setdefault(call["dataset"], []).append(call)

    assert sorted(per_dataset) == list(FIXED_NAMES)
    for calls in per_dataset.values():
        assert {c["data_seed"] for c in calls} == {0}     # same split
        assert sorted(c["seed"] for c in calls) == [1, 2]  # different inits
        assert {c["lr"] for c in calls} == {0.02}          # tuned per group


def test_fixed_dataset_cleanup_prunes_the_shared_folder_once(tmp_path, monkeypatch,
                                                             fake_registry):
    """
    Every seed of a fixed group maps to one folder, so a naive loop would prune
    it N+1 times and over-report the reclaimed size.
    """
    pruned = []
    monkeypatch.setattr(dsp, "prune_dag_arrays",
                        lambda path: pruned.append(Path(path).name) or 1000)

    exp_dir, _plan_obj = _prepared_fixed_experiment(
        tmp_path, monkeypatch, data_seeds=(0, 1), model_seeds=(1, 2))

    dsp.cleanup_stage(str(exp_dir))

    assert sorted(pruned) == list(FIXED_NAMES)



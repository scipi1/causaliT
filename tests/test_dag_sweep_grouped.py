"""
Tests for the grouped DAG sweep (``euler_sweep.opt_train_sweep`` + ``dag_provider``).

The point of the grouped sweep is a *cost* claim: tune once per DAG size and
reuse those hyper-parameters for every seed, i.e.

    runs = (#groups studies) + (#groups x #seeds)      instead of
    runs = #groups x #seeds x #trials

These tests therefore pin the behaviour that makes the claim true:

* seeds are members of a group, never an axis of the group product;
* the optimisation seed must be disjoint from the evaluation seeds;
* every seed of a group receives the *same* best params, on its *own* DAG;
* ``dag_seeds`` and ``model_seeds`` are decoupled: several initialisations can
  train on ONE sampled DAG with ONE data split (the basis of edge stability);
* heavy arrays (``ds.npz``) are pruned after each run but the dataset stays
  regenerable from ``dag_recipe.json``.

Training is stubbed out - the sweep's job is orchestration, and real training is
covered elsewhere.  DAG sampling is *not* stubbed: it is cheap at n_nodes=4 and
it is exactly the integration point that used to break.
"""

import json
from os.path import exists, join

import pytest
from omegaconf import OmegaConf

from causaliT.euler_sweep.euler_sweep import dag_provider as dp
from causaliT.euler_sweep.euler_sweep import opt_train_sweep as ots


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DAG_BLOCK = {
    "degree": 2,
    "linearity": "linear",
    "noise": "gaussian",
    "n_samples": 200,
}


def _write_experiment(tmp_path, seeds=(0, 1), n_nodes=(4,), optuna_enabled=False,
                      extra_spec=None, seed_key="seeds"):
    """Create a minimal experiment folder: base config + dagsweep spec."""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True)

    OmegaConf.save(
        OmegaConf.create({
            "experiment": {"name": "test"},
            "data": {"dataset": "PLACEHOLDER"},
            "training": {"seed": 0, "learning_rate": 0.001},
        }),
        exp_dir / "config_test.yaml",
    )

    spec = {
        "group_axes": {"n_nodes": list(n_nodes)},
        seed_key: list(seeds),
        "dag": dict(DAG_BLOCK),
        "optuna": {"enabled": optuna_enabled, "opt_seed": 1000},
        "training": {"trainer": "standard"},
        "dataset_derived": {
            "experiment.n_source": "variable_info.source_labels",
            "experiment.n_input": "variable_info.input_labels",
        },
    }
    spec.update(extra_spec or {})
    OmegaConf.save(OmegaConf.create(spec), exp_dir / "dagsweep.yaml")
    return exp_dir


@pytest.fixture
def stub_training(monkeypatch):
    """Replace the trainer + run wrapper with a recorder."""
    calls = []

    def fake_run_single_combination(config, save_dir, train_fn, data_dir, cluster,
                                    **kwargs):
        calls.append({
            "save_dir": str(save_dir),
            "data_dir": str(data_dir),
            "dataset": config["data"]["dataset"],
            "data_root": config["data"]["data_root"],
            "seed": config["training"]["seed"],
            "data_seed": config["training"].get("data_seed"),
            "config": OmegaConf.to_container(config, resolve=False),
        })

    monkeypatch.setattr(ots, "run_single_combination", fake_run_single_combination)
    monkeypatch.setattr(ots, "resolve_trainer",
                        lambda name: (lambda **kw: None, "mod", "attr"))
    return calls


# ---------------------------------------------------------------------------
# Spec / grouping semantics
# ---------------------------------------------------------------------------

def test_seeds_are_members_not_axes(tmp_path):
    """A 2-size x 3-seed sweep must yield 2 groups of 3 seeds, not 6 groups."""
    exp_dir = _write_experiment(tmp_path, seeds=(0, 1, 2), n_nodes=(4, 5))
    groups = ots.build_groups(ots.load_dagsweep_spec(str(exp_dir)))

    assert len(groups) == 2
    assert [g["name"] for g in groups] == ["n_nodes_4", "n_nodes_5"]
    assert all(g["seeds"] == [0, 1, 2] for g in groups)


def test_opt_seed_overlapping_eval_seeds_is_rejected(tmp_path):
    """Tuning on a DAG that is later evaluated would leak hyper-parameters."""
    exp_dir = _write_experiment(tmp_path, seeds=(0, 1000))
    with pytest.raises(ValueError, match="opt_seed"):
        ots.load_dagsweep_spec(str(exp_dir))


def test_spec_requires_group_axes_and_seeds(tmp_path):
    """A sampled-DAG sweep must say how to GROUP; only 'datasets' may omit it."""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    OmegaConf.save(OmegaConf.create({"seeds": [0], "dag": {"n_nodes": 4}}),
                   exp_dir / "dagsweep.yaml")
    with pytest.raises(ValueError, match="group_axes"):
        ots.load_dagsweep_spec(str(exp_dir))


def test_spec_without_any_data_source_is_rejected(tmp_path):
    """Neither 'dag' nor 'datasets' means the sweep has no data to train on."""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    OmegaConf.save(OmegaConf.create({"group_axes": {"n_nodes": [4]}, "seeds": [0]}),
                   exp_dir / "dagsweep.yaml")
    with pytest.raises(ValueError, match="no data source"):
        ots.load_dagsweep_spec(str(exp_dir))



def test_dag_seeds_is_accepted_and_seeds_stays_an_alias(tmp_path):
    """``dag_seeds`` is the current name; ``seeds`` must keep working."""
    exp_dir = _write_experiment(tmp_path, seeds=(3, 4), seed_key="dag_seeds")
    spec = ots.load_dagsweep_spec(str(exp_dir))

    assert ots.dag_seeds_of(spec) == [3, 4]
    assert ots.model_seeds_of(spec) is None  # unspecified -> follow the DAG seed
    assert ots.build_groups(spec)[0]["dag_seeds"] == [3, 4]


def test_declaring_both_seed_keys_is_rejected(tmp_path):
    """Two sources of truth for the DAG seeds could silently disagree."""
    exp_dir = _write_experiment(tmp_path, seeds=(0, 1),
                                extra_spec={"dag_seeds": [5, 6]})
    with pytest.raises(ValueError, match="ONE seed list"):
        ots.load_dagsweep_spec(str(exp_dir))



def test_empty_model_seeds_is_rejected(tmp_path):
    """An empty list means "no runs" - almost certainly a mistake."""
    exp_dir = _write_experiment(tmp_path, seeds=(0,),
                                extra_spec={"model_seeds": []})
    with pytest.raises(ValueError, match="model_seeds"):
        ots.load_dagsweep_spec(str(exp_dir))


def test_run_plan_is_the_dag_x_model_product(tmp_path):
    """2 DAGs x 3 inits = 6 runs, named so the DAG grouping is recoverable."""
    exp_dir = _write_experiment(tmp_path, seeds=(0, 1),
                                extra_spec={"model_seeds": [7, 8, 9]})
    group = ots.build_groups(ots.load_dagsweep_spec(str(exp_dir)))[0]
    plan = ots.run_plan(group)

    assert len(plan) == 6
    assert [p[0] for p in plan] == [0, 0, 0, 1, 1, 1]      # grouped by DAG seed
    assert [p[1] for p in plan[:3]] == [7, 8, 9]           # inits within a DAG
    assert plan[0][3] == "dag_0_model_7"

    # Without model_seeds the legacy one-run-per-DAG naming is preserved.
    legacy = ots.run_plan(ots.build_groups(
        ots.load_dagsweep_spec(str(_write_experiment(tmp_path / "b", seeds=(0, 1))))
    )[0])
    assert [p[1] for p in legacy] == [None, None]
    assert [p[3] for p in legacy] == ["seed_0", "seed_1"]


def test_unknown_dag_key_is_rejected():
    """A typo like ``n_node`` must fail loudly, not silently change nothing."""
    with pytest.raises(ValueError, match="Unknown key"):
        dp.split_dag_block({"n_node": 10})


# ---------------------------------------------------------------------------
# Dataset provisioning
# ---------------------------------------------------------------------------

def test_dataset_name_is_seed_specific():
    """Seeds of one group must not collide in the same dataset folder."""
    names = {
        dp.dag_dataset_name(dp.build_dag_config(DAG_BLOCK, seed=s, n_nodes=4))
        for s in (0, 1, 2)
    }
    assert len(names) == 3


def test_materialized_dag_prunes_but_stays_regenerable(tmp_path):
    """After the scope exits, ds.npz is gone yet the dataset is rebuildable."""
    cfg = dp.build_dag_config(DAG_BLOCK, seed=0, n_nodes=4)
    root = tmp_path / "datasets"
    root.mkdir()

    with dp.materialized_dag(cfg, str(root), {"n_samples": 200}) as name:
        dataset_dir = join(str(root), name)
        assert dp.has_arrays(dataset_dir), "arrays must exist during the scope"
        assert exists(join(dataset_dir, "dataset_metadata.json"))

    assert not dp.has_arrays(dataset_dir), "arrays must be pruned after the scope"
    assert dp.is_materialized(dataset_dir), "light artefacts must survive pruning"

    recipe = dp.read_recipe(dataset_dir)
    assert recipe["random_scm_config"]["seed"] == 0
    assert recipe["random_scm_config"]["n_nodes"] == 4

    dp.regenerate_from_recipe(dataset_dir, verbose=False)
    assert dp.has_arrays(dataset_dir), "recipe must fully rebuild the arrays"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def test_dry_run_reports_plan_without_generating(tmp_path, stub_training):
    exp_dir = _write_experiment(tmp_path, seeds=(0, 1), n_nodes=(4, 5))
    result = ots.run_dag_sweep(str(exp_dir), dry_run=True)

    assert result["n_runs"] == 4
    assert result["dry_run"] is True
    assert not (exp_dir / "groups").exists()
    assert stub_training == []


def test_each_seed_trains_on_its_own_dag_with_shared_best_params(tmp_path, stub_training):
    """The core contract: one tuned param set, N seeds, N distinct DAGs."""
    exp_dir = _write_experiment(tmp_path, seeds=(0, 1), n_nodes=(4,))

    # Pretend a study already ran for this group.
    group_dir = exp_dir / "groups" / "n_nodes_4"
    group_dir.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create({"params": {"training.learning_rate": 0.042}}),
        group_dir / "best_trial.yaml",
    )

    ots.run_dag_sweep(str(exp_dir), skip_optuna=True)

    assert len(stub_training) == 2

    # Same hyper-parameters for both seeds ...
    assert {c["config"]["training"]["learning_rate"] for c in stub_training} == {0.042}
    # ... but different DAGs, and the training seed follows the DAG seed.
    assert len({c["dataset"] for c in stub_training}) == 2
    assert sorted(c["seed"] for c in stub_training) == [0, 1]

    # Data lives inside the experiment folder, never in the shared data/ dir.
    for call in stub_training:
        assert call["data_root"].endswith(join("n_nodes_4", "datasets"))
        assert "n_nodes_4" in call["save_dir"]

    # dataset_derived refreshed the variable counts from the sampled DAG.
    for call in stub_training:
        exp_block = call["config"]["experiment"]
        assert exp_block["n_source"] >= 1
        assert exp_block["n_input"] >= 1
        assert exp_block["n_source"] + exp_block["n_input"] == 4
        assert exp_block["n_nodes"] == 4


def test_model_seeds_share_one_dag_and_one_data_split(tmp_path, stub_training):
    """
    Edge stability needs the graph AND the split held fixed across inits.

    So several ``model_seeds`` on one ``dag_seed`` must produce runs that differ
    ONLY in ``training.seed``: same dataset folder (generated once) and the same
    ``training.data_seed`` (= the DAG seed) driving the train/val/test split.
    """
    exp_dir = _write_experiment(tmp_path, seeds=(0, 1), n_nodes=(4,),
                                extra_spec={"model_seeds": [7, 8]})

    results = ots.run_dag_sweep(str(exp_dir), skip_optuna=True)

    assert len(stub_training) == 4  # 2 DAGs x 2 inits

    per_dag = {}
    for call in stub_training:
        per_dag.setdefault(call["data_seed"], []).append(call)

    assert sorted(per_dag) == [0, 1], "data_seed must follow the DAG seed"
    for dag_seed, calls in per_dag.items():
        # One DAG -> one dataset shared by every initialisation.
        assert len({c["dataset"] for c in calls}) == 1
        # ... and only the model seed varies.
        assert sorted(c["seed"] for c in calls) == [7, 8]

    # Distinct DAG seeds still give distinct datasets.
    assert len({c["dataset"] for c in stub_training}) == 2

    # Result keys and run folders name both seeds.
    seeds = results["groups"]["n_nodes_4"]["seeds"]
    assert set(seeds) == {"dag_0_model_7", "dag_0_model_8",
                          "dag_1_model_7", "dag_1_model_8"}
    assert seeds["dag_0_model_7"]["dag_seed"] == 0
    assert seeds["dag_0_model_7"]["model_seed"] == 7
    assert all(r["status"] == "ok" for r in seeds.values())


def test_legacy_single_seed_run_sets_matching_data_seed(tmp_path, stub_training):
    """Without model_seeds both seeds coincide, so nothing changes downstream."""
    exp_dir = _write_experiment(tmp_path, seeds=(0, 1), n_nodes=(4,))
    ots.run_dag_sweep(str(exp_dir), skip_optuna=True)

    for call in stub_training:
        assert call["seed"] == call["data_seed"]


def test_failing_run_does_not_abort_the_sweep(tmp_path, monkeypatch):
    """One pathological DAG must not cost us the other 39 runs."""
    exp_dir = _write_experiment(tmp_path, seeds=(0, 1), n_nodes=(4,))

    seen = []

    def flaky(config, save_dir, train_fn, data_dir, cluster, **kwargs):
        seen.append(int(config["training"]["seed"]))
        if int(config["training"]["seed"]) == 0:
            raise RuntimeError("boom")

    monkeypatch.setattr(ots, "run_single_combination", flaky)
    monkeypatch.setattr(ots, "resolve_trainer",
                        lambda name: (lambda **kw: None, "mod", "attr"))

    results = ots.run_dag_sweep(str(exp_dir), skip_optuna=True)
    seeds = results["groups"]["n_nodes_4"]["seeds"]

    assert seen == [0, 1]
    assert seeds[0]["status"] == "failed"
    assert seeds[1]["status"] == "ok"


def test_resolve_trainer_registry():
    for name in ("standard", "staged", "anm", "adaptive"):
        fn, module, attr = ots.resolve_trainer(name)
        assert callable(fn)
        assert module == "causaliT.euler_sweep.euler_sweep.cli"
        assert attr.endswith("_for_sweep")

    with pytest.raises(ValueError, match="Unknown trainer"):
        ots.resolve_trainer("nope")


def test_group_config_is_staged_for_optuna(tmp_path):
    """``optimize_group`` expects a config folder; check what we hand it."""
    exp_dir = _write_experiment(tmp_path, seeds=(0,), n_nodes=(4,))
    spec = ots.load_dagsweep_spec(str(exp_dir))
    base_config, base_optuna = ots._find_base_files(str(exp_dir))
    assert base_optuna is None

    group_dir = tmp_path / "group"
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    cfg = dp.build_dag_config(DAG_BLOCK, seed=1000, n_nodes=4)
    with dp.materialized_dag(cfg, str(datasets_dir), {"n_samples": 200}) as name:
        staged = ots.stage_group_config(
            base_config, base_optuna, group_dir,
            {"axes": {"n_nodes": 4}, "name": "n_nodes_4", "seeds": [0]},
            spec, name, str(datasets_dir),
        )

    assert staged["data"]["dataset"] == name
    assert staged["data"]["data_root"] == str(datasets_dir)
    assert (group_dir / "config_test.yaml").exists()

    saved = OmegaConf.load(group_dir / "config_test.yaml")
    assert saved["experiment"]["n_nodes"] == 4


def test_recipe_backed_scm_is_rebuilt_for_ate(tmp_path):
    """
    ``eval_ate_mc`` needs a live SCM.  Sampled DAGs are not in the registry, so
    the recipe is the only way to get one - and it must work *after* pruning.
    """
    from causaliT.evaluation.eval_funs.eval_interventions import get_scm_for_dataset

    cfg = dp.build_dag_config(DAG_BLOCK, seed=7, n_nodes=4)
    root = tmp_path / "datasets"
    root.mkdir()
    with dp.materialized_dag(cfg, str(root), {"n_samples": 200}) as name:
        pass  # arrays pruned on exit

    scm_dataset = get_scm_for_dataset(name, datadir_path=str(root))
    assert scm_dataset is not None
    assert hasattr(scm_dataset, "scm")

    # Unknown dataset with no recipe must still raise (no silent fallback).
    with pytest.raises(ValueError, match="not found in SCM registry"):
        get_scm_for_dataset("does_not_exist", datadir_path=str(root))


def test_resolve_datadir_precedence(tmp_path):
    """Evaluations must find the group-local data root, not the repo default."""
    from causaliT.evaluation.eval_funs.helpers.datadir import resolve_datadir
    from causaliT.paths import DATA_DIR

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    OmegaConf.save(
        OmegaConf.create({"data": {"dataset": "d", "data_root": "/abs/root"}}),
        run_dir / "config_test.yaml",
    )

    # 1. explicit wins
    assert resolve_datadir(explicit="/x", experiment=str(run_dir)) == "/x"
    # 2. config wins over the saved file
    cfg = OmegaConf.create({"data": {"data_root": "/from/cfg"}})
    assert resolve_datadir(config=cfg, experiment=str(run_dir)) == "/from/cfg"
    # 3. saved run config is used when no config is passed
    assert resolve_datadir(experiment=str(run_dir)) == "/abs/root"
    # 4. legacy default
    assert resolve_datadir() == str(DATA_DIR)


def test_recipe_json_is_self_contained(tmp_path):
    """A recipe must pin every field needed to rebuild the dataset."""
    cfg = dp.build_dag_config(DAG_BLOCK, seed=3, n_nodes=4)
    root = tmp_path / "datasets"
    root.mkdir()
    name = dp.generate_dag_dataset(cfg, str(root), {"n_samples": 200}, verbose=False)

    with open(join(str(root), name, dp.RECIPE_FILENAME), encoding="utf-8") as fh:
        recipe = json.load(fh)

    assert set(recipe) >= {"random_scm_config", "generation"}
    assert recipe["generation"]["n_samples"] == 200
    assert recipe["random_scm_config"]["degree"] == 2
    assert recipe["random_scm_config"]["linearity"] == "linear"

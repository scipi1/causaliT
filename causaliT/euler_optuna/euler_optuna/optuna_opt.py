"""
Generic Optuna Optimisation Framework

This module is the framework core — it is project-agnostic and should not be
modified.  causaliT-specific logic (training function, sampling bounds, metric
extraction) lives exclusively in cli.py.

Copied from euler_workflow/euler_optuna/euler_optuna/optuna_opt.py.
"""

from omegaconf import OmegaConf
import torch
import yaml
from pathlib import Path
import optuna
import os
from optuna.exceptions import DuplicatedStudyError
from optuna.study import MaxTrialsCallback
from os.path import dirname, abspath, join
import sys
import re
from functools import partial
from typing import Callable, Dict, Any

ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)


# =============================================================================
# TEMPLATE STUBS  (replaced by passing callables into OptunaStudy)
# =============================================================================

def sample_params_template(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Template — replace by passing ``sample_params_fn`` to OptunaStudy.

    Returns a flat dict of dotted config keys → sampled values.
    """
    return {
        "model.hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=64),
        "model.num_layers": trial.suggest_int("num_layers", 2, 6),
        "training.lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "training.dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
    }


def train_function_template(
    config: dict,
    save_dir: Path,
    data_dir: Path,
    experiment_tag: str,
    cluster: bool,
    **kwargs,
) -> Any:
    """Template — replace by passing ``train_fn`` to OptunaStudy."""
    raise NotImplementedError(
        "Implement train_function_template or pass train_fn= to OptunaStudy."
    )


def get_metrics_template(train_results: Any) -> Dict[str, Any]:
    """Template — replace by passing ``get_metrics_fn`` to OptunaStudy."""
    raise NotImplementedError(
        "Implement get_metrics_template or pass get_metrics_fn= to OptunaStudy."
    )


# =============================================================================
# Helper Functions
# =============================================================================

def get_config_run(base_config, exp_path: Path, params: Dict[str, Any], trial_id: int):
    """
    Create a trial-specific configuration.

    Applies parameter overrides to a copy of the base config, saves it to
    ``exp_path/optuna/run_<trial_id>/config.yaml``.

    Args:
        base_config: Base OmegaConf configuration.
        exp_path:    Experiment directory.
        params:      Dict of dotted keys → values to override.
        trial_id:    Trial number.

    Returns:
        Tuple of (config_path, save_dir).
    """
    config_ = base_config.copy()
    OmegaConf.set_struct(config_, False)

    for dotted_key, val in params.items():
        OmegaConf.update(config_, dotted_key, val, merge=True)

    save_dir = Path(join(exp_path, "optuna", f"run_{trial_id}"))
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / "config.yaml"
    OmegaConf.save(config_, config_path)

    return config_path, save_dir


def objective_extended(
    trial: optuna.Trial,
    sample_params: Callable,
    train_function: Callable,
    get_metrics: Callable,
    config: dict,
    exp_path: Path,
    data_dir: Path,
    experiment_tag: str,
    cluster: bool,
    optimization_metric: str = "val_loss",
    optimization_direction: str = "minimize",
):
    """
    Optuna objective function.

    Orchestrates:
    1. Sampling hyperparameters via ``sample_params``
    2. Creating a trial-specific config via ``get_config_run``
    3. Training via ``train_function``
    4. Extracting + logging metrics via ``get_metrics``
    5. Returning the scalar optimisation metric

    Args:
        trial:                Optuna trial object.
        sample_params:        Function (trial) → dict of param overrides.
        train_function:       Function (config, save_dir, data_dir, …) → results.
        get_metrics:          Function (train_results) → dict of metric values.
        config:               Base OmegaConf configuration.
        exp_path:             Experiment directory.
        data_dir:             Data directory.
        experiment_tag:       Experiment tag for manifests.
        cluster:              Whether running on a cluster.
        optimization_metric:  Key to optimise (must be in metrics dict).
        optimization_direction: ``"minimize"`` or ``"maximize"``.

    Returns:
        Scalar value of the optimisation metric.
    """
    params = sample_params(trial)

    config_path, save_dir = get_config_run(config, exp_path, params, trial.number)
    config_run = OmegaConf.load(config_path)

    if cluster and torch.cuda.is_available():
        torch.cuda.set_device(0)

    try:
        train_results = train_function(
            config=config_run,
            data_dir=data_dir,
            save_dir=save_dir,
            experiment_tag=experiment_tag,
            cluster=cluster,
            resume_ckpt=None,
            debug=False,
        )
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        # Persist the reason on the trial.  Optuna keeps FAILED trials but not
        # their exception, so without this a study where every trial crashed can
        # only report "no completed trial" and the real cause is lost.
        try:
            trial.set_user_attr("failure", f"{type(e).__name__}: {e}")
        except Exception:
            pass
        raise

    metrics = get_metrics(train_results)

    for metric_name, metric_value in metrics.items():
        trial.set_user_attr(metric_name, float(metric_value))
    trial.set_user_attr("config_path", str(config_path))

    if optimization_metric not in metrics:
        raise ValueError(
            f"Optimisation metric '{optimization_metric}' not found. "
            f"Available: {list(metrics.keys())}"
        )

    return metrics[optimization_metric]


# =============================================================================
# Main Optuna Study Class
# =============================================================================

class OptunaStudy:
    """
    Manages an Optuna hyperparameter optimisation study.

    Handles study creation, resumption, and results summary.  All
    project-specific logic is injected via callable arguments.

    Args:
        exp_dir:              Experiment directory (must contain a ``config*.yaml``).
        data_dir:             Data directory.
        cluster:              Whether running on a cluster.
        study_name:           Name for the Optuna study.
        manifest_tag:         Tag stored in experiment manifests.
        study_path:           Optional path to store the SQLite database.
        sample_params_fn:     (trial) → dict of param overrides.
        train_fn:             Training callable.
        get_metrics_fn:       Metric extraction callable.
        optimization_metric:  Metric to optimise (default ``"val_loss"``).
        optimization_direction: ``"minimize"`` or ``"maximize"``.
    """

    def __init__(
        self,
        exp_dir: Path,
        data_dir: Path,
        cluster: bool,
        study_name: str,
        manifest_tag: str,
        study_path: str = None,
        sample_params_fn: Callable = sample_params_template,
        train_fn: Callable = train_function_template,
        get_metrics_fn: Callable = get_metrics_template,
        optimization_metric: str = "val_loss",
        optimization_direction: str = "minimize",
    ):
        self.exp_dir = exp_dir
        self.data_dir = data_dir
        self.cluster = cluster
        self.study_name = study_name
        self.manifest_tag = manifest_tag
        self.optimization_metric = optimization_metric
        self.optimization_direction = optimization_direction

        # Load experiment config
        pattern_config = re.compile(r"config.*\.yaml")
        config_files = [f for f in os.listdir(exp_dir) if pattern_config.match(f)]
        if len(config_files) == 1:
            config = OmegaConf.load(join(exp_dir, config_files[0]))
        elif len(config_files) == 0:
            raise ValueError(f"No config file found in {exp_dir}")
        else:
            raise ValueError(f"Multiple config files in {exp_dir}: {config_files}")

        # Load optional optuna settings
        pattern_optuna = re.compile(r"optuna.*\.yaml")
        optuna_files = [f for f in os.listdir(exp_dir) if pattern_optuna.match(f)]
        self.optuna_settings = (
            OmegaConf.load(join(exp_dir, optuna_files[0]))
            if len(optuna_files) == 1
            else None
        )

        # Build objective
        obj_kwargs = dict(
            sample_params=sample_params_fn,
            train_function=train_fn,
            get_metrics=get_metrics_fn,
            config=config,
            exp_path=exp_dir,
            data_dir=data_dir,
            experiment_tag=manifest_tag,
            cluster=cluster,
            optimization_metric=optimization_metric,
            optimization_direction=optimization_direction,
        )
        self.objective = partial(objective_extended, **obj_kwargs)

        # Storage path
        if study_path is None:
            study_path = (Path(exp_dir) / "optuna").resolve()
            study_path.mkdir(parents=True, exist_ok=True)

        self.study_file_path = join(study_path, "study.db")
        self.storage = f"sqlite:///{self.study_file_path}?timeout=60"

        self.max_trials = self._setting("n_trials", 50)
        self.direction = self._setting("direction", optimization_direction)
        self.pruner = self._setting("pruner", "none")

    # ------------------------------------------------------------------

    def create(self):
        """Create a new Optuna study (raises DuplicatedStudyError if it exists)."""
        try:
            study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                sampler=self._build_sampler(),
                pruner=self._build_pruner(),
                storage=self.storage,
            )
            study.set_user_attr("n_trials_total", self.max_trials)
            print(f"Created study '{self.study_name}' at {self.storage}")
            print(f"Target: {self.max_trials} trials | optimising '{self.optimization_metric}' ({self.direction})")
        except DuplicatedStudyError:
            print(f"Study '{self.study_name}' already exists — use --mode resume to continue.")

    def resume(self, storage=None):
        """Resume an existing study, running until ``max_trials`` is reached."""
        stor = storage or self.storage
        study = optuna.load_study(study_name=self.study_name, storage=stor)
        print(f"Resuming '{self.study_name}': {len(study.trials)} / {self.max_trials} trials done.")

        study.optimize(
            self.objective,
            n_trials=None,
            callbacks=[MaxTrialsCallback(self.max_trials, states=None)],
            gc_after_trial=True,
            catch=(RuntimeError,),
        )
        print(f"Study complete. Total trials: {len(study.trials)}")

    def summary(self):
        """Print and save a YAML summary of the best trial."""
        study = optuna.load_study(study_name=self.study_name, storage=self.storage)
        best = study.best_trial

        print(f"\n{'='*60}")
        print("Best Trial Summary")
        print(f"{'='*60}")
        print(f"Trial number : {best.number}")
        print(f"Metric ({self.optimization_metric}): {best.value:.6f}")
        print("\nBest Parameters:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")

        to_dump = {
            "trial_number": best.number,
            "optimization_metric": self.optimization_metric,
            "optimization_value": float(best.value),
            "config_path": best.user_attrs.get("config_path"),
            "params": best.params,
            "metrics": {},
        }
        for k, v in best.user_attrs.items():
            if k != "config_path":
                to_dump["metrics"][k] = float(v) if isinstance(v, (int, float)) else v

        summary_path = Path(self.exp_dir) / "best_trial.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(to_dump, f, default_flow_style=False, sort_keys=False)

        print(f"\nSummary saved to: {summary_path}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setting(self, key: str, default: Any) -> Any:
        return (self.optuna_settings or {}).get(key, default)

    def _build_sampler(self) -> optuna.samplers.BaseSampler:
        if self.optuna_settings is None or "sampler" not in self.optuna_settings:
            return optuna.samplers.QMCSampler(qmc_type="sobol")
        cfg = dict(self.optuna_settings["sampler"])
        name = cfg.pop("name", "sobol").lower()
        if name == "sobol":
            return optuna.samplers.QMCSampler(qmc_type="sobol", **cfg)
        elif name == "tpe":
            return optuna.samplers.TPESampler(**cfg)
        else:
            raise ValueError(f"Unknown sampler: '{name}'. Use 'sobol' or 'tpe'.")

    def _build_pruner(self) -> optuna.pruners.BasePruner:
        name = self._setting("pruner", "none")
        if name == "none":
            return None
        elif name == "median":
            n_warmup = self._setting("pruner_warmup", 5)
            return optuna.pruners.MedianPruner(n_warmup_steps=n_warmup)
        elif name == "hyperband":
            return optuna.pruners.HyperbandPruner()
        else:
            print(f"Warning: unknown pruner '{name}', using none.")
            return None

"""
euler_optuna — inner package for causaliT hyperparameter optimisation.

Key Components:
- optuna_opt.py       : Core framework (OptunaStudy class, objective_extended)
- optuna_parallel.py  : Parallel execution orchestrator (SLURM job arrays)
- cli.py              : causaliT-specific CLI + sampling functions
- optuna_worker.py    : SLURM array task worker
"""

from .optuna_opt import (
    OptunaStudy,
    objective_extended,
    get_config_run,
    sample_params_template,
    train_function_template,
    get_metrics_template,
)

from .optuna_parallel import run_parallel_optuna

__all__ = [
    "OptunaStudy",
    "objective_extended",
    "get_config_run",
    "sample_params_template",
    "train_function_template",
    "get_metrics_template",
    "run_parallel_optuna",
]

"""
DAG sweep worker - the entry point every SLURM job of the chain calls.

One thin CLI per stage of :mod:`dagsweep_parallel`, so the generated SLURM
scripts contain a single ``python -m ... <stage>`` line and all logic stays
testable in-process::

    python -m causaliT.euler_sweep.euler_sweep.dagsweep_worker prepare --exp_dir DIR
    python -m ... dagsweep_worker trial   --exp_dir DIR --task_id $SLURM_ARRAY_TASK_ID
    python -m ... dagsweep_worker select  --exp_dir DIR
    python -m ... dagsweep_worker train   --exp_dir DIR --task_id $SLURM_ARRAY_TASK_ID
    python -m ... dagsweep_worker cleanup --exp_dir DIR

Every stage reads ``<exp_dir>/dagsweep/plan.json`` (written by the driver) and
writes its outcome to ``<exp_dir>/dagsweep/progress/``, so a killed job leaves a
readable trace of planned-vs-reached work.
"""

import logging
import sys
from pathlib import Path

import click

# Make the repository importable when the worker is launched as a bare script.
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from causaliT.euler_sweep.euler_sweep.dagsweep_parallel import (  # noqa: E402
    cleanup_stage,
    format_progress,
    prepare_stage,
    rebuild_progress,
    select_stage,
    train_task,
    trial_task,
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


@click.group()
def cli():
    """Stages of a parallel DAG sweep (called by the generated SLURM scripts)."""
    _setup_logging()


@cli.command()
@click.option("--exp_dir", required=True, help="Experiment folder holding dagsweep/plan.json")
def prepare(exp_dir):
    """Generate every DAG, stage the group configs, create the Optuna studies."""
    prepared = prepare_stage(exp_dir)
    for name, entry in prepared["groups"].items():
        print(f"[prepare] {name}: n_keys={entry.get('n_keys')} "
              f"opt={entry.get('opt_dataset')} "
              f"eval_datasets={len(entry.get('datasets', {}))}")


@cli.command()
@click.option("--exp_dir", required=True, help="Experiment folder holding dagsweep/plan.json")
@click.option("--task_id", required=True, type=int, help="SLURM_ARRAY_TASK_ID")
def trial(exp_dir, task_id):
    """Run ONE Optuna trial (array task of the search phase)."""
    trial_task(exp_dir, task_id)


@cli.command()
@click.option("--exp_dir", required=True, help="Experiment folder holding dagsweep/plan.json")
def select(exp_dir):
    """Select each group's winning trial and write its best_trial.yaml."""
    summary = select_stage(exp_dir)
    for name, params in summary.items():
        print(f"[select] {name}: {len(params)} tuned param(s)")


@cli.command()
@click.option("--exp_dir", required=True, help="Experiment folder holding dagsweep/plan.json")
@click.option("--task_id", required=True, type=int, help="SLURM_ARRAY_TASK_ID")
@click.option("--force", is_flag=True, default=False,
              help="Re-train even when this run is already marked ok")
def train(exp_dir, task_id, force):
    """Train ONE (dag_seed, model_seed) run (array task of the seed sweep)."""
    train_task(exp_dir, task_id, force=force)


@cli.command()
@click.option("--exp_dir", required=True, help="Experiment folder holding dagsweep/plan.json")
def cleanup(exp_dir):
    """Prune the heavy dataset arrays and write the final progress report."""
    cleanup_stage(exp_dir)


@cli.command()
@click.option("--exp_dir", required=True, help="Experiment folder holding dagsweep/plan.json")
def status(exp_dir):
    """Print the planned-vs-reached report (also refreshes the rollup file)."""
    print(format_progress(rebuild_progress(exp_dir)))


if __name__ == "__main__":
    cli()

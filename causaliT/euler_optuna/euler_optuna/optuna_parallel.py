"""
Parallel Optuna Execution — SLURM Job Array Support

Each SLURM array task runs one trial.  The worker module is configurable so
this file remains project-agnostic.

Copied from euler_workflow/euler_optuna/euler_optuna/optuna_parallel.py.
"""

import subprocess
import os
import re
from pathlib import Path
from os import makedirs
from os.path import exists, join
import optuna
from omegaconf import OmegaConf


def run_parallel_optuna(
    exp_dir: str,
    home_exp_dir: str,
    experiment_id: str,
    study_name: str,
    n_trials: int,
    data_dir: str,
    worker_module: str = "causaliT.euler_optuna.euler_optuna.optuna_worker",
    scratch_path: str = None,
    slurm_params: dict = None,
    cluster: bool = True,
    optimization_metric: str = "val_loss",
    optimization_direction: str = "minimize",
    study_path: str = None,
):
    """
    Load an existing study, compute remaining trials, generate a SLURM array
    script, and submit it.

    Args:
        exp_dir:              Experiment directory (may be on SCRATCH).
        home_exp_dir:         Home experiment directory (holds config files).
        experiment_id:        Experiment identifier (used in job name).
        study_name:           Name of the Optuna study (must already exist).
        n_trials:             CLI fallback for total trials if not in study/config.
        data_dir:             Data directory.
        worker_module:        Python module path to the worker script.
        scratch_path:         Optional SCRATCH path override.
        slurm_params:         Dict with SLURM resource parameters.
        cluster:              Whether running on a cluster.
        optimization_metric:  Metric to optimise.
        optimization_direction: ``"minimize"`` or ``"maximize"``.
        study_path:           Optional path to the study database.

    Raises:
        ValueError: If the study does not exist (create it first).
    """
    if slurm_params is None:
        slurm_params = {}
    slurm_params.setdefault("max_concurrent_jobs", 6)
    slurm_params.setdefault("walltime", "5-00:00:00")
    slurm_params.setdefault("gpu_type", "rtx_4090")
    slurm_params.setdefault("mem_per_cpu", "23g")

    print(f"\nParallel Optuna: {experiment_id} / {study_name}")

    # Resolve study path
    if study_path is None:
        study_path = Path(exp_dir) / "optuna"
        study_path.mkdir(parents=True, exist_ok=True)
    else:
        study_path = Path(study_path)

    storage = f"sqlite:///{join(study_path, 'study.db')}?timeout=60"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Loaded study: {len(study.trials)} trials completed.")
    except KeyError:
        raise ValueError(
            f"Study '{study_name}' not found. Create it first:\n"
            f"  python -m causaliT.euler_optuna.euler_optuna.cli paramsopt "
            f"--exp_id {experiment_id} --study_name {study_name} --mode create"
        )

    n_trials_total = _get_trial_limit(study, home_exp_dir, n_trials)

    current_trials = len(study.trials)
    remaining_trials = n_trials_total - current_trials

    if remaining_trials <= 0:
        print(f"\nStudy already complete ({current_trials}/{n_trials_total}). No jobs submitted.")
        return

    print(f"\nPlan:")
    print(f"  Total trials configured : {n_trials_total}")
    print(f"  Already completed       : {current_trials}")
    print(f"  Remaining to submit     : {remaining_trials}")
    print(f"  Max concurrent jobs     : {slurm_params['max_concurrent_jobs']}")

    script_path = _generate_slurm_script(
        exp_dir=exp_dir,
        home_exp_dir=home_exp_dir,
        experiment_id=experiment_id,
        study_name=study_name,
        data_dir=data_dir,
        n_trials=remaining_trials,
        worker_module=worker_module,
        slurm_params=slurm_params,
        optimization_metric=optimization_metric,
        optimization_direction=optimization_direction,
        cluster=cluster,
    )
    print(f"Script: {script_path}")

    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True, cwd=exp_dir)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"\nSubmitted! Job ID: {job_id}")
        print(f"Monitor : squeue -u $USER")
        print(f"Logs    : {exp_dir}/optuna/slurm_logs/\n")
        with open(join(exp_dir, "optuna", "job_id.txt"), "w") as f:
            f.write(job_id)
    else:
        print(f"sbatch error: {result.stderr}")


# =============================================================================
# Internal helpers
# =============================================================================

def _get_trial_limit(study, home_exp_dir: str, cli_n_trials: int) -> int:
    """Determine trial limit: study metadata → optuna*.yaml → CLI fallback."""
    # 1. Study metadata (new studies)
    try:
        n = study.user_attrs.get("n_trials_total")
        if n is not None:
            print(f"Trial limit from study metadata: {n}")
            return n
    except Exception:
        pass

    # 2. optuna*.yaml in experiment directory
    try:
        optuna_files = [f for f in os.listdir(home_exp_dir) if re.match(r"optuna.*\.yaml", f)]
        if len(optuna_files) == 1:
            cfg = OmegaConf.load(join(home_exp_dir, optuna_files[0]))
            n = cfg.get("n_trials", None)
            if n is not None:
                print(f"Trial limit from optuna*.yaml: {n}")
                return n
    except Exception:
        pass

    # 3. CLI parameter
    print(f"Warning: trial limit not found in study or config. Using CLI value: {cli_n_trials}")
    return cli_n_trials


def _generate_slurm_script(
    exp_dir: str,
    home_exp_dir: str,
    experiment_id: str,
    study_name: str,
    data_dir: str,
    n_trials: int,
    worker_module: str,
    slurm_params: dict,
    optimization_metric: str,
    optimization_direction: str,
    cluster: bool,
) -> str:
    """Generate a SLURM job-array script and return its path."""
    logs_dir = join(exp_dir, "optuna", "slurm_logs")
    if not exists(logs_dir):
        makedirs(logs_dir)

    cluster_flag = "--cluster" if cluster else ""
    root_dir = str(Path(home_exp_dir).parent.parent)

    script = f"""#!/bin/bash
#SBATCH --job-name=opt_{experiment_id}
#SBATCH --output={logs_dir}/opt_%A_%a.out
#SBATCH --error={logs_dir}/opt_%A_%a.err
#SBATCH --array=0-{n_trials - 1}%{slurm_params['max_concurrent_jobs']}
#SBATCH --ntasks=1
#SBATCH --time={slurm_params['walltime']}
#SBATCH --gpus={slurm_params['gpu_type']}:1
#SBATCH --mem-per-cpu={slurm_params['mem_per_cpu']}

set -euo pipefail

echo "[$(date)] Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID  |  Array Task: $SLURM_ARRAY_TASK_ID"

# ── Environment ──────────────────────────────────────────────────────────────
# TODO: customise module loads for your cluster
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

# TODO: update path to your virtual environment
VENV_PATH="$HOME/myenv"
source "$VENV_PATH/bin/activate"

if [[ -z "${{VIRTUAL_ENV:-}}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"

# ── Run trial ────────────────────────────────────────────────────────────────
cd "{root_dir}"

echo "[$(date)] Running trial $SLURM_ARRAY_TASK_ID ..."

python -m {worker_module} \\
    --exp_dir "{exp_dir}" \\
    --home_exp_dir "{home_exp_dir}" \\
    --study_name "{study_name}" \\
    --data_dir "{data_dir}" \\
    --optimization_metric "{optimization_metric}" \\
    --optimization_direction "{optimization_direction}" \\
    --task_id $SLURM_ARRAY_TASK_ID {cluster_flag}

# ── Wrap-up ──────────────────────────────────────────────────────────────────
deactivate
echo "[$(date)] Job finished."
"""

    script_path = join(exp_dir, "optuna", "run_optuna_array.sh")
    with open(script_path, "w") as f:
        f.write(script)

    return script_path

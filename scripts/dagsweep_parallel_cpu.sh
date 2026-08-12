#!/bin/bash
#SBATCH --job-name=causaliT_dagsweep_parallel
#SBATCH --output=dagsweep_parallel_output_%j.log
#SBATCH --error=dagsweep_parallel_error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4g

# ---------------------------------------------------------------------------
# Submit a PARALLEL grouped DAG sweep with:  sbatch scripts/dagsweep_parallel.sh
#
# This job only PLANS and SUBMITS (no DAG is generated, nothing is trained here).
# The chain it creates is
#
#   prep  ->  trials[array]  ->  select  ->  train[array]  ->  cleanup
#
# where `select` is the barrier that guarantees every training run reads the
# winning trial of its group (why the two phases are not merged).
# ---------------------------------------------------------------------------

set -euo pipefail

echo "[$(date)] Parallel DAG sweep submission job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ---------------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------------
EXPERIMENT_ID="dagma_nonlinear_nongaussian_ER4_10_80"
MAX_CONCURRENT_JOBS=6
# Walltime of ONE array task (a single Optuna trial / a single training run),
# not of the whole sweep: the chain runs as long as it needs to.
WALLTIME="36:00:00"
MEM_PER_CPU="256g"
GPU_MEM="none"
# Python environment; also passed on, so the worker jobs activate the same one.
VENV_PATH="$HOME/myenv"

# Project root and experiment folder in $HOME
PROJ_HOME="$HOME/causaliT"
HOME_EXP="$PROJ_HOME/experiments/$EXPERIMENT_ID"

# Scratch locations
RUN_DIR="$SCRATCH/${EXPERIMENT_ID}_${SLURM_JOB_ID}"
SCRATCH_EXP="$RUN_DIR"

mkdir -p "$SCRATCH_EXP"

echo "[$(date)] Experiment ID   : $EXPERIMENT_ID"
echo "[$(date)] Home exp folder : $HOME_EXP"
echo "[$(date)] Scratch folder  : $SCRATCH_EXP"
echo "[$(date)] Max concurrent  : $MAX_CONCURRENT_JOBS"
echo "[$(date)] Task walltime   : $WALLTIME"
echo "[$(date)] GPU memory      : $GPU_MEM"
echo "[$(date)] Memory per CPU  : $MEM_PER_CPU"

# ---------------------------------------------------------------------------
# COPY INPUTS TO SCRATCH
# ---------------------------------------------------------------------------
# Datasets, checkpoints and sweep state are then BORN in scratch, so $HOME only
# ever holds the spec files (config*.yaml, dagsweep*.yaml, optuna*.yaml).
# `groups/` is EXCLUDED on purpose: a leftover folder from an earlier local run
# would be treated as a finished study and its best_trial.yaml silently reused.
# Drop the --exclude (or use --skip_optuna) to deliberately resume a study.
rsync -av --exclude 'groups/' "$HOME_EXP/" "$SCRATCH_EXP/"

# ---------------------------------------------------------------------------
# ENVIRONMENT SETUP
# ---------------------------------------------------------------------------
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

source "$VENV_PATH/bin/activate"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"

# ---------------------------------------------------------------------------
# BUDGET CALIBRATION (optional, once per partition)
# ---------------------------------------------------------------------------
# `size_derived.experiment.batch_size` with {rule: activation_budget, C: auto}
# reads the activation budget cached per GPU model; without it a conservative
# default is assumed.  Measure it once on a node of the target partition:
#
#   srun --gpus=1 --time=10 python -m causaliT.euler_sweep.euler_sweep.cli calibrate-batch-budget

# ---------------------------------------------------------------------------
# SUBMIT PARALLEL DAG SWEEP
# ---------------------------------------------------------------------------
cd "$SCRATCH_EXP"

echo "[$(date)] Submitting parallel DAG sweep..."

# Add --dry_run to write the job scripts into $SCRATCH_EXP/dagsweep/scripts/ and
# print the plan WITHOUT submitting anything.
python -m causaliT.euler_sweep.euler_sweep.cli dagsweep --exp_id "$EXPERIMENT_ID" --cluster --scratch_path "$SCRATCH_EXP" --max_concurrent_jobs "$MAX_CONCURRENT_JOBS" --walltime "$WALLTIME" --gpu_mem "$GPU_MEM" --mem_per_cpu "$MEM_PER_CPU" --venv_path "$VENV_PATH"

deactivate
echo "[$(date)] Python environment deactivated"
echo "[$(date)] Submission completed - results will be in $SCRATCH_EXP"
echo "[$(date)] Follow the sweep with:"
echo "  python -m causaliT.euler_sweep.euler_sweep.cli dagsweep-status --exp_id $EXPERIMENT_ID --scratch_path $SCRATCH_EXP"
echo "[$(date)] Stage logs: $SCRATCH_EXP/dagsweep/slurm_logs/, plan: $SCRATCH_EXP/dagsweep/plan.json"

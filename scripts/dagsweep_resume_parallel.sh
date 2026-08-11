#!/bin/bash
#SBATCH --job-name=causaliT_dagsweep_resume
#SBATCH --output=dagsweep_resume_output_%j.log
#SBATCH --error=dagsweep_resume_error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4g

# ---------------------------------------------------------------------------
# Re-run ONLY phase 2 of a grouped DAG sweep, reusing a finished Optuna study.
#
#   prep  ->  train[array]  ->  cleanup       (no trials, no select barrier)
#
# Use this when the search succeeded but the training array failed (bad trainer,
# code regression, walltime).  `--skip_optuna` disables study creation and lets
# the training worker load the cached best_trial.yaml of each group; `prep`
# regenerates the pruned ds*.npz from the kept recipe, so the data is identical.
#
# Difference from dagsweep_parallel.sh: `groups/` is NOT excluded from the rsync,
# because that folder is exactly what carries best_trial.yaml.
# ---------------------------------------------------------------------------

set -euo pipefail

echo "[$(date)] DAG sweep RESUME submission job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ---------------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------------
EXPERIMENT_ID="7_PUBLISH/ATE/cheater"

# Scratch folder of the ORIGINAL sweep (the one holding groups/*/best_trial.yaml).
# Point it at a fresh folder only if you copy those files in yourself.
SCRATCH_EXP="$SCRATCH/cheater_9427393"

MAX_CONCURRENT_JOBS=6
WALLTIME="2-00:00:00"
# GPU memory of one train task.  Set to null to run the array on CPU nodes
# (no --gpus / --gres=gpumem is emitted), e.g. for the CPU-only benchmarks.
GPU_MEM="11g"
MEM_PER_CPU="10g"

VENV_PATH="$HOME/myenv"
PROJ_HOME="$HOME/causaliT"
HOME_EXP="$PROJ_HOME/experiments/$EXPERIMENT_ID"

mkdir -p "$SCRATCH_EXP"

echo "[$(date)] Experiment ID   : $EXPERIMENT_ID"
echo "[$(date)] Home exp folder : $HOME_EXP"
echo "[$(date)] Scratch folder  : $SCRATCH_EXP"

# ---------------------------------------------------------------------------
# COPY THE (UPDATED) SPEC FILES, KEEP THE STUDY RESULTS
# ---------------------------------------------------------------------------
rsync -av "$HOME_EXP/" "$SCRATCH_EXP/"

echo "[$(date)] Cached best trials found:"
find "$SCRATCH_EXP/groups" -name best_trial.yaml -print || \
    echo "  NONE -- the runs would train the base config UNTUNED, aborting" >&2
if [[ -z "$(find "$SCRATCH_EXP/groups" -name best_trial.yaml 2>/dev/null)" ]]; then
    exit 1
fi

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
python -c "import torch, torchmetrics; print('torch', torch.__version__, '| torchmetrics', torchmetrics.__version__)"

# ---------------------------------------------------------------------------
# SUBMIT (PHASE 2 ONLY)
# ---------------------------------------------------------------------------
cd "$SCRATCH_EXP"

echo "[$(date)] Submitting phase-2-only DAG sweep..."

# Add --dry_run to write the plan and job scripts WITHOUT submitting; check that
# plan.json has optuna.enabled=false and the expected number of train tasks.
python -m causaliT.euler_sweep.euler_sweep.cli dagsweep --exp_id "$EXPERIMENT_ID" --cluster --scratch_path "$SCRATCH_EXP" --skip_optuna --max_concurrent_jobs "$MAX_CONCURRENT_JOBS" --walltime "$WALLTIME" --gpu_mem "$GPU_MEM" --mem_per_cpu "$MEM_PER_CPU" --venv_path "$VENV_PATH"

deactivate
echo "[$(date)] Submission completed - results will be in $SCRATCH_EXP"
echo "[$(date)] Follow the sweep with:"
echo "  python -m causaliT.euler_sweep.euler_sweep.cli dagsweep-status --exp_id $EXPERIMENT_ID --scratch_path $SCRATCH_EXP"

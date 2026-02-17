#!/bin/bash
#SBATCH --job-name=causaliT_sweep_parallel
#SBATCH --output=sweep_parallel_output_%j.log
#SBATCH --error=sweep_parallel_error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4g

set -euo pipefail

echo "[$(date)] Parallel Sweep submission job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ───────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ───────────────────────────────────────────────────────────────────────
EXPERIMENT_ID="your_experiment_id"
SWEEP_MODE="combination"
MAX_CONCURRENT_JOBS=6
WALLTIME="4:00:00"
GPU_MEM="11g"
MEM_PER_CPU="10g"

# Project root and experiment folder in $HOME
PROJ_HOME="$HOME/causaliT"
HOME_EXP="$PROJ_HOME/experiments/$EXPERIMENT_ID"

# Scratch locations
RUN_DIR="$SCRATCH/${EXPERIMENT_ID}_${SLURM_JOB_ID}"
SCRATCH_EXP="$RUN_DIR"

mkdir -p "$SCRATCH_EXP"

echo "[$(date)] Experiment ID   : $EXPERIMENT_ID"
echo "[$(date)] Sweep mode      : $SWEEP_MODE"
echo "[$(date)] Home exp folder : $HOME_EXP"
echo "[$(date)] Scratch folder  : $SCRATCH_EXP"
echo "[$(date)] Max concurrent  : $MAX_CONCURRENT_JOBS"
echo "[$(date)] Walltime        : $WALLTIME"
echo "[$(date)] GPU memory      : $GPU_MEM"
echo "[$(date)] Memory per CPU  : $MEM_PER_CPU"

# ───────────────────────────────────────────────────────────────────────
# COPY INPUTS TO SCRATCH
# ───────────────────────────────────────────────────────────────────────
rsync -av "$HOME_EXP/" "$SCRATCH_EXP/"

# ───────────────────────────────────────────────────────────────────────
# ENVIRONMENT SETUP
# ───────────────────────────────────────────────────────────────────────
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

source "$HOME/myenv/bin/activate"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"

# ───────────────────────────────────────────────────────────────────────
# SUBMIT PARALLEL SWEEP
# ───────────────────────────────────────────────────────────────────────
cd "$SCRATCH_EXP"

echo "[$(date)] Submitting parallel sweep..."

python -m causaliT.euler_sweep.euler_sweep.cli sweep --exp_id "$EXPERIMENT_ID" --sweep_mode "$SWEEP_MODE" --parallel --cluster --scratch_path "$SCRATCH_EXP" --max_concurrent_jobs "$MAX_CONCURRENT_JOBS" --walltime "$WALLTIME" --gpu_mem "$GPU_MEM" --mem_per_cpu "$MEM_PER_CPU"

deactivate
echo "[$(date)] Python environment deactivated"
echo "[$(date)] Parallel sweep submission completed – results will be in $SCRATCH_EXP"

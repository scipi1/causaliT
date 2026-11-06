#!/bin/bash
# CPU-only variant of adaptivetrain_cluster_scratch_job.sh.
# No GPU requested; multiple cores + matching torch thread count.
# Note: --mem-per-cpu is multiplied by --cpus-per-task (8 x 4g = 32g total).
#SBATCH --job-name=${EXPERIMENT_ID:-my_job}
#SBATCH --output=my_job_output_%j.log
#SBATCH --error=my_job_error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4g

set -euo pipefail

echo "[$(date)] Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# -----------------------------------------------
# 1)  EXPERIMENT SELECTION
# -----------------------------------------------
EXPERIMENT_ID="noDARTS_bkd_recon_cpu"

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

# -----------------------------------------------
# 2)  COPY INPUTS TO SCRATCH
# -----------------------------------------------
rsync -av "$HOME_EXP/" "$SCRATCH_EXP/"

# -----------------------------------------------
# 3)  ENVIRONMENT
# -----------------------------------------------
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6   # CUDA torch build runs fine on CPU

# Activate virtual-env (use absolute path)
source "$HOME/myenv/bin/activate"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"

# CPU-only: hide any GPU and match torch threads to allocated cores
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "[$(date)] CPU threads: $OMP_NUM_THREADS"

# -----------------------------------------------
# 4)  RUN
# -----------------------------------------------
cd "$SCRATCH_EXP"

echo "[$(date)] Running script..."

python -m causaliT.cli adaptivetrain --exp_id "$EXPERIMENT_ID" --cluster True --scratch_path "$SCRATCH_EXP"

# -----------------------------------------------
# 5)  WRAP-UP
# -----------------------------------------------
deactivate
echo "[$(date)] Python environment deactivated"

echo "[$(date)] Job finished - results are still in $SCRATCH_EXP"

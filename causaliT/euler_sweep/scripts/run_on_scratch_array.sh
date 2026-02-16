#!/bin/bash
#SBATCH --job-name=my_job                      # generic at submit-time
#SBATCH --output=my_job_output_%A_%a.log       # %A = job-ID, %a = array index
#SBATCH --error=my_job_error_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --time=15-00:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=10g
#SBATCH --gres=gpumem:24g
#SBATCH --array=0-99%10                        # edit or override at sbatch

set -euo pipefail

echo "[$(date)] Job started on $(hostname)"
echo "Job ID        : $SLURM_JOB_ID"
echo "Array index   : $SLURM_ARRAY_TASK_ID"

# ───────────────────────────────────────────────
# 1)  EXPERIMENT SELECTION  (⇦ only real change)
# ───────────────────────────────────────────────
PROJ_HOME="$HOME/prochain_transformer"

# Folder that *contains* all experiment sub-folders you want to run
JOBS_ROOT="$PROJ_HOME/experiments/training/jobs_array"

# Build a bash array of absolute paths, sorted for reproducibility
mapfile -t ALL_EXPERIMENTS < <(find "$JOBS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
NUM_EXPS=${#ALL_EXPERIMENTS[@]}

if (( SLURM_ARRAY_TASK_ID >= NUM_EXPS )); then
    echo "[$(date)] Array index $SLURM_ARRAY_TASK_ID out of range (0-$((NUM_EXPS-1)))" >&2
    exit 1
fi

HOME_EXP="${ALL_EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
EXPERIMENT_ID="$(basename "$HOME_EXP")"

# (Optional) – rename the running job so squeue shows EXPERIMENT_ID
scontrol update JobId="$SLURM_JOB_ID" JobName="$EXPERIMENT_ID" || true

# Scratch locations (kept identical to original layout)
RUN_DIR="$SCRATCH/${EXPERIMENT_ID}_${SLURM_JOB_ID}"
SCRATCH_EXP="$RUN_DIR/experiments/training/$EXPERIMENT_ID"

mkdir -p "$SCRATCH_EXP"

echo "[$(date)] Experiment ID   : $EXPERIMENT_ID"
echo "[$(date)] Home exp folder : $HOME_EXP"
echo "[$(date)] Scratch folder  : $SCRATCH_EXP"

# ───────────────────────────────────────────────
# 2)  COPY INPUTS TO SCRATCH
# ───────────────────────────────────────────────
rsync -av "$HOME_EXP/" "$SCRATCH_EXP/"

# ───────────────────────────────────────────────
# 3)  ENVIRONMENT
# ───────────────────────────────────────────────
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

# Activate virtual-env (use absolute path)
source "$PROJ_HOME/venv/bin/activate"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"

# ───────────────────────────────────────────────
# 4)  RUN
# ───────────────────────────────────────────────
cd "$SCRATCH_EXP"

echo "[$(date)] Running script…"
python "$PROJ_HOME/prochain_transformer/cli.py" train --exp_id "$EXPERIMENT_ID" --cluster True --scratch_path "$SCRATCH_EXP" --plot_pred_check True

# ───────────────────────────────────────────────
# 5)  WRAP-UP
# ───────────────────────────────────────────────
deactivate
echo "[$(date)] Python environment deactivated"
echo "[$(date)] Job finished – results are still in $SCRATCH_EXP"

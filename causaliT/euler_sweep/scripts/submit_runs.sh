#!/usr/bin/env bash
# Submit all experiment folders under jobs_array/ as a single Slurm job array.
# ---------------------------------------------------------------------------
# Usage examples
#   ./submit_experiments.sh                # default concurrency = 10
#   ./submit_experiments.sh  --max-par 4   # run at most 4 tasks concurrently
#   ./submit_experiments.sh  --dry-run     # just show the sbatch command
# ---------------------------------------------------------------------------

set -euo pipefail

#############################################################################
# 0)  User-tweakable settings
#############################################################################
PROJ_HOME="$HOME/prochain_transformer"
JOBS_ROOT="$PROJ_HOME/experiments/training/jobs_array"
SCRIPT="$PROJ_HOME/scripts/run_on_scratch_array.sh"

DEFAULT_MAX_PAR=10        # fallback concurrency cap
#############################################################################

# ----- Parse optional flags ------------------------------------------------
MAX_PAR="$DEFAULT_MAX_PAR"
DRY_RUN=false

while [[ "${1:-}" =~ ^- ]]; do
  case "$1" in
    -m|--max-par) MAX_PAR="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ----- Count experiment sub-folders ---------------------------------------
N=$(find "$JOBS_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)

if (( N == 0 )); then
  echo "No experiment folders found under $JOBS_ROOT" >&2
  exit 1
fi

ARRAY_SPEC="0-$((N-1))%${MAX_PAR}"

echo "[submit_experiments] Found $N experiments ⇒ --array=$ARRAY_SPEC"

# ----- Build sbatch command -----------------------------------------------
SBATCH_CMD=(
  sbatch
  --array="$ARRAY_SPEC"
  "$SCRIPT"
)

if $DRY_RUN; then
  echo "[submit_experiments] Dry-run mode. Would execute:"
  printf '  %q ' "${SBATCH_CMD[@]}"; echo
  exit 0
fi

# ----- Fire! --------------------------------------------------------------
"${SBATCH_CMD[@]}"

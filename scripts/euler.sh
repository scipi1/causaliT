#!/bin/bash
# Common launcher. Configure this section, then submit with sbatch (not sh):
#   sbatch "$HOME/causaliT/scripts/euler.sh"
# Settings are written to a run spec YAML that is frozen in the run folder.
# Euler rules: see euler_instructions.txt (LF endings, --mem-per-cpu only).
#SBATCH --job-name=ew_launcher
#SBATCH --output=ew_launcher_%j.out
#SBATCH --error=ew_launcher_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=00:30:00

set -euo pipefail

# ---- Paths --------------------------------------------------------------
PROJ_HOME="$HOME/causaliT" # Absolute project root on Euler (causaliT installed with pip install -e .); on PYTHONPATH in jobs
EXPERIMENT_ID="exp1" # Folder relative to $PROJ_HOME/experiments; flat or nested path
INPUTS=("config*.yaml" "dagsweep*.yaml" "atesweep*.yaml" "sweeper/sweep.yaml") # Globs (relative to experiment) frozen into the run
DATA_DIR="$PROJ_HOME/data" # Folder holding <data.dataset>/ (as causaliT.cli); never copied; ignored by dagsweep
SCRATCH_ROOT="$SCRATCH" # A new run folder is created here
VENV="$HOME/myenv" # Folder with bin/activate; "" to skip
DRY_RUN=0 # 1 = prepare the run folder only; submit later with "submit --run-dir"
CHECK=1 # 1 = pre-flight check (CPU, seconds: config, dataset, model build) before submitting; 0 = skip

# ---- Function -----------------------------------------------------------
FUNCTION="training" # training | sweep | dagsweep | custom plugin "module[:attribute]"
TRAIN_FN="causaliT.euler_funs.train_fns:train" # training/sweep: ...train_fns:train (k-fold) | ...train_fns:adaptive_train
CONFIG="config.yaml" # Frozen base config, relative to the run folder
TASK_MODE="fold" # Training only: experiment (one task) | fold (one task per fold)
FOLDS_KEY="training.k_fold" # Training fold mode: dotted config key holding the number of folds
SWEEP_FILE="sweeper/sweep.yaml" # Sweep only: {category: {param: [values]}}
SWEEP_MODE="combination" # Sweep only: combination (Cartesian product) | independent (one at a time)
DAGSWEEP_TRAINER="" # Dagsweep only: "" (dagsweep*.yaml training.trainer) | standard | adaptive
DAGSWEEP_KEEP_DATA=false # Dagsweep only: true keeps dataset arrays at finalize (recipes always kept)
PLUGIN_OPTIONS="{}" # Custom plugin only: YAML flow mapping, e.g. "{alpha: 1, beta: x}"

# ---- Resources (per worker job) -----------------------------------------
WALLTIME="04:00:00" # HH:MM:SS or D-HH:MM:SS; <= 04:00:00 lands in the 4h partition (best GPU availability)
SIGNAL_MARGIN=600 # [s] pause request before walltime; must cover one epoch + checkpoint
CPUS_PER_TASK=1 # Integer >= 1
MEM_PER_CPU="10g" # Slurm memory per CPU; node-level --mem is not supported on Euler
ACCELERATOR="gpu" # cpu | gpu; cpu omits all GPU directives
GPUS=1 # GPU mode: GPUs per job
GPU_TYPE="" # GPU mode: "" (any) or a model, e.g. rtx_4090
GPU_MEM="24g" # GPU mode: Euler gpumem, e.g. 24g; "" no constraint; not CPU RAM
PARTITION="" # "" lets Slurm route by walltime, or an explicit partition name
MODULES=("stack/2024-06" "gcc/12.2.0" "python_cuda/3.11.6") # Loaded here and in worker jobs
SETUP=() # Extra raw shell lines run before the worker, e.g. ("conda activate env")
ENV_VARS=() # KEY=VALUE exported in worker jobs, e.g. ("OMP_NUM_THREADS=1")
EXTRA_SBATCH=() # Extra sbatch options, e.g. ("--tmp=20g")

# ---- Scheduling ---------------------------------------------------------
SCHEDULE="chain" # single (one job per lane) | chain (successor jobs resume unfinished work)
LANES=6 # Integer >= 1; maximum concurrent worker jobs, capped by the number of tasks
MAX_CHUNKS=40 # Integer >= 1; jobs per lane including the first; resume can extend
MAX_FAILURES=2 # Integer >= 1; crashes or exceptions per task before FAILED
JOB_PREFIX="ew" # 1-16 alphanumeric chars; job names <prefix>_<run>_L<lane>

# ---- End of configuration -----------------------------------------------

q() { local s=${1//\\/\\\\}; s=${s//\"/\\\"}; printf '"%s"' "$s"; } # YAML double-quoted string
list() { local out="" x; for x in "$@"; do out+="$(q "$x"), "; done; printf '[%s]' "${out%, }"; }
opt() { if [[ -n "$1" ]]; then q "$1"; else printf null; fi; } # "" -> null

case "$FUNCTION" in
  training) OPTIONS="{train_fn: $(q "$TRAIN_FN"), config: $(q "$CONFIG"), task_mode: $(q "$TASK_MODE"), folds_key: $(q "$FOLDS_KEY")}" ;;
  sweep) OPTIONS="{train_fn: $(q "$TRAIN_FN"), config: $(q "$CONFIG"), sweep: $(q "$SWEEP_FILE"), sweep_mode: $(q "$SWEEP_MODE")}" ;;
  dagsweep) OPTIONS="{trainer: $(opt "$DAGSWEEP_TRAINER"), keep_data: $DAGSWEEP_KEEP_DATA}" ;;
  *) OPTIONS="$PLUGIN_OPTIONS" ;;
esac
ENV_MAP=""
for pair in ${ENV_VARS[@]+"${ENV_VARS[@]}"}; do ENV_MAP+="${pair%%=*}: $(q "${pair#*=}"), "; done

SPEC=$(mktemp "${TMPDIR:-/tmp}/run_spec_XXXXXX")
trap 'rm -f "$SPEC"' EXIT
cat > "$SPEC" <<EOF
run:
  experiment: $(q "$PROJ_HOME/experiments/$EXPERIMENT_ID")
  inputs: $(list ${INPUTS[@]+"${INPUTS[@]}"})
  scratch_root: $(q "$SCRATCH_ROOT")
  data_dir: $(opt "$DATA_DIR")
  project_dir: $(q "$PROJ_HOME")
plugin:
  path: $(q "$FUNCTION")
  options: $OPTIONS
resources:
  walltime: $(q "$WALLTIME")
  signal_margin: $SIGNAL_MARGIN
  cpus_per_task: $CPUS_PER_TASK
  mem_per_cpu: $(q "$MEM_PER_CPU")
  accelerator: $(q "$ACCELERATOR")
  gpus: $GPUS
  gpu_type: $(q "$GPU_TYPE")
  gpu_mem: $(q "$GPU_MEM")
  partition: $(q "$PARTITION")
  modules: $(list ${MODULES[@]+"${MODULES[@]}"})
  setup: $(list ${SETUP[@]+"${SETUP[@]}"})
  venv: $(opt "$VENV")
  env: {${ENV_MAP%, }}
  extra_sbatch: $(list ${EXTRA_SBATCH[@]+"${EXTRA_SBATCH[@]}"})
scheduling:
  schedule: $(q "$SCHEDULE")
  lanes: $LANES
  max_chunks: $MAX_CHUNKS
  max_failures: $MAX_FAILURES
  job_prefix: $(q "$JOB_PREFIX")
EOF
if [[ "${EW_PRINT_SPEC:-0}" == 1 ]]; then cat "$SPEC"; exit 0; fi # debug: print spec only

for module in "${MODULES[@]}"; do module load "$module"; done
if [[ -n "$VENV" ]]; then source "$VENV/bin/activate"; fi
if [[ -n "$VENV" && -z "${VIRTUAL_ENV:-}" ]]; then echo "Python environment activation failed" >&2; exit 1; fi
cd "$PROJ_HOME"
FLAGS=()
if [[ "$DRY_RUN" == 1 ]]; then FLAGS+=(--dry-run); fi
if [[ "$CHECK" != 1 ]]; then FLAGS+=(--no-check); fi
python -m causaliT.euler_funs.cli launch --spec "$SPEC" ${FLAGS[@]+"${FLAGS[@]}"}
if [[ -n "$VENV" ]]; then deactivate; fi
#!/bin/bash
#SBATCH --job-name=install_daguerreo
#SBATCH --output=install_daguerreo_%j.out
#SBATCH --error=install_daguerreo_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4g

# Build DAGuerreotype (commit 47fc2ae) + lp-sparsemap into the causaliT venv.
# Skips the optional R/SID part of the upstream linux-install.sh (needs sudo).
# Usage: sbatch install_daguerreo.sh
# Afterwards add the printed PYTHONPATH line to your job scripts.

set -euo pipefail
echo "[$(date)] DAGuerreotype install on $(hostname)"

# ---------------------------------------------------------------------------
# ENVIRONMENT SETUP
# TODO: Customize these module loads / paths for your cluster
# ---------------------------------------------------------------------------
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load eth_proxy 2>/dev/null || true   # internet on compute nodes (Euler)

VENV_PATH="$HOME/myenv"
EXT_DIR="$HOME/ext"                # where external code is downloaded/built
DGE_COMMIT="47fc2ae"

source "$VENV_PATH/bin/activate"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV ($(python --version))"

export MAX_JOBS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$EXT_DIR"
cd "$EXT_DIR"

# ---------------------------------------------------------------------------
# 1) Python build tools and runtime deps (no requirements.txt: avoids
#    downgrading torch/numpy in the causaliT venv)
# ---------------------------------------------------------------------------
pip install --upgrade wheel cython pybind11
# Runtime deps from DAGuerreotype requirements.txt, without its pins
# (scikit-learn==1.1.1, deprecated "sklearn" stub); already installed
# packages are left untouched.
pip install wandb causaldag cdt optuna igraph networkx entmax tqdm matplotlib scikit-learn

# ---------------------------------------------------------------------------
# 2) Eigen 3.4.0 (header only). Pinned: Eigen master moved the version
#    macros out of Macros.h, so lp-sparsemap's find_eigen() rejects it.
# ---------------------------------------------------------------------------
EIGEN_VER="3.4.0"
rm -rf eigen-master                      # leftover from a previous attempt
if [[ ! -f "eigen-$EIGEN_VER/Eigen/Dense" ]]; then
    rm -rf "eigen-$EIGEN_VER"
    wget -q --no-check-certificate \
        "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VER/eigen-$EIGEN_VER.tar.gz"
    tar -xzf "eigen-$EIGEN_VER.tar.gz" && rm "eigen-$EIGEN_VER.tar.gz"
fi
export EIGEN_DIR="$(readlink -f "eigen-$EIGEN_VER")"
# Fallback in case setup.py still does not pick it up
export CPLUS_INCLUDE_PATH="$EIGEN_DIR${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
echo "[$(date)] EIGEN_DIR=$EIGEN_DIR"

# ---------------------------------------------------------------------------
# 3) lp-sparsemap (compiled, editable install)
# ---------------------------------------------------------------------------
if [[ ! -d lp-sparsemap-master ]]; then
    wget -q --no-check-certificate --content-disposition \
        https://github.com/deep-spin/lp-sparsemap/archive/refs/heads/master.zip
    unzip -q lp-sparsemap-master.zip && rm lp-sparsemap-master.zip
fi
cd lp-sparsemap-master
# Patch: FactorTreeTurbo.h uses std::tie without <tuple> (fails on gcc >= 11)
TURBO_H="lpsmap/ad3ext/FactorTreeTurbo.h"
if ! grep -q '#include <tuple>' "$TURBO_H"; then
    sed -i 's|#include "DependencyDecoder.h"|#include "DependencyDecoder.h"\n#include <tuple>|' "$TURBO_H"
fi
grep -q '#include <tuple>' "$TURBO_H" || { echo "Patch of $TURBO_H failed" >&2; exit 1; }
rm -rf build                             # drop artifacts of failed builds
python setup.py build_clib
pip install --no-build-isolation -e .
cd "$EXT_DIR"

# ---------------------------------------------------------------------------
# 4) DAGuerreotype (compiled in place at the pinned commit)
# ---------------------------------------------------------------------------
if [[ ! -d DAGuerreotype ]]; then
    git clone https://github.com/vzantedeschi/DAGuerreotype.git
fi
cd DAGuerreotype
git fetch --quiet origin
git checkout --quiet "$DGE_COMMIT"
# Cython resolves "cimport lpsmap..." via sys.path; the PEP 660 editable
# install uses an import hook that Cython does not see, so put the
# lp-sparsemap source tree on PYTHONPATH (as upstream linux-install.sh does).
DGE_PYTHONPATH="$EXT_DIR/DAGuerreotype:$EXT_DIR/lp-sparsemap-master"
export PYTHONPATH="$DGE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
# Patch: the torch extension _kbest is built with -std=c++14, but recent
# torch headers require C++17. Only the CppExtension block is changed.
git checkout --quiet -- setup.py
sed -i "/_kbest/,/)/ s/-std=c++14/-std=c++17/" setup.py
grep -q -- "-std=c++17" setup.py || { echo "Patch of setup.py failed" >&2; exit 1; }
rm -f daguerreo/permutahedron/_sparsemap.cpp   # drop failed Cython output
rm -rf build
python setup.py build_ext --inplace
cd "$EXT_DIR"

# ---------------------------------------------------------------------------
# 5) Smoke test
# ---------------------------------------------------------------------------
python -c "import wandb, torch, lpsmap, daguerreo; \
from daguerreo.models import Daguerro; print('daguerreo import OK')"

echo "[$(date)] Done. Add this line to your job scripts (after venv activation):"
echo "export PYTHONPATH=\"\${PYTHONPATH:-}:$DGE_PYTHONPATH\""

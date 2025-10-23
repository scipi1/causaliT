# ProT Installation Guide

## 🎯 Quick Setup (5 minutes)

### The Problem We're Solving
After restructuring, running `python proT/training/trainer.py` causes:
```
ModuleNotFoundError: No module named 'proT'
```

### The Solution
Install proT as an **editable package** in your environment. This creates a link so Python can find `proT` from anywhere.

---

## 📦 Local Setup (Windows)

### Step 1: Activate Your Virtual Environment
```powershell
# Navigate to project root
cd C:\Users\ScipioneFrancesco\Documents\Projects\causaliT

# Activate venv
.\venv\Scripts\activate
```

### Step 2: Install ProT in Editable Mode
```powershell
pip install -e .
```

### Step 3: Verify Installation
```powershell
python -c "from proT.core import ProT; print('✅ Success!')"
```

**That's it!** You only need to do this once per virtual environment.

---

## 🖥️ Cluster Setup (Linux)

### Step 1: SSH and Activate Environment
```bash
# SSH to cluster
ssh username@cluster.address

# Navigate to project
cd /scratch/username/causaliT  # or wherever your project is

# Activate your venv
source /path/to/your/venv/bin/activate
```

### Step 2: Install ProT in Editable Mode
```bash
pip install -e .
```

### Step 3: Verify Installation
```bash
python -c "from proT.core import ProT; print('✅ Success!')"
```

**Done!** Repeat for each cluster node/environment where you need proT.

---

## 🎮 Toy Tutorial: "Hello From Anywhere!"

### Create a Test Script Anywhere

**On Local (in any directory):**
```powershell
# Create test script in your Documents folder
cd ~/Documents
New-Item test_prot.py
```

**On Cluster (e.g., from $HOME):**
```bash
# Create test script in $HOME
cd ~
nano test_prot.py
```

### Add This Code:
```python
# test_prot.py
from proT.core import ProT
from proT.training import TransformerForecaster
from proT.training.callbacks import BestCheckpointCallback

print("✅ All imports successful!")
print(f"✅ ProT model class: {ProT}")
print(f"✅ TransformerForecaster class: {TransformerForecaster}")
print(f"✅ Callback class: {BestCheckpointCallback}")
print("\n🎉 You can now import proT from ANY directory!")
```

### Run It:
```bash
python test_prot.py
```

**Expected Output:**
```
✅ All imports successful!
✅ ProT model class: <class 'proT.core.model.ProT'>
✅ TransformerForecaster class: <class 'proT.training.forecasters.transformer_forecaster.TransformerForecaster'>
✅ Callback class: <class 'proT.training.callbacks.training_callbacks.BestCheckpointCallback'>

🎉 You can now import proT from ANY directory!
```

---

## 🎯 Real-World Usage Examples

### Example 1: Run Trainer Directly
```bash
# From anywhere (even $HOME on cluster)
python /path/to/causaliT/proT/training/trainer.py
```

### Example 2: Import in Your Scripts
```python
# In any Python script, any directory
from proT.core import ProT
from proT.training import TransformerForecaster, ProcessDataModule
from proT.training.callbacks import BestCheckpointCallback

# Use them as normal
model = TransformerForecaster(config)
```

### Example 3: SLURM Job Script
```bash
#!/bin/bash
#SBATCH --job-name=prot_train
#SBATCH --time=24:00:00

# Activate venv (which has proT installed with pip install -e .)
source /home/username/venv/bin/activate

# Run from anywhere - imports will work!
cd $HOME
python /scratch/username/causaliT/proT/training/trainer.py \
    --exp_id my_experiment \
    --cluster True
```

---

## ❓ FAQ

### Q: Do I need to reinstall after changing code?
**A: NO!** That's what "editable" means. All changes are live immediately.

### Q: What if I have multiple virtual environments?
**A: Run `pip install -e .` in each venv** where you want to use proT.

### Q: What actually happens during installation?
**A:** Pip creates a tiny link file in your venv's `site-packages/` pointing to your project. No code is copied.

### Q: How do I uninstall?
**A:** `pip uninstall proT` removes the link (your code stays intact).

### Q: Does this work with Jupyter notebooks?
**A: Yes!** Once installed in the venv:
```python
# In any notebook
from proT.core import ProT  # Works!
```

---

## 🐛 Troubleshooting

### Issue: Still get ModuleNotFoundError
**Solution:** Make sure you're using the correct Python environment:
```bash
# Check which Python
which python  # or: where python on Windows

# Should point to your venv's Python
# If not, activate venv again
```

### Issue: "pip: command not found"
**Solution:** Your venv might not be activated:
```bash
# Linux/Mac
source /path/to/venv/bin/activate

# Windows
.\venv\Scripts\activate
```

### Issue: Permission denied on cluster
**Solution:** Make sure you have write access to the project directory and venv:
```bash
# Check permissions
ls -la /path/to/causaliT

# If needed, use --user flag (not recommended but works)
pip install -e . --user
```

---

## 📋 What Changed?

### Before (with sys.path.append):
```python
# Every file needed this:
import sys
from os.path import dirname, abspath
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)

from proT.core import ProT  # Then this worked
```

**Problems:**
- ❌ Repetitive code in every file
- ❌ Breaks when running from different directories
- ❌ Hard to maintain ROOT_DIR path
- ❌ Doesn't work well on clusters

### After (with pip install -e .):
```python
# Just import directly:
from proT.core import ProT  # Works from anywhere!
```

**Benefits:**
- ✅ Clean code (no sys.path hacks)
- ✅ Works from ANY directory
- ✅ Industry standard approach
- ✅ IDE autocomplete works perfectly
- ✅ Cluster-friendly

---

## 🎉 Summary

1. **One-time setup:** Run `pip install -e .` in each environment
2. **Done!** Now you can:
   - Import proT from anywhere
   - Run scripts from any directory
   - No more sys.path.append needed
   - Works identically on local and cluster

**Questions?** Check the FAQ above or refer to `proT/README_RESTRUCTURE.md` for more details about the project structure.

---

## 📁 Using Project Paths

### The Problem
After removing `sys.path.append`, you might wonder: "How do I reference `data/` and `experiments/` directories?"

### The Solution: `proT.paths`

We've created a centralized paths module that defines all standard directories:

```python
# Import paths anywhere in your code
from proT.paths import ROOT_DIR, DATA_DIR, EXPERIMENTS_DIR, LOGS_DIR

# Use them with pathlib (modern Python):
data_file = DATA_DIR / "example" / "ds.npz"
config_file = EXPERIMENTS_DIR / "example" / "config.yaml"

# Or convert to string for compatibility:
data_dir_str = str(DATA_DIR)
```

### Available Paths

```python
from proT.paths import (
    ROOT_DIR,          # Project root (causaliT/)
    DATA_DIR,          # data/
    EXPERIMENTS_DIR,   # experiments/
    LOGS_DIR,          # logs/
    CONFIG_DIR,        # proT/config/
)
```

### Quick Import from proT

For convenience, paths are also exported from the main package:

```python
# Short version - import from proT directly
from proT import ROOT_DIR, DATA_DIR, EXPERIMENTS_DIR

# Also works
import proT
data_path = proT.DATA_DIR / "example"
```

### Example Usage

**Before (with ROOT_DIR calculation):**
```python
from os.path import dirname, abspath, join
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
data_dir = join(ROOT_DIR, "data", "input")
exp_dir = join(ROOT_DIR, "experiments", "example")
```

**After (with proT.paths):**
```python
from proT.paths import DATA_DIR, EXPERIMENTS_DIR

data_dir = DATA_DIR / "input"
exp_dir = EXPERIMENTS_DIR / "example"
```

### Advanced: Environment Variable Override

For cluster flexibility, you can override paths with environment variables:

```bash
# On cluster, set custom paths
export PROT_ROOT=/scratch/username/causaliT
export PROT_DATA=/fast_storage/data

# Now Python will use these paths
python my_script.py
```

In Python, the paths automatically pick up these overrides:
```python
from proT.paths import ROOT_DIR, DATA_DIR
# These now point to your custom locations!
```

This is useful when:
- Data is on fast/scratch storage
- Running on different clusters with different mount points
- Testing with different data locations

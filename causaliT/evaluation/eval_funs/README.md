# Eval Funs CLI - Quick Start

**Location:** `causaliT/evaluation/eval_funs/`

---

## Run All Evaluations on Experiment(s)

Don't forget the `--no-show` option to avoid pop-up windows. 

```bash
# Single experiment
python -m causaliT.evaluation.eval_funs.eval_fun_cli evaluate -e experiments/my_exp --no-show

# Multiple experiments
python -m causaliT.evaluation.eval_funs.eval_fun_cli evaluate -e experiments/exp1 -e experiments/exp2 --no-show

# All experiments in a folder (auto-discovers subdirectories)
python -m causaliT.evaluation.eval_funs.eval_fun_cli evaluate -f experiments/euler --no-show
```

---

## Run Specific Evaluation Functions

```bash
python -m causaliT.evaluation.eval_funs.eval_fun_cli evaluate -e experiments/my_exp \
  --functions eval_train_metrics eval_attention_scores --no-show
```

**Available functions:**

| Function | Description |
|----------|-------------|
| `eval_train_metrics` | Training curves, loss analysis, and instability metrics |
| `eval_attention_scores` | DAG recovery metrics (phi, attention) |
| `eval_embed` | Embedding evolution analysis |
| `eval_interventions` | Causal intervention tests |
| `eval_embedding_dag_correlation` | Embedding-DAG correlation |
| `eval_dyconex_predictions` | Dyconex-specific prediction evaluation |
| `fix_kfold_summary` | Fix tensor strings in kfold_summary.json |
| `enrich_kfold_summary` | Add aggregated statistics to kfold_summary |

---

## Update Manifest Only (No Re-evaluation)

```bash
# Update manifest for specific experiments
python -m causaliT.evaluation.eval_funs.eval_fun_cli manifest -e experiments/my_exp

# Update for all experiments in folder
python -m causaliT.evaluation.eval_funs.eval_fun_cli manifest -f experiments/euler
```

**Manifest location:** `experiments/experiments_manifest.csv`

---

## Options

| Flag | Description |
|------|-------------|
| `-e, --experiment` | Experiment path (can repeat) |
| `-f, --folder` | Folder containing experiments (auto-discovers) |
| `--no-show` | Don't display plots (save to files only) |
| `--update-manifest` | Update manifest after running evaluations |
| `--functions` | Specific functions to run (default: all) |

---

## Evaluate Attention Necessity Score (ANS)

```bash
python -m causaliT.evaluation.eval_funs.eval_ans experiments/ANS_sweep_exp --no-show
```

---

## Python API

```python
from causaliT.evaluation.eval_funs import (
    eval_train_metrics,
    eval_attention_scores,
    eval_embed,
    eval_interventions,
    run_all_evaluations,
    update_experiments_manifest,
)

# Run specific evaluation
eval_train_metrics("experiments/my_exp", show_plots=False)

# Run all evaluations
run_all_evaluations("experiments/my_exp", show_plots=False)

# Update manifest
update_experiments_manifest("experiments/my_exp")
```

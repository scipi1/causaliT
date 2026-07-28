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
  --functions eval_attention_scores eval_interventions --no-show
```

**Available functions:**

| Function | Description | Status |
|----------|-------------|--------|
| `eval_attention_scores` | DAG recovery metrics from attention (tables only, no figures) | ✓ Active |
| `eval_interventions` | Causal intervention tests (ATE) | ✓ Active |
| `eval_seed_sweep` | Aggregate DAG + ATE metrics across seeds for paper reporting | ✓ Active |
| `fix_kfold_summary` | Fix tensor strings in kfold_summary.json | ✓ Active |
| `enrich_kfold_summary` | Add aggregated statistics to kfold_summary | ✓ Active |

> **Scope note:** this package now covers **DAG recovery** and **interventions** only.
> The following evaluations were retired to `_OLD/` and are no longer importable or
> callable through the CLI: `eval_train_metrics`, `eval_attention_evolution`,
> `eval_embed`, `eval_embedding_dag_correlation`, `eval_ans`,
> `eval_anm_residual_hsic`, `eval_d_model_sweep`, `eval_dyconex_predictions`,
> plus the plotting library `eval_plot_lib`.
>
> Some config templates still list `eval_train_metrics` / `eval_dyconex_predictions`
> under `evaluation.functions`; those entries now log `✗ Unknown function` and are
> skipped. Remove them from the templates when convenient.

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

## Evaluate Seed Sweep (Paper Reporting)

Aggregate metrics across multiple training seeds for paper tables:

```bash
# CLI usage
python -m causaliT.evaluation.eval_funs.eval_seed_sweep experiments/baseline/euler/vanilla_transformer_scm1_61555008

# Python API
from causaliT.evaluation.eval_funs import eval_seed_sweep
df = eval_seed_sweep("experiments/baseline/euler/vanilla_transformer_scm1_61555008")
```

**Output files:**
- `summary_stats.csv` - Summary table with mean, std, min, max (one row per metric)
- `ate_by_intervention.csv` - ATE errors per intervention × variable × seed
- `raw_per_seed.csv` - Raw per-seed data for custom analysis
- `ate_by_intervention_{exp_id}.png` - Bar chart of ATE errors
- `dag_metrics_{exp_id}.png` - Bar chart of DAG recovery metrics (SHD, MEC)

**Metrics aggregated:**
- **Test Performance:** test_loss, test_mae, test_r2, test_rmse
- **ATE Errors:** Overall and per-intervention (S1, S2, S3, S5)
- **DAG Recovery:** SHD_cross, SHD_self, MEC_distance, MEC_in_class

---

## Checkpoint Selection

`eval_attention_scores` and `eval_interventions` both resolve the checkpoint via
`infer_checkpoint_type(config)`: causal models are read from `best_causal`,
baselines from `best_reconstruction`. The rationale for not simply taking the
prediction-"best" checkpoint:

- **"best"** selects on prediction loss, not causal correctness
- Causal regularizers (HSIC, sparsity) may need more epochs to converge

---

## Python API

```python
from causaliT.evaluation.eval_funs import (
    eval_attention_scores,
    eval_interventions,
    eval_seed_sweep,
    run_all_evaluations,
    update_experiments_manifest,
)

# DAG recovery metrics for one experiment
eval_attention_scores("experiments/my_exp", show_plots=False)

# Run all evaluations (DAG + interventions + kfold_summary maintenance)
run_all_evaluations("experiments/my_exp", show_plots=False)

# Update manifest
update_experiments_manifest("experiments/my_exp")
```

---

## Architecture-agnostic DAG extraction

`eval_attention_scores` has a single extraction path for every architecture:

```
best checkpoint -> predict_test_from_ckpt -> query_dag_blocks -> compute_dag_metrics
```

`eval_dag_query.query_dag_blocks(attention, L_S, L_X)` classifies each attention
tensor **by its shape**, so no architecture registry is involved:

| Shape | Interpretation |
|-------|----------------|
| `(L_X, L_S)` | `cross` block (S->X) |
| `(L_X, L_X)` | `self` block (X->X) |
| `(L_X, L_S + L_X)` | combined block, split at `L_S` into `cross` + `self` |

Every architecture therefore emits the same canonical block names, and
`dag_metrics.json` / `learned_dag_edges.json` are comparable across models.
MEC metrics require both blocks and are skipped for cross-only models.

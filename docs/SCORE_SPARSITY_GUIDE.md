# Score Sparsity: Lambda Selection and Pipeline Integration

## Background

Toeplitz attention computes edge weights proportional to the **dot product of
embeddings**. An edge is zero only when the two embeddings are
**anti-parallel** (dot product = −1). Without explicit regularisation, the
model can learn near-uniform attention (all edges non-zero) even when the true
causal graph is sparse.

The **score sparsity penalty** adds an L1 regularisation term on the raw
attention score matrix (before softmax / sigmoid):

```
L_total = L_recon  +  λ_group · L_group_l1  +  λ_hsic · L_hsic  +  λ_score · L_score_sparse
```

- `L_score_sparse = ||A_raw||_1`  (mean absolute value of raw scores)  
- `λ_score` forces embeddings to become **anti-parallel** for non-causal
  variable pairs.

---

## Two Approaches to λ_score Selection

### Approach A: Integrated CV (Recommended — Staged Training Stage 2)

The **preferred** approach integrates λ_score selection directly into the
staged training pipeline as **Stage 2: Score Sparsity CV**. This runs k-fold
cross-validation over a grid of λ_score candidates, selecting the best by
lowest mean validation HSIC (or reconstruction loss for baselines).

```
Stage 0 – CALIBRATION         → λ_group*, λ_hsic*
Stage 1 – CAUSAL INIT         → structural checkpoint
Stage 2 – SCORE SPARSITY CV   → λ_score* (via k-fold CV)
Stage 3 – MAIN TRAINING       → final model with λ_score* applied
```

See `docs/STAGED_TRAINING.md` for details. Enable with:
```yaml
staged_training:
  use_score_sparsity_cv: true
  score_sparsity_lambda_candidates: [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
  score_sparsity_selection_rule: "min_hsic"  # or "min_recon"
```

### Approach B: Sweep-Based (for detailed diagnostics)

For richer diagnostic outputs (LASSO-path plots, variable importance paths),
use the `euler_sweep` approach described below. This is heavier but produces
more detailed analysis artifacts.

## Order of Operations (Sweep Approach)

The full pipeline for a new dataset/model configuration is:

```
Step 0 – CALIBRATION (once per dataset)
    ↓  finds: lambda_group*, lambda_hsic*
Step 1 – SCORE-SPARSITY SWEEP (once per dataset + model)
    ↓  sweeps: lambda_cross_score_sparse in [0, 0.001, 0.01, 0.1, ...]
    ↓  selects: lambda_score* via HSIC-path / 1-SE rule
Step 2 – CAUSAL INITIALIZATION (once per lambda_score*)
    ↓  trains: HSIC-dominated short run inside the sparse landscape
Step 3 – MAIN TRAINING (k-fold)
    ↓  trains: standard pipeline with lambda_score* fixed
```

**Why this order matters:**

- **Calibration first**: λ_group is dataset-specific (complex datasets can
  tolerate lower sparsity and still get a non-flat HSIC landscape). Calibration
  must run before the score-sparsity sweep so that every sweep combination uses
  the correct λ_group.

- **Score sweep before causal init**: Causal init is a training-stabilisation
  step. It must run *inside* the sparse landscape selected by the score sweep,
  so that the HSIC-dominated pre-training operates in the same parameter space
  as the final model.

---

## Step 0: Calibration

Calibration is described in detail in `docs/STAGED_TRAINING.md`.  
For the score-sparsity pipeline it is run **automatically** as a pre-sweep
action by the `calibrated_sweep` CLI command.

Manual usage:
```python
from causaliT.training.calibration import calibrate_group_l1
from causaliT.training.config_operations import apply_calibration_to_config

cal_result = calibrate_group_l1(config, data_dir, save_dir)
config = apply_calibration_to_config(config, cal_result)
```

---

## Step 1: Score-Sparsity Sweep

### Create the sweep experiment

```
experiments/
└── my_score_sparsity/
    ├── config_*.yaml        # your model config (with lambda_hsic, lambda_group set)
    └── sweeper/
        └── sweep.yaml       # score lambda grid
```

**`sweeper/sweep.yaml`**:
```yaml
training:
  lambda_cross_score_sparse: [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
  # optionally sweep seeds for robustness:
  # seed: [0, 1, 2, 3, 4]
```

### Run the calibrated sweep

```bash
# Full pipeline: calibrate first, then sweep, then analyse
python -m causaliT.euler_sweep.euler_sweep.cli calibrated_sweep \
    --exp_id my_score_sparsity

# If calibration was already run:
python -m causaliT.euler_sweep.euler_sweep.cli calibrated_sweep \
    --exp_id my_score_sparsity --skip_calibration

# Analysis only (after a completed sweep):
python -m causaliT.euler_sweep.euler_sweep.cli calibrated_sweep \
    --exp_id my_score_sparsity --analysis_only

# Use min-HSIC rule instead of 1-SE:
python -m causaliT.euler_sweep.euler_sweep.cli calibrated_sweep \
    --exp_id my_score_sparsity --selection_rule min_hsic
```

### Directory structure after sweep

```
experiments/my_score_sparsity/
├── calibration/                   # from pre-sweep calibration
│   ├── calibration_summary.json
│   └── lambda_1.00e-03/           # per-trial runs
├── pre_sweep_calibration.json     # calibration results (for resumption)
├── sweeper/
│   ├── sweep.yaml
│   ├── score_sparsity_path.png    # LASSO-path: HSIC vs lambda_score
│   ├── variable_importance_path.png  # per-variable attention norms
│   ├── score_sparsity_analysis.json  # selected lambda*
│   └── runs/
│       └── combinations/
│           ├── combo_lambda_cross_score_sparse_0.0/
│           ├── combo_lambda_cross_score_sparse_0.001/
│           └── ...
```

---

## Step 2: Reading the LASSO-path plots

### `score_sparsity_path.png`

- **Left panel**: HSIC (cross and self) vs log10(λ_score)  
  - HSIC should stay low as λ_score increases (causal signal is preserved)  
  - A sharp HSIC *increase* signals that the model is losing its ability to
    capture the causal mechanism — the graph is being over-pruned

- **Right panel**: Validation MAE vs log10(λ_score)  
  - MAE should remain stable until λ_score is too large, then degrade

The **green vertical line** marks the selected λ*.

### `variable_importance_path.png`

Each line is one source variable.  As λ_score increases, attention norms shrink
toward zero.  Variables that go to zero early are less causally important.
This is the direct analogue of the LASSO coefficient path.

---

## Step 3: Lambda Selection Rules

### 1-SE rule (default, recommended)

Select the **smallest** λ_score such that HSIC does not increase by more than
`tolerance` (default 5%) relative to the minimum HSIC:

```
λ* = min{ λ : HSIC(λ) ≤ HSIC_min × (1 + tolerance) }
```

This is analogous to the "1-standard-error rule" in LASSO: prefer the
**sparsest model that is still within tolerance of the causal optimum**.

### min-HSIC rule

Select the λ that achieves the lowest HSIC.  More aggressive — may over-prune
if HSIC has a flat minimum.

### Programmatic selection

```python
from causaliT.evaluation.eval_score_sparsity import (
    collect_score_sparsity_results,
    select_lambda_score,
    plot_score_sparsity_path,
)

results = collect_score_sparsity_results("experiments/my_score_sparsity")
lambda_star = select_lambda_score(results, rule="1se", tolerance=0.05)
fig = plot_score_sparsity_path(results, selected_lambda=lambda_star)
```

---

## Step 4: Causal Initialization with Selected λ_score

After selecting λ*, update your experiment config:

```yaml
# In your main experiment config
training:
  lambda_cross_score_sparse: 0.01   # ← from score-sparsity sweep
  lambda_self_score_sparse: 0.001   # ← optional
  lambda_group_l1: 1.23e-03         # ← from calibration
  lambda_hsic_cross: 0.847          # ← from calibration

staged_training:
  use_calibration: false    # already done
  use_causal_init: true
  causal_init_epochs: 30
  causal_init_hsic_multiplier: 10.0
```

The causal initialization reads `lambda_cross_score_sparse` from
`config["training"]` and passes it through unchanged to `train_single_fold`.
This means causal init runs **inside the sparse landscape** selected by the
sweep — the model is initialised with the same sparsity pressure it will
experience during main training.

Run the staged pipeline:
```bash
python -m causaliT.cli train --exp_dir experiments/my_main_exp
```

---

## Config Keys Reference

| Key | Section | Description |
|-----|---------|-------------|
| `lambda_cross_score_sparse` | `training` | L1 penalty on cross-attention score matrix |
| `lambda_self_score_sparse` | `training` | L1 penalty on self-attention score matrix |
| `lambda_group_l1` | `training` | Group-L1 sparsity (from calibration) |
| `lambda_hsic_cross` | `training` | HSIC loss weight for S→X attention |
| `lambda_hsic_self` | `training` | HSIC loss weight for X→X attention |
| `use_calibration` | `staged_training` | Run calibration stage (Stage 0) |
| `use_causal_init` | `staged_training` | Run causal init stage (Stage 1) |
| `use_score_sparsity_cv` | `staged_training` | Run score sparsity CV (Stage 2) |
| `causal_init_epochs` | `staged_training` | Epochs for causal init |
| `causal_init_hsic_multiplier` | `staged_training` | HSIC boost factor for causal init |
| `score_sparsity_lambda_candidates` | `staged_training` | Grid of λ_score values for CV |
| `score_sparsity_cv_folds` | `staged_training` | Number of CV folds |
| `score_sparsity_cv_epochs` | `staged_training` | Epochs per CV fold |
| `score_sparsity_selection_rule` | `staged_training` | "min_hsic" or "min_recon" |
| `lambda_score_suggested` | `staged_training` | Auto-populated by Stage 2 |
| `calibration_epochs` | `staged_training` | Epochs per calibration trial |
| `calibration_lambda_group_range` | `staged_training` | Search range `[low, high]` |
| `calibration_max_iterations` | `staged_training` | Max binary-search iterations |
| `calibration_balance_threshold` | `staged_training` | Convergence criterion (e.g., 2.0) |

---

## API Reference

### `causaliT.training.config_operations`

```python
# Apply calibration results to config (returns new config, does not mutate)
config = apply_calibration_to_config(config, cal_result)

# Set score sparsity lambdas (returns new config)
config = apply_score_sparsity_to_config(config, lambda_cross=0.01, lambda_self=0.001)

# Wire HSIC annealing for main training after causal init (returns new config)
config = configure_main_training_from_staged(config)
```

### `causaliT.evaluation.eval_score_sparsity`

```python
# Collect results from sweep directory
results_df = collect_score_sparsity_results(sweep_dir)

# Select optimal lambda
lambda_star = select_lambda_score(results_df, rule="1se", tolerance=0.05)

# Plot LASSO paths
fig = plot_score_sparsity_path(results_df, selected_lambda=lambda_star)
fig = plot_variable_importance_path(sweep_dir, results_df)

# Full pipeline in one call
output = run_score_sparsity_analysis(sweep_dir, rule="1se")
```

### `causaliT.euler_sweep.euler_sweep.pre_sweep_actions`

```python
# Factory for calibration pre-sweep hook
pre_fn = make_calibration_pre_sweep(seed=42)

# Load existing calibration (for resumption)
overrides = load_pre_sweep_calibration(save_dir)
```

---

## Relation to REGULARIZATION_GUIDE.md

The `REGULARIZATION_GUIDE.md` describes the full regularisation landscape.
This document focuses specifically on:

1. **When** to select λ_score (before causal init, after calibration)
2. **How** to select λ_score (LASSO-path / HSIC-minimisation)
3. **How** λ_score flows through the pipeline (`config_operations.py`)

The key insight is that λ_score is **model-architecture-specific** (unlike
λ_group which is dataset-specific).  A larger model with more capacity needs
a larger λ_score to achieve the same edge sparsity.  The sweep procedure
automatically accounts for this by selecting λ* based on the HSIC response
rather than a fixed value.

# Regularization and Score Sparsity

## Regularization Overview

CausaliT supports various regularization techniques to encourage:

- **Sparsity**: Fewer edges in the learned DAG
- **Acyclicity**: No cycles in the DAG (NOTEARS constraint)
- **Decisiveness**: Edge probabilities pushed toward 0 or 1
- **Independence**: Residuals independent of source variables (HSIC)
- **Focused Attention**: Low entropy = concentrated attention weights

### Key Design Principles

1. **Unified Score Tensor**: Each attention type exposes `score_tensor_for_sparsity`
2. **Architecture-Specific Support**: Not all regularizers work with all architectures
3. **Attention-Type Compatibility**: Some regularizers only work with specific types

---

## Regularization Parameters

### Structure Learning (DAG)

| Parameter | Target | Description |
|-----------|--------|-------------|
| `kappa` | NOTEARS | Acyclicity constraint on self-attention DAG |
| `lambda_sparse` | σ(phi).mean() | Sparsity on self-attention edge probabilities |
| `lambda_sparse_cross` | σ(phi).mean() | Sparsity on cross-attention edge probabilities |
| `lambda_decisive` | Binary entropy | Pushes self-attention edges away from 0.5 |
| `lambda_decisive_cross` | Binary entropy | Pushes cross-attention edges away from 0.5 |

### Score Sparsity (Attention Sharpening)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lambda_self_score_sparse` | L1/entropy on self-attention scores | 0.0 |
| `lambda_cross_score_sparse` | L1/entropy on cross-attention scores | 0.0 |
| `self_sparsity_regularizer` | Mode: `"l1"` or `"entropy"` | `"l1"` |
| `cross_sparsity_regularizer` | Mode: `"l1"` or `"entropy"` | `"entropy"` |

**Mode Selection**:
- `"l1"`: Uses `score_tensor_for_sparsity` property (effective for GeLU-based attention)
- `"entropy"`: Uses attention entropy -Σ p log(p) (effective for **all** attention types)

**Automatic Fallback**: If `"l1"` mode is selected but `score_tensor_for_sparsity` returns
`None` (softmax-based attention), the forecaster falls back to entropy with a warning.

### Independence / HSIC

| Parameter | Description |
|-----------|-------------|
| `lambda_hsic_cross` | HSIC between S and cross-attention residuals |
| `lambda_hsic_self` | HSIC between X and self-attention residuals |
| `hsic_adaptive_bandwidth` | Use median heuristic for kernel bandwidth (recommended) |

### Other

| Parameter | Description |
|-----------|-------------|
| `lambda_kl` | KL divergence from running average prior |
| `lambda_tau` | Penalizes high Gumbel-Softmax temperature |
| `lambda_embed_l1` | L1 on embedding parameters (for ANS experiments) |
| `lambda_noise_prior` | Prior on noise parameters (noise-aware only) |
| `lambda_group_l1` | Group L1 (L2,1) on embedding columns (staged training) |

---

## Architecture Support Matrix

| Regularizer | SingleCausal | NoiseAware | StageCausal |
|-------------|:------------:|:----------:|:-----------:|
| `kappa` (NOTEARS) | ✅ | ✅ | ✅ |
| `lambda_sparse[_cross]` | ✅ | ✅ | ✅ |
| `lambda_entropy_*` | ✅ | ✅ | ✅ |
| `lambda_l1_*_scores` | ✅ | ✅ | ✅ |
| `lambda_hsic` | ✅ | ✅ | ❌ |
| `lambda_kl` | ✅ | ✅ | ❌ |
| `lambda_decisive[_cross]` | ✅ | ✅ | ❌ |
| `lambda_tau` | ✅ | ✅ | ❌ |
| `lambda_embed_l1` | ✅ | ❌ | ❌ |
| `lambda_noise_prior` | ❌ | ✅ | ❌ |
| All annealing schedules | ✅ | ✅ | ❌ |

---

## Attention Type Compatibility

| Attention Type | Activation | `lambda_l1_*_scores` | `lambda_sparse` | `lambda_entropy_*` |
|----------------|------------|:--------------------:|:---------------:|:------------------:|
| ScaledDotProduct | Softmax | ❌ Ineffective | ❌ No phi | ✅ **Use this** |
| PhiSoftMax | Softmax + phi | ❌ Ineffective | ✅ Works | ✅ Works |
| LieAttention | GeLU | ✅ Works | ✅ Works | ✅ Works |
| CausalCrossAttention | GeLU(Tanh) | ✅ Works | ✅ Works | ✅ Works |
| ToeplitzLieAttention | ReLU(Tanh) | ✅ Works | ✅ Works on gate | ✅ Works |

### Recommended Combinations

| Attention Type | Recommended Sparsity |
|----------------|---------------------|
| ScaledDotProduct | `lambda_entropy_*` (only effective option) |
| PhiSoftMax | `lambda_sparse` + `lambda_entropy_*` |
| LieAttention | `lambda_l1_*_scores` + `lambda_sparse` |
| ToeplitzLieAttention | `lambda_l1_*_scores` (applies to gate) |

### ToeplitzLieAttention Specifics

ToeplitzLieAttention decomposes attention into:
- **Gate (symmetric)**: Controls edge existence - P(edge exists)
- **Direction (antisymmetric)**: Controls flow direction - P(i→j | edge exists)

The `lambda_l1_*_scores` regularizer applies to gate probabilities, encouraging
sparse edge existence. This is recommended because:
1. We want to penalize "is there an edge?", not "which direction?"
2. The gate controls edge existence (both directions)
3. Sparse gates → sparse DAG

---

## Annealing Schedules

### Gumbel-Softmax Temperature (`tau_gs`)

```yaml
training:
  use_tau_gs_annealing: true
  tau_gs_start: 2.0          # High = exploration (soft masks)
  tau_gs_end: 0.2            # Low = exploitation (hard masks)
  tau_gs_anneal_epochs: 80
```

Schedule: Exponential τ(t) = τ_start × (τ_end/τ_start)^(t/T). Recommended for ToeplitzLieAttention.

### Activation Temperature (`tau_gate`, `tau_dir`)

For ToeplitzLieAttention only:

```yaml
training:
  use_tau_act_annealing: true
  tau_gate_start: 1.0
  tau_gate_end: 0.2
  tau_dir_start: 0.5
  tau_dir_end: 0.1
  tau_act_anneal_epochs: 80
```

### HSIC Annealing

```yaml
training:
  use_hsic_annealing: true
  hsic_lambda_start: 1.0     # Strong early
  hsic_lambda_end: 0.0       # Disabled late
  hsic_anneal_epochs: 50
```

Schedule: Linear annealing.

### Conflicting Mechanisms

| Mechanism A | Mechanism B | Conflict |
|-------------|-------------|----------|
| `lambda_tau` | `use_tau_gs_annealing` | Both control τ_gs |
| `lambda_hsic` (fixed) | `use_hsic_annealing` | Both control HSIC strength |

---

## Unified Score Tensor

All attention types expose `score_tensor_for_sparsity`:

| Attention Type | Returns | Notes |
|----------------|---------|-------|
| LieAttention | `sigmoid(phi)` | Edge existence probabilities |
| CausalCrossAttention | `sigmoid(phi)` | Edge existence probabilities |
| PhiSoftMax | `None` | Use entropy instead |
| ScaledDotAttention | `None` | Use entropy instead |
| ToeplitzAttention | `P_edge_for_reg` | Edge existence probabilities |
| ToeplitzLieAttention | `gate_probs_for_reg` | Gate probabilities |

---

## Configuration Examples

### Minimal Baseline

```yaml
training:
  kappa: 0.0
  lambda_sparse: 0.0
  lambda_entropy_self: 0.0
  lambda_l1_self_scores: 0.0
  lambda_hsic_cross: 0.0
  lambda_hsic_self: 0.0
  lambda_decisive: 0.0
  lambda_tau: 0.0
  lambda_kl: 0.0
```

### DAG Learning with ToeplitzLieAttention

```yaml
experiment:
  dec_self_attention_type: "ToeplitzLieAttention"

training:
  use_tau_gs_annealing: true
  tau_gs_start: 2.0
  tau_gs_end: 0.2
  tau_gs_anneal_epochs: 80

  lambda_l1_self_scores: 0.1
  lambda_decisive: 0.05
  kappa: 0.1
```

### ScaledDotProduct with Entropy

```yaml
experiment:
  dec_self_attention_type: "ScaledDotProduct"

training:
  lambda_entropy_self: 0.1
  lambda_entropy_cross: 0.1
  lambda_sparse: 0.0
  lambda_decisive: 0.0
```

---

## Debugging Tips

| Problem | Cause | Solution |
|---------|-------|----------|
| DAG values stuck at ~0.5 | Insufficient decisiveness | `lambda_decisive: 0.1`, lower `tau_gs_end` |
| DAG too sparse (all edges pruned) | Over-regularization | Reduce `lambda_sparse`, `lambda_l1_self_scores` |
| HSIC too high throughout | Model not learning causal structure | Use HSIC annealing: start at 0, ramp up |

---

## Score Sparsity Selection

### The Problem

Toeplitz attention computes edge weights proportional to the dot product of
embeddings. Without explicit regularization, the model can learn near-uniform
attention (all edges non-zero) even when the true causal graph is sparse.

The **score sparsity penalty** adds L1 regularization on the raw attention score matrix:

```
L_total = L_recon + λ_group · L_group_l1 + λ_hsic · L_hsic + λ_score · L_score_sparse
```

### λ_score Selection via Staged Training (Recommended)

The preferred approach integrates λ_score selection into the staged training
pipeline as Stage 2 (Score Sparsity CV):

```
Stage 0 – CALIBRATION         → λ_group*, λ_hsic*
Stage 1 – CAUSAL INIT         → structural checkpoint
Stage 2 – SCORE SPARSITY CV   → λ_score* (via k-fold CV)
Stage 3 – MAIN TRAINING       → final model with λ_score* applied
```

Enable with:

```yaml
staged_training:
  use_score_sparsity_cv: true
  score_sparsity_lambda_candidates: [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
  score_sparsity_selection_rule: "min_hsic"  # or "min_recon"

training:
  # Asymmetric sparsity weights compensate for Toeplitz double-sigmoid
  lambda_self_to_cross_score_ratio: 0.5   # default 1.0
```

### Selection Rules

**1-SE rule** (default for sweep-based analysis): Select the smallest λ_score
such that HSIC does not increase by more than tolerance (5%) relative to the
minimum HSIC. Analogous to the 1-standard-error rule in LASSO.

**min-HSIC rule**: Select the λ that achieves the lowest HSIC. More aggressive
— may over-prune if HSIC has a flat minimum.

### Sweep-Based Selection (for Diagnostics)

For richer diagnostic outputs (LASSO-path plots, variable importance paths),
use the `euler_sweep` approach:

```bash
python -m causaliT.euler_sweep.euler_sweep.cli calibrated_sweep \
    --exp_id my_score_sparsity
```

Output:
```
experiments/my_score_sparsity/
├── calibration/                     # Pre-sweep calibration results
├── sweeper/
│   ├── score_sparsity_path.png      # LASSO-path: HSIC vs lambda_score
│   ├── variable_importance_path.png # Per-variable attention norms
│   ├── score_sparsity_analysis.json # Selected lambda*
│   └── runs/combinations/           # Per-combination results
```

### Config Keys Reference

| Key | Section | Description |
|-----|---------|-------------|
| `lambda_cross_score_sparse` | `training` | L1 penalty on cross-attention score matrix |
| `lambda_self_score_sparse` | `training` | L1 penalty on self-attention score matrix |
| `lambda_self_to_cross_score_ratio` | `staged_training` | Asymmetry ratio for Toeplitz compensation |
| `use_score_sparsity_cv` | `staged_training` | Run score sparsity CV (Stage 2) |
| `score_sparsity_lambda_candidates` | `staged_training` | Grid of λ_score values |
| `score_sparsity_selection_rule` | `staged_training` | "min_hsic" or "min_recon" |
| `lambda_score_suggested` | `staged_training` | Auto-populated by Stage 2 |

---

## Logging Configuration

```yaml
training:
  log_entropy: true          # Attention entropy
  log_acyclicity: true       # NOTEARS value
  log_sparsity: true         # Sparsity regularizer value
  log_l1_scores: true        # L1 attention scores
  log_hsic: true             # HSIC value
  log_decisiveness: true     # Decisiveness metrics
  log_embed_l1: true         # Embedding L1 norm
  log_tau_annealing: true    # Annealed temperature values
  log_hsic_annealing: true   # Annealed HSIC coefficient
  log_noise_params: true     # σ_A, σ_R (noise-aware only)
```

---

## Related Documents

- `docs/STAGED_TRAINING.md` — Full staged training pipeline (calibration, causal init, CV)
- `docs/TOEPLITZ_DECOMPOSITION.md` — Toeplitz decomposition and gate/direction semantics
- `docs/SVFA.md` — Structure-Value Factorized Attention
- `docs/HSIC_ADAPTIVE_BANDWIDTH.md` — Why adaptive HSIC bandwidth matters
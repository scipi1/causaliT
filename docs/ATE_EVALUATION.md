# ATE (Average Treatment Effect) Evaluation

## Definition

The **Average Treatment Effect (ATE)** measures the causal effect of an intervention:

$$\text{ATE}(S \to X, s) = E[X | \text{do}(S=s)] - E[X | \text{do}(S=0)]$$

Where:
- $E[X | \text{do}(S=s)]$ = Expected value of X when S is **intervened** to value s
- $E[X | \text{do}(S=0)]$ = Expected value of X when S is **intervened** to baseline (0)
- ATE = The **change** in expectation caused by the treatment

**Key insight**: ATE is a **difference**, not an absolute value.

---

## Implementation in CausaliT

### 1. Ground Truth Generation (`scm_ds/scm.py`)

The `compute_ate_ground_truth()` method computes ATE for each source variable:

```python
# For each intervention S=s:
baseline = compute_interventional_expectation(intervention={S: 0})    # do(S=0)
treated = compute_interventional_expectation(intervention={S: s})     # do(S=s)
ATE[S=s][X] = treated[X] - baseline[X]                                # difference
```

**Methods**:
- `analytical`: Symbolic computation assuming E[ε] = 0
- `monte_carlo`: Empirical mean from N samples

### 2. Model Evaluation (`eval_interventions.py`)

The `compute_ate_metrics()` function computes model ATE:

```python
# Get predictions for same test samples
pred_baseline = model.forward(S=0, ...)    # baseline predictions
pred_treated = model.forward(S=s, ...)     # treated predictions

model_ATE = mean(pred_treated) - mean(pred_baseline)  # difference
```

**Comparison**:
```python
error = |model_ATE - true_ATE|
```

---

## Why This Matters for Hard-Mask Models

With **correct hard masks** blocking S1→X (S1 has no children):

| Quantity | Value | Explanation |
|----------|-------|-------------|
| E[X \| do(S1=0)] | some_value | Baseline prediction |
| E[X \| do(S1=0.5)] | **same_value** | Attention blocked! No change |
| Model ATE | **0** | Difference is zero |
| True ATE | **0** | S1 doesn't cause X |
| Error | **0** | Perfect agreement ✓ |

If we compared **absolute** expectations instead:
- Model prediction: some_value (from test data)
- True expectation: 0 (analytical, assumes E[S]=0)
- Error: **non-zero** ✗ (even though causally correct!)

---

## Baseline Specification

For this project, the **baseline** intervention value is **0** for all source variables:

$$\text{baseline} = \text{do}(S = 0)$$

This is chosen because:
1. S variables are sampled from distributions centered at 0 (Uniform[-1,1])
2. Setting S=0 represents "no deviation from center"
3. ATE then measures "effect of deviating S from its center"

---

## File Outputs

### `ate_ground_truth.json`

```json
{
  "analytical": {
    "S1=0.5": {"X1": 0.0, "X2": 0.0, ...},  // ATE values (difference)
    "S2=-1.7": {"X1": -1.7, "X2": 0.0, ...}
  },
  "monte_carlo": {
    // Same structure, empirical estimates
  },
  "baseline": {
    "analytical": {"X1": 0.0, "X2": 0.0, ...},  // E[X | do(S=0)]
    "monte_carlo": {"X1": ..., "X2": ..., ...}
  }
}
```

### `ate_metrics.csv`

| intervention | variable | model_ate | true_ate | abs_error |
|--------------|----------|-----------|----------|-----------|
| S1=0.5 | X1 | 0.001 | 0.0 | 0.001 |
| S2=-1.7 | X1 | -1.68 | -1.7 | 0.02 |

---

## Normalization Handling

The model operates in **normalized space**. There are TWO normalization steps:

### 1. Input Normalization (Intervention Values)

Intervention values in `ate_ground_truth.json` are in **RAW scale** (e.g., S2=-1.7).
The model expects **normalized** inputs. We must normalize intervention values before passing to the model:

```python
# For minmax normalization (source: [-3, 3] → [0, 1])
val_normalized = (val_raw - min) / (max - min)

# Example: S2=-1.7 with source range [-3, 3]
# S2_norm = (-1.7 - (-3)) / (3 - (-3)) = 1.3 / 6 = 0.217
```

### 2. Output Denormalization (Predictions)

Model predictions are denormalized to raw scale for comparison:

```python
pred_raw = denormalize(pred_norm)
```

### Summary

| Step | Direction | Applied To |
|------|-----------|------------|
| Normalize | raw → normalized | Intervention values (input to model) |
| Denormalize | normalized → raw | Model predictions (for ATE comparison) |

Ground truth ATE is in **raw scale**, so both must be converted properly.

---

## Expected Results by Intervention Type

| Intervention | Expected ATE | Why |
|--------------|--------------|-----|
| S1=0.5 (dangling) | 0 for all X | S1 has no children |
| S2=-1.7 (one-to-one) | -1.7 for X1, 0 for others | S2 → X1 only |
| S3 (one-to-many) | Non-zero for X2, X3 | S3 → X2, X3 |
| S5 (many-to-one) | Non-zero for X4 | S5 → X4 |

---

## Debugging Checklist

If ATE errors are high for hard-mask models:

1. **Check mask loading**: Are masks correctly applied? (print attention scores)
2. **Check baseline**: Is baseline computed with do(S=0)?
3. **Check difference**: Is ATE = treated - baseline (not just treated)?
4. **Check scale**: Are both model and ground truth in same scale?
5. **Check seed**: Are treated/baseline predictions on same samples?

---

## References

- Pearl, J. (2009). Causality: Models, Reasoning, and Inference
- `scm_ds/scm.py`: `SCMDataset.compute_ate_ground_truth()`
- `causaliT/evaluation/eval_funs/eval_interventions.py`: `compute_ate_metrics()`

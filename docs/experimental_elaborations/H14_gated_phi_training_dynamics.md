# H14: Gated Φ Parametrization - Training Dynamics Analysis

**Hypothesis:** Gated Φ parametrization improves DAG learning consistency by introducing a gating mechanism for edges that better controls edge ON/OFF states.

**Dataset:** scm6 (linear Gaussian SCM)

---

## Overview

This document collects the loss vs epoch plots for the models of interest in H14 to compare training dynamics across:
1. **Baseline (independent)** - Standard Φ parametrization with independent edge learning
2. **Antisymmetric** - Φ parametrization with P(i→j) = 1 - P(j→i) constraint
3. **Gated** - Gated Φ parametrization with learnable edge gates

---

## 1. Lie Attention

### 1.1 Baseline (Lie)
**Experiment:** `single_Lie_SM_scm6_59251043`

![Loss vs Epoch - Lie Baseline](../../experiments/single/scm6/euler/single_Lie_SM_scm6_59251043/eval/eval_train_metrics/fig/loss_single_Lie_SM_scm6_59251043.png)

### 1.2 Antisymmetric (Lie)
**Experiment:** `single_Lie_SM_scm6_antisym_59251000`

![Loss vs Epoch - Lie Antisym](../../experiments/single/scm6/euler/single_Lie_SM_scm6_antisym_59251000/eval/eval_train_metrics/fig/loss_single_Lie_SM_scm6_antisym_59251000.png)

### 1.3 Gated (Lie)
**Experiment:** `single_Lie_SM_scm6_gated_59250941`

![Loss vs Epoch - Lie Gated](../../experiments/single/scm6/euler/single_Lie_SM_scm6_gated_59250941/eval/eval_train_metrics/fig/loss_single_Lie_SM_scm6_gated_59250941.png)

---

## 2. PhiSM (Phi SoftMax) Attention

### 2.1 Baseline (PhiSM Independent)
**Experiment:** `single_PhiSM_SM_scm6_indep_59250811`

![Loss vs Epoch - PhiSM Indep](../../experiments/single/scm6/euler/single_PhiSM_SM_scm6_indep_59250811/eval/eval_train_metrics/fig/loss_single_PhiSM_SM_scm6_indep_59250811.png)

### 2.2 Antisymmetric (PhiSM)
**Experiment:** `single_PhiSM_SM_scm6_antisym_59250822`

![Loss vs Epoch - PhiSM Antisym](../../experiments/single/scm6/euler/single_PhiSM_SM_scm6_antisym_59250822/eval/eval_train_metrics/fig/loss_single_PhiSM_SM_scm6_antisym_59250822.png)

### 2.3 Gated (PhiSM)
**Experiment:** `single_PhiSM_SM_scm6_gated_59970947`

![Loss vs Epoch - PhiSM Gated](../../experiments/single/scm6/euler/single_PhiSM_SM_scm6_gated_59970947/eval/eval_train_metrics/fig/loss_single_PhiSM_SM_scm6_gated_59970947.png)

---

## 3. Toeplitz-Lie Attention

### 3.1 Baseline (Toeplitz Independent)
**Experiment:** `single_Toeplitz_SM_scm6_indep_59250653`

![Loss vs Epoch - Toeplitz Indep](../../experiments/single/scm6/euler/single_Toeplitz_SM_scm6_indep_59250653/eval/eval_train_metrics/fig/loss_single_Toeplitz_SM_scm6_indep_59250653.png)

### 3.2 Antisymmetric (Toeplitz)
**Experiment:** `single_Toeplitz_SM_scm6_antisym_59250699`

![Loss vs Epoch - Toeplitz Antisym](../../experiments/single/scm6/euler/single_Toeplitz_SM_scm6_antisym_59250699/eval/eval_train_metrics/fig/loss_single_Toeplitz_SM_scm6_antisym_59250699.png)

### 3.3 Gated (Toeplitz)
**Experiment:** `single_Toeplitz_SM_scm6_gated_59250683`

![Loss vs Epoch - Toeplitz Gated](../../experiments/single/scm6/euler/single_Toeplitz_SM_scm6_gated_59250683/eval/eval_train_metrics/fig/loss_single_Toeplitz_SM_scm6_gated_59250683.png)

---

## Summary Table

| Attention Type | Parametrization | Experiment ID | Best Test R² | Soft Hamming Self (mean) | DAG Confidence Self |
|----------------|-----------------|---------------|--------------|--------------------------|---------------------|
| Lie | Baseline | single_Lie_SM_scm6_59251043 | 0.995 | - | - |
| Lie | Antisymmetric | single_Lie_SM_scm6_antisym_59251000 | 0.995 | 0.236 | 0.90 |
| Lie | **Gated** | single_Lie_SM_scm6_gated_59250941 | 0.995 | 0.236 | 0.99 |
| PhiSM | Baseline | single_PhiSM_SM_scm6_indep_59250811 | 0.995 | 0.275 | 0.90 |
| PhiSM | Antisymmetric | single_PhiSM_SM_scm6_antisym_59250822 | 0.995 | - | - |
| PhiSM | **Gated** | single_PhiSM_SM_scm6_gated_59970947 | 0.995 | 0.499 | 0.92 |
| Toeplitz | Baseline | single_Toeplitz_SM_scm6_indep_59250653 | 0.995 | 0.258 | 0.91 |
| Toeplitz | Antisymmetric | single_Toeplitz_SM_scm6_antisym_59250699 | 0.995 | 0.261 | 0.82 |
| Toeplitz | **Gated** | single_Toeplitz_SM_scm6_gated_59250683 | 0.995 | 0.257 | 0.88 |

---

## Observations

### From H14 in experiments_analysis.ipynb:

> **Francesco 11/03/2026:** For Lie attention gating helps, as the mechanism is strictly antisymmetric and there is no option "no edge". For Toeplitz, which already has such a mechanism, the improvement is weaker.

### Key Training Dynamics Observations:

1. **Two-Phase Training Pattern (Gated Lie):**
   - The loss curve shows a first plateau followed by a second descent
   - The k-fold that skipped the first phase (went straight to low loss) learned the wrong information flow
   - This suggests the first phase may involve DAG learning, while the second phase fits the data

2. **Training Instability (Gated):**
   - Gating made training unstable: for gated parametrization, 3/5 folds loss raised and plateaued at ~100
   - The folds that actually trained also had the best DAG

3. **Cross-Attention Density:**
   - In the fold with wrong DAG (k2), cross-attention is very dense
   - The model may have used cross-attention heavily to fit data, bypassing proper DAG learning

---

## AI Analysis Notes

*To be filled by AI assistant during analysis*

---

**Document created:** 2026-03-12
**Last updated:** 2026-03-12

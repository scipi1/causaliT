# Evaluation of experiments in tests/critical

The experiments aim to characterize the HSIC potential in causal learning.


## On the Bandwidth

### Adaptive bandwidth (median)
It allows to avoid that HSIC decreases with the residual magnitude, which naturally decreases as the reconstruction improves during training.

- No difference between with (`test_D3_2_hsic_high_bs_hard_64321366`) and without (`test_D3_1_hsic_high_bs_64315144`) hard mask: the bottom noise level is at the same order of magnitude in both cases


### Fixed bandwidth
Used in HSIC-bottleneck

- No difference between with (`test_D3_1_hsic_high_bs_sigma_fixed_hard_64329182`) and without (`test_D3_1_hsic_high_bs_sigma_fixed_64349255`) mask: the HSIC decreases because of the reconstruction, not because of better causal learning


### Overall: 
- the HSIC doesn't seem to be better when the causal mask is given/oracle, no matter the bandwidth choice
- Why does HSIC oscillate during training in the first place? Too high LR? Adam is not the right opt?


---

## D1 Extended: Discrete vs Continuous S — Kernel Analysis

**Date**: 2026-04-22

### Hypothesis
The paper's SCM datasets sample S from **discrete** distributions (3–11 values). The RBF kernel may not be appropriate for discrete variables. A **Dirac delta kernel** (k(s_i, s_j) = δ(s_i = s_j)) is the natural kernel for categorical/discrete data.

### Tests Run (`test_D1_hsic/test_hsic_sanity.py`, Tests 6–9)

#### Test 6: Detection Power
RBF kernel can **detect** dependence with discrete S (even better ratios than continuous):

| Config | HSIC(indep) | HSIC(dep) | Ratio |
|--------|-------------|-----------|-------|
| continuous_rbf | 0.000588 | 0.077517 | 131.9x |
| discrete_3_rbf | 0.000290 | 0.139144 | 479.9x |
| discrete_3_dirac | 0.000463 | 0.145050 | 313.1x |
| discrete_11_rbf | 0.000325 | 0.090107 | 277.3x |
| discrete_11_dirac | 0.000662 | 0.036645 | 55.4x |

**Conclusion**: Detection is NOT the problem. RBF detects dependence fine with discrete S.

#### Test 7: W Recovery (fixed paper values)
Per-variable HSIC with RBF recovers W correctly even with paper's discrete S values:

| Config | Final W error | Converged? |
|--------|--------------|------------|
| continuous_rbf | 0.0081 | ✓ |
| discrete_3_rbf (S1 values) | 0.0096 | ✓ |
| discrete_11_rbf (S3 values) | 0.0061 | ✓ |

**Conclusion**: With the exact paper distribution values and n=500, RBF works.

#### Test 9: Discretization Sweep (evenly spaced values) — THE KEY RESULT

| N levels | RBF error | Dirac error |
|----------|-----------|-------------|
| 3 | 5.015 ✗ | 3.686 ✗ |
| 4 | 0.999 ✗ | 0.005 ✓ |
| 6 | 0.405 ✗ | 0.010 ✓ |
| 11 | 1.793 ✗ | 0.005 ✓ |
| 20 | 0.640 ✗ | 0.009 ✓ |
| 50 | 0.009 ✓ | 0.007 ✓ |
| 100 | 1.040 ✗ | 0.009 ✓ |
| ∞ (cont.) | 0.004 ✓ | N/A |

### Key Finding

**RBF kernel with discrete S creates an unstable optimization landscape.** The results are erratic — sometimes works, sometimes fails catastrophically. The RBF kernel on discrete data creates a low-rank K matrix with block structure that introduces saddle points and flat regions in the HSIC loss landscape.

**Dirac kernel is stable and consistent** for ≥4 discrete levels. It fails at 3 levels due to fundamental identifiability limits (3 values × 3 variables = too few degrees of freedom for a 3×3 W matrix).

The apparent contradiction between Test 7 (works) and Test 9 (fails) for the same discretization levels reveals that **the RBF optimization is seed-sensitive and unreliable** — it can succeed under favorable conditions but fails unpredictably.

### Why This Matters for the Cluster Experiments

The cluster experiments use:
- Batch size 128 (smaller than test's n=500)
- Discrete S with 3–11 values
- Nonlinear SCM (harder than linear W)
- Full transformer model (compensation effects)

The combination of unreliable RBF kernel + small batch + nonlinear structure + model compensation explains why HSIC oscillates without converging in the cluster experiments.

### Action Plan

1. **Wire Dirac kernel for S** into training pipeline (keep RBF for residuals)
2. **Re-run cluster experiments** with Dirac kernel
3. 3-value S variables (like S1) may need ≥4 values for identifiability
4. **New config option**: `hsic_kernel_source: "dirac"` (default: "rbf")

### Implementation

Added to `causaliT/utils/hsic_utils.py`:
- `dirac_kernel(x, tolerance)` — Dirac delta kernel matrix
- `hsic_from_kernels(K, L)` — HSIC from pre-computed kernel matrices (enables mixing kernel types)

---

## D3 Continuous: End-to-End Validation with Continuous S

**Date**: 2026-04-22

### Purpose

The D1 tests show RBF+discrete S creates unstable optimization in isolated settings (synthetic W recovery). But does this translate to the full transformer model? This experiment answers: **Does HSIC work better in the full pipeline when S is continuous?**

### Dataset

`data/scm3_continuous/` — Same SCM3 structure (nonlinear, non-Gaussian) but with:
- S sampled uniformly (continuous) instead of discrete
- 80/20 random train/test split (no holdout — diagnostic only)

### Experiments

| Experiment | Structure | Key Question |
|-----------|-----------|-------------|
| `test_D3_cont_learned` | Learned (HSIC drives attention) | Can HSIC learn the DAG with continuous S? |
| `test_D3_cont_oracle` | Oracle (hard mask = true DAG) | What's the HSIC floor with perfect structure? |

Both use: batch_size=2048, per-variable HSIC, gradient routing ON, adaptive bandwidth (RBF).

### Decision Matrix

| Oracle HSIC vs Learned HSIC | Conclusion |
|----------------------------|------------|
| Oracle << Learned | ✅ HSIC learns structure with continuous S → discrete was the bottleneck → wire Dirac kernel for paper datasets |
| Oracle ≈ Learned | ❌ Deeper issue — model compensation dominates → need architectural fix before kernel choice matters |

### Status: RESULTS AVAILABLE

### Observations

**On the "oracle" comparison:**
- The model + mask is not a true oracle because the model can still choose not to use an allowed edge. With Toeplitz and CausalCross Attention, the attention parameters control *how much* information flows through each allowed edge, and gradient descent can effectively zero-out an edge. Therefore, the hard mask is a **necessary but not sufficient** condition for the correct DAG.
- Oracle HSIC is indeed smaller than learned HSIC, but in the **same order of magnitude**. This means the HSIC floor is not primarily determined by the structure — it's dominated by other factors (model capacity, noise, estimation error).
- A stronger oracle would **freeze attention scores** at the true adjacency matrix values, not just mask illegal edges.

**On adaptive bandwidth + continuous S:**
- `test_D3_cont_learned_64428116` shows a nicely decreasing HSIC with adaptive bandwidth — not observed with discrete S. This suggests the structure is being trained with HSIC.
- **Caveat**: Decreasing HSIC ≠ improving structure. HSIC can decrease due to reconstruction improvement leaking through gradient routing (weight sharing, LayerNorm, residual connections can leak despite `use_gradient_routing: true`).
- The dag_metrics show: `soft_hamming_self=0.25` (decent for X→X), `soft_hamming_cross=0.57` (poor for S→X), `mec_distance=0.95`, `skeleton_recall=0.31`, `in_mec=false`. The structure is not correct but may improve with more training budget.
- A 1000-epoch run is in progress to test whether more budget helps.

**On the Dirac kernel (`test_D3_disc_dirac_64431751`):**
- Dirac kernel on scm3 does not perform better than RBF in the full pipeline, despite being clearly better in the isolated D1 sanity tests (Test 9). 
- The gap between D1 (isolated W recovery) and the full pipeline likely comes from model compensation — the transformer routes information through alternative pathways (residual connections, multiple layers, cross-attention).
- Possible remedies: increase the number of discrete levels for S, or focus on continuous S first.

### Recommended Control Experiment
Run `test_D3_cont_learned` with `lambda_hsic_cross: 0.0` and `lambda_hsic_self: 0.0` (pure reconstruction, no HSIC gradient). If HSIC still decreases during training, then the decrease observed in the learned experiment is driven by reconstruction artifacts, not structure learning.

---

## Next Steps: Normalized HSIC (nHSIC)

**Date**: 2026-04-22

### Motivation

The standard biased HSIC estimator has known issues:
1. With adaptive bandwidth, it tends to create a **noise floor** because the bandwidth self-normalizes
2. The HSIC magnitude depends on residual and source scales, making it hard to set λ_hsic
3. Gradients can vanish as residuals shrink during training

The **Normalized HSIC** (Ma et al., AAAI 2020) addresses all three:
- nHSIC = tr(K̃ · L̃) where K̃ = K̄(K̄ + nεI)^{-1} (Tikhonov regularization)
- Scale-invariant (bounded statistic)
- Better gradients at small residual magnitudes
- Reduces noise sensitivity at small sample sizes

### Implementation Status: ✅ ALREADY IMPLEMENTED

The nHSIC is already fully wired through the codebase:
- `causaliT/utils/hsic_utils.py`: All HSIC functions support `mode="normalized"` and `nhsic_epsilon`
- `single_causal_forecaster.py` and `noise_aware_forecaster.py` both read from config
- Config fields (with defaults, so they're optional in YAML):
  ```yaml
  training:
    hsic_mode: "normalized"      # "biased" (default) or "normalized"
    nhsic_epsilon: 0.01          # Tikhonov regularization (default: 0.01)
  ```

### Experiment Plan

1. **Continuous S + nHSIC**: Re-run `test_D3_cont_learned` with `hsic_mode: "normalized"` — the cleanest test
2. **Discrete S + nHSIC + Dirac**: Test if nHSIC helps the Dirac kernel case
3. **Compare nHSIC trajectories**: oracle vs learned to see if nHSIC provides a clearer signal

### Results: `test_D3_cont_nhsic_64447703` (100 epochs, continuous S)

**Date**: 2026-04-22

Compared to biased HSIC at 100 epochs (`test_D3_cont_learned_64428116`) and 1000 epochs (`test_D3_cont_learned_64429712`):

| Metric | nHSIC (100ep) | HSIC (1000ep) | Winner |
|--------|--------------|---------------|--------|
| SH cross | **0.241** | 0.419 | ✅ nHSIC |
| SH self | 0.320 | **0.211** | ❌ HSIC |
| MEC distance | **0.522** | 0.909 | ✅ nHSIC |
| Skeleton recall | **0.651** | 0.372 | ✅ nHSIC |
| Test R² | **0.566** | 0.502 | ✅ nHSIC |
| Time/epoch | 52.3s | 10.5s | ❌ ~5× slower |

**ATE evaluation** reveals a tradeoff:
- nHSIC is **better at blocking non-causal paths** (null interventions are cleaner)
- nHSIC is **worse at transmitting multi-hop causal effects** (e.g., S3→X2→X4→X5 severely underestimated)
- Root cause: worse self-attention SH means X→X chains that mediate multi-hop effects are less accurate

**Key insight**: nHSIC improves *structure discovery* (which edges exist) but the normalization may over-regularize the *value pathway* (functional relationships along those edges). The worse self-attention is consistent with the Toeplitz mechanism needing stronger gradients to move from 0.5 initialization to 0/1.

### Strategy

Present continuous S results first (eliminates kernel choice confound). If discrete S results with nHSIC + Dirac are not satisfactory, the continuous case can be the primary result, with discrete as a "practical extension" discussion.

**Open questions:**
- Does nHSIC with 1000 epochs close the self-attention gap?
- Would hybrid nHSIC-cross + HSIC-self work?
- Is the ATE degradation from worse self-attention, or from nHSIC limiting the reconstruction quality?

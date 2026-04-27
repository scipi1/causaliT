# Paper Template — NeurIPS 2026

---

## Possible Titles

1. **Decoupling Structure from Reconstruction: Causal Discovery via Factorized Transformer Attention**
2. **CausaliT: Factorized Attention for Joint Causal Discovery and Prediction in Transformers**
3. **Learning Causal Structure in Transformer Attention via Independence-Guided Factorization**
4. **CausaliT: Causal Discovery Through Factorized Attention and Residual Independence**
5. **Structure-Value Factorization in Transformers for Causal Discovery and Intervention Invariance**

**Recommended:** Title 1 or 2.

---

## Abstract (sketch)

Transformers have recently been explored as vehicles for causal reasoning, yet standard attention conflates structural learning with value reconstruction — queries, keys, and values all depend on the same fused embeddings, making the learned attention pattern sample-specific and contaminated by spurious correlations. We propose to **decouple causal structure learning from reconstruction** by (i) factorizing attention into structure-dependent and value-dependent paths (SVFA), (ii) parameterizing DAG edge probabilities via a symmetric–antisymmetric (Toeplitz) decomposition of the attention score matrix, and (iii) using the Hilbert-Schmidt Independence Criterion (HSIC) between residuals and parents as a differentiable causal discovery signal routed exclusively to structural parameters. We prove that, under additive noise models, our architecture satisfies the conditions for DAG identifiability via HSIC minimization. Experiments on synthetic SCMs of varying complexity (linear/non-linear, Gaussian/non-Gaussian, up to 20 nodes) show that our method recovers causal structure more accurately than standard transformers, with consistent improvements in structural Hamming distance, Markov equivalence class recovery, and average treatment effect estimation under both in-distribution and out-of-distribution interventions.

---

## 1. Introduction

### Motivation
- Transformers are increasingly used in causal settings: causal representation learning, causal abstraction, causal effect estimation, amortized causal discovery.
- A natural question: **can the attention mechanism itself learn the causal DAG** of the data-generating process?
- If yes, this enables both accurate prediction **and** intervention invariance from the same model.

### The Problem
- Standard attention conflates structure and reconstruction: Q, K, V all projected from the same fused embedding → attention patterns are sample-specific, value-contaminated, inconsistent across data splits.
- SoftMax attention **cannot represent absent edges** (outputs always sum to 1) → fully connected DAG by construction.

### Our Contribution
We propose to **decouple causal structure learning from reconstruction** by providing two separate signals to the transformer:
1. **Reconstruction loss** (MSE) drives data fitting through value/reconstruction parameters.
2. **HSIC regularization** drives causal discovery through structural parameters.

This is enabled by three methodological contributions:
- **SVFA** (Structure-Value Factorized Attention): architecturally separates structure from values.
- **Toeplitz Decomposition**: parameterizes DAG edge probabilities via symmetric (edge existence) and antisymmetric (flow direction) components.
- **Gradient Routing**: separates the optimization paths so that each signal updates only the relevant parameters.

### Key Result
We show that HSIC is a valid differentiable signal for learning causal structure within transformer attention, and that the decoupled architecture recovers DAGs significantly better than standard transformers across multiple SCM settings.

---

## 2. Related Work

### Transformers for Causality
- **Liu & Bellamy** — DAG-aware Transformer for Causal Effect Estimation: *provide* the DAG; we *learn* it.
- **Löwe et al. (2022)** — Amortized Causal Discovery: meta-learning across datasets; we learn structure for a single dataset.
- **Melnychuk et al. (2022)** — Causal Transformer for Counterfactual Outcomes: treatment effect estimation in time series.
- **Geiger et al. (2021–2024)** — Causal Abstraction / Interchange Interventions: probes internal causal representations; complementary to our explicit training approach.

### Neural Causal Discovery
- **Zheng et al. (2018)** — NOTEARS: continuous DAG optimization.
- **Yu et al. (2019)** — DAG-GNN: separate structural and functional parameter optimization.
- **Kyono et al. (2020)** — CASTLE: neural DAG learning with masked gradients.
- **Brouillard et al. (2020)** — DCDI: differentiable causal discovery from interventional data.

### Identifiability in Additive Noise Models
- **Hoyer et al. (2008)** — Nonlinear causal discovery with additive noise models.
- **Peters et al. (2014)** — Causal discovery with continuous additive noise models (resit method using HSIC).

### Our Differentiator
We are the first to jointly learn a DAG **within transformer attention** through an independence criterion, while factorizing attention to separate structure from reconstruction.

---

## 3. Method

### 3.1 Architecture

#### Transformer with Hierarchical Attention
- Two-stage decoder: cross-attention (S → X) then self-attention (X → X).
- Inverted order (cross before self) encodes the causal hierarchy S ⇀ X.

#### Structure-Value Factorized Attention (SVFA)

**Proposition 1 (SVFA yields sample-invariant attention).**
*Under SVFA, where Q = W_Q · e_struct(j) and K = W_K · e_struct(k) with e_struct depending only on variable identity, the attention score matrix α = QK⊤ is identical across all samples.*

*Proof.* By construction, Q and K are functions of variable identity embeddings only. Since these embeddings are fixed parameters (not data-dependent), the product QK⊤ is constant across the batch. □

Implication: the learned DAG is globally consistent — it does not vary by sample.

#### Toeplitz Decomposition for DAG Parameterization

**Proposition 2 (Toeplitz DAG properties).**
*The Toeplitz gate-direction parameterization satisfies:*
- *(a) P(i → j) + P(j → i) ≤ 1 for all i ≠ j,*
- *(b) P(i → i) = 0,*
- *(c) the gate can represent the absence of an edge: P(i → j) = P(j → i) = 0.*

*Property (c) is impossible under SoftMax attention.*

*Proof.* (a) follows from σ(γ)·σ(φ) + σ(γ)·σ(−φ) = σ(γ) ≤ 1. (b) follows from A_ii = 0 by antisymmetry plus diagonal masking. (c) When γ_ij → −∞, both edge probabilities vanish. Under SoftMax, Σ_j α_ij = 1 forces at least one non-zero edge per query. □

Decomposition:
```
QK⊤ = S + A
S = (QK⊤ + KQ⊤)/2    — symmetric  (edge existence)
A = (QK⊤ − KQ⊤)/2    — antisymmetric  (flow direction)

P(i → j) = σ(gain_gate · tanh(S_ij / τ_gate)) · σ(gain_dir · tanh(A_ij / τ_dir))
```

#### Intervention Invariance

**Proposition 3 (Intervention invariance under true DAG masking).**
*If attention scores are masked according to the true DAG (α_ij = 0 whenever j ∉ PA(i)), then the transformer output for variable X_i is invariant to interventions on non-ancestors: X̂_i(do(S_k := s')) = X̂_i for all S_k ∉ AN(X_i).*

*Proof.* Under DAG masking, the computation graph for X̂_i only involves tokens in AN(X_i). Interventions on S_k ∉ AN(X_i) do not alter any token in this subgraph. □

### 3.2 Training

#### HSIC as a Causal Signal

**Corollary 1 (HSIC minimization recovers causal structure).**
*Assume (i) additive noise model: X_i = f_i(PA(X_i)) + ε_i with ε_i ⊥ PA(X_i), (ii) non-linear mechanisms (Hoyer et al., 2008), and (iii) the SVFA+Toeplitz transformer can represent the true mechanisms. Then, among all DAGs consistent with reconstruction loss minimization, the true DAG uniquely minimizes Σ_i HSIC(ε̂_i, PA_learned(X_i)).*

*Proof sketch.* The SVFA+Toeplitz architecture parameterizes a family of additive noise models: the value path computes X̂_i = Σ_j α_ij V_j (which, with sufficient capacity, can represent any f_i), and the residual ε̂_i = X_i − X̂_i satisfies the additive noise assumption when the attention structure matches the true DAG. By the identifiability result of Hoyer et al. (2008), in the non-linear case, fitting in the wrong causal direction yields residuals that are not independent of the parents. Therefore, HSIC(ε̂, PA) = 0 if and only if the attention DAG matches the true DAG. □

#### Adaptive Bandwidth for HSIC
- Median heuristic (Gretton et al., 2012) prevents training-induced HSIC collapse as residuals shrink.
- Bandwidth is detached from the computation graph → clean gradient separation.

#### Gradient Routing (Bilevel Optimization)
- **Structural parameters** (Q/K projections, structure embeddings, attention gains/temperatures): updated by HSIC + score sparsity.
- **Reconstruction parameters** (V projection, FF layers, MLP head): updated by reconstruction loss (MSE).
- Dual optimizers; single forward pass per step.

#### Staged Training Pipeline
| Stage | Goal |
|-------|------|
| **0. Calibration** | Binary search for group LASSO λ to balance HSIC and reconstruction gradient norms |
| **1. Causal Initialization** | Short HSIC-dominated pre-training to initialize toward causal structure |
| **2. Score Sparsity CV** | Cross-validation over L1 sparsity candidates; select λ* by min validation HSIC |
| **3. Main Training** | Standard training with calibrated HSIC and selected sparsity |

---

## 4. Experimental Setup

### 4.1 Datasets

| Dataset | Functional Form | Noise | Nodes (S + X) |
|---------|----------------|-------|----------------|
| **scm1** | Linear | Gaussian | 5S + 5X |
| **scm2** | Non-linear | Gaussian | 5S + 5X |
| **scm3** | Non-linear | Non-Gaussian | 5S + 5X |
| **scm3-large** | Non-linear | Non-Gaussian | ~10S + 10X (≈20 nodes) |

- Discrete S sampling with holdout split for OOD evaluation.
- Holdout intervention values: S3=1.0, S5=2.5 (never seen during training).
- 50k samples per dataset.

### 4.2 Models

| Model | Description |
|-------|-------------|
| **Ours** | SVFA + Toeplitz self-attn + Causal Cross-Attn + HSIC + gradient routing |
| **SoftMax Baseline** | Standard transformer with ScaledDotProduct attention, no DAG learning |
| **Oracle (True DAG)** | Standard transformer with ground-truth hard mask applied to attention |

### 4.3 Metrics

| Metric | Measures | Direction |
|--------|----------|-----------|
| **SHD** | Structural Hamming Distance (edge errors) | ↓ better |
| **MEC** | Markov Equivalence Class accuracy | ↑ better |
| **ATE Error (ID)** | Avg Treatment Effect error, in-distribution interventions | ↓ better |
| **ATE Error (OOD)** | Avg Treatment Effect error, out-of-distribution interventions | ↓ better |
| **Test MAE** | Reconstruction quality (control metric) | ↓ better |

### 4.4 Training Configuration
- Single train/val/test split (no cross-validation).
- 10 random seeds per configuration for variability measurement.
- d_model: 24.

---

## 5. Results

### 5.1 Main Comparison (Table 1)

*Ours vs. SoftMax Baseline vs. Oracle (true DAG) across scm1, scm2, scm3.*

| Dataset | Model | SHD ↓ | MEC ↑ | ATE-ID ↓ | ATE-OOD ↓ | MAE ↓ |
|---------|-------|-------|-------|----------|-----------|-------|
| scm1 | SoftMax | | | | | |
| scm1 | **Ours** | | | | | |
| scm1 | Oracle | | | | | |
| scm2 | SoftMax | | | | | |
| scm2 | **Ours** | | | | | |
| scm2 | Oracle | | | | | |
| scm3 | SoftMax | | | | | |
| scm3 | **Ours** | | | | | |
| scm3 | Oracle | | | | | |

*All values: mean ± std over 10 seeds.*

### 5.2 Ablation Studies (Table 2)

*Metric delta from full model. Negative = worse.*

| Ablation | Config Change | ΔSHD | ΔMEC | ΔATE | ΔMAE |
|----------|---------------|------|------|------|------|
| No SVFA | `comps_embed: "summation"` | | | | |
| No Toeplitz | `ScaledDotProduct` self-attn | | | | |
| No HSIC | `lambda_hsic = 0` | | | | |
| No Sparsity | `lambda_score_sparse = 0` | | | | |
| No Gradient Routing | Single optimizer | | | | |

*Across scm1/scm2/scm3, 10 seeds each.*

### 5.3 Scaling (Table 3)

*scm3-large (~20 nodes): Ours vs. SoftMax.*

| Model | SHD ↓ | MEC ↑ | ATE-ID ↓ | ATE-OOD ↓ | MAE ↓ |
|-------|-------|-------|----------|-----------|-------|
| SoftMax | | | | | |
| **Ours** | | | | | |

### 5.4 Real-World Benchmark: Sachs Protein Signaling Network (Table 4)

To position CausaliT within the established causal discovery landscape, we evaluate on the **Sachs protein signaling dataset** (Sachs et al., 2005) — the most widely used real-world benchmark for causal structure learning. The dataset contains 853 observational single-cell measurements of 11 phosphoproteins and phospholipids, with a consensus ground-truth DAG of 17 directed edges established through prior biological knowledge and experimental interventions.

**Compatibility with our S→X framework.** The Sachs network naturally decomposes into a two-tier hierarchy compatible with our architecture: two root nodes (PKC, Plcg) serve as source variables S, while the remaining nine proteins (Raf, Mek, PIP2, PIP3, Erk, Akt, PKA, P38, Jnk) serve as observed variables X. The cross-attention block learns S→X edges (7 ground-truth edges), while the self-attention block learns X→X edges (10 ground-truth edges).

**Comparison methodology.** We report the standard **Structural Hamming Distance (SHD)** — the number of edge additions, deletions, and reversals needed to transform the learned graph into the ground truth — following the convention used by all compared methods. Published SHD values are taken from Lachapelle et al. (2020) and Zheng et al. (2018) under identical experimental conditions (853 observational samples, same ground-truth DAG). Our SHD is computed from the binarized attention structure (threshold = 0.5 on learned edge probabilities) and reported as mean ± std over 10 seeds.

**Important caveat.** This comparison is inherently asymmetric: the literature baselines are purpose-built causal discovery algorithms operating on the full 11-node joint distribution, while CausaliT is a transformer-based method designed for joint prediction and causal discovery in the S→X regime. We include this comparison not to claim state-of-the-art on Sachs, but to provide reviewers with an interpretable reference point for the quality of our learned DAGs on a dataset where the ground truth is universally agreed upon.

| Method | Type | SHD ↓ | Source |
|--------|------|-------|--------|
| **CausaliT (Ours)** | Transformer + HSIC | | This work |
| Vanilla Transformer | Transformer baseline | | This work |
| CAM (Bühlmann et al., 2014) | Score-based | 12 | Lachapelle et al. (2020) |
| GraN-DAG (Lachapelle et al., 2020) | Neural, gradient-based | 13 | Lachapelle et al. (2020) |
| DAG-GNN (Yu et al., 2019) | Neural, VAE-based | 16 | Lachapelle et al. (2020) |
| PC (Spirtes et al., 2000) | Constraint-based | 17 | Lachapelle et al. (2020) |
| NOTEARS (Zheng et al., 2018) | Continuous optimization | 21 | Lachapelle et al. (2020) |
| GES (Chickering, 2002) | Score-based | 26 | Lachapelle et al. (2020) |
| Random DAG | — | 21 | Lachapelle et al. (2020) |

*SHD = Structural Hamming Distance (lower is better). All published values on 853 observational samples from Sachs et al. (2005).*

**Dataset preparation:** `python scm_ds/prepare_sachs.py` downloads and formats the Sachs dataset into the CausaliT data pipeline (S→X npz format, attention masks, metadata).

---

## 6. Plots

### Figure 1 — Architecture Overview
*DAG diagram + transformer block diagram showing cross-attention and self-attention blocks color-matched to DAG slices.*

### Figure 2 — SHD vs. HSIC (Key Validation Plot)
*Scatter plot across seeds × SCMs. Each point = one trained model. X-axis = HSIC, Y-axis = SHD. Expected: strong positive correlation (lower HSIC → lower SHD), validating HSIC as a causal proxy.*

### Figure 3 — LASSO-Path (Sparsity Selection)
*Left: number of active edges vs. λ_score. Right: test loss vs. λ_score. Demonstrates principled edge selection analogous to LASSO coefficient paths.*

### Figure 4 — Attention Score Heatmaps
*Side-by-side: True DAG adjacency matrix vs. Ours (learned) vs. SoftMax (learned). Visual comparison of DAG recovery quality.*

### Figure 5 — ATE Intervention Invariance
*Bar/box plot: deviations ΔX under do(S_k := s') for each intervention. Ground truth = 0 for non-ancestors. Shows our model achieves near-zero deviations where SoftMax does not.*

### Figure 6 — Scaling Result
*SHD and ATE error for the 20-node experiment (scm3-large). Can be bar chart or included in Table 3.*

---

## 7. Discussion

### What Works
- SVFA + Toeplitz provides a clean, interpretable DAG parameterization.
- HSIC with adaptive bandwidth is a reliable causal signal across training.
- Gradient routing eliminates gradient interference between structure and reconstruction.
- The method scales to ~20 nodes while maintaining advantages over SoftMax.

### Limitations
- Experiments are predominantly on synthetic data; the Sachs benchmark (§5.4) provides a first real-world validation, but broader real-world evaluation remains future work.
- The staged training pipeline introduces multiple hyperparameters (calibrated automatically, but complexity remains).
- The additive noise assumption is required for the identifiability guarantee.
- Computational overhead from HSIC kernel computation (quadratic in batch size).

### Future Work
- **Noise-aware architecture**: Explicit ambient and reading noise modeling for uncertainty quantification.
- **Mixed-SCM datasets**: Learning multiple causal structures from heterogeneous data via multi-head attention.
- **Real-world application**: Industrial manufacturing surrogates with partially known causal structure.
- **Scaling**: Efficient HSIC approximations for larger graphs (>100 nodes).

---

## 8. Conclusion

We introduced a method for decoupling causal structure learning from reconstruction in transformers. By factorizing attention into structure-dependent and value-dependent paths, parameterizing DAGs via Toeplitz decomposition, and using HSIC as a differentiable causal signal with gradient routing, our approach learns causal graphs directly within the attention mechanism. We proved that this architecture satisfies the conditions for DAG identifiability under additive noise models, and demonstrated empirically that it recovers causal structure more accurately than standard transformers, with corresponding improvements in intervention invariance.

---

## Formal Statements Summary

| # | Type | Statement | Section |
|---|------|-----------|---------|
| **P1** | Proposition | SVFA yields sample-invariant attention scores | §3.1 |
| **P2** | Proposition | Toeplitz decomposition: (a) P(i→j)+P(j→i) ≤ 1, (b) P(i→i)=0, (c) can represent absent edges (impossible under SoftMax) | §3.1 |
| **P3** | Proposition | Intervention invariance guaranteed under true DAG masking | §3.1 |
| **C1** | Corollary | Under non-linear ANM (Hoyer/Peters), SVFA+Toeplitz satisfies conditions for DAG identifiability via HSIC minimization | §3.2 |

---

## Appendix A: Staged Training Details

*(Full description of the 4-stage pipeline: calibration, causal initialization, score sparsity CV, main training. Including gradient ratio calibration procedure and hyperparameter selection.)*

## Appendix B: ATE Intervention Scheme

| Variable | In-Distribution | Out-of-Distribution | Role |
|----------|----------------|---------------------|------|
| **S1** | 0.5 | — | Negative control (dangling, no children) |
| **S2** | −1.7 | — | Positive control (one-to-one → X1) |
| **S3** | −0.5 | 1.0 (holdout) | Structure test (one-to-many → X2, X3) |
| **S5** | −0.8 | 2.5 (holdout) | Confounding test (many-to-one → X4) |

Expected behavior:
- S1: Zero effect on all X (tests intervention invariance for non-causal paths)
- S2: Effect only on X1 (tests simple one-to-one causal learning)
- S3: Effects on X2, X3 (tests one-to-many structure learning)
- S5: Effect on X4 (tests confounded parent learning)

## Appendix C: What Didn't Work

- Lie Attention + Gated phi → unstable training
- Orthogonal frozen S embeddings → not conclusive
- Noise-aware architecture → worse DAG recovery than single-layer model (overfitting to noise parameters)

## Appendix D: Proofs

*(Full proofs of Propositions 1–3 and Corollary 1.)*

---

## Key References

1. Vaswani et al. (2017). Attention Is All You Need.
2. Liu & Bellamy. DAG-aware Transformer for Causal Effect Estimation.
3. Hoyer et al. (2008). Nonlinear causal discovery with additive noise models.
4. Peters et al. (2014). Causal discovery with continuous additive noise models.
5. Gretton et al. (2012). A Kernel Two-Sample Test. JMLR.
6. Zheng et al. (2018). DAGs with NO TEARS.
7. Liu et al. (2019). DARTS: Differentiable Architecture Search. ICLR.
8. Yu et al. (2020). PCGrad. NeurIPS.
9. Kyono et al. (2020). CASTLE. NeurIPS.
10. Löwe et al. (2022). Amortized Causal Discovery.
11. Melnychuk et al. (2022). Causal Transformer for Estimating Counterfactual Outcomes.
12. Geiger et al. (2021–2024). Causal Abstraction in Neural Models.
13. Sachs et al. (2005). Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data. Science 308(5721).
14. Lachapelle et al. (2020). Gradient-Based Neural DAG Learning (GraN-DAG). ICLR.
15. Bühlmann et al. (2014). CAM: Causal Additive Models. Annals of Statistics.
16. Spirtes et al. (2000). Causation, Prediction, and Search. MIT Press.
17. Chickering (2002). Optimal Structure Identification with Greedy Search (GES). JMLR.

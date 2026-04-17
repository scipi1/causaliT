# Adaptive Bandwidth for HSIC: Why Not Simply Divide by Residual Norm?

## 1. Background: What HSIC Computes and Why It Needs Pairwise Kernels

### 1.1 The Goal

In causal structure learning, we want the model's residual $\varepsilon_i = x_i - \hat{x}_i$ to be **statistically independent** of the parent variables (S for cross-attention, other X's for self-attention).  If the model has correctly captured the causal mechanism $x = f(S) + \text{noise}$, then the residual should contain only noise — independent of S.

The **Hilbert-Schmidt Independence Criterion** (HSIC) is a kernel-based measure of statistical dependence between two random variables.  It has a key property: $\text{HSIC}(X, Y) = 0$ if and only if $X \perp Y$ (for characteristic kernels like the RBF).  This makes it ideal for our purpose — it detects *any* form of dependence, not just linear correlation.

### 1.2 From Samples to Dependence: The Role of Kernel Matrices

Given $n$ paired observations $\{(s_i, \varepsilon_i)\}_{i=1}^n$ (parent values and residuals from a batch), HSIC needs to measure how much knowing $s_i$ tells us about $\varepsilon_i$.

The challenge is that dependence is a **distributional** property — it lives in the joint distribution $p(s, \varepsilon)$, not in any single sample.  To estimate it from finite samples, HSIC uses **kernel matrices** that encode how similar each pair of samples is:

$$K_{ij} = k(s_i, s_j) = \exp\!\Bigl(-\frac{(s_i - s_j)^2}{2\sigma_s^2}\Bigr), \qquad L_{ij} = k(\varepsilon_i, \varepsilon_j) = \exp\!\Bigl(-\frac{(\varepsilon_i - \varepsilon_j)^2}{2\sigma_\varepsilon^2}\Bigr)$$

The intuition is:

- **$K_{ij}$** measures whether samples $i$ and $j$ have similar parent values.
- **$L_{ij}$** measures whether samples $i$ and $j$ have similar residuals.

If $s$ and $\varepsilon$ are **dependent**, then samples that are similar in $s$-space will also tend to be similar in $\varepsilon$-space — high $K_{ij}$ will coincide with high $L_{ij}$.  HSIC detects exactly this pattern by computing a centered trace:

$$\text{HSIC} = \frac{1}{(n-1)^2} \operatorname{tr}(KHLH)$$

where $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ is the centering matrix (removes the mean, so that HSIC measures dependence rather than just shared location).

### 1.3 Why Every Pair Matters

The kernel matrix $L$ is computed over **all** $n \times n$ pairs of residuals — not just between residuals and parents, but between every residual sample and every other residual sample.  This is essential because:

1. **Dependence is relational**: To detect that "similar parents → similar residuals", we need to compare *all pairs* of samples.  A single residual value tells us nothing about dependence; it is the *pattern of co-variation across samples* that reveals it.

2. **Non-linear detection**: The RBF kernel maps each scalar residual $\varepsilon_i$ into an infinite-dimensional feature space.  By computing $L_{ij}$ for all pairs, we are effectively computing the **Gram matrix** of these feature vectors — the full inner-product structure needed to detect any functional relationship, not just linear ones.

3. **The centering step**: After centering ($LH$), the kernel matrix encodes how residuals **deviate from their mean** in a distributional sense.  The trace $\operatorname{tr}(KHLH)$ then measures the alignment between the centered parent structure and the centered residual structure across the entire batch.

### 1.4 The Bandwidth Controls Resolution

The bandwidth $\sigma$ in the RBF kernel determines the **scale at which the kernel distinguishes samples**:

- **$\sigma$ too large** (relative to $|\varepsilon_i - \varepsilon_j|$): All kernel entries $L_{ij} \approx 1$.  The kernel sees all residuals as "the same" — it has no resolution.  After centering, $LH \approx 0$ and HSIC collapses to zero regardless of actual dependence.

- **$\sigma$ too small**: Only self-similarities survive ($L_{ii} \gg L_{ij}$ for $i \neq j$).  The kernel matrix approaches the identity — every sample looks unique.  HSIC becomes noisy and overly sensitive to individual samples.

- **$\sigma$ well-matched to data spread** (median heuristic): The kernel preserves the relative geometry of the residuals.  Samples that are genuinely close get high similarity; distant samples get low similarity.  This is the regime where HSIC has maximum statistical power.

This is the crux of the problem: as training proceeds and residuals shrink, a fixed $\sigma$ becomes "too large" and the residual kernel degenerates.

---

## 2. The Training-Induced Collapse

During training, the model's reconstruction error decreases:

$$\varepsilon_i = x_i - \hat{x}_i \;\longrightarrow\; 0$$

Because HSIC is computed between the **parents** (S or X) and the **residuals** (ε), its magnitude naturally shrinks — not because the causal structure is better learned, but simply because ε becomes smaller.  This artificial decay makes the HSIC signal unreliable as a regularizer across training epochs.

A natural first idea is to **normalize HSIC by the residual norm**:

$$\text{HSIC}_{\text{norm}} = \frac{\text{HSIC}(S, \varepsilon)}{\|\varepsilon\|^2}$$

Below, we explain why **adaptive kernel bandwidth** (the median heuristic) is a more principled and robust solution.

---

## 3. Why Dividing by Residual Norm Fails

### 3.1 It Treats the Symptom, Not the Cause

The HSIC collapse is not merely a scaling issue — it is a **kernel degeneracy** problem.  When ε shrinks, the RBF kernel matrix on the residuals becomes:

$$L_{ij} = \exp\!\Bigl(-\frac{(\varepsilon_i - \varepsilon_j)^2}{2\sigma^2}\Bigr) \;\longrightarrow\; 1 \quad \forall\, i, j$$

because all pairwise distances $|\varepsilon_i - \varepsilon_j| \ll \sigma$.  A kernel matrix of all-ones is rank-1 and carries **no distributional information**: after centering ($H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$), we get $LH \approx 0$.

Dividing $\text{HSIC} \approx 0$ by $\|\varepsilon\|^2 \approx 0$ does not recover the lost distributional structure — it just amplifies numerical noise.  The fundamental issue is that the kernel can no longer distinguish between different residual samples, and no post-hoc rescaling can fix that.

### 3.2 Numerical Instability

When residuals are small, both numerator (HSIC) and denominator ($\|\varepsilon\|^2$) approach zero, creating a $0/0$ indeterminate form.  In practice this leads to:

- **Gradient explosions** when the denominator shrinks faster than the numerator.
- **Noisy gradients** from dividing two near-zero quantities, each with their own finite-precision errors.
- **Sensitivity to outlier batches** where a single large residual can dominate the norm.

Adding a stabilizing constant $\|\varepsilon\|^2 + c$ mitigates explosions but introduces an arbitrary hyperparameter $c$ that interacts with the training dynamics in unpredictable ways.

### 3.3 Loss of Dependence Information

HSIC is designed to measure **statistical dependence** — the full joint relationship between two random variables.  The residual norm $\|\varepsilon\|^2$ measures only **marginal variance** (second moment of ε).  Normalizing by it conflates two conceptually distinct quantities:

- HSIC detects *any* functional relationship: linear, non-linear, heteroscedastic.
- $\|\varepsilon\|^2$ captures only the overall scale.

If the residuals are small but still structurally dependent on a parent (e.g., $\varepsilon_i \propto 0.001 \cdot S_i$), the raw HSIC collapses, but so does the norm — the ratio is undefined or dominated by noise.  Adaptive bandwidth, by contrast, **rescales the kernel to match the data's actual spread**, correctly detecting the residual dependence regardless of its absolute magnitude.

### 3.4 Breaks Calibration

In our staged training pipeline, we calibrate $\lambda_{\text{HSIC}}$ by matching the Frobenius norms of the reconstruction and HSIC gradients:

$$\|\nabla_\theta \mathcal{L}_{\text{recon}}\| \;\approx\; \lambda_{\text{HSIC}} \cdot \|\nabla_\theta \text{HSIC}\|$$

If HSIC is divided by $\|\varepsilon\|^2$, the gradient $\nabla_\theta \text{HSIC}_\text{norm}$ picks up chain-rule terms from the denominator, coupling the HSIC gradient to the reconstruction loss in complex ways.  This makes the calibrated $\lambda_{\text{HSIC}}$ less interpretable and less transferable across training stages.

---

## 4. Why Adaptive Bandwidth Works

### 4.1 The Median Heuristic

The **median heuristic** (Gretton et al., 2012) sets the kernel bandwidth to:

$$\sigma = \text{median}\bigl\{ |z_i - z_j| : i < j \bigr\}$$

computed **separately** for each variable: $\sigma_x$ from the parents, $\sigma_\varepsilon$ from the residuals.

This ensures:

- Roughly half the kernel entries are above $e^{-1/2} \approx 0.61$ and half below.
- The kernel matrix is **well-conditioned**: neither all-ones (too flat) nor near-identity (too peaked).
- The centered kernel $LH$ retains full distributional information.

### 4.2 Kernel Theory: Why Bandwidth Matters

The RBF kernel $k_\sigma(z_i, z_j) = \exp(-\|z_i - z_j\|^2 / 2\sigma^2)$ defines a feature map $\phi_\sigma$ into a reproducing kernel Hilbert space (RKHS) $\mathcal{H}_\sigma$.  The HSIC is:

$$\text{HSIC}(X, Y) = \| C_{XY} \|_{\text{HS}}^2$$

where $C_{XY}$ is the cross-covariance operator between the RKHS embeddings of X and Y.

**When $\sigma$ is too large relative to the data spread** (our problem: residuals ε ≪ σ):

- The feature map $\phi_\sigma(\varepsilon)$ becomes approximately **constant** — all residuals map to nearly the same point in RKHS.
- The empirical embedding $\hat{\mu}_\varepsilon = \frac{1}{n}\sum_i \phi_\sigma(\varepsilon_i)$ coincides with every individual embedding.
- The centered embedding $\phi_\sigma(\varepsilon_i) - \hat{\mu}_\varepsilon \approx 0$, so HSIC ≈ 0 regardless of the true dependence structure.

**When $\sigma$ adapts to the data scale** (median heuristic):

- The feature map preserves the **relative geometry** of the data points.
- The centered embeddings span a meaningful subspace of RKHS.
- HSIC correctly reflects the dependence between parents and residuals.

Crucially, the adaptive bandwidth **does not change what HSIC measures** — it remains a consistent test of independence.  It only ensures the kernel has enough resolution to detect the dependence at the current data scale.

### 4.3 Theoretical Guarantees

The median heuristic is not an ad-hoc fix.  It has been studied extensively in the kernel two-sample testing and independence testing literature:

1. **Consistency**: HSIC with the median heuristic is a consistent estimator of the population HSIC.  As $n \to \infty$, the empirical HSIC converges to the true HSIC, and the median bandwidth converges to a value that maximizes test power (Gretton et al., 2012).

2. **Minimax optimality**: Adaptive bandwidth selection achieves minimax optimal rates for kernel-based independence tests under smoothness assumptions on the joint density (Ramdas et al., 2015).

3. **Scale invariance**: Because the bandwidth tracks the data scale, the resulting HSIC is naturally **invariant to affine rescaling** of either variable.  This is precisely the property we need: as residuals shrink by a factor $\alpha$, the bandwidth shrinks proportionally, and the HSIC value reflects only the **dependence structure**, not the scale.

### 4.4 Gradient Properties

The bandwidth is computed from detached data (no gradient flows through the median computation).  This means:

- Gradients of HSIC flow only through the kernel evaluations $k_\sigma(z_i, z_j)$, not through $\sigma$ itself.
- The optimization landscape remains smooth — no discontinuities from the median operation.
- The gradient magnitude scales naturally with the dependence strength, not with the residual scale.

This is fundamentally different from dividing by $\|\varepsilon\|^2$, which introduces gradient terms $\propto \nabla_\theta \|\varepsilon\|^2$ that couple the HSIC regularizer to the reconstruction loss.

---

## 5. Comparison Summary

| Aspect | Divide by $\|\varepsilon\|^2$ | Adaptive Bandwidth |
|--------|-------------------------------|-------------------|
| **Kernel degeneracy** | Not addressed — kernel still all-ones | Fixed — kernel stays well-conditioned |
| **Numerical stability** | 0/0 risk, gradient explosions | Stable (median is robust, clamped ≥ 1e-5) |
| **Dependence detection** | Lost when kernel collapses | Preserved at any residual scale |
| **Gradient coupling** | Couples HSIC gradient to recon loss | Clean separation (σ detached) |
| **Calibration compatibility** | Breaks gradient-norm balancing | Fully compatible |
| **Theoretical backing** | Ad-hoc normalization | Gretton et al. (2012), 20+ years of kernel literature |
| **Hyperparameters** | Needs stabilization constant $c$ | None (fully data-driven) |
| **Implementation** | Trivial but fragile | Simple and robust |

---

## 6. Implementation

In our codebase, adaptive bandwidth is enabled via config:

```yaml
training:
  hsic_adaptive_bandwidth: true   # Enable median heuristic
  hsic_sigma: 1.0                 # Ignored when adaptive_bandwidth=true
```

The implementation in `causaliT/utils/hsic_utils.py`:

1. **`_median_bandwidth(x)`**: Computes σ = median(|x_i − x_j|) for i < j, detached from the computation graph, clamped at 1e-5 for numerical safety.
2. **`hsic(x, y, adaptive_bandwidth=True)`**: Computes separate σ_x and σ_y, then builds the RBF kernel matrices K and L with their respective bandwidths.
3. All higher-level functions (`hsic_per_token`, `hsic_per_x_pair`, `hsic_attention_weighted`) propagate the `adaptive_bandwidth` flag transparently.

---

## 7. References

1. **Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).**
   *A Kernel Two-Sample Test.*
   Journal of Machine Learning Research, 13, 723–773.
   — Foundational paper on kernel-based hypothesis testing; introduces the median heuristic for bandwidth selection and proves consistency of HSIC.

2. **Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B., & Smola, A. J. (2008).**
   *A Kernel Statistical Test of Independence.*
   Advances in Neural Information Processing Systems (NeurIPS) 20.
   — Establishes HSIC as a non-parametric independence criterion and analyzes its statistical properties.

3. **Ramdas, A., Reddi, S. J., Póczos, B., Singh, A., & Wasserman, L. (2015).**
   *On the Decreasing Power of Kernel and Distance Based Nonparametric Hypothesis Tests in High Dimensions.*
   AAAI Conference on Artificial Intelligence.
   — Analyzes the power of kernel-based tests and the role of bandwidth selection.

4. **Schölkopf, B. & Smola, A. J. (2002).**
   *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond.*
   MIT Press.
   — Comprehensive reference on RKHS theory, kernel methods, and the role of bandwidth in kernel function design.

5. **Fukumizu, K., Gretton, A., Sun, X., & Schölkopf, B. (2008).**
   *Kernel Measures of Conditional Dependence.*
   Advances in Neural Information Processing Systems (NeurIPS) 20.
   — Extends HSIC to conditional independence testing; discusses kernel choice and bandwidth.

6. **Song, L., Smola, A., Gretton, A., Bedo, J., & Borgwardt, K. (2012).**
   *Feature Selection via Dependence Maximization.*
   Journal of Machine Learning Research, 13, 1393–1434.
   — Uses HSIC for feature selection with adaptive bandwidth; demonstrates practical effectiveness of the median heuristic.

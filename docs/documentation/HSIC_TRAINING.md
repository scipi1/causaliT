# HSIC in Training: Adaptive Bandwidth and Batch Size

## Background

In causal structure learning, we want the model's residual $\varepsilon_i = x_i - \hat{x}_i$ to be **statistically independent** of the parent variables (S for cross-attention, other X's for self-attention). The **Hilbert-Schmidt Independence Criterion** (HSIC) is a kernel-based measure of dependence with the property $\text{HSIC}(X, Y) = 0 \iff X \perp Y$ for characteristic kernels like the RBF.

Given $n$ paired observations $\{(s_i, \varepsilon_i)\}_{i=1}^n$, HSIC uses RBF kernel matrices:

$$K_{ij} = \exp\!\Bigl(-\frac{(s_i - s_j)^2}{2\sigma_s^2}\Bigr), \qquad L_{ij} = \exp\!\Bigl(-\frac{(\varepsilon_i - \varepsilon_j)^2}{2\sigma_\varepsilon^2}\Bigr)$$

$$\text{HSIC} = \frac{1}{(n-1)^2} \operatorname{tr}(KHLH)$$

where $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ is the centering matrix.

Two practical issues arise during training: the **bandwidth collapse** as residuals shrink, and **estimation variance** from small batch sizes.

---

## 1. Training-Induced Bandwidth Collapse

### The Problem

During training, reconstruction error decreases: $\varepsilon_i = x_i - \hat{x}_i \to 0$. With a fixed kernel bandwidth $\sigma$, the RBF kernel on residuals degenerates:

$$L_{ij} = \exp\!\Bigl(-\frac{(\varepsilon_i - \varepsilon_j)^2}{2\sigma^2}\Bigr) \longrightarrow 1 \quad \forall\, i, j$$

because all pairwise distances $|\varepsilon_i - \varepsilon_j| \ll \sigma$. After centering, $LH \approx 0$, and HSIC collapses to zero regardless of actual dependence structure.

### Why Normalizing by Residual Norm Fails

A natural workaround, $\text{HSIC}_{\text{norm}} = \text{HSIC}(S, \varepsilon) / \|\varepsilon\|^2$, fails for several reasons:

1. **Doesn't fix kernel degeneracy**: The kernel is still all-ones — no post-hoc rescaling recovers the lost distributional structure.
2. **Numerical instability**: Both numerator and denominator approach zero (0/0 indeterminate), causing gradient explosions.
3. **Conflates dependence with scale**: HSIC detects any functional relationship; $\|\varepsilon\|^2$ captures only marginal variance.
4. **Breaks calibration**: The gradient $\nabla_\theta \text{HSIC}_{\text{norm}}$ picks up chain-rule terms from the denominator, coupling the HSIC gradient to the reconstruction loss.

### Solution: Adaptive Bandwidth (Median Heuristic)

The **median heuristic** (Gretton et al., 2012) sets the bandwidth to:

$$\sigma = \text{median}\bigl\{ |z_i - z_j| : i < j \bigr\}$$

computed separately for parents ($\sigma_s$) and residuals ($\sigma_\varepsilon$).

| Aspect | Divide by $\|\varepsilon\|^2$ | Adaptive Bandwidth |
|--------|-------------------------------|-------------------|
| Kernel degeneracy | Not addressed | Fixed — kernel stays well-conditioned |
| Numerical stability | 0/0 risk, gradient explosions | Stable (median is robust, clamped ≥ 1e-5) |
| Dependence detection | Lost when kernel collapses | Preserved at any residual scale |
| Gradient coupling | Couples HSIC gradient to recon loss | Clean separation (σ detached from graph) |
| Hyperparameters | Needs stabilization constant | None (fully data-driven) |

**Theoretical guarantees**: HSIC with the median heuristic is a consistent estimator (Gretton et al., 2012), achieves minimax optimal rates (Ramdas et al., 2015), and is scale-invariant — as residuals shrink by factor α, the bandwidth shrinks proportionally.

---

## 2. Batch Size and HSIC Estimation Variance

### The Problem

With small batch sizes (n=128), the HSIC kernel estimator has high variance. HSIC estimation variance scales as O(1/n²):

| N samples | Relative variance (vs N=128) |
|-----------|------------------------------|
| 128 | 1.0× |
| 512 | 0.06× |
| 1024 | 0.016× |
| 2048 | 0.004× |

Observed symptoms with n=128: HSIC values oscillate in a narrow range (~0.002–0.005) across training without converging, score sparsity CV selection becomes dominated by noise, and HSIC fails to guide structure learning. This is a variance problem, not a bandwidth problem — with correct adaptive bandwidth, HSIC should decrease when true structure is learned, but the estimate is too noisy at n=128 to detect the independence pattern.

### Practical Mitigations

**Increase batch size**: `batch_size: 1024` — trivial for small models (48K params, HSIC kernel ~4 MB). Often sufficient to reduce variance enough for HSIC to drive structure learning.

**DARTS-style bilevel optimization** (for larger models where big batches are expensive):

| | DARTS | Our approach |
|---|---|---|
| **Upper level** (slow) | Architecture params α | Structural params θ_S (Q, K, embeddings) |
| **Lower level** (fast) | Network weights w | Reconstruction params θ_R (V, FF, MLP head) |
| **Upper objective** | L_val(w*(α), α) | HSIC on large accumulated batch |
| **Lower objective** | L_train(w, α) | MSE reconstruction loss |
| **Update pattern** | w per-batch, α per-epoch | θ_R per-batch, θ_S per-epoch (large batch) |

Within each epoch: update θ_R via MSE on small batches while buffering raw (S, X) data on CPU. At epoch end: fresh forward pass through the buffered large batch, compute HSIC on residuals, backward through θ_S only, step structural optimizer. Each structural update is less frequent but far cleaner.

---

## 3. Implementation

Enabled via config:

```yaml
training:
  hsic_adaptive_bandwidth: true   # Enable median heuristic
  hsic_sigma: 1.0                 # Ignored when adaptive_bandwidth=true
```

Implementation in `causaliT/utils/hsic_utils.py`:

- `_median_bandwidth(x)`: Computes σ = median(|x_i − x_j|) for i < j, detached from the computation graph, clamped at 1e-5.
- `hsic(x, y, adaptive_bandwidth=True)`: Computes separate σ_x and σ_y, builds RBF kernel matrices K and L with respective bandwidths.
- All higher-level functions (`hsic_per_token`, `hsic_per_x_pair`, `hsic_attention_weighted`) propagate the `adaptive_bandwidth` flag transparently.

---

## References

- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). *A Kernel Two-Sample Test.* JMLR 13, 723–773.
- Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B., & Smola, A. J. (2008). *A Kernel Statistical Test of Independence.* NeurIPS 20.
- Ramdas, A., Reddi, S. J., Póczos, B., Singh, A., & Wasserman, L. (2015). *On the Decreasing Power of Kernel and Distance Based Nonparametric Hypothesis Tests in High Dimensions.* AAAI.
- Liu, H., Simonyan, K., & Yang, Y. (2019). *DARTS: Differentiable Architecture Search.* ICLR.
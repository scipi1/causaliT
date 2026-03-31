# Paper Outline

- Motivate the need of a surrogate that is intervention invariance
- Intervention invariance is guaranteed if the true DAG is given to the attention (maybe not even), 
    - how does the ETA look like?
    - shall we keep the hard-mask model in the benchmarks?

- Since the causal structure is not always available, learning the DAG through attention is desired.

- Standard attention has fundamental problems in learning the DAG (from CausaliT report)

### Proposal
- Architectural changes designed for learning the causal structure
    - SVFA 
    - Toeplitz Attention + Cross Causal
    - noise_aware architecture

- Regularization & Training
    - HSIC regularization (potentially with cross-splitting) with initial calibration in `staged_training` ✅
    - L1 (sparsity) on scores (still don't know how to select it) ⌛


### Results

- Benchmark models
    - Toeplitz + SVFA (Ours)
    - Vanilla Transformer
    - Vanilla Transformer true DAG

- Datasets
    - scm 1
    - scm 2
    - scm 3
    - scm_mix

- Metrics (variability given by 10 seeds)
    - ETA
    - SHD
    - Markov Equivalence Class
    - test MAE (as a control metric)

- Ablation: record metrics drop from benchmark (variability given by 10 seeds)
    - No SVFA
    - SoftMax attention
    - No noise aware architecture
    - No HSIC regularization

- lambda_L1 vs edge score Plot vs test loss

### Appendix 1: How to use CausaliT
- Criterion for d_model: HSIC under random masks --> propose a starting value of lambda_hsic
- Criterion for lambda_sparse: minimum number of edges such that the loss doesn't change
- Warning for user for too large lambda_hsic when training gets unstable

### Appendix 2: ATE Intervention Scheme

Paper uses non-binary treatments with carefully selected intervention values:

| Variable | In-Distribution | Out-of-Distribution | Role |
|----------|-----------------|---------------------|------|
| **S1** | 0.5 | - | Negative control (dangling, no children) |
| **S2** | -1.7 | - | Positive control (one-to-one → X1) |
| **S3** | -0.5 | 1.0 (holdout) | Structure test (one-to-many → X2, X3) |
| **S5** | -0.8 | 2.5 (holdout) | Confounding test (many-to-one → X4) |

**Total: 6 interventions**

Expected behavior:
- S1: Zero effect on all X (tests intervention invariance for non-causal paths)
- S2: Effect only on X1 (tests simple one-to-one causal learning)
- S3: Effects on X2, X3 (tests one-to-many structure learning)
- S5: Effect on X4 (tests confounded parent learning)

### Appendix 2: What didn't work
- Lie Attention + Gated phi --> unstable training
- Orthogonal frozen S embeddings (maybe it works, still to check)



# TODO today

## Finalized Action Plan for Paper Experiments

### Datasets
All experiments use **discrete S sampling with holdout split**:
- `data/scm1/` - Linear Gaussian (discrete holdout)
- `data/scm2/` - Non-linear Gaussian (discrete holdout)
- `data/scm3/` - Non-linear Non-Gaussian (discrete holdout)

Holdout values: S3=1.0, S5=2.5 → OOD test evaluation

**To generate datasets:**
```bash
python -m scm_ds.datasets
```

### Training Configuration
- `k_fold: 1` (no cross-validation, single train/val/test split)
- `d_model: 24` (from template, adjust based on past experiments)
- 10 seeds for variability measurement

### Step 1: Sparsity Sweep (YOUR model)
**Config:** `config_noise_aware.yaml` + `sweeps/sweep_sparsity_joint.yaml`
- Sweeps λ_self × λ_cross: [0.0, 0.01, 0.05, 0.1, 0.5]²
- 3 seeds × 3 datasets = 225 runs total
- **Output:** Lasso-style 2D heatmap, select λ_sparse*

### Step 2: HSIC Sweep (YOUR model)
**Config:** `config_noise_aware.yaml` + `sweeps/sweep_hsic.yaml`
- Sweeps λ_hsic_cross × λ_hsic_self
- 3 seeds × 3 datasets = 135 runs total
- **Output:** Stability upper bound, select λ_hsic*

### Step 3: Main Experiments (10 seeds × 3 datasets)
Run in parallel:
1. **Vanilla baseline (no reg):** `config_vanilla_transformer.yaml`
2. **Vanilla baseline (same λ*):** `config_vanilla_transformer.yaml` with λ* overrides
3. **Our model (λ*):** `config_noise_aware.yaml` with λ* overrides

**Output:** Comparison table (ETA, SHD, MEC, MAE ± std)

### Step 4: Ablation Studies (10 seeds × 3 datasets)
| Ablation | Config Change |
|----------|---------------|
| No SVFA | `comps_embed_S/X: "summation"` |
| No Toeplitz | `dec_self_attention_type: "ScaledDotProduct"` |
| No noise-aware | Use `SingleCausalLayer` |
| No HSIC | `lambda_hsic_cross/self: 0.0` |
| No sparsity | `lambda_self/cross_score_sparse: 0.0` |

**Output:** Metric drop table from main model

### Step 5: Analysis & Plots
1. Lasso-style edge selection plot (from Step 1)
2. Comparison table (from Step 3)
3. Ablation metric drop table (from Step 4)

---

# Important notes
- the d_model sweep will tell if we would benefit from cross-splitting
- the experiments with different architectures will tell us 




# TODOs in priority order


## (!!!) How to aggregate k-fold results?
- stack? https://arxiv.org/pdf/2401.01645
- if we use cross split, we need some sort of aggregation
- don't use cross-validation at all
- select the best fold but not according to loss but HSIC, which is the causality metric used during training




## Cross-validation inconsistency SOLVED ✅
It was a bug in the trainer where the model was not initialized in the same way for each fold, the seed was at the beginning and, as training kept going it changed/got consumed, changing the model initialization


## Multi-head ❔
So far we are using only one head. As the inconsistency bug is solved, we can focus on stability, i.e. precision with different initialization. Multiple heads could explore several causal path coming from such randomness at the same time, so that the final attention path is overall robust against such variations.

**Fair Point:** MHA could be lazy and make all head converge to the same representation, which could then, in turn, change with seed. Why not directly training different models with different seeds at this point?



## Improve Teoplitz ✅
The symmetric part should represent the total probability of finding an edge. The antisymmetric component informs on the direction of the information flow. The direction should be bound to the total probability of the edge, else the model can represent the same situation with a large anti-symmetry and a low symmetry or a low anti-symmetry and a high symmetry.

**Result:** SA is now very dense, because the gates stay open with zero alignment, since Sigmoid(0)=0.5. We need to L1 regularize it!

## Markov Equivalence Class ✅
Are we retrieving DAGs that are, at least in the Markov equivalence class of the true one? --> Added to attention metrics.

## Alignment evaluation 
Currently we don't have an evaluation function that shows the learned alignment, i.e. the product of key and queries. This was motivated by the fact that this quantity can be proxied by the attention score, which is evaluated, nevertheless, different attention design use the alignment differently.


## Complexity vs HSIC ⌛
For the HSIC to be informative on the correct causal structure, it is important that the model is not too complex. Carry on a parametric sweep by changing the model dimension, seed (self-attention?) and logging HSIC. Changing the seed will lead to different causal scores (SHD) and we can observe the coupling with HSIC at different complexity regimes.

### Uniform vs discrete $S$ sampling
In addition of not being realistic, uniformly sampling the source nodes could make the HSIC regularizer not so effective. In previous experiments (when preparing the report for Causality) we noticed that the u-shape of the Gaussian sampling for the $S$ nodes was reflecting on the HSIC and was gone with uniform.

![HSIC vs SHD Uniform sampling](../experiments/noise_aware_single/scm1/euler/sweep_d_model_60863422/eval/eval_d_model_sweep/hsic_shd.png)


## Fix eval_emb, not working for noise_aware (or remove from sweep, are they important?)

## Run all experiments
All experiments need to be re-run due to the seed bug in the trainer which led to wrong cross-validation consistency. We can leverage some knowledge gained so far to make sure the new experiments are "fair", i.e. the control variable is the main source of performance improvement/degradation. To achieve this, all experiments should have the same macro settings.

### Training
- All regularization OFF
- Tau set to reasonable value for exploration exploitation tradeoff (no annealing)
- Model dimension: HSIC and model dimension are related, as for a more rigid model, HSIC becomes a better proxy for correct causal learning --> sweep running⌛
- Learning rate: define a rate which is safe, i.e. stable training throughout all configurations
- Substitute ToeplitzLie with Toeplitz
- No gating, only

### Sweeps
- HSIC lambda
- L1 sparsity: applying sparsity only on SA will probably activate spurious edges from the cross attention to compensate. By sweeping lambda, we can see which spurious connection arise in CA due to edge suppression in SA. This will be problematic to plot


## How to determine d_model?
Currently we do sweeps with seeds and d_model but, in practice, this is not feasible. We can propose an heuristic method based on the following fact: instead of hoping the model to learn a very wrong DAG, we just randomly provide it and train for some epochs. If the HSIC is independent of the DAG, the model is too complex. If the loss is very bad, the model is too simple, therefore we choose d_model by optimizing Loss (the model has to fit) and HSIC independence, the HSIC loss should respond to different random enforced DAGs (noise additive assumption).

## How to determine lambda?
Also lambda we should do a sweep and find the combination that lowers the test loss.

## Test loss as a proxy for generalization
In the case of discrete $S$ sampling we can leave out one choice for testing. This would be 100% unseen during training, ensuring that the test metrics capture the generalization. What if we use a sort of cross split where we keep out one "label" at a time, and use it to calculate the HSIC instead of the validation loss, which contains S data that we also used for training? In this way HSIC is unbiased.


## Clean Code
- old experiment ID in training
- make many loggins ON by default and delete option, always log
- trim very long comments/explanations
- adjust config templates
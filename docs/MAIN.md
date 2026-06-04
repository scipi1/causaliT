# MAIN

This document aims to collect the project status main findings, open problems and the next steps.

## Status
- The machinery of calibration, causal initialization, sparsity cross-validation is temporary on pause. They offer a nice practical framework to select some hyperparameters but can hide some bugs that we need to address first. For example, we realized that the decreasing of HSIC was not because of the HSIC itself but an artefact of sparsity

## Findings

- F1: Providing the true DAG in the attention (cite the work that proposed that) made the model better for zero-expected effects (no edges). In practice, the mask made the model worse on prediction on other edges, which resulted in worse ATE (Not documented). 

- F2: *"On using the self (X)-/-cross (S) attention HSIC"*. Regarding the HSIC optimization, in normal SVFA, removing the contribution from the S helped because the system was not forced to learn a spurious edge. With the `SVFA_residual`, ameliorated this (Not documented). 

- F3: In gradient routing, the structure optimization from the HSIC is challenging. It was observed that lower learning rate (1E-4) improves stability and noise but the optimization objective flattens. The optimization has also shown signs of chaotic dynamics: for the same seeds it led to very different results.

- F4: The `noise-aware` architecture and its variants underperform in reconstruction. From `experiments\2_ARCH_STUDY\OPTUNA_STUDY`, the R2 of those models saturates at 0.6 

- F5: The oracle condition scores the minimum structural signal (from HSIC) for the noise-aware architecture. The single architecture shows better structural signal when it can cheat with spurious edges.

## Problems
- P1: Optimizing the structure is challenging (see F3)
- P2: The model structure doesn't converge to the true DAG
- P3: No documented advantage of the SVFA-family of method or separating the gradient. The model works well and trains but no quantitative improvements have been documented.

## Open questions
1. We see during training that often the SHD decreases. Is it because the model learns the causal structure or because the attention scores move away from the initialization during training? This is important because reporting a decreasing SHD $\neq$ the model is learning the causal structure.
2. Is a low SHD actually needed for good ATE performance?


## Elaborations
With the noise-aware model, we saw that the oracle is heuristically at the minimum of the structural supervisory signal (F5). Therefore, optimizing the structure of the noise-aware architecture should get the model there, solving (P2). Nevertheless, this is not so easy due to (P1). According to this reasoning, P1 should have the highest priority. 

## Next steps
1. Test a simple model providing both $S,X$ to see if cross-attention recovers the causal structure.
2. Batch size study on HSIC
3. Extensive experiment to support F1 in `experiments\1_FOUNDATIONS\0_BASELINE`
4. Extensive experiment to support F2
5. Continue the study on the `experiments\5_EXPLORATORY\PARTIAL_ANM` to tackle P1
6. Better understand why F4
7. Optimize reconstruction-optimized models on the structure learning with Optuna.

# Paper Outline

- Motivate the need of a surrogate that is intervention invariance
- Intervention invariance is guaranteed if the true DAG is given to the attention
- Learning the DAG through attention is desired

- Standard attention has problems
- Architectural changes designed for learning the causal structure
- Show that we are more intervention invariant than vanilla transformer from the ETA
- Show the learned causal structure


# TODO today
- dataset test split on unique single choices (generalization test)
- Add a metric for Markov Equivalence class calculation



# TODOs in priority order

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


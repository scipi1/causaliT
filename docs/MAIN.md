# Paper Outline

- Motivate the need of a surrogate that is intervention invariance
- Intervention invariance is guaranteed if the true DAG is given to the attention
- Learning the DAG through attention is desired

- Standard attention has problems
- Architectural changes designed for learning the causal structure
- Show that we are more intervention invariant than vanilla transformer from the ETA
- Show the learned causal structure


# TODOs in priority order

### Cross-validation inconsistency SOLVED ✅
It was a bug in the trainer where the model was not initialized in the same way for each fold, the seed was at the beginning and, as training kept going it changed/got consumed, changing the model initialization


### Multi-head
So far we are using only one head. As the inconsistency bug is solved, we can focus on stability, i.e. precision with different initialization. Multiple heads could explore several causal path coming from such randomness at the same time, so that the final attention path is overall robust against such variations.

### Complexity vs HSIC ⌛
For the HSIC to be informative on the correct causal structure, it is important that the model is not too complex. Carry on a parametric sweep by changing the model dimension, seed (self-attention?) and logging HSIC. Changing the seed will lead to different causal scores (SHD) and we can observe the coupling with HSIC at different complexity regimes.

### Improve Teoplitz
The symmetric part should represent the total probability of finding an edge. The antisymmetric component informs on the direction of the information flow. The direction should be bound to the total probability of the edge, else the model can represent the same situation with a large anti-symmetry and a low symmetry or a low anti-symmetry and a high symmetry.

### Alignment evaluation
Currently we don't have an evaluation function that shows the learned alignment, i.e. the product of key and queries. This was motivated by the fact that this quantity can be proxied by the attention score, which is evaluated, nevertheless, different attention design use the alignment differently.


### Make a universal config template and add default value

### Add ETA in the manifest ✅

### Fix eval_emb, not working for noise_aware (or remove from sweep, are they important?)

### Run all experiments
All experiments need to be re-run due to the seed bug in the trainer which led to wrong cross-validation consistency. We can leverage some knowledge gained so far to make sure the new experiments are "fair", i.e. the control variable is the main source of performance improvement/degradation. To achieve this, all experiments should have the same macro settings.
    - All regularization OFF
    - Tau annealing to a reasonable value: the model should not collapse to too confident/sharp edge probability in every case, only when the fit is sincerely confident.
    - Model dimension: so far we saw that, no matter how big is the model, attention is needed. Nonetheless, HSIC and model dimension are related, as for a more rigid model, HSIC becomes a better proxy for correct causal learning.
    - Learning rate: define a rate which is safe: stable training throughout all configurations 

Define tests, check name convention and run all


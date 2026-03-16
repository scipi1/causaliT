# Paper Outline

- Motivate the need of a surrogate that is intervention invariance
- Intervention invariance is guaranteed if the true DAG is given to the attention
- Learning the DAG through attention is desired

- Standard attention has problems
- Architectural changes designed for learning the causal structure
- Show that we are more intervention invariant than vanilla transformer
- Show the learned causal structure


# TODOs in priority order



- Toeplitz L1-regularization at the symmetric part. It promotes sparse representations.

- Add the following regularization **annealing**
    - Toeplitz/Lie tau, defines how steep is the tanh, i.e. how decisive the edge should be. It naturally promotes binary edges towards the training end.
    - HSIC: start with a very high regularization, to avoid learning very wrong DAGs and anneal it to zero

### Cross-validation inconsistency

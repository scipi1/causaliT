"""
Support modules for the ``eval_funs`` evaluation entry points.

Nothing here is an evaluation in its own right; these are the shared
building blocks the entry points (``eval_attention``, ``eval_interventions``,
``eval_seed_sweep``) are assembled from:

    eval_utils:      eval directories, dataset metadata, true-DAG masks,
                     soft Hamming / SHD / DAG-confidence / MEC primitives
    eval_lib:        checkpoint and config discovery
    eval_dag_query:  model-free, shape-based extraction of the canonical
                     DAG blocks (``cross`` = S->X, ``self`` = X->X)
    eval_dag_scores: the shared metric core built on the above

This module deliberately re-exports nothing: import the submodule you need
explicitly, e.g. ``from .helpers.eval_utils import _load_true_dag_mask``.
That keeps the import graph readable and avoids pulling the whole support
layer in just to reach one helper.
"""

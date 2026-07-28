"""
[DEPRECATED] AttentionSelectorLayer DAG evaluation.

This module used to contain a *parallel* implementation of DAG-recovery
evaluation for ``AttentionSelectorForecaster``: it re-implemented checkpoint
discovery, ran its own ``ds_test.npz`` forward loop, and assumed the model's
combined attention block always had width ``L_S + L_X``.  That assumption broke
once the layer gained an optional dedicated self-attention module, and the two
code paths drifted apart (different metrics, different logging, duplicated MEC
code).

Both paths are now unified in :func:`eval_attention.eval_attention_scores`:

    checkpoint -> predictor -> eval_dag_query.query_dag_blocks -> metrics

``eval_dag_query.query_dag_blocks`` classifies each attention tensor by its
shape and assembles the canonical blocks (``cross`` = S->X, ``self`` = X->X),
handling the case where both, or only one of the two, are present.
``eval_dag_scores.compute_dag_metrics`` then computes identical metrics for
every architecture.

This module is kept only so that existing call sites and scripts keep working.
"""

import warnings


def eval_attention_selector_scores(experiment: str, show_plots: bool = False) -> dict:
    """
    [DEPRECATED] Use :func:`eval_attention.eval_attention_scores` instead.

    Forwards to the unified evaluator, which identifies the attention layout
    from the tensor shapes (combined cross-attention, cross + self-attention, or
    cross only) and writes the same ``dag_metrics.json`` /
    ``learned_dag_edges.json`` artefacts.

    Args:
        experiment: Path to the experiment folder (contains ``k_*`` subdirs).
        show_plots: Ignored; the unified evaluator produces tables only.

    Returns:
        dag_metrics dict, as returned by ``eval_attention_scores``.
    """
    warnings.warn(
        "eval_attention_selector_scores() is deprecated: eval_attention_scores() "
        "now handles AttentionSelectorLayer (with or without self-attention) "
        "through the shared DAG query helper.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Local import to avoid a circular import at module load time.
    from ..eval_attention import eval_attention_scores

    return eval_attention_scores(experiment, show_plots=False)

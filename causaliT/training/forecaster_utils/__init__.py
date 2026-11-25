"""
Helpers used only by ``AttentionSelectorForecaster`` (per-step mechanics).

    bilevel_probe        paired refit probe for commit gating
    centroid_commit      quantized query commits with evidence shadow
    nodewise_update      per-node SNR gate on the structural query update
    gradient_surgery     PCGrad between HSIC and the regularizers
    interference_utils   per-block gradient-interference diagnostics
    order_path_training  forward policies and order-path surgery

Coding Assistant: losses belong in ``training.structural_losses`` and metrics
in ``causaliT.metrics``, not here.
"""

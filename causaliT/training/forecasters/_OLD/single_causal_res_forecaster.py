"""SingleCausalResForecaster: Lightning wrapper for SingleCausalLayerRes.

Identical to ``SingleCausalForecaster`` (same loss, regularizers, gradient
routing, schedulers) except the inner model is the dual-residual variant
``SingleCausalLayerRes`` (see
``causaliT/core/architectures/single_causal_res/``).

The two forecasters can share configs verbatim — only the
``model.model_object`` field selects between them
(``SingleCausalLayer`` vs ``SingleCausalLayerRes``).

Why a separate class instead of a flag on the parent?
    Keeping the dual-residual model behind a dedicated forecaster makes
    A/B comparisons unambiguous in checkpoints, logs, and the trainer
    registry, and avoids a nested ``isinstance`` check inside the parent's
    long ``__init__``.
"""

from causaliT.training.forecasters.single_causal_forecaster import (
    SingleCausalForecaster,
)
from causaliT.core.architectures.single_causal_res import SingleCausalLayerRes


class SingleCausalResForecaster(SingleCausalForecaster):
    """Lightning wrapper for ``SingleCausalLayerRes`` (SVFA dual-residual).

    Behaviour, regularizers, and configuration schema are identical to
    ``SingleCausalForecaster``. Only the inner model class differs.
    """

    def __init__(self, config, data_dir: str = None):
        # Trick the parent's __init__ into instantiating the dual-residual
        # model: temporarily monkey-patch the symbol the parent looks up.
        # This is safer than copy-pasting the parent's >1000-line __init__.
        import causaliT.training.forecasters.single_causal_forecaster as _scf

        original_cls = _scf.SingleCausalLayer
        _scf.SingleCausalLayer = SingleCausalLayerRes
        try:
            super().__init__(config, data_dir=data_dir)
        finally:
            _scf.SingleCausalLayer = original_cls

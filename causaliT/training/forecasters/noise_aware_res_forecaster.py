"""NoiseAwareCausalResForecaster: Lightning wrapper for NoiseAwareSingleCausalLayerRes.

Identical to ``NoiseAwareCausalForecaster`` (same Gaussian NLL loss,
regularizers, noise-prior, gradient routing, schedulers) except the
inner model is the dual-residual variant ``NoiseAwareSingleCausalLayerRes``
(see ``causaliT/core/architectures/noise_aware_res/``).

The two forecasters can share configs verbatim — only the
``model.model_object`` field selects between them
(``NoiseAwareSingleCausalLayer`` vs ``NoiseAwareSingleCausalLayerRes``).

Why a separate class instead of a flag on the parent?
    Keeping the dual-residual model behind a dedicated forecaster makes
    A/B comparisons unambiguous in checkpoints, logs, and the trainer
    registry, and avoids nested ``isinstance`` checks inside the parent's
    long ``__init__``.
"""

from causaliT.training.forecasters.noise_aware_forecaster import (
    NoiseAwareCausalForecaster,
)
from causaliT.core.architectures.noise_aware_res import NoiseAwareSingleCausalLayerRes


class NoiseAwareCausalResForecaster(NoiseAwareCausalForecaster):
    """Lightning wrapper for ``NoiseAwareSingleCausalLayerRes`` (noise-aware SVFA dual-residual).

    Behaviour, regularizers, and configuration schema are identical to
    ``NoiseAwareCausalForecaster``. Only the inner model class differs:
    cross- and self-attention both update X_struct via a residual connection
    in addition to the usual value-stream update, while ambient noise
    injection on the value path is preserved unchanged.
    """

    def __init__(self, config: dict, data_dir: str = None):
        # Trick the parent's __init__ into instantiating the dual-residual
        # model: temporarily monkey-patch the symbol the parent looks up.
        import causaliT.training.forecasters.noise_aware_forecaster as _naf

        original_cls = _naf.NoiseAwareSingleCausalLayer
        _naf.NoiseAwareSingleCausalLayer = NoiseAwareSingleCausalLayerRes
        try:
            super().__init__(config, data_dir=data_dir)
        finally:
            _naf.NoiseAwareSingleCausalLayer = original_cls

"""Predictor for ``NoiseAwareCausalResForecaster`` (noise-aware SVFA dual-residual).

Mirrors :class:`NoiseAwareCausalPredictor` but loads checkpoints into the
dual-residual forecaster.  Inheriting from ``NoiseAwareCausalPredictor`` and
overriding only ``_load_model`` keeps the post-processing (mu/log_var/std,
attention extraction, hard-mask handling, predict loop, …) in one place.
"""

from .noise_aware_predictor import NoiseAwareCausalPredictor
from causaliT.training.forecasters.noise_aware_res_forecaster import (
    NoiseAwareCausalResForecaster,
)


class NoiseAwareCausalResPredictor(NoiseAwareCausalPredictor):
    """Predictor for ``NoiseAwareSingleCausalLayerRes`` checkpoints.

    Identical behaviour to :class:`NoiseAwareCausalPredictor`; only the
    forecaster class used for ``load_from_checkpoint`` differs.
    """

    def _load_model(self) -> NoiseAwareCausalResForecaster:
        model = NoiseAwareCausalResForecaster.load_from_checkpoint(
            self.checkpoint_path
        )

        if model is None:
            raise RuntimeError("Model failed to load from checkpoint.")

        if not any(param.requires_grad for param in model.parameters()):
            raise RuntimeError(
                "Model parameters seem uninitialized. Check the checkpoint path."
            )

        if model.use_hard_masks and not model._hard_masks_loaded:
            self._load_hard_masks_for_model(model)

        return model

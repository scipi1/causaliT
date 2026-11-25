"""Predictor for ``SingleCausalResForecaster`` (SVFA dual-residual variant).

Mirrors :class:`SingleCausalPredictor` but loads checkpoints into the
dual-residual forecaster.  Inheriting from ``SingleCausalPredictor`` and
overriding only ``_load_model`` keeps the post-processing (attention
extraction, hard-mask handling, intervention helpers, …) in one place.
"""

from .single_causal_predictor import SingleCausalPredictor
from causaliT.training.forecasters.single_causal_res_forecaster import (
    SingleCausalResForecaster,
)


class SingleCausalResPredictor(SingleCausalPredictor):
    """Predictor for ``SingleCausalLayerRes`` checkpoints.

    Identical behaviour to :class:`SingleCausalPredictor`; only the
    forecaster class used for ``load_from_checkpoint`` differs.
    """

    def _load_model(self) -> SingleCausalResForecaster:
        model = SingleCausalResForecaster.load_from_checkpoint(self.checkpoint_path)

        if model is None:
            raise RuntimeError("Model failed to load from checkpoint.")

        if not any(param.requires_grad for param in model.parameters()):
            raise RuntimeError(
                "Model parameters seem uninitialized. Check the checkpoint path."
            )

        if model.use_hard_masks and not model._hard_masks_loaded:
            self._load_hard_masks_for_model(model)

        return model

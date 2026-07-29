"""
Predictor for the AttentionSelectorLayer architecture.

``AttentionSelectorForecaster`` was previously not reachable through
``create_predictor()``, which forced every evaluation function that needed it
(DAG recovery, ATE, ...) to re-implement checkpoint discovery and a raw
``ds_test.npz`` forward loop.  This predictor plugs the architecture into the
standard evaluation workflow:

    checkpoint  ->  predictor  ->  attention/DAG query  ->  metrics

Data format and prediction loop are identical to ``SingleCausalPredictor``
(three-input ``(S, X, Y)`` npz, ``Y`` ignored, blanking of ``X`` handled inside
the forecaster), so that class is reused as the base and only the
architecture-specific hooks are overridden.
"""

from typing import Any, Dict

from .single_causal_predictor import SingleCausalPredictor
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)


class AttentionSelectorPredictor(SingleCausalPredictor):
    """
    Predictor for ``AttentionSelectorForecaster`` (``AttentionSelectorLayer``).

    Handles:
    - Single combined cross-attention block: ``Q = X_blanked``,
      ``K/V = [S_actual, X_actual]`` -> attention ``(B, L_X, L_S + L_X)``.
    - Split mode (``self_attention_type`` set / ``split_xx=True``): the model
      itself re-concatenates the S->X and X->X posteriors into the same
      ``(B, L_X, L_S + L_X)`` layout, so a single extraction path covers both.
    - Automatic reload of the combined GT oracle mask when the model was
      trained with ``use_hard_masks=True``.

    The returned ``attention_weights`` dict contains the single key
    ``"att_combined"``.  Splitting into the canonical ``cross`` (S->X) and
    ``self`` (X->X) DAG blocks is *not* done here: it is the job of
    ``eval_funs.helpers.eval_dag_query.query_dag_blocks()``, which inspects which
    attention modules actually exist on the model.  Keeping one tensor here
    avoids storing the same values three times for the whole test split.
    """

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> AttentionSelectorForecaster:
        """
        Load ``AttentionSelectorForecaster`` from checkpoint.

        ``data_dir`` is not persisted in the checkpoint hyperparameters, so a
        model trained with hard masks comes back with
        ``use_hard_masks=True`` but ``_hard_masks_loaded=False``.  In that case
        the combined oracle mask is rebuilt from the data directory so the
        inference forward pass replays exactly the training-time masking
        (including the wrong-DAG corruption, which is derived deterministically
        from the config seed inside ``_load_combined_oracle_mask``).
        """
        model = AttentionSelectorForecaster.load_from_checkpoint(
            self.checkpoint_path,
            map_location="cpu",
        )

        if model is None:
            raise RuntimeError("Model failed to load from checkpoint.")

        if getattr(model, "use_hard_masks", False) and not model._hard_masks_loaded:
            if self.datadir_path is None:
                print(
                    "Warning: model was trained with use_hard_masks=True but no "
                    "datadir_path was provided - oracle mask not reloaded."
                )
            else:
                model._load_combined_oracle_mask(self.config, str(self.datadir_path))

        return model

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, S, X, Y=None, disable_hard_masks: bool = False, **kwargs) -> Any:
        """
        Forward pass through ``AttentionSelectorForecaster``.

        Args:
            S: Source tensor ``(B, L_S, F)``.
            X: Intermediate tensor ``(B, L_X, F)`` with actual values; the value
               column is blanked internally for the query path.
            Y: Ignored (kept for signature compatibility).
            disable_hard_masks: Accepted for interface compatibility with
               ``SingleCausalPredictor.predict()``.  When True the oracle mask
               buffer is bypassed by temporarily clearing the loaded flag.

        Returns:
            Tuple ``(pred_x, attention_weights, aux)``.
        """
        if disable_hard_masks and getattr(self.model, "_hard_masks_loaded", False):
            # setattr (not direct assignment) keeps static type checkers happy:
            # nn.Module.__setattr__ is typed for Tensor/Module values only.
            setattr(self.model, "_hard_masks_loaded", False)
            try:
                return self.model.forward(data_source=S, data_intermediate=X)
            finally:
                setattr(self.model, "_hard_masks_loaded", True)

        return self.model.forward(data_source=S, data_intermediate=X)

    # ------------------------------------------------------------------
    # Output processing
    # ------------------------------------------------------------------

    def _process_forward_output(self, output: Any) -> Dict[str, Any]:
        """
        Process ``AttentionSelectorForecaster`` output.

        Args:
            output: ``(pred_x, attention_weights, aux)`` where
                ``attention_weights`` is the combined ``(B, L_X, L_S + L_X)``
                posterior (or ``(B, H, L_X, L_S + L_X)`` when
                ``shared_dag_across_heads=False``).  With
                ``homogeneous_nodes=True`` it is the square ``(B, N, N)``
                posterior over all ``N = L_S + L_X`` nodes — ``split_combined_
                attention`` in ``eval_dag_query`` classifies that shape, so the
                attention dict is forwarded unchanged.

        Returns:
            Dict with ``pred_x`` and ``attention_weights={"att_combined": ...}``.
            In homogeneous mode ``pred_x`` is restricted to the X rows so it
            still aligns with the X ground truth used by every downstream
            reconstruction metric, and the S rows are additionally exposed as
            ``pred_s`` for the (new) S-reconstruction diagnostics.
        """
        pred_x = output[0]
        attention_weights = output[1] if len(output) > 1 else None

        result: Dict[str, Any] = {}

        # Homogeneous mode: the model reconstructs ALL N nodes, so rows
        # 0..L_S-1 are the S variables and rows L_S..N-1 are the X variables.
        if getattr(self.model, "homogeneous_nodes", False) and pred_x is not None:
            L_S = int(getattr(self.model, "S_seq_len", 0))
            if L_S > 0 and pred_x.shape[1] == int(getattr(self.model, "N", -1)):
                result["pred_s"] = pred_x[:, :L_S, ...]
                pred_x = pred_x[:, L_S:, ...]

        result["pred_x"] = pred_x
        result["attention_weights"] = {"att_combined": attention_weights}
        return result



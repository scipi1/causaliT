"""Shared helpers for the staged / adaptive trainers.

Trainer-agnostic building blocks used by
``causaliT.training.adaptive_trainer``:

* ``_to_plain_container``      - OmegaConf -> plain python containers (resolved);
* ``_compute_score_margin``    - DAG score-margin diagnostics at a phase switch;
* ``_partition_train_indices`` - honest (cross-fit) recon/struct split;
* ``_json_default``            - JSON fallback for numpy / torch scalars.

They used to live in ``anm_staged_trainer`` (the rigid ANM stage schedule,
removed in favour of the metric-driven ``adaptive_trainer``).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def _to_plain_container(obj: Any) -> Any:
    """
    Convert OmegaConf containers to resolved plain Python dict/list objects.

    ANM stage specs often come from YAML as ``DictConfig`` / ``ListConfig``.
    Those are mapping-like but do not satisfy ``isinstance(x, dict)``, which
    previously caused nested keys such as ``evaluation.functions`` to be
    silently ignored.  Resolving here also materialises interpolations such as
    ``batch_key_dropout_p: ${experiment.batch_key_dropout}``.
    """
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _compute_score_margin(
    pl_module: pl.LightningModule,
    config: dict,
    data_dir: str,
) -> Dict[str, Optional[float]]:
    """
    Compute ``score(true edges) - score(false edges)`` for cross and self.

    Uses the batch-averaged score tensor stored on the inner attention module
    after the last forward pass.  Returns ``None`` for each attention type
    when the true DAG mask or the score tensor is unavailable.

    This is the primary diagnostic for H2 (does alternating improve edge
    separation?) and H3 (does the curriculum improve margin over stages?).

    Returns:
        Dict with keys ``"cross"`` and ``"self"``, each a float or None.
    """
    result: Dict[str, Optional[float]] = {"cross": None, "self": None}
    try:
        inner_model = getattr(pl_module, "model", pl_module)
        dataset_name = config.get("data", {}).get("dataset", "")
        if not dataset_name or not data_dir:
            return result

        from causaliT.evaluation.eval_funs.helpers.eval_utils import _load_true_dag_mask

        # ------------------------------------------------------------------
        # AttentionSelectorLayer branch
        # Single combined cross-attention block; no ``decoder`` attribute.
        # The attention module lives at ``inner_model.attention`` and its
        # inner_attention exposes a combined (L_X, L_S+L_X) score tensor.
        # We split it into the S→X (cross) and X→X (self) sub-matrices and
        # compute separate margins against the respective GT DAG masks.
        # ------------------------------------------------------------------
        if hasattr(inner_model, "attention") and not hasattr(inner_model, "decoder"):
            combined_inner = inner_model.attention.inner_attention
            # Accept both public and private attribute names for robustness.
            combined_score_t = getattr(combined_inner, "score_tensor_for_sparsity", None)
            if combined_score_t is None:
                combined_score_t = getattr(
                    combined_inner, "_score_tensor_for_sparsity", None
                )
            if combined_score_t is not None:
                scores_np = combined_score_t.detach().cpu().numpy()
                S_seq_len = getattr(inner_model, "S_seq_len", None)
                if S_seq_len is not None and scores_np.ndim == 2:
                    # ------------------------------------------------------
                    # Shape-aware row selection.
                    #   * split mode        -> (L_X, L_S+L_X): rows are already
                    #     the X children, slice the columns directly.
                    #   * homogeneous_nodes -> (N, N) with N = L_S+L_X: EVERY
                    #     node is a child, so first keep the X child rows
                    #     ``[S_seq_len:, :]`` and only then split the columns,
                    #     which restores the (L_X, L_S+L_X) layout below.
                    # ------------------------------------------------------
                    X_seq_len = getattr(inner_model, "X_seq_len", None)
                    if (
                        X_seq_len is not None
                        and scores_np.shape[0] == scores_np.shape[1] == S_seq_len + X_seq_len
                    ):
                        scores_np = scores_np[S_seq_len:, :]      # (L_X, N)

                    cross_scores_np = scores_np[:, :S_seq_len]    # (L_X, L_S)
                    self_scores_np  = scores_np[:, S_seq_len:]    # (L_X, L_X)


                    # --- S→X (cross) margin ---
                    true_cross = _load_true_dag_mask(
                        data_dir, dataset_name, "dec_cross"
                    )
                    if (
                        true_cross is not None
                        and cross_scores_np.shape == true_cross.shape
                    ):
                        true_mask  = true_cross.astype(bool)
                        false_mask = ~true_mask
                        if true_mask.any() and false_mask.any():
                            result["cross"] = float(
                                cross_scores_np[true_mask].mean()
                                - cross_scores_np[false_mask].mean()
                            )

                    # --- X→X (self) margin ---
                    true_self = _load_true_dag_mask(
                        data_dir, dataset_name, "dec_self"
                    )
                    if (
                        true_self is not None
                        and self_scores_np.shape == true_self.shape
                    ):
                        true_mask  = true_self.astype(bool)
                        false_mask = ~true_mask
                        np.fill_diagonal(false_mask, False)  # no self-loops
                        if true_mask.any() and false_mask.any():
                            result["self"] = float(
                                self_scores_np[true_mask].mean()
                                - self_scores_np[false_mask].mean()
                            )
            return result

        # ------------------------------------------------------------------
        # SingleCausalLayer (and related) branch
        # Separate cross- and self-attention blocks inside ``decoder.layers``.
        # ------------------------------------------------------------------
        if not hasattr(inner_model, "decoder"):
            return result

        layer = inner_model.decoder.layers[0]

        # --- Cross-attention score margin ---
        true_cross = _load_true_dag_mask(data_dir, dataset_name, "dec_cross")
        cross_inner = layer.global_cross_attention.inner_attention
        cross_score_t = getattr(cross_inner, "_score_tensor_for_sparsity", None)
        if cross_score_t is not None and true_cross is not None:
            scores_np = cross_score_t.detach().cpu().numpy()
            if scores_np.shape == true_cross.shape:
                true_mask = true_cross.astype(bool)
                false_mask = ~true_mask
                if true_mask.any() and false_mask.any():
                    result["cross"] = float(
                        scores_np[true_mask].mean() - scores_np[false_mask].mean()
                    )

        # --- Self-attention score margin ---
        true_self = _load_true_dag_mask(data_dir, dataset_name, "dec_self")
        self_inner = layer.global_self_attention.inner_attention
        # ToeplitzAttention exposes P_edge_for_reg; others use _score_tensor_for_sparsity
        self_score_t = getattr(self_inner, "P_edge_for_reg", None)
        if self_score_t is None:
            self_score_t = getattr(self_inner, "_score_tensor_for_sparsity", None)
        if self_score_t is not None and true_self is not None:
            scores_np = self_score_t.detach().cpu().numpy()
            if scores_np.ndim == 2 and scores_np.shape == true_self.shape:
                true_mask = true_self.astype(bool)
                false_mask = ~true_mask
                np.fill_diagonal(false_mask, False)   # exclude diagonal (no self-loops)
                if true_mask.any() and false_mask.any():
                    result["self"] = float(
                        scores_np[true_mask].mean() - scores_np[false_mask].mean()
                    )

    except Exception as exc:
        logger.debug(f"_compute_score_margin failed (non-critical): {exc}")

    return result


def _partition_train_indices(
    train_local_idx: np.ndarray,
    ratio: float,
    seed: int,
) -> tuple:
    """
    Split ``train_local_idx`` into two disjoint subsets (recon, struct).

    Deterministic given ``seed`` so reruns are reproducible.  ``ratio`` is the
    fraction of samples assigned to the reconstruction subset; the remainder go
    to the structure subset.  Both subsets are guaranteed non-empty.

    This implements the DML / DARTS-style honest cross-fit: the reconstruction
    regressor is fit on one subset and residual-based structural signals (HSIC)
    are computed on the other, so residuals used for structure learning are
    out-of-sample w.r.t. the reconstruction fit.

    Args:
        train_local_idx: 1-D array of local training indices to partition.
        ratio:           Fraction assigned to the recon subset.  Values in the
                         open interval (0, 1) produce a genuine disjoint split;
                         values <= 0 or >= 1 deactivate cross-fitting (both
                         subsets = the full training set).
        seed:            RNG seed for the deterministic permutation.

    Returns:
        (recon_idx, struct_idx): two disjoint sub-arrays of ``train_local_idx``.
    """
    train_local_idx = np.asarray(train_local_idx)

    # --- Cross-fit deactivation: ratio outside (0, 1) -> use full set for both ---
    # A ratio of exactly 0 or 1 (or out-of-range) is not a valid disjoint split;
    # treat it as an explicit "turn cross-fitting off" signal and return the full
    # training set for both subsets (identical to every stage using
    # ``data_split: full``).
    if not (0.0 < ratio < 1.0):
        return train_local_idx, train_local_idx

    n = len(train_local_idx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(ratio * n))
    # Keep both subsets non-empty
    cut = max(1, min(n - 1, cut))
    recon_idx = train_local_idx[perm[:cut]]
    struct_idx = train_local_idx[perm[cut:]]
    return recon_idx, struct_idx


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return str(obj)

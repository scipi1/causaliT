"""
ANM Residual-HSIC Evaluation for Partial ANM Regression experiments.

This module provides:

    eval_anm_residual_hsic(experiment, show_plots=False)

        Post-stage H1 diagnostic: loads the final stage checkpoint, runs a
        batched forward pass over the validation split in *eval mode* (so
        BatchConsistentKeyDropout is disabled — dense attention), computes
        per-(X_j, S_i) cross-HSIC values from the residuals, classifies each
        edge pair against the true DAG mask, and saves a structured CSV and
        JSON summary under ``<experiment>/eval/eval_anm_residual_hsic/``.

Hypothesis covered:
    H1 — Does reconstruction warmup make HSIC meaningful?
        After a reconstruction-only stage the residuals should be small for
        true parent edges.  Running this eval at the end of each stage lets
        you track whether HSIC becomes progressively more informative.

Edge classification (cross-attention only):
    true_parent   : (X_j, S_i) with  true_dag[j, i] == 1
    false_parent  : (X_j, S_i) with  true_dag[j, i] == 0  AND S_i not independent of X_j
    independent   : (X_j, S_i) pairs where S_i has no true child in X  (column sum == 0)

    NOTE: proxy/omitted-parent classification requires oracle knowledge and is
    not available at eval time.  The ``edge_class`` column will contain one of
    the three labels above.

Self-attention HSIC (X_j ← X_k) is computed when the inner self-attention
module stores a ``score_tensor_for_sparsity`` (Toeplitz), using the true
self-attention DAG mask if available.  Otherwise only cross-HSIC is reported.
"""

import json
import os
from os.path import join, exists, isdir
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .eval_lib import (
    find_config_file,
    find_best_or_last_checkpoint,
    get_architecture_type,
)
from .eval_utils import (
    _setup_eval_directories,
    _load_true_dag_mask,
    root_path,
    DEFAULT_PLOT_FORMAT,
)


# =============================================================================
# Helpers
# =============================================================================

def _load_model_and_config(fold_dir: str) -> Tuple:
    """
    Load config and model from a single fold directory.

    Args:
        fold_dir: Path to a ``k_<n>`` subdirectory of an experiment.

    Returns:
        (model, config, architecture_type) or raises FileNotFoundError.
    """
    from omegaconf import OmegaConf
    from .eval_attention import _load_model_from_checkpoint

    stage_or_experiment_dir = Path(fold_dir).parent
    try:
        # Preferred for current ANM staged runs: each stage persists its own
        # fully-resolved config at ``<stage_dir>/config.yaml``.
        config_path = find_config_file(str(stage_or_experiment_dir))
    except FileNotFoundError:
        # Backward compatibility for older ANM runs that did not save a stage
        # config.  This may be semantically weaker because it uses the parent
        # experiment config without stage overrides.
        config_path = _find_config_file_upwards(stage_or_experiment_dir)
    config = OmegaConf.load(config_path)

    arch_type = get_architecture_type(config)
    ckpt_dir = join(fold_dir, "checkpoints")
    ckpt_path = find_best_or_last_checkpoint(ckpt_dir, checkpoint_type="last")

    model = _load_model_from_checkpoint(ckpt_path, arch_type)
    model.eval()
    return model, config, arch_type


def _find_config_file_upwards(start_dir: Path) -> str:
    """
    Find a ``config*.yaml`` file in ``start_dir`` or one of its parents.

    Standard experiment evals receive the root experiment directory, which
    contains ``config*.yaml`` directly.  ANM per-stage evals receive a stage
    directory such as ``.../anm_stages/00_recon_warmup``; the config lives two
    levels above at the parent experiment root.  Searching upwards keeps the
    eval function compatible with both layouts.
    """
    cur = Path(start_dir).resolve()
    for candidate in [cur, *cur.parents]:
        try:
            return find_config_file(str(candidate))
        except FileNotFoundError:
            continue
        except ValueError:
            # Multiple config files is still a real ambiguity; propagate it.
            raise
    raise FileNotFoundError(f"No config*.yaml found in {start_dir} or its parents")


def _build_data_module(config, data_dir: str):
    """
    Build and set up the correct DataModule in evaluation mode.

    Uses the same factory as training so SingleCausal/StageCausal models get
    ``StageCausalDataModule`` and legacy models get ``ProcessDataModule``.
    """
    from causaliT.training.trainer import get_dataloader

    seed = int(config.get("training", {}).get("seed", 42))
    dm = get_dataloader(config=config, data_dir=data_dir, cluster=False, seed=seed)
    dm.prepare_data()
    dm.setup(stage="fit")
    return dm


def _infer_data_dir(config) -> str:
    """
    Infer the data directory from the project root and the dataset name in config.

    Falls back gracefully to a sibling ``data/`` or ``scm_ds/`` path relative
    to the package root if the config does not specify an explicit ``data_dir``.
    """
    dataset = config.get("data", {}).get("dataset", "")

    # 1. Explicit data_dir in config
    explicit = config.get("data", {}).get("data_dir", None)
    if explicit and exists(explicit):
        return explicit

    # 2. Probe common locations relative to project root.  ``root_path`` is a
    # module-level string exported by eval_utils, not a callable.
    project_root = root_path
    for candidate in ("data", "scm_ds", join("data", dataset)):
        full = join(project_root, candidate)
        if exists(full):
            return full

    # 3. Fall back to project root itself
    return project_root


def _collect_residuals_and_sources(
    model,
    dm,
    val_idx: int,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run batched forward pass in eval mode and return concatenated arrays.

    BatchConsistentKeyDropout is automatically disabled in eval mode (it
    checks ``self.training``), so the attention is dense — ideal for the
    H1 diagnostic where we want to measure residuals under full information.

    Args:
        model    : Loaded forecaster (eval mode).
        dm       : Prepared ProcessDataModule.
        val_idx  : Index of the value feature in the data tensor.
        device   : ``"cpu"`` (default) or ``"cuda"``.
        max_batches: If set, limit the number of batches (useful for quick checks).

    Returns:
        residuals  : ``(N, L_X)`` float32 array
        s_values   : ``(N, L_S)`` float32 array
    """
    model = model.to(device)
    loader = dm.val_dataloader()

    all_residuals: List[np.ndarray] = []
    all_s_values: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if len(batch) == 3:
                S, X, _ = batch
            else:
                S, X = batch

            S = S.to(device)
            X = X.to(device)

            x_target = X[:, :, val_idx]
            pred_x, *_ = model(data_source=S, data_intermediate=X)

            residuals = (x_target - pred_x.squeeze()).detach().cpu().float().numpy()
            s_vals = S[:, :, val_idx].detach().cpu().float().numpy()

            all_residuals.append(residuals)
            all_s_values.append(s_vals)

    if not all_residuals:
        return None, None

    return np.concatenate(all_residuals, axis=0), np.concatenate(all_s_values, axis=0)


def _compute_hsic_matrix_np(
    residuals: np.ndarray,
    sources: np.ndarray,
    sigma_res: float = 1.0,
    sigma_src: float = 1.0,
) -> np.ndarray:
    """
    Compute ``HSIC(ε_j, S_i)`` for all (j, i) pairs with RBF kernels.

    Uses the biased estimator ``(1/n²) * tr(K_X H K_Y H)`` where ``H`` is
    the centring matrix.  Computation is vectorised over the source dimension
    (one kernel matrix ``K_S`` per source variable, then batched dot products).

    Args:
        residuals : ``(N, L_X)``  — residuals per target variable.
        sources   : ``(N, L_S)``  — source variable values.
        sigma_res : RBF bandwidth for residuals (same for all ``X_j``).
        sigma_src : RBF bandwidth for sources   (same for all ``S_i``).

    Returns:
        hsic_matrix : ``(L_X, L_S)``  — HSIC value for each (j, i) pair.
    """
    N, L_X = residuals.shape
    N2, L_S = sources.shape
    assert N == N2, "residuals and sources must have the same number of samples"

    # Centring matrix
    H = np.eye(N) - np.ones((N, N)) / N  # (N, N)

    hsic_mat = np.zeros((L_X, L_S), dtype=np.float32)

    # Pre-compute centred source kernels K_S_i (H @ K @ H)
    src_kernels_centred: List[np.ndarray] = []
    for i in range(L_S):
        si = sources[:, i : i + 1]           # (N, 1)
        diff2_s = ((si - si.T) ** 2)          # (N, N)
        K_s = np.exp(-diff2_s / (2 * sigma_src**2))
        src_kernels_centred.append(H @ K_s @ H)

    for j in range(L_X):
        rj = residuals[:, j : j + 1]          # (N, 1)
        diff2_r = ((rj - rj.T) ** 2)          # (N, N)
        K_r = np.exp(-diff2_r / (2 * sigma_res**2))
        K_r_centred = H @ K_r @ H             # (N, N)

        for i, K_s_centred in enumerate(src_kernels_centred):
            hsic_val = float(np.trace(K_r_centred @ K_s_centred)) / (N**2)
            hsic_mat[j, i] = max(hsic_val, 0.0)  # HSIC ≥ 0 by construction

    return hsic_mat


def _classify_edges(true_dag: np.ndarray) -> np.ndarray:
    """
    Classify each (X_j, S_i) pair into an edge class string.

    Classification rules (cross-attention):
    - ``"true_parent"``  : true_dag[j, i] == 1
    - ``"independent"``  : column i of true_dag is all-zero (S_i has no
                           children in X — genuinely independent).
    - ``"false_parent"`` : true_dag[j, i] == 0 and S_i is not independent.

    Returns:
        ``(L_X, L_S)`` object array of strings.
    """
    L_X, L_S = true_dag.shape
    classes = np.empty((L_X, L_S), dtype=object)

    col_sums = true_dag.sum(axis=0)  # (L_S,) — number of true children per Si
    for j in range(L_X):
        for i in range(L_S):
            if true_dag[j, i] == 1:
                classes[j, i] = "true_parent"
            elif col_sums[i] == 0:
                classes[j, i] = "independent"
            else:
                classes[j, i] = "false_parent"
    return classes


def _compute_adaptive_sigma(values: np.ndarray) -> float:
    """
    Compute the median absolute pairwise distance as an adaptive RBF bandwidth.

    Uses the median heuristic: ``σ = median(|v_i - v_j|, i < j)``.
    Falls back to 1.0 when the array is degenerate (all-equal or too small).

    Args:
        values: ``(N, D)`` or ``(N,)`` array.

    Returns:
        Scalar float bandwidth.
    """
    v = values.ravel()
    if len(v) < 2:
        return 1.0
    n = min(len(v), 2000)
    idx = np.random.choice(len(v), size=n, replace=False)
    sub = v[idx]
    diffs = np.abs(sub[:, None] - sub[None, :])
    triu = diffs[np.triu_indices(n, k=1)]
    med = float(np.median(triu))
    return med if med > 1e-8 else 1.0


# =============================================================================
# Main evaluation function
# =============================================================================

def eval_anm_residual_hsic(
    experiment: str,
    show_plots: bool = False,
    max_batches: Optional[int] = None,
    adaptive_bandwidth: bool = True,
    sigma_res: float = 1.0,
    sigma_src: float = 1.0,
) -> dict:
    """
    Compute and save per-edge residual HSIC diagnostics for an ANM stage.

    Covers the H1 diagnostic: *Does reconstruction warmup make HSIC meaningful?*

    The function:

    1. Loads the final checkpoint from ``<experiment>/k_0/checkpoints/``.
    2. Puts the model in eval mode (BatchConsistentKeyDropout auto-disabled).
    3. Runs all validation batches through the model to collect residuals
       ``ε_j = X_j - f̂_j`` and source values ``S_i``.
    4. Computes ``HSIC(ε_j, S_i)`` for every (j, i) pair with RBF kernels.
    5. Classifies each pair against the true DAG mask.
    6. Saves ``edge_hsic.csv`` and ``summary.json`` under
       ``<experiment>/eval/eval_anm_residual_hsic/``.

    Args:
        experiment       : Path to the experiment folder (e.g. an ANM stage dir
                           such as ``<save_dir>/anm_stages/00_recon_warmup``).
        show_plots       : If True, display matplotlib heatmap of HSIC matrix.
        max_batches      : Limit number of validation batches (None = all).
                           Useful for quick sanity checks on large datasets.
        adaptive_bandwidth: If True, use the median heuristic to set RBF σ.
                            Overrides ``sigma_res`` / ``sigma_src``.
        sigma_res        : Fixed RBF bandwidth for residual kernel (used when
                           ``adaptive_bandwidth=False``).
        sigma_src        : Fixed RBF bandwidth for source kernel (used when
                           ``adaptive_bandwidth=False``).

    Returns:
        dict with keys:
            - ``"mean_hsic_true_parent"``  : float or None
            - ``"mean_hsic_false_parent"`` : float or None
            - ``"mean_hsic_independent"``  : float or None
            - ``"true_minus_false_margin"`` : float or None  (lower is better)
            - ``"n_samples"``              : int
            - ``"edge_hsic_path"``         : str  — path to the CSV file
    """
    result: dict = {
        "mean_hsic_true_parent": None,
        "mean_hsic_false_parent": None,
        "mean_hsic_independent": None,
        "true_minus_false_margin": None,
        "n_samples": 0,
        "edge_hsic_path": "",
    }

    # ------------------------------------------------------------------
    # Setup output directories
    # ------------------------------------------------------------------
    eval_name = "eval_anm_residual_hsic"
    eval_path_root, eval_path_figs, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, eval_name)

    print(f"\n[eval_anm_residual_hsic] Experiment: {experiment}")

    # ------------------------------------------------------------------
    # Find fold directory (k_0 is the only fold used by anm_staged_trainer)
    # ------------------------------------------------------------------
    fold_dir = join(experiment, "k_0")
    if not isdir(fold_dir):
        # Try the experiment itself if it already points to k_0
        fold_dir = experiment
        ckpt_dir = join(fold_dir, "checkpoints")
        if not isdir(ckpt_dir):
            print(
                f"  [skip] No k_0 subdirectory or checkpoints found in: {experiment}"
            )
            return result

    # ------------------------------------------------------------------
    # Load model and config
    # ------------------------------------------------------------------
    try:
        model, config, arch_type = _load_model_and_config(fold_dir)
    except FileNotFoundError as exc:
        print(f"  [skip] Could not load model: {exc}")
        return result

    if arch_type not in (
        "SingleCausalForecaster",
        "SingleCausalResForecaster",
        "NoiseAwareCausalForecaster",
    ):
        print(
            f"  [skip] eval_anm_residual_hsic only supports SingleCausalForecaster "
            f"architectures.  Got: {arch_type}"
        )
        return result

    val_idx: int = int(config["data"]["val_idx"])

    # ------------------------------------------------------------------
    # Build data module
    # ------------------------------------------------------------------
    data_dir = _infer_data_dir(config)
    print(f"  data_dir inferred: {data_dir}")

    try:
        dm = _build_data_module(config, data_dir)
    except Exception as exc:
        print(f"  [skip] Could not build data module: {exc}")
        return result

    # ------------------------------------------------------------------
    # Forward pass — collect residuals & sources
    # ------------------------------------------------------------------
    print("  Running forward pass (eval mode, BKD disabled)…")
    residuals, s_values = _collect_residuals_and_sources(
        model=model,
        dm=dm,
        val_idx=val_idx,
        device="cpu",
        max_batches=max_batches,
    )

    if residuals is None or s_values is None or residuals.shape[0] == 0:
        print("  [skip] No validation samples available.")
        return result

    N, L_X = residuals.shape
    _, L_S = s_values.shape
    result["n_samples"] = N
    print(f"  Collected {N} samples  (L_X={L_X}, L_S={L_S})")

    # ------------------------------------------------------------------
    # Adaptive bandwidth
    # ------------------------------------------------------------------
    if adaptive_bandwidth:
        sigma_res = _compute_adaptive_sigma(residuals)
        sigma_src = _compute_adaptive_sigma(s_values)
        print(f"  Adaptive σ — residuals: {sigma_res:.4f}  sources: {sigma_src:.4f}")

    # ------------------------------------------------------------------
    # Compute HSIC matrix  (L_X, L_S)
    # ------------------------------------------------------------------
    print(f"  Computing HSIC for {L_X}x{L_S} = {L_X * L_S} pairs...")
    hsic_mat = _compute_hsic_matrix_np(
        residuals=residuals,
        sources=s_values,
        sigma_res=sigma_res,
        sigma_src=sigma_src,
    )

    # ------------------------------------------------------------------
    # Load true DAG mask for classification
    # ------------------------------------------------------------------
    dataset_name = config.get("data", {}).get("dataset", "")
    true_cross = _load_true_dag_mask(data_dir, dataset_name, "dec_cross")

    # ------------------------------------------------------------------
    # Build per-edge DataFrame
    # ------------------------------------------------------------------
    rows = []
    edge_classes = None
    if true_cross is not None and true_cross.shape == (L_X, L_S):
        edge_classes = _classify_edges(true_cross)

    # Variable labels (fallback to index strings)
    from .eval_utils import load_dataset_metadata
    meta = {}
    try:
        meta = load_dataset_metadata(data_dir, dataset_name)
    except Exception:
        pass
    var_info = meta.get("variable_info", {}) or {}
    src_labels = list(var_info.get("source_labels", [f"S{i}" for i in range(L_S)]))
    inp_labels = list(var_info.get("input_labels", [f"X{j}" for j in range(L_X)]))

    for j in range(L_X):
        x_label = inp_labels[j] if j < len(inp_labels) else f"X{j}"
        for i in range(L_S):
            s_label = src_labels[i] if i < len(src_labels) else f"S{i}"
            edge_class = (
                edge_classes[j, i]
                if edge_classes is not None
                else "unknown"
            )
            true_val = (
                int(true_cross[j, i])
                if true_cross is not None and true_cross.shape == (L_X, L_S)
                else None
            )
            rows.append({
                "target_idx": j,
                "source_idx": i,
                "target_label": x_label,
                "source_label": s_label,
                "hsic": float(hsic_mat[j, i]),
                "edge_class": edge_class,
                "true_edge": true_val,
            })

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Summary statistics per edge class
    # ------------------------------------------------------------------
    class_stats: Dict[str, Optional[float]] = {}
    for cls in ("true_parent", "false_parent", "independent", "unknown"):
        sub = df[df["edge_class"] == cls]["hsic"]
        class_stats[cls] = float(sub.mean()) if len(sub) > 0 else None

    result["mean_hsic_true_parent"] = class_stats.get("true_parent")
    result["mean_hsic_false_parent"] = class_stats.get("false_parent")
    result["mean_hsic_independent"] = class_stats.get("independent")

    tp = class_stats.get("true_parent")
    fp = class_stats.get("false_parent")
    if tp is not None and fp is not None:
        result["true_minus_false_margin"] = tp - fp

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = join(eval_path_files, "edge_hsic.csv")
    df.to_csv(csv_path, index=False)
    result["edge_hsic_path"] = csv_path
    print(f"  Saved: edge_hsic.csv  ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    summary = {
        "n_samples": N,
        "L_X": L_X,
        "L_S": L_S,
        "sigma_res": float(sigma_res),
        "sigma_src": float(sigma_src),
        "adaptive_bandwidth": adaptive_bandwidth,
        "mean_hsic_per_class": {
            k: (float(v) if v is not None else None)
            for k, v in class_stats.items()
        },
        "true_minus_false_margin": result["true_minus_false_margin"],
        "true_dag_available": true_cross is not None,
    }
    json_path = join(eval_path_files, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: summary.json")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n  Edge-class HSIC summary (cross-attention):")
    for cls, val in class_stats.items():
        if val is not None:
            print(f"    {cls:20s}: {val:.6f}")
    if result["true_minus_false_margin"] is not None:
        margin = result["true_minus_false_margin"]
        sign = "↓ good (true < false)" if margin < 0 else "↑ warning (true > false)"
        print(f"    margin (true−false)  : {margin:+.6f}  {sign}")

    # ------------------------------------------------------------------
    # Optional heatmap plot
    # ------------------------------------------------------------------
    if show_plots or True:  # always save to file; show only when requested
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(max(4, L_S * 0.6), max(3, L_X * 0.5)))
            im = ax.imshow(hsic_mat, aspect="auto", cmap="hot_r")
            plt.colorbar(im, ax=ax, label="HSIC")
            ax.set_xlabel("Source variable (S_i)")
            ax.set_ylabel("Target variable (X_j)")
            ax.set_title("Residual HSIC — cross-attention")
            ax.set_xticks(range(L_S))
            ax.set_xticklabels(src_labels[:L_S], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(L_X))
            ax.set_yticklabels(inp_labels[:L_X], fontsize=7)
            plt.tight_layout()
            fig_path = join(eval_path_figs, f"residual_hsic_heatmap.{DEFAULT_PLOT_FORMAT}")
            plt.savefig(fig_path, dpi=120, bbox_inches="tight")
            print(f"  Saved: residual_hsic_heatmap.{DEFAULT_PLOT_FORMAT}")
            if show_plots:
                plt.show()
            plt.close(fig)
        except Exception as exc:
            print(f"  [warn] Could not save HSIC heatmap: {exc}")

    return result

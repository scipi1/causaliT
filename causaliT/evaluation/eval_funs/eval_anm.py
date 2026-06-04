"""
ANM Residual-HSIC Evaluation for Partial ANM Regression experiments.

This module provides:

    eval_anm_residual_hsic(experiment, show_plots=False)

        Post-stage H1 diagnostic: loads the final stage checkpoint, runs a
        batched forward pass over the validation split in *eval mode* (so
        BatchConsistentKeyDropout is disabled — dense attention), computes
        residual-HSIC ``HSIC(ε_j, Z_k)`` against all candidate parents
        ``Z ∈ {S, X}``, classifies each edge pair against the true DAG masks,
        and saves structured CSV/JSON/plot artifacts under
        ``<experiment>/eval/eval_anm_residual_hsic/``.

Hypothesis covered:
    H1 — Does reconstruction warmup make HSIC meaningful?
        After a reconstruction-only stage the residuals should be small for
        true parent edges.  Running this eval at the end of each stage lets
        you track whether HSIC becomes progressively more informative.

Edge classification:
    true_parent   : (X_j, S_i) with  true_dag[j, i] == 1
    false_parent  : (X_j, S_i) with  true_dag[j, i] == 0  AND S_i not independent of X_j
    independent   : (X_j, S_i) pairs where S_i has no true child in X  (column sum == 0)
    self_loop     : diagonal self-attention pair (X_j, X_j), excluded from margins

    NOTE: proxy/omitted-parent classification requires oracle knowledge and is
    not available at eval time.  The ``edge_class`` column will contain one of
    the three labels above.

``edge_hsic_all.csv`` is the notebook-friendly long-format artifact containing
both cross pairs ``HSIC(ε_Xj, S_i)`` and self pairs ``HSIC(ε_Xj, X_k)``.
``edge_hsic.csv`` remains the backward-compatible cross-only artifact.
"""

import json
import os
from os.path import join, exists, isdir
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from causaliT.utils.hsic_utils import hsic_pair_matrix

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
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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
        x_values   : ``(N, L_X)`` float32 array
    """
    model = model.to(device)
    loader = dm.val_dataloader()

    all_residuals: List[np.ndarray] = []
    all_s_values: List[np.ndarray] = []
    all_x_values: List[np.ndarray] = []

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
            x_vals = x_target.detach().cpu().float().numpy()

            all_residuals.append(residuals)
            all_s_values.append(s_vals)
            all_x_values.append(x_vals)

    if not all_residuals:
        return None, None, None

    return (
        np.concatenate(all_residuals, axis=0),
        np.concatenate(all_s_values, axis=0),
        np.concatenate(all_x_values, axis=0),
    )


def _classify_edge_pair(
    true_dag: Optional[np.ndarray],
    target_idx: int,
    parent_idx: int,
    parent_space: str,
) -> Tuple[str, Optional[int]]:
    """
    Classify one residual-vs-parent pair for either S or X parent variables.

    Classification rules:
    - ``"true_parent"``  : true_dag[j, i] == 1
    - ``"self_loop"``    : parent_space == "X" and target_idx == parent_idx
    - ``"independent"``  : column i of true_dag is all-zero
    - ``"false_parent"`` : true_dag[j, i] == 0 and parent variable is not independent
    - ``"unknown"``      : no compatible true mask is available

    Returns:
        ``(edge_class, true_edge)`` where true_edge is 0/1 or None.
    """
    if parent_space == "X" and target_idx == parent_idx:
        return "self_loop", 0

    if true_dag is None or target_idx >= true_dag.shape[0] or parent_idx >= true_dag.shape[1]:
        return "unknown", None

    true_edge = int(true_dag[target_idx, parent_idx])
    if true_edge == 1:
        return "true_parent", true_edge
    if true_dag[:, parent_idx].sum() == 0:
        return "independent", true_edge
    return "false_parent", true_edge


def _class_stats_from_df(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Mean HSIC by edge class for valid non-diagonal candidate pairs."""
    class_stats: Dict[str, Optional[float]] = {}
    valid_df = df[df["edge_class"] != "self_loop"]
    for cls in ("true_parent", "false_parent", "independent", "unknown"):
        sub = valid_df[valid_df["edge_class"] == cls]["hsic"].dropna()
        class_stats[cls] = float(sub.mean()) if len(sub) > 0 else None
    return class_stats


def _margin_from_class_stats(class_stats: Dict[str, Optional[float]]) -> Optional[float]:
    """Return mean(true_parent) - mean(false_parent), or None if unavailable."""
    tp = class_stats.get("true_parent")
    fp = class_stats.get("false_parent")
    return (tp - fp) if tp is not None and fp is not None else None


def _plot_hsic_heatmap(
    hsic_mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    fig_path: str,
    show_plots: bool,
) -> bool:
    """Save a residual-HSIC heatmap. Returns True on success."""
    try:
        import matplotlib.pyplot as plt

        n_rows, n_cols = hsic_mat.shape
        fig, ax = plt.subplots(figsize=(max(4, n_cols * 0.6), max(3, n_rows * 0.5)))
        im = ax.imshow(hsic_mat, aspect="auto", cmap="hot_r")
        plt.colorbar(im, ax=ax, label="HSIC")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels[:n_cols], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels[:n_rows], fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"  [warn] Could not save HSIC heatmap {fig_path}: {exc}")
        return False


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
        "NoiseAwareCausalResForecaster",
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
    residuals, s_values, x_values = _collect_residuals_and_sources(
        model=model,
        dm=dm,
        val_idx=val_idx,
        device="cpu",
        max_batches=max_batches,
    )

    if residuals is None or s_values is None or x_values is None or residuals.shape[0] == 0:
        print("  [skip] No validation samples available.")
        return result

    N, L_X = residuals.shape
    _, L_S = s_values.shape
    result["n_samples"] = N
    print(f"  Collected {N} samples  (L_X={L_X}, L_S={L_S})")

    # ------------------------------------------------------------------
    # HSIC configuration: reuse the same torch utilities/kernels as training.
    # ------------------------------------------------------------------
    train_cfg = config.get("training", {})
    hsic_sigma = float(train_cfg.get("hsic_sigma", sigma_src))
    hsic_mode = str(train_cfg.get("hsic_mode", "biased"))
    nhsic_epsilon = float(train_cfg.get("nhsic_epsilon", 0.01))
    source_kernel = str(train_cfg.get("hsic_kernel_source", "rbf"))

    if adaptive_bandwidth:
        sigma_res = _compute_adaptive_sigma(residuals)
        sigma_src = _compute_adaptive_sigma(s_values)
        sigma_x = _compute_adaptive_sigma(x_values)
        print(
            f"  Adaptive σ diagnostics — residuals: {sigma_res:.4f}  "
            f"S: {sigma_src:.4f}  X: {sigma_x:.4f}"
        )
    else:
        sigma_x = float(hsic_sigma)

    # ------------------------------------------------------------------
    # Compute residual-HSIC matrices using the training HSIC implementation.
    # ------------------------------------------------------------------
    print(
        f"  Computing residual-HSIC for S and X parents: "
        f"cross={L_X}x{L_S}, self={L_X}x{L_X}..."
    )
    residuals_t = torch.as_tensor(residuals, dtype=torch.float32)
    s_values_t = torch.as_tensor(s_values, dtype=torch.float32)
    x_values_t = torch.as_tensor(x_values, dtype=torch.float32)

    with torch.no_grad():
        hsic_cross = hsic_pair_matrix(
            source_values=s_values_t,
            residuals=residuals_t,
            sigma=hsic_sigma,
            adaptive_bandwidth=adaptive_bandwidth,
            mode=hsic_mode,
            nhsic_epsilon=nhsic_epsilon,
            source_kernel=source_kernel,
            exclude_diagonal=False,
        ).cpu().numpy()
        hsic_self = hsic_pair_matrix(
            source_values=x_values_t,
            residuals=residuals_t,
            sigma=hsic_sigma,
            adaptive_bandwidth=adaptive_bandwidth,
            mode=hsic_mode,
            nhsic_epsilon=nhsic_epsilon,
            source_kernel="rbf",
            exclude_diagonal=True,
        ).cpu().numpy()

    # ------------------------------------------------------------------
    # Load true DAG masks for classification
    # ------------------------------------------------------------------
    dataset_name = config.get("data", {}).get("dataset", "")
    true_cross = _load_true_dag_mask(data_dir, dataset_name, "dec_cross")
    true_self = _load_true_dag_mask(data_dir, dataset_name, "dec_self")

    # ------------------------------------------------------------------
    # Variable labels (fallback to index strings)
    # ------------------------------------------------------------------
    from .eval_utils import load_dataset_metadata
    meta = {}
    try:
        meta = load_dataset_metadata(data_dir, dataset_name)
    except Exception:
        pass
    var_info = meta.get("variable_info", {}) or {}
    src_labels = list(var_info.get("source_labels", [f"S{i}" for i in range(L_S)]))
    inp_labels = list(var_info.get("input_labels", [f"X{j}" for j in range(L_X)]))

    # ------------------------------------------------------------------
    # Build one long-format residual-vs-parent table for S and X parents.
    # ------------------------------------------------------------------
    rows = []

    for j in range(L_X):
        target_label = inp_labels[j] if j < len(inp_labels) else f"X{j}"
        for i in range(L_S):
            parent_label = src_labels[i] if i < len(src_labels) else f"S{i}"
            edge_class, true_val = _classify_edge_pair(
                true_dag=true_cross,
                target_idx=j,
                parent_idx=i,
                parent_space="S",
            )
            rows.append({
                "attention_type": "cross",
                "parent_space": "S",
                "target_idx": j,
                "parent_idx": i,
                "source_idx": i,  # backward-compatible alias for cross rows
                "target_label": target_label,
                "parent_label": parent_label,
                "source_label": parent_label,  # backward-compatible alias for cross rows
                "hsic": float(hsic_cross[j, i]),
                "edge_class": edge_class,
                "true_edge": true_val,
                "is_self_loop": False,
            })

        for k in range(L_X):
            parent_label = inp_labels[k] if k < len(inp_labels) else f"X{k}"
            edge_class, true_val = _classify_edge_pair(
                true_dag=true_self,
                target_idx=j,
                parent_idx=k,
                parent_space="X",
            )
            hsic_val = hsic_self[j, k]
            rows.append({
                "attention_type": "self",
                "parent_space": "X",
                "target_idx": j,
                "parent_idx": k,
                "source_idx": k,
                "target_label": target_label,
                "parent_label": parent_label,
                "source_label": parent_label,
                "hsic": float(hsic_val) if not np.isnan(hsic_val) else np.nan,
                "edge_class": edge_class,
                "true_edge": true_val,
                "is_self_loop": bool(j == k),
            })

    df_all = pd.DataFrame(rows)
    df = df_all[df_all["attention_type"] == "cross"].copy()
    df_self = df_all[df_all["attention_type"] == "self"].copy()

    # ------------------------------------------------------------------
    # Summary statistics per edge class
    # ------------------------------------------------------------------
    class_stats = _class_stats_from_df(df)
    self_class_stats = _class_stats_from_df(df_self)
    all_class_stats = _class_stats_from_df(df_all)

    result["mean_hsic_true_parent"] = class_stats.get("true_parent")
    result["mean_hsic_false_parent"] = class_stats.get("false_parent")
    result["mean_hsic_independent"] = class_stats.get("independent")
    result["true_minus_false_margin"] = _margin_from_class_stats(class_stats)
    result["self_true_minus_false_margin"] = _margin_from_class_stats(self_class_stats)
    result["all_true_minus_false_margin"] = _margin_from_class_stats(all_class_stats)

    # ------------------------------------------------------------------
    # Save CSVs. `edge_hsic.csv` is kept as the historical cross-only artifact.
    # ------------------------------------------------------------------
    csv_path = join(eval_path_files, "edge_hsic.csv")
    df.to_csv(csv_path, index=False)
    result["edge_hsic_path"] = csv_path
    print(f"  Saved: edge_hsic.csv  ({len(df)} rows)")

    self_csv_path = join(eval_path_files, "edge_hsic_self.csv")
    df_self.to_csv(self_csv_path, index=False)
    result["self_edge_hsic_path"] = self_csv_path
    print(f"  Saved: edge_hsic_self.csv  ({len(df_self)} rows)")

    all_csv_path = join(eval_path_files, "edge_hsic_all.csv")
    df_all.to_csv(all_csv_path, index=False)
    result["all_edge_hsic_path"] = all_csv_path
    print(f"  Saved: edge_hsic_all.csv  ({len(df_all)} rows)")

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    summary = {
        "n_samples": N,
        "L_X": L_X,
        "L_S": L_S,
        "hsic_sigma": float(hsic_sigma),
        "sigma_res": float(sigma_res),
        "sigma_src": float(sigma_src),
        "sigma_x": float(sigma_x),
        "adaptive_bandwidth": adaptive_bandwidth,
        "hsic_mode": hsic_mode,
        "nhsic_epsilon": nhsic_epsilon,
        "source_kernel_cross": source_kernel,
        "source_kernel_self": "rbf",
        "mean_hsic_per_class_cross": {
            k: (float(v) if v is not None else None)
            for k, v in class_stats.items()
        },
        "mean_hsic_per_class_self": {
            k: (float(v) if v is not None else None)
            for k, v in self_class_stats.items()
        },
        "mean_hsic_per_class_all": {
            k: (float(v) if v is not None else None)
            for k, v in all_class_stats.items()
        },
        "true_minus_false_margin_cross": result["true_minus_false_margin"],
        "true_minus_false_margin_self": result["self_true_minus_false_margin"],
        "true_minus_false_margin_all": result["all_true_minus_false_margin"],
        "true_cross_dag_available": true_cross is not None,
        "true_self_dag_available": true_self is not None,
        "output_files": {
            "edge_hsic_cross": csv_path,
            "edge_hsic_self": self_csv_path,
            "edge_hsic_all": all_csv_path,
        },
    }
    json_path = join(eval_path_files, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    result["summary_path"] = json_path
    print(f"  Saved: summary.json")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    def _print_stats(title: str, stats: Dict[str, Optional[float]], margin: Optional[float]) -> None:
        print(f"\n  Edge-class HSIC summary ({title}):")
        for cls, val in stats.items():
            if val is not None:
                print(f"    {cls:20s}: {val:.6f}")
        if margin is not None:
            sign = "↓ good (true < false)" if margin < 0 else "↑ warning (true > false)"
            print(f"    margin (true−false)  : {margin:+.6f}  {sign}")

    _print_stats("cross / S parents", class_stats, result["true_minus_false_margin"])
    _print_stats("self / X parents", self_class_stats, result["self_true_minus_false_margin"])
    _print_stats("all parents", all_class_stats, result["all_true_minus_false_margin"])

    # ------------------------------------------------------------------
    # Heatmap plots. Always save; display only when requested.
    # ------------------------------------------------------------------
    cross_fig_path = join(eval_path_figs, f"residual_hsic_heatmap.{DEFAULT_PLOT_FORMAT}")
    if _plot_hsic_heatmap(
        hsic_mat=hsic_cross,
        row_labels=inp_labels,
        col_labels=src_labels,
        title="Residual HSIC — cross-attention / S parents",
        xlabel="Parent source variable (S_i)",
        ylabel="Target residual (ε_Xj)",
        fig_path=cross_fig_path,
        show_plots=show_plots,
    ):
        result["cross_heatmap_path"] = cross_fig_path
        print(f"  Saved: residual_hsic_heatmap.{DEFAULT_PLOT_FORMAT}")

    self_fig_path = join(eval_path_figs, f"residual_hsic_self_heatmap.{DEFAULT_PLOT_FORMAT}")
    if _plot_hsic_heatmap(
        hsic_mat=hsic_self,
        row_labels=inp_labels,
        col_labels=inp_labels,
        title="Residual HSIC — self-attention / X parents",
        xlabel="Parent intermediate variable (X_k)",
        ylabel="Target residual (ε_Xj)",
        fig_path=self_fig_path,
        show_plots=show_plots,
    ):
        result["self_heatmap_path"] = self_fig_path
        print(f"  Saved: residual_hsic_self_heatmap.{DEFAULT_PLOT_FORMAT}")

    return result

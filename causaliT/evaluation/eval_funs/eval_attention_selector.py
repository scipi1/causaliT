"""
Attention Selector evaluation: DAG recovery metrics for AttentionSelectorLayer.

The AttentionSelectorLayer uses a single combined cross-attention block whose
output matrix has shape (B, L_X, L_S + L_X).  Splitting at L_S gives:

    phi_sx  (L_X, L_S)   — S → X learned edges  (compare to dec1_cross_att_mask.csv)
    phi_xx  (L_X, L_X)   — X → X learned edges  (compare to dec1_self_att_mask.csv)

This module is intentionally **self-contained**: it directly loads
``AttentionSelectorForecaster`` from its checkpoint and bypasses the
``load_attention_data`` / ``predict_test_from_ckpt`` machinery used by
``eval_attention_scores``.  The output, however, is written to the **same**
``eval/eval_attention_scores/files/dag_metrics.json`` path so that
``eval_seed_sweep`` and ``update_experiments_manifest`` require no changes.

Public API
----------
eval_attention_selector_scores(experiment, show_plots=False) -> dict
"""

import json
import traceback
from os.path import join, exists, isdir
from os import listdir

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from omegaconf import OmegaConf

from .eval_utils import (
    root_path,
    _setup_eval_directories,
    _save_readme,
    _save_variable_labels,
    _create_cline_template,
    _compute_soft_hamming,
    _compute_standard_shd,
    _compute_zeroness_metrics,
    _load_true_dag_mask,
    _compute_dag_confidence,
    _combine_attention_to_full_dag,
    _load_full_true_dag,
    _compute_mec_distance,
    _check_mec_membership,
    _compute_mec_threshold,
    _find_v_structures,
    load_dataset_metadata,
    DEFAULT_PLOT_FORMAT,
)
from .eval_interventions import infer_checkpoint_type


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_k_fold_dirs(experiment: str):
    """Return sorted list of k_* fold directories inside *experiment*."""
    fold_dirs = sorted(
        join(experiment, d)
        for d in listdir(experiment)
        if d.startswith("k_") and isdir(join(experiment, d))
    )
    return fold_dirs


def _best_checkpoint(fold_dir: str, ckpt_type: str) -> str:
    """
    Return the best/last checkpoint path for a fold directory.

    Priority (same as ``find_best_or_last_checkpoint`` in eval_lib):
        best_causal   → best_causal_checkpoint.ckpt
        best_recon    → best_reconstruction_checkpoint.ckpt
        last          → last.ckpt or epoch=N-….ckpt (highest epoch)
    """
    ckpts_dir = join(fold_dir, "checkpoints")
    if not exists(ckpts_dir):
        raise FileNotFoundError(f"No checkpoints dir: {ckpts_dir}")

    files = [f for f in listdir(ckpts_dir) if f.endswith(".ckpt")]
    if not files:
        raise FileNotFoundError(f"No .ckpt files in {ckpts_dir}")

    if ckpt_type == "best_causal" and "best_causal_checkpoint.ckpt" in files:
        return join(ckpts_dir, "best_causal_checkpoint.ckpt")
    if ckpt_type in ("best_causal", "best_reconstruction"):
        if "best_reconstruction_checkpoint.ckpt" in files:
            return join(ckpts_dir, "best_reconstruction_checkpoint.ckpt")
        if "best_checkpoint.ckpt" in files:
            return join(ckpts_dir, "best_checkpoint.ckpt")

    # Fall back to highest-epoch checkpoint
    import re
    epoch_pat = re.compile(r"epoch=(\d+)")
    best_epoch, best_path = -1, None
    for f in files:
        if f in ("best_checkpoint.ckpt", "best_causal_checkpoint.ckpt",
                 "best_reconstruction_checkpoint.ckpt"):
            continue
        m = epoch_pat.search(f)
        if m and int(m.group(1)) > best_epoch:
            best_epoch = int(m.group(1))
            best_path = join(ckpts_dir, f)
    if best_path is None:
        # last resort: use last.ckpt if present
        if "last.ckpt" in files:
            return join(ckpts_dir, "last.ckpt")
        return join(ckpts_dir, files[0])
    return best_path


def _extract_combined_attention_from_data(
    forecaster,
    data_path: str,
    max_samples: int = 4096,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run a batched forward pass over test data and return mean attention weights.

    The model's Q/K projections encode the structural information; phi-learning
    has been deprecated.  The mean attention over a large test batch is the
    empirical proxy for the learned DAG.

    Parameters
    ----------
    forecaster : AttentionSelectorForecaster  (already loaded, eval mode)
    data_path  : str   path to ds_test.npz (keys 's', 'x')
    max_samples: int   maximum number of samples to average over (default 4096)
    batch_size : int   mini-batch size for forward pass (default 256)

    Returns
    -------
    att_mean : np.ndarray, shape (L_X, L_S + L_X)
        Mean attention weights averaged over all samples.
        Returns None if the data file is missing.
    """
    import torch

    if not exists(data_path):
        print(f"    ✗ Test data not found: {data_path}")
        return None

    data = np.load(data_path)
    S_np = data["s"].astype(np.float32)   # (N, L_S, features)
    X_np = data["x"].astype(np.float32)   # (N, L_X, features)
    N = min(len(S_np), max_samples)

    S_t = torch.tensor(S_np[:N])
    X_t = torch.tensor(X_np[:N])

    att_list = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            S_b = S_t[start : start + batch_size]
            X_b = X_t[start : start + batch_size]
            _, att, _ = forecaster.forward(S_b, X_b)   # att: (B, L_X, L_S+L_X)
            att_list.append(att.cpu().numpy())

    all_att = np.concatenate(att_list, axis=0)  # (N, L_X, L_S+L_X)  or  (N, H, L_X, L_S+L_X)

    att_mean = all_att.mean(axis=0)             # (L_X, L_S+L_X)  or  (H, L_X, L_S+L_X)

    # When shared_dag_across_heads=False, each head gets its own Q/K projection
    # and the inner_attention enters the is_multihead=True path, producing a 4-D
    # attention tensor (B, H, L_X, L_S+L_X).  Collapse the head dimension by
    # averaging so the returned matrix is always 2-D (L_X, L_S+L_X).
    if att_mean.ndim == 3:
        att_mean = att_mean.mean(axis=0)        # (H, L_X, L_S+L_X) → (L_X, L_S+L_X)

    return att_mean


def _plot_heatmaps(
    phi_sx_list, phi_xx_list,
    true_sx, true_xx,
    fold_names,
    save_path,
    show_plots=False,
):
    """
    2-panel heatmap per fold: Learned att (S→X) | Learned att (X→X).

    True edges are highlighted directly on each attention panel with green
    cell-border overlays (``mpatches.FancyBboxPatch``) instead of being shown
    as a separate column.  Per-cell attention values are annotated as text.

    Layout: n_folds rows × 2 columns.
    """
    n_folds = len(fold_names)
    n_cols = 2

    # Dynamically size: wider panels because we need readable cell labels
    fig_w = max(10, 5.5 * n_cols)
    fig_h = max(4, 4.0 * n_folds)
    fig, axes = plt.subplots(n_folds, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    col_titles = ["Learned att (S→X)", "Learned att (X→X)"]
    col_true   = [true_sx, true_xx]
    col_xlabel = ["Source S (key)", "Source X (key)"]

    for row, (fold_name, phi_sx, phi_xx) in enumerate(
        zip(fold_names, phi_sx_list, phi_xx_list)
    ):
        att_blocks = [phi_sx, phi_xx]

        for col, (att, true_mask, title, xlabel) in enumerate(
            zip(att_blocks, col_true, col_titles, col_xlabel)
        ):
            ax = axes[row][col]
            if att is None:
                ax.axis("off")
                ax.set_title(title if row == 0 else "", fontsize=10)
                continue

            n_rows_att, n_cols_att = att.shape
            vmax = att.max() if att.max() > 0 else 1.0

            im = ax.imshow(att, vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # ── Per-cell text annotations ──────────────────────────────
            for i in range(n_rows_att):
                for j in range(n_cols_att):
                    v = att[i, j]
                    text_color = "white" if v > 0.55 * vmax else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color=text_color)

            # ── Green border overlay for true edges ────────────────────
            if true_mask is not None and true_mask.shape == att.shape:
                for i in range(n_rows_att):
                    for j in range(n_cols_att):
                        if true_mask[i, j] == 1:
                            ax.add_patch(mpatches.FancyBboxPatch(
                                (j - 0.48, i - 0.48), 0.96, 0.96,
                                boxstyle="square,pad=0",
                                linewidth=2.5,
                                edgecolor="limegreen",
                                facecolor="none",
                            ))

            # ── Axis labels (S1/S2... or X1/X2...) ───────────────────
            x_labels = (
                [f"S{k+1}" for k in range(n_cols_att)]
                if col == 0
                else [f"X{k+1}" for k in range(n_cols_att)]
            )
            y_labels = [f"X{k+1}" for k in range(n_rows_att)]

            ax.set_xticks(range(n_cols_att))
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_yticks(range(n_rows_att))
            ax.set_yticklabels(y_labels, fontsize=8)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Target X (query)" if col == 0 else "", fontsize=8)

            ax.set_title(
                (f"{title}" if row == 0 else f"{fold_name}  —  {title}"),
                fontsize=10,
            )
            if row == 0 and col == 0:
                ax.set_ylabel(fold_name, fontsize=9)

    # Legend patch explaining the green border
    legend_patch = mpatches.Patch(
        facecolor="none", edgecolor="limegreen", linewidth=2.5,
        label="True parent edge",
    )
    fig.legend(handles=[legend_patch], loc="lower center",
               ncol=1, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(save_path, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Public evaluation function
# ---------------------------------------------------------------------------

def eval_attention_selector_scores(
    experiment: str,
    show_plots: bool = False,
) -> dict:
    """
    Evaluate DAG recovery for an AttentionSelectorLayer experiment.

    Extracts the learned phi from the best causal checkpoint of every k-fold,
    splits it into S→X and X→X sub-matrices, computes SHD / soft-Hamming /
    zeroness metrics versus the ground-truth adjacency masks, and writes the
    results to ``eval/eval_attention_scores/files/dag_metrics.json`` (same
    location as ``eval_attention_scores`` so downstream consumers are unchanged).

    Args:
        experiment: Path to the experiment folder (contains k_* subdirs).
        show_plots: If True, display matplotlib figures interactively.

    Returns:
        dag_metrics dict (also written to disk).
    """
    # ------------------------------------------------------------------
    # 0. Setup output directories (mirrors eval_attention_scores layout)
    # ------------------------------------------------------------------
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_attention_scores")

    dag_metrics_filename = "dag_metrics.json"
    attention_labels_filename = "attention_labels.json"

    print(f"\n{'='*60}")
    print(f"eval_attention_selector_scores")
    print(f"Experiment: {experiment}")
    print(f"Experiment ID: {exp_id}")
    print('='*60)

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config_files = [
        f for f in listdir(experiment)
        if f.startswith("config") and f.endswith(".yaml")
    ]
    if not config_files:
        raise ValueError(f"No config*.yaml found in {experiment}")
    config = OmegaConf.load(join(experiment, config_files[0]))

    dataset_name = config.get("data", {}).get("dataset")
    if not dataset_name:
        raise ValueError("No 'data.dataset' in config.")

    datadir_path = join(root_path, "data")
    metadata = load_dataset_metadata(datadir_path, dataset_name)

    # Read L_S / L_X from dataset metadata (same source as eval_attention_scores).
    # Fall back to config if the dataset has no metadata.json (S_seq_len is filled
    # at runtime by the datamodule and is null in the on-disk YAML).
    if metadata and "variable_info" in metadata:
        L_S = int(metadata["variable_info"]["n_source"])
        L_X = int(metadata["variable_info"]["n_input"])
    else:
        L_S = config["data"].get("S_seq_len")
        L_X = config["data"].get("X_seq_len")
        if L_S is None or L_X is None:
            raise ValueError(
                f"S_seq_len / X_seq_len not set in config and no dataset_metadata.json "
                f"found for dataset '{dataset_name}'. "
                f"Either add a metadata.json or set data.S_seq_len / data.X_seq_len explicitly."
            )
        L_S, L_X = int(L_S), int(L_X)

    ckpt_type = infer_checkpoint_type(config)
    print(f"  Dataset     : {dataset_name}  (L_S={L_S}, L_X={L_X})")
    print(f"  Checkpoint  : {ckpt_type}")

    # Test-data path for attention extraction (phi-learning deprecated)
    data_path = join(datadir_path, dataset_name, "ds_test.npz")
    if not exists(data_path):
        data_path = join(datadir_path, dataset_name, "ds.npz")
    print(f"  Data path   : {data_path}")

    # ------------------------------------------------------------------
    # 2. Save README / labels (matches eval_attention_scores pattern)
    # ------------------------------------------------------------------
    attention_labels = {
        "description": (
            "AttentionSelectorLayer: single combined cross-attention "
            "(Q=X_blanked, K/V=[S_actual, X_actual]).  "
            "phi columns 0..L_S-1 = S→X; columns L_S..end = X→X."
        ),
        "dataset": dataset_name,
        "L_S": L_S,
        "L_X": L_X,
    }
    if metadata and "variable_descriptions" in metadata:
        attention_labels["variable_mapping"] = metadata["variable_descriptions"]
    if metadata and "causal_structure" in metadata:
        cs = metadata["causal_structure"]
        if "edges" in cs:
            attention_labels["dag_structure"] = ", ".join(
                f"{s}->{t}" for s, t in cs["edges"]
            )
    _save_variable_labels(eval_path_files, attention_labels, attention_labels_filename)

    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description=(
            "AttentionSelectorLayer DAG evaluation: combined cross-attention "
            "phi split at L_S into S→X and X→X sub-matrices."
        ),
        files_info={
            dag_metrics_filename: (
                "Soft Hamming / SHD / zeroness metrics for S→X and X→X "
                "sub-matrices (JSON)"
            ),
            attention_labels_filename: "Variable labels and edge list (JSON)",
        },
    )
    _create_cline_template(eval_path_cline, "eval_attention_selector_scores", exp_id)

    # ------------------------------------------------------------------
    # 3. Load ground-truth masks
    # ------------------------------------------------------------------
    true_sx = _load_true_dag_mask(datadir_path, dataset_name, "dec_cross")   # (L_X, L_S)
    true_xx = _load_true_dag_mask(datadir_path, dataset_name, "dec_self")    # (L_X, L_X)

    if true_sx is None:
        print("  Warning: dec_cross mask not found — skipping S→X metrics.")
    if true_xx is None:
        print("  Warning: dec_self mask not found — skipping X→X metrics.")

    # ------------------------------------------------------------------
    # 4. Iterate over k-fold directories
    # ------------------------------------------------------------------
    fold_dirs = _find_k_fold_dirs(experiment)
    if not fold_dirs:
        raise ValueError(f"No k_* fold directories found in {experiment}")
    print(f"\n  Found {len(fold_dirs)} fold(s): {[d.split('/')[-1] for d in fold_dirs]}")

    phi_sx_per_fold = []   # (L_X, L_S) per fold
    phi_xx_per_fold = []   # (L_X, L_X) per fold
    fold_names = []

    # Lazy import — avoids circular imports at module level
    from causaliT.training.forecasters.attention_selector_forecaster import (
        AttentionSelectorForecaster,
    )

    for fold_dir in fold_dirs:
        fold_name = fold_dir.split("\\")[-1].split("/")[-1]  # portable on win/posix
        fold_names.append(fold_name)
        print(f"\n  --- {fold_name} ---")

        try:
            ckpt_path = _best_checkpoint(fold_dir, ckpt_type)
            print(f"    Checkpoint: {ckpt_path}")
        except FileNotFoundError as e:
            print(f"    ✗ {e}")
            phi_sx_per_fold.append(None)
            phi_xx_per_fold.append(None)
            continue

        try:
            forecaster = AttentionSelectorForecaster.load_from_checkpoint(
                ckpt_path,
                map_location="cpu",
            )
            forecaster.eval()
        except Exception as e:
            print(f"    ✗ Failed to load checkpoint: {e}")
            phi_sx_per_fold.append(None)
            phi_xx_per_fold.append(None)
            continue

        # Extract combined attention (L_X, L_S+L_X) via batched forward pass
        # Wrapped in try/except: any forward-pass or data-loading error is caught
        # so that dag_metrics.json is always written even if a fold fails.
        try:
            att_combined = _extract_combined_attention_from_data(forecaster, data_path)
        except Exception as e:
            print(f"    ✗ Failed to extract attention weights: {e}")
            traceback.print_exc()
            phi_sx_per_fold.append(None)
            phi_xx_per_fold.append(None)
            continue
        if att_combined is None:
            print("    ✗ Could not extract attention weights (data file missing).")
            phi_sx_per_fold.append(None)
            phi_xx_per_fold.append(None)
            continue

        if att_combined.shape != (L_X, L_S + L_X):
            print(
                f"    ✗ Unexpected attention shape {att_combined.shape}; "
                f"expected ({L_X}, {L_S + L_X}).  Skipping fold."
            )
            phi_sx_per_fold.append(None)
            phi_xx_per_fold.append(None)
            continue

        phi_sx = att_combined[:, :L_S]        # (L_X, L_S)
        phi_xx = att_combined[:, L_S:]        # (L_X, L_X)

        phi_sx_per_fold.append(phi_sx)
        phi_xx_per_fold.append(phi_xx)

        print(f"    att_sx  shape={phi_sx.shape}  "
              f"range=[{phi_sx.min():.3f}, {phi_sx.max():.3f}]  "
              f"mean={phi_sx.mean():.3f}")
        print(f"    att_xx  shape={phi_xx.shape}  "
              f"range=[{phi_xx.min():.3f}, {phi_xx.max():.3f}]  "
              f"mean={phi_xx.mean():.3f}")

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    print("\n--- Computing DAG Recovery Metrics ---")

    dag_metrics = {
        "dataset": dataset_name,
        "architecture": "AttentionSelectorForecaster",
    }

    def _metric_block(phi_list, true_mask, block_name, is_cross):
        """
        Compute per-fold metrics for one sub-matrix (sx or xx).

        Returns a dict of aggregated metrics, or {} if no valid fold.
        """
        fold_sh_list = []
        fold_std_shd_list = []
        fold_zeroness_list = []
        per_fold_sh = {}
        per_fold_shd = {}
        per_fold_zeroness = {}

        valid_phis = []

        for fold_name, phi in zip(fold_names, phi_list):
            if phi is None or true_mask is None:
                per_fold_sh[fold_name] = None
                per_fold_shd[fold_name] = None
                per_fold_zeroness[fold_name] = None
                continue

            if phi.shape != true_mask.shape:
                print(
                    f"    Shape mismatch for {block_name} fold={fold_name}: "
                    f"phi={phi.shape}  true={true_mask.shape} — skipping."
                )
                per_fold_sh[fold_name] = None
                per_fold_shd[fold_name] = None
                per_fold_zeroness[fold_name] = None
                continue

            sh = _compute_soft_hamming(phi, true_mask)
            shd = _compute_standard_shd(phi, true_mask,
                                        threshold=0.5, is_cross_attention=is_cross)
            zeroness = _compute_zeroness_metrics(phi, true_mask)

            per_fold_sh[fold_name] = sh
            per_fold_shd[fold_name] = shd
            per_fold_zeroness[fold_name] = zeroness
            fold_sh_list.append(sh)
            fold_std_shd_list.append(shd["shd"])
            fold_zeroness_list.append(zeroness)
            valid_phis.append(phi)

            print(
                f"  {fold_name} [{block_name}]:  "
                f"SoftHamming={sh:.4f}  "
                f"SHD={shd['shd']:d} (miss={shd['missing']}, extra={shd['extra']}, "
                f"rev={shd['reversed']})  "
                f"contrast={zeroness['contrast']:.3f}  "
                f"mean_nonedge={zeroness['mean_nonedge']:.3f}  "
                f"min_edge={zeroness['min_edge']:.3f}"
            )

        result = {}
        if fold_sh_list:
            arr = np.array(fold_sh_list)
            result[f"soft_hamming_{block_name}"] = {
                "best": float(np.min(arr)),
                "mean": float(np.mean(arr)),
                "worst": float(np.max(arr)),
                "std": float(np.std(arr)),
                "per_fold": per_fold_sh,
            }
            result[f"soft_hamming_{block_name}_source"] = "attention"

            print(
                f"  Soft Hamming [{block_name}]: "
                f"mean={np.mean(arr):.4f}  std={np.std(arr):.4f}  "
                f"best={np.min(arr):.4f}"
            )

        if fold_std_shd_list:
            arr = np.array(fold_std_shd_list, dtype=float)
            result[f"standard_shd_{block_name}"] = {
                "best": int(np.min(arr)),
                "mean": float(np.mean(arr)),
                "worst": int(np.max(arr)),
                "std": float(np.std(arr)),
                "per_fold": {k: v["shd"] if v else None
                             for k, v in per_fold_shd.items()},
                "per_fold_details": per_fold_shd,
            }
            print(
                f"  Standard SHD [{block_name}]: "
                f"mean={np.mean(arr):.1f}  std={np.std(arr):.1f}  "
                f"best={int(np.min(arr))}"
            )

        if fold_zeroness_list:
            zeroness_agg = {}
            for field in ["mean_nonedge", "max_nonedge", "mean_edge", "min_edge", "contrast"]:
                vals = [z[field] for z in fold_zeroness_list]
                zeroness_agg[field] = float(np.mean(vals))
            zeroness_agg["per_fold"] = per_fold_zeroness
            result[f"zeroness_{block_name}"] = zeroness_agg

        if len(valid_phis) >= 2:
            confidence = _compute_dag_confidence(valid_phis)
            result[f"dag_confidence_{block_name}"] = confidence
            print(f"  DAG Confidence [{block_name}]: {confidence:.4f}")

        return result

    if true_sx is not None:
        dag_metrics.update(_metric_block(phi_sx_per_fold, true_sx, "cross", is_cross=True))
    if true_xx is not None:
        dag_metrics.update(_metric_block(phi_xx_per_fold, true_xx, "self", is_cross=False))

    # ------------------------------------------------------------------
    # 5b. Compute MEC / skeleton / v-structure metrics
    #
    # Combines S→X and X→X sub-matrices into a single full DAG and
    # computes MEC distance + membership, skeleton recall/precision, and
    # v-structure recall/precision — exactly the fields expected by
    # eval_seed_sweep._extract_dag_metrics_per_seed.
    # ------------------------------------------------------------------
    print("\n--- Computing MEC Metrics ---")

    true_full_dag = _load_full_true_dag(datadir_path, dataset_name)

    if true_full_dag is None:
        print("  Warning: full true DAG not available — skipping MEC metrics.")
    else:
        mec_distances = []
        mec_memberships = []
        mec_per_fold = {}

        for fold_name, phi_sx, phi_xx in zip(fold_names, phi_sx_per_fold, phi_xx_per_fold):
            if phi_sx is None or phi_xx is None:
                mec_per_fold[fold_name] = None
                continue

            if phi_sx.shape != (L_X, L_S) or phi_xx.shape != (L_X, L_X):
                print(
                    f"  {fold_name}: unexpected sub-matrix shapes "
                    f"(sx={phi_sx.shape}, xx={phi_xx.shape}) — skipping MEC for this fold."
                )
                mec_per_fold[fold_name] = None
                continue

            # Build continuous full learned DAG: (L_S + L_X) × (L_S + L_X)
            full_learned_dag = _combine_attention_to_full_dag(
                cross_adj=phi_sx,
                self_adj=phi_xx,
                n_source=L_S,
                n_intermediate=L_X,
            )

            mec_dist, mec_details = _compute_mec_distance(full_learned_dag, true_full_dag)
            in_mec, _ = _check_mec_membership(full_learned_dag, true_full_dag)
            mec_thresh, _ = _compute_mec_threshold(full_learned_dag, true_full_dag)

            mec_distances.append(mec_dist)
            mec_memberships.append(in_mec)
            mec_per_fold[fold_name] = {
                "mec_distance": mec_dist,
                "in_mec": in_mec,
                "mec_threshold": mec_thresh,
                "skeleton_recall":      mec_details["skeleton_recall"],
                "skeleton_precision":   mec_details["skeleton_precision"],
                "v_structure_recall":   mec_details["v_structure_recall"],
                "v_structure_precision": mec_details["v_structure_precision"],
            }

            thresh_str = f"{mec_thresh:.4f}" if mec_thresh is not None else "N/A"
            print(
                f"  {fold_name}: mec_dist={mec_dist:.4f}  in_mec={in_mec}  "
                f"mec_thresh={thresh_str}  "
                f"skel_recall={mec_details['skeleton_recall']:.3f}  "
                f"skel_prec={mec_details['skeleton_precision']:.3f}  "
                f"vstr_recall={mec_details['v_structure_recall']:.3f}  "
                f"vstr_prec={mec_details['v_structure_precision']:.3f}"
            )

        if mec_distances:
            dist_arr = np.array(mec_distances)
            dag_metrics["mec_distance"] = {
                "best":  float(np.min(dist_arr)),
                "mean":  float(np.mean(dist_arr)),
                "worst": float(np.max(dist_arr)),
                "std":   float(np.std(dist_arr)),
                "per_fold": mec_per_fold,
            }
            dag_metrics["mec_membership_rate"] = float(np.mean(mec_memberships))

            # mec_threshold: mean over folds, ignoring folds where no threshold works (NaN).
            thresh_vals = [
                d["mec_threshold"]
                for d in mec_per_fold.values()
                if isinstance(d, dict) and d.get("mec_threshold") is not None
            ]
            if thresh_vals:
                thresh_arr = np.array(thresh_vals)
                dag_metrics["mec_threshold"] = {
                    "mean":  float(np.mean(thresh_arr)),
                    "std":   float(np.std(thresh_arr)) if len(thresh_arr) > 1 else 0.0,
                    "best":  float(np.max(thresh_arr)),   # higher = better
                    "worst": float(np.min(thresh_arr)),
                    "per_fold": {
                        k: d["mec_threshold"]
                        for k, d in mec_per_fold.items()
                        if isinstance(d, dict)
                    },
                }
                print(
                    f"  MEC threshold: mean={dag_metrics['mec_threshold']['mean']:.4f}  "
                    f"std={dag_metrics['mec_threshold']['std']:.4f}  "
                    f"best={dag_metrics['mec_threshold']['best']:.4f}"
                )
            else:
                dag_metrics["mec_threshold"] = None
                print("  MEC threshold: no fold reached MEC membership at any threshold.")

            print(
                f"  MEC distance: mean={dag_metrics['mec_distance']['mean']:.4f}  "
                f"std={dag_metrics['mec_distance']['std']:.4f}  "
                f"best={dag_metrics['mec_distance']['best']:.4f}"
            )
            print(f"  MEC membership rate: {dag_metrics['mec_membership_rate']:.4f}")
        else:
            print("  No valid fold produced a full learned DAG — MEC metrics skipped.")

    # ------------------------------------------------------------------
    # 6. Save dag_metrics.json
    # ------------------------------------------------------------------
    def _make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_json_serializable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    dag_metrics_path = join(eval_path_files, dag_metrics_filename)
    with open(dag_metrics_path, "w") as f:
        json.dump(_make_json_serializable(dag_metrics), f, indent=2)
    print(f"\n  Saved dag_metrics.json → {dag_metrics_path}")

    # ------------------------------------------------------------------
    # 6b. Save learned_dag_edges.json
    # Consumed by eval_seed_sweep._aggregate_learned_dag_across_seeds to
    # build the mean±std aggregate-DAG heatmap across seeds.
    # Format mirrors the output of eval_attention_scores so the same
    # plotting helper (plot_aggregate_dag) works for both.
    # ------------------------------------------------------------------
    s_labels = [f"S{i+1}" for i in range(L_S)]
    x_labels = [f"X{i+1}" for i in range(L_X)]

    valid_sx = [phi for phi in phi_sx_per_fold if phi is not None]
    valid_xx = [phi for phi in phi_xx_per_fold if phi is not None]

    edges_blocks = {}
    if valid_sx:
        mean_sx = np.mean(np.stack(valid_sx, axis=0), axis=0)   # (L_X, L_S)
        edges_blocks["att_cross"] = {
            "learned_mean": mean_sx.tolist(),
            "row_labels": x_labels,
            "col_labels": s_labels,
            "true": true_sx.astype(int).tolist() if true_sx is not None else [],
            "mask_type": "cross",
        }
    if valid_xx:
        mean_xx = np.mean(np.stack(valid_xx, axis=0), axis=0)   # (L_X, L_X)
        edges_blocks["att_self"] = {
            "learned_mean": mean_xx.tolist(),
            "row_labels": x_labels,
            "col_labels": x_labels,
            "true": true_xx.astype(int).tolist() if true_xx is not None else [],
            "mask_type": "self",
        }

    if edges_blocks:
        edges_payload = {
            "dataset": dataset_name,
            "architecture": "AttentionSelectorForecaster",
            "blocks": edges_blocks,
        }
        edges_path = join(eval_path_files, "learned_dag_edges.json")
        with open(edges_path, "w") as f:
            json.dump(_make_json_serializable(edges_payload), f, indent=2)
        print(f"  Saved learned_dag_edges.json → {edges_path}")

    # ------------------------------------------------------------------
    # 7. Plot heatmaps
    # ------------------------------------------------------------------
    valid_fold_plot = [
        (fn, psx, pxx)
        for fn, psx, pxx in zip(fold_names, phi_sx_per_fold, phi_xx_per_fold)
        if psx is not None and pxx is not None
    ]

    if valid_fold_plot:
        fn_list, psx_list, pxx_list = zip(*valid_fold_plot)
        heatmap_path = join(
            eval_path_fig, f"attention_selector_heatmaps_{exp_id}.{DEFAULT_PLOT_FORMAT}"
        )
        _plot_heatmaps(
            phi_sx_list=list(psx_list),
            phi_xx_list=list(pxx_list),
            true_sx=true_sx,
            true_xx=true_xx,
            fold_names=list(fn_list),
            save_path=heatmap_path,
            show_plots=show_plots,
        )
        print(f"  Saved heatmaps → {heatmap_path}")
    else:
        print("  Warning: no valid phi extracted from any fold — no heatmap saved.")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("eval_attention_selector_scores complete.")
    for key in ["soft_hamming_cross", "standard_shd_cross",
                "soft_hamming_self", "standard_shd_self",
                "mec_distance"]:
        if key in dag_metrics:
            v = dag_metrics[key]
            if isinstance(v, dict) and "mean" in v:
                print(f"  {key:35s}: mean={v['mean']:.4f}  std={v.get('std', float('nan')):.4f}")
    if "mec_membership_rate" in dag_metrics:
        print(f"  {'mec_membership_rate':35s}: {dag_metrics['mec_membership_rate']:.4f}")

    return dag_metrics

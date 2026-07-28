"""
Shared DAG-recovery metric core.

Consumes the canonical blocks produced by ``eval_dag_query.query_dag_blocks``
(``{fold_name: {"cross": (L_X, L_S), "self": (L_X, L_X), ...}}``) and computes
every DAG metric the project reports, independently of how the model stores its
attention:

- soft Hamming distance          (continuous, per block)
- standard SHD + TPR/FDR         (thresholded, literature-comparable)
- zeroness / contrast            (edge vs non-edge separation)
- DAG confidence                 (agreement across folds)
- MEC distance / membership      (only when both ``cross`` and ``self`` exist)

Blocks that a model does not provide are simply skipped: a cross-attention-only
model yields cross metrics and no MEC section, instead of crashing or silently
reporting zeros.
"""

from typing import Dict, Optional, Tuple, List, Any

import numpy as np

from .eval_utils import (
    _compute_soft_hamming,
    _compute_standard_shd,
    _compute_zeroness_metrics,
    _load_true_dag_mask,
    _compute_dag_confidence,
    _load_full_true_dag,
    _compute_mec_distance,
    _check_mec_membership,
    _compute_mec_threshold,
    _find_v_structures,
    _dag_to_skeleton,
)
from .eval_dag_query import (
    CROSS,
    SELF,
    block_mask_type,
    assemble_full_dag,
)


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays so ``json.dump`` accepts them."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def load_true_masks_for_blocks(
    block_names,
    datadir_path: str,
    dataset_name: str,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load the ground-truth mask for each canonical block name present.

    Returns a dict keyed by block name (``cross``, ``self_L1``, ...); blocks
    whose mask CSV is missing are omitted.
    """
    masks: Dict[str, np.ndarray] = {}
    cache: Dict[str, Optional[np.ndarray]] = {}

    for block in block_names:
        mask_type = block_mask_type(block)
        if mask_type is None:
            continue
        if mask_type not in cache:
            cache[mask_type] = _load_true_dag_mask(datadir_path, dataset_name, mask_type)
            if cache[mask_type] is None and verbose:
                print(f"  Warning: no true DAG mask for '{mask_type}' - skipping {block} metrics.")
        cached = cache[mask_type]
        if cached is not None:
            masks[block] = cached

    return masks


def _metric_block(
    block: str,
    per_fold_blocks: Dict[str, Dict[str, np.ndarray]],
    true_mask: np.ndarray,
    dag_threshold: float,
    source: str,
    verbose: bool,
) -> Tuple[dict, List[dict]]:
    """
    Compute all per-fold metrics for one canonical block.

    Returns ``(metrics, comparison_rows)`` where ``comparison_rows`` feed the
    heatmap plots and the portable ``learned_dag_edges.json`` export.
    """
    is_cross = block.split("_L")[0] == CROSS

    per_fold_sh, per_fold_shd, per_fold_zero = {}, {}, {}
    sh_list, shd_list, zero_list = [], [], []
    valid_matrices, comparison_rows = [], []

    for fold_name, blocks in per_fold_blocks.items():
        learned = blocks.get(block)

        if learned is None or learned.shape != true_mask.shape:
            if learned is not None and verbose:
                print(
                    f"    {fold_name} [{block}]: shape mismatch "
                    f"learned={learned.shape} true={true_mask.shape} - skipped."
                )
            per_fold_sh[fold_name] = None
            per_fold_shd[fold_name] = None
            per_fold_zero[fold_name] = None
            continue

        sh = _compute_soft_hamming(learned, true_mask)
        shd = _compute_standard_shd(
            learned, true_mask, threshold=dag_threshold, is_cross_attention=is_cross
        )
        zero = _compute_zeroness_metrics(learned, true_mask)

        per_fold_sh[fold_name] = sh
        per_fold_shd[fold_name] = shd
        per_fold_zero[fold_name] = zero
        sh_list.append(sh)
        shd_list.append(shd["shd"])
        zero_list.append(zero)
        valid_matrices.append(learned)

        comparison_rows.append({
            "fold_name": fold_name,
            "block": block,
            "mask_type": block_mask_type(block),
            "learned": learned,
            "true": true_mask,
            "soft_hamming": sh,
            "source": source,
        })

        if verbose:
            print(
                f"    {fold_name} [{block}]: SoftHamming={sh:.4f}"
                f"  | SHD={shd['shd']} (miss={shd['missing']}, extra={shd['extra']},"
                f" rev={shd['reversed']})"
                f"  | TPR={shd['tpr']:.1f}% FDR={shd['fdr']:.1f}%"
                f"  | contrast={zero['contrast']:.3f}"
                f" mean_nonedge={zero['mean_nonedge']:.3f}"
                f" min_edge={zero['min_edge']:.3f}"
            )

    metrics: dict = {}
    if sh_list:
        arr = np.array(sh_list)
        metrics[f"soft_hamming_{block}"] = {
            "best": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "worst": float(np.max(arr)),
            "std": float(np.std(arr)),
            "per_fold": per_fold_sh,
        }
        metrics[f"soft_hamming_{block}_source"] = source

    if shd_list:
        arr = np.array(shd_list, dtype=float)
        metrics[f"standard_shd_{block}"] = {
            "best": int(np.min(arr)),
            "mean": float(np.mean(arr)),
            "worst": int(np.max(arr)),
            "std": float(np.std(arr)),
            "per_fold": {k: (v["shd"] if v else None) for k, v in per_fold_shd.items()},
            "per_fold_details": per_fold_shd,
        }
        if verbose:
            print(
                f"    Standard SHD [{block}]: mean={np.mean(arr):.1f} "
                f"+/- {np.std(arr):.1f}"
            )

    if zero_list:
        agg: Dict[str, Any] = {
            field: float(np.mean([z[field] for z in zero_list]))
            for field in ("mean_nonedge", "max_nonedge", "mean_edge", "min_edge", "contrast")
        }
        agg["per_fold"] = per_fold_zero
        metrics[f"zeroness_{block}"] = agg

    if len(valid_matrices) >= 2:
        confidence = _compute_dag_confidence(valid_matrices)
        metrics[f"dag_confidence_{block}"] = confidence
        if verbose:
            print(f"    DAG Confidence [{block}]: {confidence:.4f}")

    return metrics, comparison_rows


def _mec_metrics(
    per_fold_blocks: Dict[str, Dict[str, np.ndarray]],
    datadir_path: str,
    dataset_name: str,
    L_S: Optional[int],
    L_X: Optional[int],
    layer_suffix: str = "",
    verbose: bool = True,
) -> dict:
    """
    MEC / skeleton / v-structure metrics from the assembled full DAG.

    Returns an empty dict when the full ground-truth DAG is unavailable or the
    model does not provide both a ``cross`` and a ``self`` block.
    """
    true_full_dag = _load_full_true_dag(datadir_path, dataset_name)
    if true_full_dag is None:
        if verbose:
            print("  Full true DAG unavailable - MEC metrics skipped.")
        return {}

    if verbose:
        skeleton = _dag_to_skeleton(true_full_dag)
        v_structures = _find_v_structures(true_full_dag)
        print(f"  True DAG: {len(skeleton)} edges, {len(v_structures)} v-structures")

    distances, memberships, per_fold = [], [], {}

    for fold_name, blocks in per_fold_blocks.items():
        full_learned = assemble_full_dag(blocks, L_S=L_S, L_X=L_X, layer_suffix=layer_suffix)

        if full_learned is None or full_learned.shape != true_full_dag.shape:
            per_fold[fold_name] = None
            continue

        mec_dist, details = _compute_mec_distance(full_learned, true_full_dag)
        in_mec, _ = _check_mec_membership(full_learned, true_full_dag)
        mec_thresh, _ = _compute_mec_threshold(full_learned, true_full_dag)

        distances.append(mec_dist)
        memberships.append(in_mec)
        per_fold[fold_name] = {
            "mec_distance": mec_dist,
            "in_mec": in_mec,
            "mec_threshold": mec_thresh,
            "skeleton_recall": details["skeleton_recall"],
            "skeleton_precision": details["skeleton_precision"],
            "v_structure_recall": details["v_structure_recall"],
            "v_structure_precision": details["v_structure_precision"],
        }

        if verbose:
            thresh_str = f"{mec_thresh:.4f}" if mec_thresh is not None else "N/A"
            print(
                f"    {fold_name}: mec_dist={mec_dist:.4f}  in_mec={in_mec}  "
                f"mec_thresh={thresh_str}  "
                f"skel_recall={details['skeleton_recall']:.3f}  "
                f"vstr_recall={details['v_structure_recall']:.3f}"
            )

    if not distances:
        if verbose:
            print("  No fold produced a full learned DAG - MEC metrics skipped.")
        return {}

    dist_arr = np.array(distances)
    metrics = {
        "mec_distance": {
            "best": float(np.min(dist_arr)),
            "mean": float(np.mean(dist_arr)),
            "worst": float(np.max(dist_arr)),
            "std": float(np.std(dist_arr)),
            "per_fold": per_fold,
        },
        "mec_membership_rate": float(np.mean(memberships)),
        "n_true_v_structures": len(_find_v_structures(true_full_dag)),
    }

    thresh_vals = [
        d["mec_threshold"] for d in per_fold.values()
        if isinstance(d, dict) and d.get("mec_threshold") is not None
    ]
    if thresh_vals:
        thresh_arr = np.array(thresh_vals)
        metrics["mec_threshold"] = {
            "mean": float(np.mean(thresh_arr)),
            "std": float(np.std(thresh_arr)) if len(thresh_arr) > 1 else 0.0,
            "best": float(np.max(thresh_arr)),    # higher = better
            "worst": float(np.min(thresh_arr)),
            "per_fold": {
                k: d["mec_threshold"] for k, d in per_fold.items() if isinstance(d, dict)
            },
        }
    else:
        metrics["mec_threshold"] = None
        if verbose:
            print("    MEC threshold: no fold reached MEC membership at any threshold.")

    if verbose:
        print(
            f"  MEC distance: mean={metrics['mec_distance']['mean']:.4f}  "
            f"std={metrics['mec_distance']['std']:.4f}  "
            f"best={metrics['mec_distance']['best']:.4f}"
        )
        print(f"  MEC membership rate: {metrics['mec_membership_rate']:.4f}")

    return metrics


def compute_dag_metrics(
    per_fold_blocks: Dict[str, Dict[str, np.ndarray]],
    datadir_path: str,
    dataset_name: str,
    architecture: str,
    dag_threshold: float = 0.5,
    L_S: Optional[int] = None,
    L_X: Optional[int] = None,
    source: str = "attention",
    verbose: bool = True,
) -> Tuple[dict, List[dict]]:
    """
    Compute all DAG-recovery metrics from canonical per-fold blocks.

    Args:
        per_fold_blocks: ``{fold_name: {block_name: matrix}}`` as returned by
            ``query_dag_blocks`` per fold.  Blocks may differ between folds; a
            missing block simply contributes no metric for that fold.
        datadir_path: Path to the ``data/`` directory (ground-truth masks).
        dataset_name: Dataset folder name.
        architecture: Architecture label, stored in the output for provenance.
        dag_threshold: Threshold for the standard (integer) SHD.
        L_S, L_X: Dimensions used when assembling the full DAG for MEC.
        source: Provenance of the matrices (``"attention"`` or ``"phi"``).
        verbose: Print per-fold diagnostics.

    Returns:
        ``(dag_metrics, comparison_rows)`` - the metrics dict written to
        ``dag_metrics.json`` and the per-fold rows used for plotting/export.
    """
    block_names = sorted({b for blocks in per_fold_blocks.values() for b in blocks})

    if verbose:
        print(f"  Blocks found: {block_names or '<none>'}")

    true_masks = load_true_masks_for_blocks(
        block_names, datadir_path, dataset_name, verbose=verbose
    )

    dag_metrics: dict = {"dataset": dataset_name, "architecture": architecture}
    comparison_rows: List[dict] = []

    for block in block_names:
        true_mask = true_masks.get(block)
        if true_mask is None:
            continue
        if verbose:
            print(f"  Evaluating block '{block}'...")
        metrics, rows = _metric_block(
            block=block,
            per_fold_blocks=per_fold_blocks,
            true_mask=true_mask,
            dag_threshold=dag_threshold,
            source=source,
            verbose=verbose,
        )
        dag_metrics.update(metrics)
        comparison_rows.extend(rows)

    # ---- MEC: needs both blocks -------------------------------------------
    has_cross = any(CROSS in blocks for blocks in per_fold_blocks.values())
    has_self = any(SELF in blocks for blocks in per_fold_blocks.values())

    if has_cross and has_self:
        if verbose:
            print("\n--- Computing MEC Metrics ---")
        dag_metrics.update(
            _mec_metrics(
                per_fold_blocks=per_fold_blocks,
                datadir_path=datadir_path,
                dataset_name=dataset_name,
                L_S=L_S,
                L_X=L_X,
                verbose=verbose,
            )
        )
    elif verbose:
        missing = "self (X->X)" if has_cross else "cross (S->X)"
        print(
            f"\n  MEC metrics skipped: model provides no {missing} block, "
            "so no full DAG can be assembled."
        )

    return dag_metrics, comparison_rows

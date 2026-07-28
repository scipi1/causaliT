"""
Simplified evaluation function for seed sweep experiments.

Computes mean and std over seeds for ATE metrics, test MAE, fit diagnostics, and
(when available) DAG recovery metrics produced by ``eval_attention_scores``.

Outputs (under ``<sweep>/eval/eval_seed_sweep/files/``):
    - ``ate_summary.csv``             : per-(intervention, variable) ATE stats
    - ``experiment_summary.csv``      : experiment-wide stats (single row, wide format)
    - ``dag_summary.csv``    *(opt.)* : DAG metrics in long format, human-readable
    - ``dag_summary.json``   *(opt.)* : DAG metrics with per-seed values, machine-readable
    - ``aggregate_dag.json`` *(opt.)* : per-edge inferred-probability mean/std/min/max
                                        across seeds, plus per-seed values and the true
                                        DAG mask. Drives plotting of the "aggregate DAG"
                                        for the paper.
    - ``aggregate_dag.csv``  *(opt.)* : same data in long format (block,row,col,...,mean,std,...)

The DAG/aggregate-DAG outputs are only emitted when at least one seed run
produced the corresponding upstream artifacts in ``eval/eval_attention_scores/files/``
(i.e. the model is one of the causal attention models, not a plain baseline).

Example:
    >>> from causaliT.evaluation.eval_funs import eval_seed_sweep
    >>> df_ate, df_exp = eval_seed_sweep("experiments/baseline/euler/vanilla_transformer_scm1_61555008")
"""

import re
import json
from os.path import join, exists, isdir
from os import makedirs, listdir
from typing import Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd


def _parse_seed_folder_name(folder_name: str) -> Optional[int]:
    """Parse seed from folder name (expected format: *_seed_{seed}*)."""
    match = re.search(r'seed_(\d+)', folder_name)
    return int(match.group(1)) if match else None


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file, return None if not found or error."""
    if not exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _parse_intervention_value(intervention: str) -> Optional[float]:
    """Parse intervention value from string (e.g., 'S2=-1.7' -> -1.7)."""
    match = re.search(r'=(-?[\d.]+)', intervention)
    return float(match.group(1)) if match else None


def _extract_fit_diagnostics(run_path: str) -> Optional[Dict[str, float]]:
    """
    Extract fit diagnostics (train_loss, generalization gap) from a seed run.
    
    Reads the training CSV log from k_0/logs/csv/version_0/metrics.csv,
    finds the epoch with best val_loss, and computes:
    - best_train_loss: first column matching ``train_loss*`` at the best val_loss epoch
    - best_val_loss: best validation loss achieved
    - generalization_gap: val_loss - train_loss at best epoch
    - gap_ratio: (val_loss - train_loss) / train_loss
    
    The train-loss column is resolved via a ``train_loss*`` regex so that
    experiments logged as ``train_loss_x`` (component loss) or ``train_loss``
    (total loss) are both handled transparently.
    
    Returns None if training CSV is not found or required metrics are missing.
    """
    # Find the k-fold directory (typically k_0 for single-fold)
    kfold_dirs = sorted([
        d for d in listdir(run_path)
        if isdir(join(run_path, d)) and d.startswith("k_")
    ])
    
    if not kfold_dirs:
        return None
    
    # Use first fold
    metrics_path = join(run_path, kfold_dirs[0], "logs", "csv", "version_0", "metrics.csv")
    if not exists(metrics_path):
        return None
    
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return None
    
    # Resolve train-loss column: prefer exact "train_loss", fall back to first
    # column matching the train_loss* pattern (e.g. "train_loss_x").
    train_loss_cols = [c for c in df.columns if re.match(r'train_loss', c)]
    if not train_loss_cols or "val_loss" not in df.columns:
        return None
    train_loss_col = train_loss_cols[0]
    
    # Group by epoch, take first non-NaN for each metric
    if "epoch" in df.columns:
        df_epoch = df.groupby("epoch").first().reset_index()
    else:
        df_epoch = df
    
    # Drop rows where either is NaN
    valid = df_epoch.dropna(subset=[train_loss_col, "val_loss"])
    if len(valid) == 0:
        return None
    
    # Find best val_loss epoch
    best_idx = valid["val_loss"].idxmin()
    best_row = valid.loc[best_idx]
    
    best_train = float(best_row[train_loss_col])
    best_val = float(best_row["val_loss"])
    gen_gap = best_val - best_train
    gap_ratio = gen_gap / best_train if best_train > 1e-12 else float('inf')
    
    return {
        "best_train_loss": best_train,
        "best_val_loss": best_val,
        "generalization_gap": gen_gap,
        "gap_ratio": gap_ratio,
    }


# DAG metric column names (per-seed scalars extracted from dag_metrics.json).
# Order here drives the column order in the dag_summary outputs.
_DAG_METRIC_COLUMNS = [
    # Soft Hamming distance (continuous, in [0,1]) for each attention block
    "soft_hamming_cross",
    "soft_hamming_self",
    "soft_hamming_total",        # cross + self (aggregate, see _extract_dag_metrics_per_seed)
    # Standard SHD (integer, literature-compatible) at threshold 0.5
    "standard_shd_cross",
    "standard_shd_self",
    "standard_shd_total",        # cross + self
    # Zeroness contrast = mean_edge - mean_nonedge (separation between true edges and non-edges)
    "zeroness_cross_contrast",
    "zeroness_self_contrast",
    # Classification metrics from thresholded confusion matrix (percentages)
    "tpr_cross",              # True Positive Rate (Recall) [%]
    "tpr_self",
    "fdr_cross",              # False Discovery Rate [%]
    "fdr_self",
    "precision_cross",        # Precision [%]
    "precision_self",
    # MEC (Markov Equivalence Class) metrics on the combined full DAG
    "mec_distance",
    "mec_membership_rate",
    # MEC threshold: max binarisation threshold θ at which the DAG is still in the MEC.
    # Analogous to a p-value: higher = scores are more discriminative (NaN = never in MEC).
    "mec_threshold",
    # Skeleton / v-structure quality (averaged across folds within each seed)
    "skeleton_recall",
    "skeleton_precision",
    "v_structure_recall",
    "v_structure_precision",
]


def _safe_get(d: Optional[Dict[str, Any]], *keys, default=None):
    """Nested dict getter that tolerates missing keys / None."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _mean_per_fold_field(per_fold: Optional[Dict[str, Any]], field: str) -> Optional[float]:
    """
    Average a single field across all folds in a per_fold dict.
    Returns None if no fold has the field.
    """
    if not isinstance(per_fold, dict):
        return None
    values = []
    for v in per_fold.values():
        if isinstance(v, dict) and field in v and v[field] is not None:
            values.append(v[field])
    return float(np.mean(values)) if values else None


def _extract_dag_metrics_per_seed(dag_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten a single seed's ``dag_metrics.json`` payload into a flat dict
    of scalars suitable for cross-seed aggregation.

    See ``_DAG_METRIC_COLUMNS`` for the list of returned keys. Missing
    fields are simply omitted (not filled) so downstream aggregation can
    detect partial coverage.
    """
    out: Dict[str, float] = {}

    # --- Soft Hamming distance (per-block "mean" across folds) -------------
    sh_cross = _safe_get(dag_data, "soft_hamming_cross", "mean")
    sh_self = _safe_get(dag_data, "soft_hamming_self", "mean")
    if sh_cross is not None:
        out["soft_hamming_cross"] = float(sh_cross)
    if sh_self is not None:
        out["soft_hamming_self"] = float(sh_self)
    # Aggregate "total" Hamming by summing the two blocks (interpretable as
    # the overall edge-wise disagreement on the joint S∪X graph).
    if sh_cross is not None and sh_self is not None:
        out["soft_hamming_total"] = float(sh_cross) + float(sh_self)

    # --- Standard (integer) SHD --------------------------------------------
    std_cross = _safe_get(dag_data, "standard_shd_cross", "mean")
    std_self = _safe_get(dag_data, "standard_shd_self", "mean")
    if std_cross is not None:
        out["standard_shd_cross"] = float(std_cross)
    if std_self is not None:
        out["standard_shd_self"] = float(std_self)
    if std_cross is not None and std_self is not None:
        out["standard_shd_total"] = float(std_cross) + float(std_self)

    # --- Zeroness contrast (separation of edges vs non-edges) ---------------
    zc = _safe_get(dag_data, "zeroness_cross", "contrast")
    zs = _safe_get(dag_data, "zeroness_self", "contrast")
    if zc is not None:
        out["zeroness_cross_contrast"] = float(zc)
    if zs is not None:
        out["zeroness_self_contrast"] = float(zs)

    # --- Classification metrics from thresholded confusion matrix -----------
    # Extracted from standard_shd_* per_fold_details (tpr, fdr, precision)
    for block in ("cross", "self"):
        per_fold_details = _safe_get(dag_data, f"standard_shd_{block}", "per_fold_details")
        for field in ("tpr", "fdr", "precision"):
            v = _mean_per_fold_field(per_fold_details, field)
            if v is not None:
                out[f"{field}_{block}"] = float(v)

    # --- MEC distance, membership, and threshold ----------------------------
    mec_mean = _safe_get(dag_data, "mec_distance", "mean")
    mec_member = _safe_get(dag_data, "mec_membership_rate")
    mec_thresh = _safe_get(dag_data, "mec_threshold", "mean")
    if mec_mean is not None:
        out["mec_distance"] = float(mec_mean)
    if mec_member is not None:
        out["mec_membership_rate"] = float(mec_member)
    # mec_threshold may be None when no fold achieved MEC membership (→ omit
    # from out so downstream dropna() naturally treats it as NaN).
    if mec_thresh is not None:
        out["mec_threshold"] = float(mec_thresh)

    # --- Skeleton / v-structure metrics (avg across folds) -----------------
    mec_per_fold = _safe_get(dag_data, "mec_distance", "per_fold")
    for field in ("skeleton_recall", "skeleton_precision",
                  "v_structure_recall", "v_structure_precision"):
        v = _mean_per_fold_field(mec_per_fold, field)
        if v is not None:
            out[field] = float(v)

    return out


def _aggregate_learned_dag_across_seeds(
    seed_to_edges: Dict[int, Dict[str, Any]],
    eval_path_files: str,
    sweep_experiment: str,
) -> None:
    """
    Aggregate per-seed ``learned_dag_edges.json`` payloads into the
    "aggregate DAG" outputs (mean/std/min/max per edge across seeds).

    Writes two files (only when ``seed_to_edges`` is non-empty):
        - aggregate_dag.json  (machine-readable, full matrices + per-seed)
        - aggregate_dag.csv   (long-format, one row per edge per block)

    Robustness:
      - Seeds whose block disagrees in shape with the first encountered
        shape are skipped (with a warning); shouldn't happen unless the
        sweep mixes datasets.
      - Block / variable labels are taken from the first seed that
        provides them; warned if subsequent seeds disagree.
    """
    if not seed_to_edges:
        return

    # Discover all block keys across seeds (e.g. "dec_cross", "dec_self",
    # "dec_cross_L0", ...).
    block_to_seed_data: Dict[str, list] = {}
    datasets = set()
    architectures = set()
    for seed, payload in seed_to_edges.items():
        if not isinstance(payload, dict):
            continue
        ds = payload.get("dataset")
        arch = payload.get("architecture")
        if ds:
            datasets.add(ds)
        if arch:
            architectures.add(arch)
        for block_name, block_data in (payload.get("blocks") or {}).items():
            block_to_seed_data.setdefault(block_name, []).append(
                (seed, block_data)
            )

    if not block_to_seed_data:
        return

    blocks_out: Dict[str, Any] = {}
    csv_rows = []

    for block_name, seed_entries in block_to_seed_data.items():
        # Reference shape and labels from the first seed.
        ref_seed, ref_block = seed_entries[0]
        try:
            ref_mean = np.asarray(ref_block["learned_mean"], dtype=float)
        except (KeyError, ValueError, TypeError):
            print(f"  [aggregate_dag] block {block_name}: missing learned_mean, skipping")
            continue
        ref_shape = ref_mean.shape
        row_labels = list(ref_block.get("row_labels", []))
        col_labels = list(ref_block.get("col_labels", []))
        true_dag = np.asarray(ref_block.get("true", []), dtype=float)
        mask_type = ref_block.get("mask_type", "")

        # Stack matrices across seeds, dropping mismatched shapes.
        per_seed: Dict[str, list] = {}
        stack = []
        used_seeds = []
        for seed, block_data in seed_entries:
            try:
                arr = np.asarray(block_data["learned_mean"], dtype=float)
            except (KeyError, ValueError, TypeError):
                continue
            if arr.shape != ref_shape:
                print(
                    f"  [aggregate_dag] block {block_name}: seed {seed} shape "
                    f"{arr.shape} != reference {ref_shape}, skipping seed"
                )
                continue
            stack.append(arr)
            per_seed[str(int(seed))] = arr.tolist()
            used_seeds.append(int(seed))

        if not stack:
            continue

        stack_arr = np.stack(stack, axis=0)
        mean_arr = stack_arr.mean(axis=0)
        std_arr = (
            stack_arr.std(axis=0)
            if stack_arr.shape[0] > 1
            else np.zeros_like(mean_arr)
        )
        min_arr = stack_arr.min(axis=0)
        max_arr = stack_arr.max(axis=0)

        n_rows, n_cols = ref_shape
        blocks_out[block_name] = {
            "mask_type": mask_type,
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "row_labels": row_labels,
            "col_labels": col_labels,
            "n_seeds": int(len(used_seeds)),
            "seeds": sorted(used_seeds),
            "true": true_dag.astype(int).tolist() if true_dag.size else [],
            "mean": mean_arr.tolist(),
            "std": std_arr.tolist(),
            "min": min_arr.tolist(),
            "max": max_arr.tolist(),
            "per_seed": per_seed,
        }

        # Long-format CSV rows, one per edge.
        for i in range(n_rows):
            for j in range(n_cols):
                row_label = row_labels[i] if i < len(row_labels) else f"row{i}"
                col_label = col_labels[j] if j < len(col_labels) else f"col{j}"
                true_val = (
                    int(true_dag[i, j])
                    if true_dag.size and true_dag.shape == ref_shape
                    else None
                )
                csv_rows.append({
                    "block": block_name,
                    "mask_type": mask_type,
                    "row": i,
                    "col": j,
                    "row_label": row_label,    # target  (cross/self)
                    "col_label": col_label,    # source  (cross) or target (self)
                    "true": true_val,
                    "mean": float(mean_arr[i, j]),
                    "std": float(std_arr[i, j]),
                    "min": float(min_arr[i, j]),
                    "max": float(max_arr[i, j]),
                    "n_seeds": int(len(used_seeds)),
                })

    if not blocks_out:
        return

    # --- Save JSON (machine-readable) -----------------------------------
    payload = {
        "sweep_experiment": sweep_experiment,
        "datasets": sorted(datasets),
        "architectures": sorted(architectures),
        "n_seeds": int(len(seed_to_edges)),
        "seeds": sorted(int(s) for s in seed_to_edges.keys()),
        "blocks": blocks_out,
    }
    json_path = join(eval_path_files, "aggregate_dag.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {json_path}")

    # --- Save long-format CSV (human-readable) --------------------------
    df_csv = pd.DataFrame(csv_rows)
    csv_path = join(eval_path_files, "aggregate_dag.csv")
    df_csv.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def eval_seed_sweep(sweep_experiment: str, show_plots: bool=False) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """
    Evaluate a seed sweep experiment and compute mean ± std over seeds.
    
    Args:
        sweep_experiment: Path to sweep experiment folder
        
    Returns:
        Tuple of:
        - df_ate: ATE metrics per (intervention, variable) with mean/std over seeds.
          ``None`` when no seed run produced ``eval_ate_mc`` results (e.g. pure
          DAG-learning experiments without ATE evaluation).
        - df_experiment: Experiment-wide metrics (test_mae, shd, mec) with mean/std over seeds
        
    Output Files:
        eval/eval_seed_sweep/files/ate_summary.csv       (only when ATE data available)
        eval/eval_seed_sweep/files/experiment_summary.csv
        eval/eval_seed_sweep/files/dag_summary.csv        (only when DAG metrics are available)
        eval/eval_seed_sweep/files/dag_summary.json       (only when DAG metrics are available)
    """
    combinations_dir = join(sweep_experiment, "sweeper", "runs", "combinations")
    
    if not exists(combinations_dir):
        raise FileNotFoundError(f"No sweep runs found at: {combinations_dir}")
    
    # Setup output directory
    eval_path_files = join(sweep_experiment, "eval", "eval_seed_sweep", "files")
    makedirs(eval_path_files, exist_ok=True)
    
    # Discover seed folders
    run_folders = sorted([
        d for d in listdir(combinations_dir)
        if isdir(join(combinations_dir, d))
    ])
    
    # Collect data
    ate_records = []
    experiment_records = []
    dag_records = []                # list of dicts {seed, <metric>: value, ...}
    dag_meta = {"datasets": set(), "architectures": set()}
    seed_to_edges: Dict[int, Dict[str, Any]] = {}   # seed -> learned_dag_edges payload
    
    for run_folder in run_folders:
        run_path = join(combinations_dir, run_folder)
        seed = _parse_seed_folder_name(run_folder)
        
        if seed is None:
            continue
        
        # Load data sources
        ate_data = _load_json(join(run_path, "eval", "eval_ate_mc", "files", "ate_metrics_mc.json"))
        kfold_data = _load_json(join(run_path, "kfold_summary.json"))
        # DAG metrics are only produced by causal-attention models. For
        # baselines the eval_attention_scores folder simply doesn't exist
        # and we silently skip the DAG aggregation for that seed.
        dag_metrics_path = join(
            run_path, "eval", "eval_attention_scores", "files", "dag_metrics.json"
        )
        dag_data = _load_json(dag_metrics_path) if exists(dag_metrics_path) else None
        # Per-edge inferred-probability matrices (for aggregate-DAG plotting).
        # Produced alongside dag_metrics.json by eval_attention_scores; older
        # cached runs may not have this file yet — re-run eval_attention_scores
        # to backfill. We tolerate its absence per-seed.
        edges_path = join(
            run_path, "eval", "eval_attention_scores", "files", "learned_dag_edges.json"
        )
        edges_data = _load_json(edges_path) if exists(edges_path) else None
        if edges_data:
            seed_to_edges[int(seed)] = edges_data
        
        # Extract experiment-wide metrics
        exp_record = {"seed": seed}
        
        if kfold_data:
            stats = kfold_data.get("statistics", {})
            if "test_x_mae" in stats:
                exp_record["test_mae"] = stats["test_x_mae"].get("mean")
            if "val_x_mae" in stats:
                exp_record["val_mae"] = stats["val_x_mae"].get("mean")
            if "val_loss_x" in stats:
                exp_record["val_loss"] = stats["val_loss_x"].get("mean")
        
        # Extract fit diagnostics from training CSV logs
        fit_diag = _extract_fit_diagnostics(run_path)
        if fit_diag:
            exp_record.update(fit_diag)
        
        # Extract DAG metrics (rich set, see _DAG_METRIC_COLUMNS)
        if dag_data:
            ds = dag_data.get("dataset")
            arch = dag_data.get("architecture")
            if ds:
                dag_meta["datasets"].add(ds)
            if arch:
                dag_meta["architectures"].add(arch)
            
            seed_dag = _extract_dag_metrics_per_seed(dag_data)
            if seed_dag:
                # Mirror DAG metrics into the experiment-wide record so they
                # also appear in experiment_summary.csv.
                exp_record.update(seed_dag)
                # And keep a dedicated record for the DAG-only summary files.
                dag_records.append({"seed": seed, **seed_dag})
        
        experiment_records.append(exp_record)
        
        # Extract per-intervention-variable ATE metrics
        if ate_data:
            per_interv = ate_data.get("per_intervention_variable", [])
            for entry in per_interv:
                intervention = entry.get("intervention", "unknown")
                abs_error = entry.get("abs_error_mean")
                
                # Compute scaled_error = abs_error / |intervention_value|
                interv_value = _parse_intervention_value(intervention)
                scaled_error = None
                if abs_error is not None and interv_value is not None and interv_value != 0:
                    scaled_error = abs_error / abs(interv_value)
                
                ate_records.append({
                    "seed": seed,
                    "intervention": intervention,
                    "variable": entry.get("variable", "unknown"),
                    "true_ate": entry.get("true_ate"),
                    "model_ate": entry.get("model_ate_mean"),
                    "abs_error": abs_error,
                    "scaled_error": scaled_error,
                })
    
    if not experiment_records:
        raise ValueError("No valid seed runs found")
    
    # Create DataFrames
    df_exp = pd.DataFrame(experiment_records)
    
    # =========================================================================
    # Aggregate ATE metrics by (intervention, variable) — optional
    # Skipped when no seed produced eval_ate_mc results (e.g. pure causal-
    # discovery experiments that never ran eval_ate_mc).
    # =========================================================================
    agg_ate: Optional[pd.DataFrame] = None
    if ate_records:
        df_ate = pd.DataFrame(ate_records)
        _agg = df_ate.groupby(["intervention", "variable"]).agg({
            "true_ate": "first",
            "model_ate": ["mean", "std"],
            "abs_error": ["mean", "std"],
            "scaled_error": ["mean", "std"],
            "seed": "count",
        })
        # Flatten column names
        _agg.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in _agg.columns
        ]
        _agg = _agg.rename(columns={"seed_count": "n_seeds", "true_ate_first": "true_ate"})
        _agg = _agg.reset_index()
        col_order_ate = [
            "intervention", "variable", "true_ate",
            "model_ate_mean", "model_ate_std",
            "abs_error_mean", "abs_error_std",
            "scaled_error_mean", "scaled_error_std",
            "n_seeds",
        ]
        agg_ate = _agg[[c for c in col_order_ate if c in _agg.columns]]
    else:
        print("No ATE data found across seeds (skipping ate_summary.csv).")
    
    # =========================================================================
    # Aggregate experiment-wide metrics
    # =========================================================================
    exp_columns = [
        "test_mae", "val_mae", "val_loss",
        "best_train_loss", "best_val_loss", "generalization_gap", "gap_ratio",
    ] + list(_DAG_METRIC_COLUMNS)
    
    exp_summary = {}
    for col in exp_columns:
        if col in df_exp.columns:
            values = df_exp[col].dropna()
            if len(values) > 0:
                exp_summary[f"{col}_mean"] = values.mean()
                exp_summary[f"{col}_std"] = values.std()
    
    exp_summary["n_seeds"] = len(df_exp)
    df_exp_summary = pd.DataFrame([exp_summary])
    
    # =========================================================================
    # Save outputs
    # =========================================================================
    exp_path = join(eval_path_files, "experiment_summary.csv")
    df_exp_summary.to_csv(exp_path, index=False)
    print(f"Saved: {exp_path}")

    if agg_ate is not None:
        ate_path = join(eval_path_files, "ate_summary.csv")
        agg_ate.to_csv(ate_path, index=False)
        print(f"Saved: {ate_path}")
    
    # =========================================================================
    # Aggregate DAG metrics across seeds (only when at least one seed produced
    # a dag_metrics.json — baselines without an attention block won't).
    # Outputs:
    #   - dag_summary.csv  : long-format, human-readable (1 row per metric)
    #   - dag_summary.json : machine-readable, includes per-seed values
    # =========================================================================
    if dag_records:
        df_dag = pd.DataFrame(dag_records).set_index("seed").sort_index()
        # Restrict / order columns by the canonical metric list (keep only
        # those that actually have data).
        metric_cols = [c for c in _DAG_METRIC_COLUMNS if c in df_dag.columns]
        df_dag = df_dag[metric_cols]
        
        # --- Long-format CSV (human-readable) -------------------------------
        long_rows = []
        for metric in metric_cols:
            values = df_dag[metric].dropna()
            if len(values) == 0:
                continue
            long_rows.append({
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std()) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "n_seeds": int(len(values)),
            })
        df_dag_summary = pd.DataFrame(long_rows)
        
        dag_csv_path = join(eval_path_files, "dag_summary.csv")
        df_dag_summary.to_csv(dag_csv_path, index=False)
        print(f"Saved: {dag_csv_path}")
        
        # --- Machine-readable JSON (per-seed + aggregate stats) -------------
        seeds_sorted = [int(s) for s in df_dag.index.tolist()]
        metrics_json: Dict[str, Any] = {}
        for metric in metric_cols:
            series = df_dag[metric]
            valid = series.dropna()
            if len(valid) == 0:
                continue
            metrics_json[metric] = {
                "mean": float(valid.mean()),
                "std": float(valid.std()) if len(valid) > 1 else 0.0,
                "min": float(valid.min()),
                "max": float(valid.max()),
                "n_seeds": int(len(valid)),
                # Cast keys to str so the JSON is valid; values may be NaN -> None.
                "per_seed": {
                    str(int(seed)): (float(v) if pd.notna(v) else None)
                    for seed, v in series.items()
                },
            }
        
        dag_json_payload = {
            "sweep_experiment": sweep_experiment,
            "datasets": sorted(dag_meta["datasets"]),
            "architectures": sorted(dag_meta["architectures"]),
            "n_seeds": int(len(df_dag)),
            "seeds": seeds_sorted,
            "metrics": metrics_json,
        }
        
        dag_json_path = join(eval_path_files, "dag_summary.json")
        with open(dag_json_path, "w") as f:
            json.dump(dag_json_payload, f, indent=2)
        print(f"Saved: {dag_json_path}")
    else:
        print("No DAG metrics found across seeds (skipping dag_summary.csv/json).")
    
    # =========================================================================
    # Aggregate per-edge inferred DAG probabilities across seeds (when
    # learned_dag_edges.json was emitted by eval_attention_scores).
    # Produces aggregate_dag.{json,csv} for plotting the "aggregate DAG" in
    # the paper. Skipped silently if no seed has the per-edge file (e.g.,
    # baselines, or older runs that need eval_attention_scores re-run).
    # =========================================================================
    if seed_to_edges:
        _aggregate_learned_dag_across_seeds(
            seed_to_edges=seed_to_edges,
            eval_path_files=eval_path_files,
            sweep_experiment=sweep_experiment,
        )
    else:
        print("No learned_dag_edges.json across seeds (skipping aggregate_dag.csv/json).")
    
    return agg_ate, df_exp_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate seed sweep (mean ± std over seeds)")
    parser.add_argument("sweep_experiment", help="Path to sweep experiment folder")
    args = parser.parse_args()
    
    df_ate, df_exp = eval_seed_sweep(args.sweep_experiment)
    print("\n--- ATE Summary ---")
    if df_ate is not None:
        print(df_ate.to_string())
    else:
        print("(not available — no eval_ate_mc results found)")
    print("\n--- Experiment Summary ---")
    print(df_exp.to_string())

"""
DAG recovery evaluation from attention scores.

This module has a single entry point, ``eval_attention_scores``:
   - Loads attention from the best checkpoint of each fold
   - Computes DAG recovery metrics (soft Hamming, SHD, zeroness, DAG
     confidence, MEC)
   - Writes quantitative artefacts only (JSON tables); no figures

It serves every architecture through one extraction path:

       best checkpoint -> predict_test_from_ckpt -> query_dag_blocks
                       -> compute_dag_metrics

``eval_dag_query.query_dag_blocks`` classifies each attention tensor by its
shape, given ``L_S`` and ``L_X``.  The same code therefore handles per-key
architectures (SingleCausal, NoiseAware, StageCausal, proT: ``dec_cross``,
``dec_self``, ``dec_cross_L0``, ...) and the AttentionSelectorLayer's single
combined ``(L_X, L_S + L_X)`` block, with or without a dedicated
self-attention module.

Every architecture emits the canonical blocks ``cross`` (S->X) and ``self``
(X->X), so ``dag_metrics.json`` / ``learned_dag_edges.json`` and their
downstream consumers (eval_seed_sweep, manifest) are identical across
architectures.  Blocks a model does not own are simply absent, and MEC
metrics are skipped rather than computed on a half-assembled DAG.

Because extraction goes through ``predict_test_from_ckpt``, this module imports
no model class of its own.  The former ``eval_attention_evolution`` /
``load_attention_evolution`` (per-epoch attention drift plots) lived here too;
they were retired to ``_OLD/`` together with ``eval_plot_lib``.
"""

import json
from os.path import join, isdir
from os import listdir
from collections import defaultdict

import numpy as np
from omegaconf import OmegaConf

# Import shared utilities
from .helpers.eval_utils import (
    root_path,
    _setup_eval_directories,
    _save_readme,
    _save_variable_labels,
    _create_cline_template,
    _load_true_dag_mask,
    load_dataset_metadata,
)

# Import from project modules
from causaliT.evaluation.predict import predict_test_from_ckpt

# Import from the local support layer
from .helpers.eval_lib import (
    get_architecture_type,
    find_best_or_last_checkpoint,
)
from .eval_interventions import infer_checkpoint_type

# Architecture-agnostic DAG query + shared metric core
from .helpers.eval_dag_query import query_dag_blocks, describe_topology
from .helpers.eval_dag_scores import compute_dag_metrics, make_json_serializable


# =============================================================================
# Helpers for portable per-edge DAG export
# =============================================================================


def _write_learned_dag_edges_json(
    eval_path_files: str,
    per_fold_comparison_data: list,
    metadata: dict,
    dataset_name: str,
    architecture: str,
    filename: str = "learned_dag_edges.json",
) -> None:
    """
    Persist per-fold learned DAG matrices in a portable JSON format
    suitable for cross-seed aggregation and aggregate-DAG plotting.

    For each evaluated attention block, we record:
        - the true DAG mask
        - the per-fold learned probability matrix
        - the fold-mean and fold-std matrices
        - row/column variable labels (from dataset metadata)

    No architecture-specific knowledge is required to consume this file -
    it's just nested numeric arrays + labels.
    """
    if not per_fold_comparison_data:
        return

    # --- Variable labels from dataset metadata ---------------------------
    var_info = (metadata or {}).get("variable_info", {}) or {}
    src_labels = list(var_info.get("source_labels", []))
    inp_labels = list(var_info.get("input_labels", []))

    def _labels_for(mask_type: str, n_rows: int, n_cols: int):
        """
        Cross-attention: rows = targets (X), cols = sources (S).
        Self-attention:  rows = cols    = targets (X).
        Falls back to row{i}/col{j} when metadata is incomplete.
        """
        is_cross = "cross" in (mask_type or "")
        rows = (
            inp_labels[:n_rows] if len(inp_labels) >= n_rows
            else [f"row{i}" for i in range(n_rows)]
        )
        if is_cross:
            cols = (
                src_labels[:n_cols] if len(src_labels) >= n_cols
                else [f"col{j}" for j in range(n_cols)]
            )
        else:
            cols = (
                inp_labels[:n_cols] if len(inp_labels) >= n_cols
                else [f"col{j}" for j in range(n_cols)]
            )
        return rows, cols

    # --- Group per-fold entries by block (att_key) -----------------------
    blocks_data = defaultdict(list)
    for entry in per_fold_comparison_data:
        blocks_data[entry["block"]].append(entry)

    blocks_out = {}
    for block_name, fold_entries in blocks_data.items():
        first = fold_entries[0]
        true_dag = np.asarray(first["true"])
        mask_type = first.get("mask_type", "")
        n_rows, n_cols = true_dag.shape
        rows, cols = _labels_for(mask_type, n_rows, n_cols)

        learned_per_fold = {}
        learned_stack = []
        for entry in fold_entries:
            arr = np.asarray(entry["learned"])
            if arr.shape != true_dag.shape:
                continue
            learned_per_fold[entry["fold_name"]] = arr.tolist()
            learned_stack.append(arr)

        if not learned_stack:
            continue

        learned_stack = np.stack(learned_stack, axis=0)
        learned_mean = learned_stack.mean(axis=0)
        learned_std = (
            learned_stack.std(axis=0)
            if learned_stack.shape[0] > 1
            else np.zeros_like(learned_mean)
        )

        blocks_out[block_name] = {
            "att_key": block_name,
            "mask_type": mask_type,
            "source": first.get("source"),
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "row_labels": rows,
            "col_labels": cols,
            "true": true_dag.astype(int).tolist(),
            "learned_mean": learned_mean.tolist(),
            "learned_std": learned_std.tolist(),
            "learned_per_fold": learned_per_fold,
        }

    payload = {
        "dataset": dataset_name,
        "architecture": architecture,
        "blocks": blocks_out,
    }

    out_path = join(eval_path_files, filename)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {filename}")


def _resolve_dag_dims(
    config,
    metadata: dict,
    datadir_path: str,
    dataset_name: str,
):
    """
    Resolve ``(L_S, L_X)`` - the number of source and intermediate variables.

    These two numbers are all ``query_dag_blocks`` needs to classify attention
    tensors, so they are read from the data side rather than from the model.
    Three independent sources are tried, in order:

    1. dataset metadata (``variable_info.source_labels`` / ``input_labels``),
    2. the experiment config (``data.S_seq_len`` / ``data.X_seq_len``),
    3. the shape of the ``dec_cross`` ground-truth mask, which is ``(L_X, L_S)``.

    Returns:
        ``(L_S, L_X, origin)`` where *origin* names the source used, for logging.

    Raises:
        ValueError: if neither dimension can be determined - continuing would
            silently mis-classify every attention block.
    """
    var_info = (metadata or {}).get("variable_info", {}) or {}
    L_S = len(var_info.get("source_labels") or []) or None
    L_X = len(var_info.get("input_labels") or []) or None
    if L_S and L_X:
        return int(L_S), int(L_X), "dataset metadata"

    data_cfg = config.get("data", {}) or {}
    L_S = L_S or data_cfg.get("S_seq_len")
    L_X = L_X or data_cfg.get("X_seq_len")
    if L_S and L_X:
        return int(L_S), int(L_X), "config (data.S_seq_len/X_seq_len)"

    cross_mask = _load_true_dag_mask(datadir_path, dataset_name, "dec_cross")
    if cross_mask is not None and cross_mask.ndim == 2:
        L_X = L_X or int(cross_mask.shape[0])
        L_S = L_S or int(cross_mask.shape[1])
        if L_S and L_X:
            return int(L_S), int(L_X), "dec_cross ground-truth mask"

    raise ValueError(
        f"Could not determine L_S/L_X for dataset '{dataset_name}' "
        f"(got L_S={L_S}, L_X={L_X}). Checked dataset metadata, "
        "config.data.S_seq_len/X_seq_len and the dec_cross mask."
    )


def _extract_dag_blocks(
    experiment: str,
    config,
    ckpt_type: str,
    datadir_path: str,
    L_S: int,
    L_X: int,
) -> dict:
    """
    Extract the canonical DAG blocks of every fold - the one extraction path.

    Workflow (one fold at a time):
        best checkpoint -> predict_test_from_ckpt (forward pass over the test
        split) -> query_dag_blocks (shape-based classification)

    This works for every architecture: ``predict_test_from_ckpt`` dispatches to
    the right predictor, and ``query_dag_blocks`` interprets whatever attention
    layout that predictor returns.

    Returns:
        ``{fold_name: {block_name: matrix}}``.  A fold that fails to load
        contributes an empty dict, so the remaining folds are still evaluated.
    """
    fold_names = sorted(
        d for d in listdir(experiment)
        if d.startswith("k_") and isdir(join(experiment, d))
    )
    if not fold_names:
        raise ValueError(f"No k_* fold directories found in {experiment}")
    print(f"  Found {len(fold_names)} fold(s): {fold_names}")

    per_fold_blocks = {}

    for fold_name in fold_names:
        print(f"\n  --- {fold_name} ---")
        per_fold_blocks[fold_name] = {}

        try:
            ckpt_path = find_best_or_last_checkpoint(
                join(experiment, fold_name, "checkpoints"), checkpoint_type=ckpt_type
            )
            print(f"    Checkpoint: {ckpt_path}")

            results = predict_test_from_ckpt(
                config=config,
                datadir_path=datadir_path,
                checkpoint_path=ckpt_path,
                dataset_label="test",
                cluster=False,
            )

            attention = getattr(results, "attention_weights", None)
            if not attention:
                print("    No attention weights returned by the predictor.")
                continue

            blocks = query_dag_blocks(attention, L_S=L_S, L_X=L_X)
            print(f"    {describe_topology(blocks, L_S, L_X)}")
            if not blocks:
                print("    No DAG blocks could be queried from this fold.")
                continue

            per_fold_blocks[fold_name] = blocks
            for name, matrix in sorted(blocks.items()):
                print(
                    f"    {name}: shape={matrix.shape} "
                    f"range=[{matrix.min():.3f}, {matrix.max():.3f}] "
                    f"mean={matrix.mean():.3f}"
                )

        except Exception as e:
            print(f"    [FAIL] {fold_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return per_fold_blocks


# =============================================================================
# Final Attention Analysis (from best checkpoint)
# =============================================================================

def eval_attention_scores(experiment: str, show_plots: bool = True) -> dict:
    """
    Evaluate final attention scores and DAG recovery metrics.

    Only the best checkpoint of each fold is loaded (~10-30 seconds).

    This function:
    - Loads attention weights from the best checkpoint of each k-fold
    - Queries the canonical DAG blocks (``cross`` = S->X, ``self`` = X->X),
      whichever the model's attention modules provide
    - Computes DAG recovery metrics (soft Hamming, standard SHD, zeroness,
      DAG confidence, MEC) and writes them as quantitative tables

    One code path serves every architecture: blocks are identified by the shape
    of the attention tensors (see ``eval_dag_query``), not by an architecture
    registry.  MEC metrics require both a ``cross`` and a ``self`` block and are
    skipped for cross-attention-only models.

    No figures are produced: this evaluation returns numbers only.  Plotting is
    the responsibility of the notebooks/report layer, which reads
    ``dag_metrics.json`` and ``learned_dag_edges.json``.

    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        show_plots: Deprecated and ignored; kept for call-site compatibility.

    Returns:
        dict: DAG recovery metrics with keys:
            - soft_hamming_cross: Soft Hamming distance for S->X edges (best/mean/worst/std/per_fold)
            - soft_hamming_self: Soft Hamming distance for X->X edges (best/mean/worst/std/per_fold)
            - dag_confidence_cross: DAG consistency across folds for S->X (1=identical, 0=max disagreement)
            - dag_confidence_self: DAG consistency across folds for X->X (1=identical, 0=max disagreement)
            - mec_distance: MEC distance metrics (only when both blocks exist)

    Output Files:
        - files/dag_metrics.json: DAG recovery metrics (soft Hamming + SHD + MEC + dag_confidence)
        - files/learned_dag_edges.json: Per-fold learned DAG matrices + true mask + labels
        - files/attention_labels.json: Descriptions of the canonical DAG blocks

    Example:
        >>> metrics = eval_attention_scores("experiments/single/local/my_experiment")
        >>> print(f"Soft Hamming (cross): {metrics['soft_hamming_cross']['mean']:.4f}")
        >>> print(f"DAG Confidence (cross): {metrics['dag_confidence_cross']:.4f}")
    """
    # Setup directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_attention_scores")

    dag_metrics_filename = "dag_metrics.json"
    attention_labels_filename = "attention_labels.json"
    learned_dag_edges_filename = "learned_dag_edges.json"

    print(f"Experiment ID: {exp_id}")

    # =========================================================================
    # Load dataset metadata for variable mappings
    # =========================================================================
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    if not config_files:
        raise ValueError(f"No config file found in {experiment}")

    config = OmegaConf.load(join(experiment, config_files[0]))
    dataset_name = config.get("data", {}).get("dataset")

    if not dataset_name:
        raise ValueError("No dataset specified in experiment config.")

    datadir_path = join(root_path, "data")
    metadata = load_dataset_metadata(datadir_path, dataset_name)

    if not metadata:
        raise ValueError(f"Dataset metadata not found for '{dataset_name}'.")

    architecture = get_architecture_type(config)
    print(f"  Architecture: {architecture}")

    # =========================================================================
    # Determine checkpoint type (same logic as ATE evaluation)
    # Causal models -> best_causal, baselines -> best_reconstruction
    # =========================================================================
    ckpt_type = infer_checkpoint_type(config)
    print(f"  Checkpoint type: {ckpt_type}")

    print(f"  Dataset: {dataset_name}")

    # =========================================================================
    # Read DAG threshold from config (default 0.5)
    # =========================================================================
    dag_threshold = config.get("evaluation", {}).get("dag_threshold", 0.5)
    print(f"  DAG threshold: {dag_threshold}")

    # =========================================================================
    # Resolve the dimensions used to classify attention blocks
    # =========================================================================
    L_S, L_X, dims_origin = _resolve_dag_dims(
        config=config,
        metadata=metadata,
        datadir_path=datadir_path,
        dataset_name=dataset_name,
    )
    print(f"  Dimensions: L_S={L_S}, L_X={L_X} (from {dims_origin})")

    # =========================================================================
    # Build attention labels for AI interpretation
    # =========================================================================
    attention_labels = {
        "description": "Canonical DAG blocks queried from the model's attention weights",
        "dag_blocks": {
            "cross": f"S->X edges, shape ({L_X}, {L_S}): rows = intermediate variables (X), columns = source variables (S)",
            "self": f"X->X edges, shape ({L_X}, {L_X}): rows and columns = intermediate variables (X)",
        },
        "block_naming": (
            "Multi-layer models append _L{i} (e.g. cross_L0). Blocks the "
            "architecture does not provide are absent from dag_metrics.json."
        ),
        "dimensions": {"L_S": L_S, "L_X": L_X, "resolved_from": dims_origin},
        "dag_metrics": {
            "soft_hamming": "Mean absolute difference between learned and true DAG. 0 = perfect, 1 = inverted",
        },
        "dataset": dataset_name,
        "architecture": architecture,
    }

    # Add variable mapping from metadata
    if "variable_descriptions" in metadata:
        attention_labels["variable_mapping"] = metadata["variable_descriptions"]
    if "causal_structure" in metadata and "edges" in metadata["causal_structure"]:
        edges = metadata["causal_structure"]["edges"]
        edge_strs = [f"{src}->{tgt}" for src, tgt in edges]
        attention_labels["dag_structure"] = ", ".join(edge_strs)

    _save_variable_labels(eval_path_files, attention_labels, attention_labels_filename)

    # Save README
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="Attention scores evaluation (FAST): final checkpoint analysis and DAG recovery metrics.",
        files_info={
            dag_metrics_filename: "Soft Hamming distance and MEC metrics comparing learned DAG to true DAG (JSON)",
            attention_labels_filename: "Descriptions of the canonical DAG blocks and interpretation guide (JSON)",
            learned_dag_edges_filename: "Per-fold learned DAG probability matrices + true mask + variable labels (JSON, portable for cross-seed aggregation)",
        },
    )

    _create_cline_template(eval_path_cline, "eval_attention_scores", exp_id)

    # =========================================================================
    # Extract the learned DAG blocks of every fold
    # =========================================================================
    per_fold_blocks = _extract_dag_blocks(
        experiment=experiment,
        config=config,
        ckpt_type=ckpt_type,
        datadir_path=datadir_path,
        L_S=L_S,
        L_X=L_X,
    )

    # =========================================================================
    # Compute DAG Recovery Metrics (incl. MEC when both blocks are available)
    # =========================================================================
    print("\n--- Computing DAG Recovery Metrics ---")
    dag_metrics, per_fold_comparison_data = compute_dag_metrics(
        per_fold_blocks=per_fold_blocks,
        datadir_path=datadir_path,
        dataset_name=dataset_name,
        architecture=architecture,
        dag_threshold=dag_threshold,
        L_S=L_S,
        L_X=L_X,
        source="attention",
    )

    all_blocks = sorted({b for blocks in per_fold_blocks.values() for b in blocks})
    dag_metrics["attention_topology"] = {
        "blocks": all_blocks,
        "L_S": L_S,
        "L_X": L_X,
        "resolved_from": dims_origin,
    }

    with open(join(eval_path_files, dag_metrics_filename), "w") as f:
        json.dump(make_json_serializable(dag_metrics), f, indent=2)
    print(f"\n  Saved: {dag_metrics_filename}")

    # =========================================================================
    # Save portable per-edge learned DAG (for cross-seed aggregation in
    # eval_seed_sweep, and for plotting the aggregate DAG in the paper).
    # =========================================================================
    _write_learned_dag_edges_json(
        eval_path_files=eval_path_files,
        per_fold_comparison_data=per_fold_comparison_data,
        metadata=metadata,
        dataset_name=dataset_name,
        architecture=architecture,
        filename=learned_dag_edges_filename,
    )

    print("\n[OK] eval_attention_scores complete!")
    return dag_metrics

"""
Shared DAG-metric report writer.

This module holds the **model-free half** of ``eval_attention_scores``: given the
canonical DAG blocks of every fold it computes the DAG recovery metrics and
writes the quantitative artefacts.  Nothing here knows about checkpoints,
predictors or attention modules, so the very same code serves

    - SVFA / AttentionSelector and every other causaliT architecture
      (``eval_attention.eval_attention_scores`` extracts the blocks from a
      checkpoint, then calls ``write_dag_report``), and
    - the external benchmarks (NOTEARS, DAGMA, PC) in ``causaliT.benchmarks``,
      which turn an estimated adjacency into the same canonical blocks.

Because both paths go through this one function, ``dag_metrics.json`` and
``learned_dag_edges.json`` are produced by identical code with identical keys,
which is what makes model and benchmark numbers directly comparable and lets
``eval_seed_sweep`` aggregate them without any special-casing.

Entry points:
    ``resolve_dag_dims``  - determine ``(L_S, L_X)`` from data-side information
    ``write_dag_report``  - metrics + JSON artefacts for a set of per-fold blocks

Output files (under ``<experiment>/eval/<eval_name>/files/``):
    ``dag_metrics.json``        soft Hamming, SHD, zeroness, DAG confidence, MEC
    ``learned_dag_edges.json``  per-fold matrices + true mask + variable labels
    ``attention_labels.json``   description of the canonical blocks
"""

import json
from collections import defaultdict
from os.path import join
from typing import Dict, Optional, Tuple

import numpy as np

from .eval_dag_query import CROSS, SELF
from .eval_dag_scores import compute_dag_metrics, make_json_serializable
from .eval_utils import (
    _create_cline_template,
    _load_true_dag_mask,
    _save_readme,
    _save_variable_labels,
    _setup_eval_directories,
)

#: Canonical artefact names, kept in one place so producers and consumers agree.
DAG_METRICS_FILENAME = "dag_metrics.json"
ATTENTION_LABELS_FILENAME = "attention_labels.json"
LEARNED_DAG_EDGES_FILENAME = "learned_dag_edges.json"

#: Default evaluation folder.  Benchmarks reuse it on purpose: ``eval_seed_sweep``
#: scans ``eval/eval_attention_scores/files/``, so writing there keeps the
#: cross-seed aggregation identical for models and benchmarks.
DEFAULT_EVAL_NAME = "eval_attention_scores"


# =============================================================================
# Dimensions (L_S, L_X)
# =============================================================================

def resolve_dag_dims(
    config,
    metadata: dict,
    datadir_path: str,
    dataset_name: str,
) -> Tuple[int, int, str]:
    """
    Resolve ``(L_S, L_X)`` - the number of source and intermediate variables.

    These two numbers are all ``query_dag_blocks`` needs to classify a learned
    structure, so they are read from the data side rather than from the model.
    Three independent sources are tried, in order:

    1. dataset metadata (``variable_info.source_labels`` / ``input_labels``),
    2. the experiment config (``data.S_seq_len`` / ``data.X_seq_len``),
    3. the shape of the ``dec_cross`` ground-truth mask, which is ``(L_X, L_S)``.

    Args:
        config: Experiment config (OmegaConf or dict); may be ``None``.
        metadata: Dataset metadata dict (``dataset_metadata.json``).
        datadir_path: Data root holding ``<dataset_name>/``.
        dataset_name: Dataset folder name.

    Returns:
        ``(L_S, L_X, origin)`` where *origin* names the source used, for logging.

    Raises:
        ValueError: if neither dimension can be determined - continuing would
            silently mis-classify every block.
    """
    var_info = (metadata or {}).get("variable_info", {}) or {}
    L_S = len(var_info.get("source_labels") or []) or None
    L_X = len(var_info.get("input_labels") or []) or None
    if L_S and L_X:
        return int(L_S), int(L_X), "dataset metadata"

    data_cfg = (config.get("data", {}) if config is not None else {}) or {}
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


# =============================================================================
# Portable per-edge DAG export
# =============================================================================

def write_learned_dag_edges_json(
    eval_path_files: str,
    per_fold_comparison_data: list,
    metadata: dict,
    dataset_name: str,
    architecture: str,
    filename: str = LEARNED_DAG_EDGES_FILENAME,
) -> None:
    """
    Persist per-fold learned DAG matrices in a portable JSON format
    suitable for cross-seed aggregation and aggregate-DAG plotting.

    For each evaluated block, we record:
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


# =============================================================================
# The shared report writer
# =============================================================================

def build_block_labels(
    L_S: int,
    L_X: int,
    dataset_name: str,
    architecture: str,
    metadata: Optional[dict] = None,
    dims_origin: str = "",
    source: str = "attention",
) -> dict:
    """
    Build the ``attention_labels.json`` payload describing the canonical blocks.

    Kept separate so the description stays identical for models and benchmarks
    (only ``source`` and ``architecture`` differ).
    """
    labels = {
        "description": (
            "Canonical DAG blocks queried from the learned structure "
            f"(source: {source})"
        ),
        "dag_blocks": {
            CROSS: (
                f"S->X edges, shape ({L_X}, {L_S}): rows = intermediate "
                "variables (X), columns = source variables (S)"
            ),
            SELF: (
                f"X->X edges, shape ({L_X}, {L_X}): rows and columns = "
                "intermediate variables (X)"
            ),
        },
        "block_naming": (
            "Multi-layer models append _L{i} (e.g. cross_L0). Blocks the "
            "architecture does not provide are absent from dag_metrics.json."
        ),
        "dimensions": {"L_S": L_S, "L_X": L_X, "resolved_from": dims_origin},
        "dag_metrics": {
            "soft_hamming": (
                "Mean absolute difference between learned and true DAG. "
                "0 = perfect, 1 = inverted"
            ),
        },
        "dataset": dataset_name,
        "architecture": architecture,
        "source": source,
    }

    metadata = metadata or {}
    if "variable_descriptions" in metadata:
        labels["variable_mapping"] = metadata["variable_descriptions"]
    causal = metadata.get("causal_structure") or {}
    if "edges" in causal:
        labels["dag_structure"] = ", ".join(
            f"{src}->{tgt}" for src, tgt in causal["edges"]
        )
    return labels


def write_dag_report(
    experiment: str,
    per_fold_blocks: Dict[str, Dict[str, np.ndarray]],
    datadir_path: str,
    dataset_name: str,
    architecture: str,
    L_S: int,
    L_X: int,
    metadata: Optional[dict] = None,
    dag_threshold: float = 0.5,
    dims_origin: str = "",
    source: str = "attention",
    eval_name: str = DEFAULT_EVAL_NAME,
    description: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Compute DAG recovery metrics and write the quantitative artefacts.

    This is the shared export path: every producer of canonical DAG blocks
    (causaliT models via ``eval_attention_scores``, external benchmarks via
    ``causaliT.benchmarks.runner``) funnels through here, so the resulting
    metrics files are structurally identical and directly comparable.

    Args:
        experiment: Run folder; artefacts go to ``<experiment>/eval/<eval_name>/``.
        per_fold_blocks: ``{fold_name: {block_name: matrix}}`` where block names
            are canonical (``cross``, ``self``, ``cross_L0``, ...).  A model or
            method that owns only one block simply omits the other; metrics that
            need both (MEC) are then skipped instead of computed on a
            half-assembled DAG.
        datadir_path: Data root holding ``<dataset_name>/`` with the true masks.
        dataset_name: Dataset folder name.
        architecture: Label written into the artefacts (e.g. ``atsel``,
            ``dagma_mlp``); used by the notebooks / manifest to group runs.
        L_S, L_X: Number of source / intermediate variables.
        metadata: Dataset metadata; only used for variable labels.
        dag_threshold: Binarisation threshold for the integer metrics.
        dims_origin: Where ``L_S``/``L_X`` came from (logging only).
        source: Provenance tag stored in the artefacts (``attention`` for
            models, the method name for benchmarks).
        eval_name: Evaluation subfolder.  Defaults to ``eval_attention_scores``
            so ``eval_seed_sweep`` finds the files unchanged.
        description: README description; a sensible default is used when None.
        verbose: Print progress.

    Returns:
        The ``dag_metrics`` dict (also written to ``dag_metrics.json``).
    """
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, eval_name)

    if verbose:
        print(f"Experiment ID: {exp_id}")
        print(f"  Architecture: {architecture}")
        print(f"  Dataset: {dataset_name}")
        print(f"  DAG threshold: {dag_threshold}")
        print(f"  Dimensions: L_S={L_S}, L_X={L_X} ({dims_origin})")

    # --- Block descriptions + README + cline template --------------------
    attention_labels = build_block_labels(
        L_S=L_S,
        L_X=L_X,
        dataset_name=dataset_name,
        architecture=architecture,
        metadata=metadata,
        dims_origin=dims_origin,
        source=source,
    )
    _save_variable_labels(eval_path_files, attention_labels, ATTENTION_LABELS_FILENAME)

    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description=description or (
            "DAG recovery metrics computed from the learned structure "
            f"(source: {source}). Tables only, no figures."
        ),
        files_info={
            DAG_METRICS_FILENAME: (
                "Soft Hamming distance, SHD and MEC metrics comparing the "
                "learned DAG to the true DAG (JSON)"
            ),
            ATTENTION_LABELS_FILENAME: (
                "Descriptions of the canonical DAG blocks and interpretation "
                "guide (JSON)"
            ),
            LEARNED_DAG_EDGES_FILENAME: (
                "Per-fold learned DAG matrices + true mask + variable labels "
                "(JSON, portable for cross-seed aggregation)"
            ),
        },
    )
    _create_cline_template(eval_path_cline, eval_name, exp_id)

    # --- Metrics ---------------------------------------------------------
    if verbose:
        print("\n--- Computing DAG Recovery Metrics ---")

    dag_metrics, per_fold_comparison_data = compute_dag_metrics(
        per_fold_blocks=per_fold_blocks,
        datadir_path=datadir_path,
        dataset_name=dataset_name,
        architecture=architecture,
        dag_threshold=dag_threshold,
        L_S=L_S,
        L_X=L_X,
        source=source,
        verbose=verbose,
    )

    all_blocks = sorted({b for blocks in per_fold_blocks.values() for b in blocks})
    dag_metrics["attention_topology"] = {
        "blocks": all_blocks,
        "L_S": int(L_S),
        "L_X": int(L_X),
        "resolved_from": dims_origin,
    }

    with open(join(eval_path_files, DAG_METRICS_FILENAME), "w") as f:
        json.dump(make_json_serializable(dag_metrics), f, indent=2)
    if verbose:
        print(f"\n  Saved: {DAG_METRICS_FILENAME}")

    # --- Portable per-edge export (cross-seed aggregation, plotting) ------
    write_learned_dag_edges_json(
        eval_path_files=eval_path_files,
        per_fold_comparison_data=per_fold_comparison_data,
        metadata=metadata or {},
        dataset_name=dataset_name,
        architecture=architecture,
        filename=LEARNED_DAG_EDGES_FILENAME,
    )

    return dag_metrics


__all__ = [
    "DAG_METRICS_FILENAME",
    "ATTENTION_LABELS_FILENAME",
    "LEARNED_DAG_EDGES_FILENAME",
    "DEFAULT_EVAL_NAME",
    "resolve_dag_dims",
    "write_learned_dag_edges_json",
    "build_block_labels",
    "write_dag_report",
]

"""
DAG recovery evaluation from attention scores.

This module has a single entry point, ``eval_attention_scores``:
   - Loads attention from the best checkpoint of each fold
   - Computes DAG recovery metrics (soft Hamming, SHD, zeroness, DAG
     confidence, MEC)
   - Writes quantitative artefacts only (JSON tables); no figures

It serves every architecture through one extraction path:

       best checkpoint -> predict_test_from_ckpt -> query_dag_blocks
                       -> write_dag_report

``eval_dag_query.query_dag_blocks`` classifies each attention tensor by its
shape, given ``L_S`` and ``L_X``.  The same code therefore handles per-key
architectures (SingleCausal, NoiseAware, StageCausal, proT: ``dec_cross``,
``dec_self``, ``dec_cross_L0``, ...) and the AttentionSelectorLayer's single
combined ``(L_X, L_S + L_X)`` block (or the homogeneous square ``(N, N)``
posterior), with or without a dedicated self-attention module.

Every architecture emits the canonical blocks ``cross`` (S->X) and ``self``
(X->X), so ``dag_metrics.json`` / ``learned_dag_edges.json`` and their
downstream consumers (eval_seed_sweep, manifest) are identical across
architectures.  Blocks a model does not own are simply absent, and MEC
metrics are skipped rather than computed on a half-assembled DAG.

**Separation of concerns.**  This module only *extracts* the blocks; the metric
computation and artefact writing live in ``helpers.eval_dag_report``, which is
model-free and shared with the external benchmarks (NOTEARS / DAGMA / PC in
``causaliT.benchmarks``).  Both paths therefore emit byte-compatible metrics
files, which is what makes model-vs-benchmark comparisons meaningful.

Because extraction goes through ``predict_test_from_ckpt``, this module imports
no model class of its own.  The former ``eval_attention_evolution`` /
``load_attention_evolution`` (per-epoch attention drift plots) lived here too;
they were retired to ``_OLD/`` together with ``eval_plot_lib``.
"""

from os import listdir
from os.path import isdir, join

from omegaconf import OmegaConf

# Import shared utilities
from .helpers.eval_utils import load_dataset_metadata

# Import from project modules
from causaliT.evaluation.predict import predict_test_from_ckpt

# Import from the local support layer
from .helpers.eval_lib import (
    find_best_or_last_checkpoint,
    get_architecture_type,
)
from .eval_interventions import infer_checkpoint_type

# Architecture-agnostic DAG query + shared metric/report core
from .helpers.datadir import resolve_datadir
from .helpers.eval_dag_query import describe_topology, query_dag_blocks
from .helpers.eval_dag_report import resolve_dag_dims, write_dag_report


# =============================================================================
# Extraction: attention -> canonical DAG blocks (the model-specific half)
# =============================================================================

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
    - Delegates to ``helpers.eval_dag_report.write_dag_report``, which computes
      the DAG recovery metrics (soft Hamming, standard SHD, zeroness, DAG
      confidence, MEC) and writes them as quantitative tables

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

    Output Files (under ``eval/eval_attention_scores/files/``):
        - dag_metrics.json: DAG recovery metrics (soft Hamming + SHD + MEC + dag_confidence)
        - learned_dag_edges.json: Per-fold learned DAG matrices + true mask + labels
        - attention_labels.json: Descriptions of the canonical DAG blocks

    Example:
        >>> metrics = eval_attention_scores("experiments/single/local/my_experiment")
        >>> print(f"Soft Hamming (cross): {metrics['soft_hamming_cross']['mean']:.4f}")
        >>> print(f"DAG Confidence (cross): {metrics['dag_confidence_cross']:.4f}")
    """
    # =========================================================================
    # Load config + dataset metadata
    # =========================================================================
    config_files = [
        f for f in listdir(experiment)
        if f.startswith("config") and f.endswith(".yaml")
    ]
    if not config_files:
        raise ValueError(f"No config file found in {experiment}")

    config = OmegaConf.load(join(experiment, config_files[0]))
    dataset_name = config.get("data", {}).get("dataset")

    if not dataset_name:
        raise ValueError("No dataset specified in experiment config.")

    # DAG-sweep runs keep their datasets inside the experiment folder, so the
    # data root comes from the run's config (falling back to <repo>/data).
    datadir_path = resolve_datadir(config=config, experiment=experiment)
    metadata = load_dataset_metadata(datadir_path, dataset_name)

    if not metadata:
        raise ValueError(f"Dataset metadata not found for '{dataset_name}'.")

    architecture = get_architecture_type(config)

    # =========================================================================
    # Determine checkpoint type (same logic as ATE evaluation)
    # Causal models -> best_causal, baselines -> best_reconstruction
    # =========================================================================
    ckpt_type = infer_checkpoint_type(config)
    print(f"  Checkpoint type: {ckpt_type}")

    # DAG threshold for the integer metrics (default 0.5)
    dag_threshold = config.get("evaluation", {}).get("dag_threshold", 0.5)

    # =========================================================================
    # Resolve the dimensions used to classify attention blocks
    # =========================================================================
    L_S, L_X, dims_origin = resolve_dag_dims(
        config=config,
        metadata=metadata,
        datadir_path=datadir_path,
        dataset_name=dataset_name,
    )

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
    # Metrics + artefacts (shared with causaliT.benchmarks)
    # =========================================================================
    dag_metrics = write_dag_report(
        experiment=experiment,
        per_fold_blocks=per_fold_blocks,
        datadir_path=datadir_path,
        dataset_name=dataset_name,
        architecture=architecture,
        L_S=L_S,
        L_X=L_X,
        metadata=metadata,
        dag_threshold=dag_threshold,
        dims_origin=dims_origin,
        source="attention",
        description=(
            "Attention scores evaluation (FAST): final checkpoint analysis "
            "and DAG recovery metrics."
        ),
    )

    print("\n[OK] eval_attention_scores complete!")
    return dag_metrics

"""
Evaluation Functions for CausaliT Experiments - Self-Contained Package.

This subpackage contains modular evaluation functions organized by category.
All dependencies are local - can be moved to causaliT/evaluation when ready.

Modules:
    eval_utils: Shared utility functions (setup directories, DAG metrics, etc.)
    eval_lib: Data loading and model utilities (AttentionData, load_*, save_*, etc.)
    eval_plot_lib: Plotting functions (plot_attention_scores, plot_attention_evolution)
    eval_training: Training metrics evaluation (eval_train_metrics)
    eval_embeddings: Embedding evaluation (eval_embed, eval_embedding_dag_correlation)
    eval_attention: Attention scores and DAG recovery (eval_attention_scores, load_attention_evolution)
    eval_interventions: Intervention evaluation (eval_interventions)
    eval_ans: Attention Necessity Score (ANS) evaluation for sweep experiments
    update_manifest: Manifest update functions (update_experiments_manifest, etc.)
    eval_funs_wraps: Evaluation wrappers (run_all_evaluations, run_evaluations_from_config)
    eval_dyconex: Dyconex-specific evaluation functions
"""

# Utility functions
from .eval_utils import (
    root_path,
    _setup_eval_directories,
    _save_readme,
    _save_variable_labels,
    _create_cline_template,
    find_all_checkpoints,
    _select_evenly_spaced_checkpoints,
    _compute_soft_hamming,
    _load_true_dag_mask,
    load_dataset_metadata,  # NEW: Load dataset metadata for dataset-agnostic evaluation
    _get_learned_dag,
    _compute_dag_confidence,
    _get_learned_dag_per_fold,
    # Metric plotting helpers
    _should_use_log_scale,
    _is_column_plottable,
    _plot_metric_pair,
    _discover_metric_pairs,
    _filter_metric_pairs,
    # Training instability metrics
    _compute_spike_ratio,
    _compute_coefficient_of_variation,
    _compute_max_jump,
    _compute_trend_instability,
    compute_instability_metrics,
    # Plot format configuration
    DEFAULT_PLOT_FORMAT,
)

# Data loading and model utilities (self-contained library)
from .eval_lib import (
    AttentionData,
    find_config_file,
    find_best_or_last_checkpoint,
    get_architecture_type,
    extract_phi_from_model,
    load_attention_data,
    load_attention_data_from_file,
    save_attention_data,
    load_embeddings_evolution,
    load_training_metrics,
    predict_from_experiment,
    predictions_to_long_df,
)

# Plotting functions (self-contained library)
from .eval_plot_lib import (
    plot_attention_scores,
    plot_attention_evolution,
    plot_phi_evolution,
)

# Training metrics evaluation
from .eval_training import eval_train_metrics, compute_training_instability

# Embedding evaluation
from .eval_embeddings import (
    eval_embed,
    eval_embedding_dag_correlation,
)

# Attention/DAG evaluation
from .eval_attention import (
    load_attention_evolution,
    eval_attention_scores,
)

# Intervention evaluation
from .eval_interventions import eval_interventions

# ANS evaluation (sweep experiments only)
from .eval_ans import eval_ans

# Manifest functions
from .update_manifest import (
    fix_kfold_summary,
    enrich_kfold_summary,
    update_experiments_manifest,
    load_experiments_manifest,
    batch_update_manifest,
    MANIFEST_PATH,
)

# Evaluation wrappers
from .eval_funs_wraps import (
    run_all_evaluations,
    run_evaluations_from_config,
)

# Dyconex-specific (optional import)
try:
    from .eval_dyconex import eval_dyconex_predictions
except ImportError:
    pass

__all__ = [
    # Utils
    "root_path",
    "_setup_eval_directories",
    "_save_readme",
    "_save_variable_labels",
    "_create_cline_template",
    "find_all_checkpoints",
    "_select_evenly_spaced_checkpoints",
    "_compute_soft_hamming",
    "_load_true_dag_mask",
    "load_dataset_metadata",
    "_get_learned_dag",
    "_compute_dag_confidence",
    "_get_learned_dag_per_fold",
    # Metric plotting helpers
    "_should_use_log_scale",
    "_is_column_plottable",
    "_plot_metric_pair",
    "_discover_metric_pairs",
    "_filter_metric_pairs",
    # Data loading (eval_lib)
    "AttentionData",
    "find_config_file",
    "find_best_or_last_checkpoint",
    "get_architecture_type",
    "extract_phi_from_model",
    "load_attention_data",
    "load_attention_data_from_file",
    "save_attention_data",
    "load_embeddings_evolution",
    "load_training_metrics",
    "predict_from_experiment",
    "predictions_to_long_df",
    # Plotting (eval_plot_lib)
    "plot_attention_scores",
    "plot_attention_evolution",
    "plot_phi_evolution",
    # Training instability metrics
    "_compute_spike_ratio",
    "_compute_coefficient_of_variation",
    "_compute_max_jump",
    "_compute_trend_instability",
    "compute_instability_metrics",
    "compute_training_instability",
    # Training
    "eval_train_metrics",
    # Embeddings
    "eval_embed",
    "eval_embedding_dag_correlation",
    # Attention
    "load_attention_evolution",
    "eval_attention_scores",
    # Interventions
    "eval_interventions",
    # ANS (sweep experiments only)
    "eval_ans",
    # Manifest
    "fix_kfold_summary",
    "enrich_kfold_summary",
    "update_experiments_manifest",
    "load_experiments_manifest",
    "batch_update_manifest",
    "MANIFEST_PATH",
    # Wrappers
    "run_all_evaluations",
    "run_evaluations_from_config",
    # Dyconex
    "eval_dyconex_predictions",
    # Plot format configuration
    "DEFAULT_PLOT_FORMAT",
]

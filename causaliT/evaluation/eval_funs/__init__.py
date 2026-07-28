"""
Evaluation Functions for CausaliT Experiments.

Scope: **DAG recovery** and **causal interventions**.  Everything else that used
to live here has been retired to ``_OLD/`` (see the bottom of this docstring).

Entry points (this level):
    eval_attention: DAG recovery from attention scores (eval_attention_scores)
    eval_interventions: Intervention / ATE evaluation (eval_interventions)
    eval_seed_sweep: Cross-seed aggregation of DAG + ATE metrics (paper reporting)
    eval_funs_wraps: Dispatchers (run_all_evaluations, run_evaluations_from_config)
    eval_fun_cli: Command-line interface

Support modules (``helpers/``, see helpers/__init__.py):
    helpers.eval_utils: Shared utilities (eval directories, true-DAG masks, metric helpers)
    helpers.eval_lib: Checkpoint/config discovery
    helpers.eval_dag_query: Model-free, shape-based extraction of the canonical DAG blocks
    helpers.eval_dag_scores: Shared metric core (soft Hamming, SHD, DAG confidence, MEC)

Also still in ``_OLD/``: update_manifest (manifest and kfold_summary maintenance),
which the CLI and wrappers continue to import from there.

Retired to ``_OLD/`` and no longer importable from this package:
``eval_train_metrics`` (eval_training), ``eval_embed`` /
``eval_embedding_dag_correlation`` (eval_embeddings), ``eval_ans``,
``eval_anm_residual_hsic``, ``eval_d_model_sweep``, ``eval_dyconex_predictions``
and the plotting library ``eval_plot_lib`` - together with the per-epoch
``eval_attention_evolution`` / ``load_attention_evolution`` pair that used it.
"""

# Utility functions
from .helpers.eval_utils import (
    root_path,
    _setup_eval_directories,
    _save_readme,
    _save_variable_labels,
    _create_cline_template,
    _compute_soft_hamming,
    _compute_standard_shd,
    _load_true_dag_mask,
    load_dataset_metadata,
    _compute_dag_confidence,
)

# Checkpoint/config discovery
from .helpers.eval_lib import (
    find_config_file,
    find_best_or_last_checkpoint,
    get_architecture_type,
    extract_phi_from_model,
)

# Architecture-agnostic DAG query (shape-based block classification)
from .helpers.eval_dag_query import (
    CROSS,
    SELF,
    canonical_block_name,
    query_dag_blocks,
    assemble_full_dag,
    describe_topology,
    block_axis_labels,
)

# Shared DAG metric core
from .helpers.eval_dag_scores import compute_dag_metrics, make_json_serializable

# Attention/DAG evaluation
from .eval_attention import eval_attention_scores

# AttentionSelector evaluation (deprecated alias of eval_attention_scores)
from ._OLD.eval_attention_selector import eval_attention_selector_scores

# Intervention evaluation
from .eval_interventions import eval_interventions

# Seed sweep evaluation (paper reporting)
from .eval_seed_sweep import eval_seed_sweep

# Manifest functions
from ._OLD.update_manifest import (
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

__all__ = [
    # Utils
    "root_path",
    "_setup_eval_directories",
    "_save_readme",
    "_save_variable_labels",
    "_create_cline_template",
    "_compute_soft_hamming",
    "_compute_standard_shd",
    "_load_true_dag_mask",
    "load_dataset_metadata",
    "_compute_dag_confidence",
    # Checkpoint / config discovery (helpers.eval_lib)
    "find_config_file",
    "find_best_or_last_checkpoint",
    "get_architecture_type",
    "extract_phi_from_model",
    # DAG query
    "CROSS",
    "SELF",
    "canonical_block_name",
    "query_dag_blocks",
    "assemble_full_dag",
    "describe_topology",
    "block_axis_labels",
    # DAG metrics
    "compute_dag_metrics",
    "make_json_serializable",
    # Attention / DAG evaluation
    "eval_attention_scores",
    "eval_attention_selector_scores",
    # Interventions
    "eval_interventions",
    # Seed sweep evaluation (paper reporting)
    "eval_seed_sweep",
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
]

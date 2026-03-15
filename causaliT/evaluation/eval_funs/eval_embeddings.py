"""
Embedding evaluation functions for CausaliT experiments.

This module provides functions for analyzing learned embeddings:
- eval_embed: Analyze embedding evolution and cosine similarities
- eval_embedding_dag_correlation: Correlate embedding similarity with DAG structure
"""

import json
from functools import partial
from os.path import join, exists
from os import listdir
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from omegaconf import OmegaConf
from matplotlib.patches import Patch

# Import shared utilities
from .eval_utils import (
    root_path,
    _setup_eval_directories,
    _save_readme,
    _save_variable_labels,
    _create_cline_template,
    _get_learned_dag,
    _load_true_dag_mask,
    load_dataset_metadata,
    DEFAULT_PLOT_FORMAT,
)

# Import from local eval_funs modules (self-contained)
from .eval_lib import (
    load_embeddings_evolution,
    load_attention_data_from_file,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _build_embedding_labels_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build embedding labels dictionary from dataset metadata.
    
    Args:
        metadata: Dataset metadata loaded from dataset_metadata.json
        
    Returns:
        Dict containing variable mappings for embedding analysis
    """
    source_labels = metadata["variable_info"]["source_labels"]
    input_labels = metadata["variable_info"]["input_labels"]
    
    # Build direct edges set for expected causal relations
    edges_set = set(tuple(e) for e in metadata["causal_structure"]["direct_edges"])
    
    # Generate cosine similarity pair descriptions
    cosine_similarity_pairs = {}
    expected_causal_relations = {}
    
    for s in source_labels:
        for x in input_labels:
            key = f"cos_{s}_{x}"
            cosine_similarity_pairs[key] = f"Cosine similarity between {s} and {x} embeddings"
            has_edge = (s, x) in edges_set
            expected_causal_relations[key] = (
                f"Should be high ({s} → {x} in true DAG)" if has_edge 
                else f"Should be low (no direct edge {s} → {x})"
            )
    
    # Add X-X pairs
    for i, x1 in enumerate(input_labels):
        for x2 in input_labels[i+1:]:
            key = f"cos_{x1}_{x2}"
            cosine_similarity_pairs[key] = f"Cosine similarity between {x1} and {x2} embeddings"
    
    return {
        "source_variables": source_labels,
        "intermediate_variables": input_labels,
        "target_variables": metadata["variable_info"].get("target_labels", []),
        "cosine_similarity_pairs": cosine_similarity_pairs,
        "expected_causal_relations": expected_causal_relations,
    }


def _generate_cosine_pairs(source_labels: List[str], input_labels: List[str]) -> List[str]:
    """
    Generate list of cosine similarity pair column names.
    
    Args:
        source_labels: List of source variable names (e.g., ["S1", "S2", "S3"])
        input_labels: List of input/intermediate variable names (e.g., ["X1", "X2"])
        
    Returns:
        List of column names like ["cos_S1_X1", "cos_S1_X2", ...]
    """
    var_pairs = []
    
    # S-X pairs
    for s in source_labels:
        for x in input_labels:
            var_pairs.append(f"cos_{s}_{x}")
    
    # X-X pairs (optional, only if multiple input variables)
    for i, x1 in enumerate(input_labels):
        for x2 in input_labels[i+1:]:
            var_pairs.append(f"cos_{x1}_{x2}")
    
    return var_pairs


def _build_cos_to_dag_map(
    source_labels: List[str], 
    input_labels: List[str]
) -> Dict[str, Tuple[int, int]]:
    """
    Build mapping from cosine similarity column names to DAG matrix indices.
    
    The DAG matrix for cross-attention (S→X) has:
    - Rows = input/intermediate variables (targets/queries)
    - Columns = source variables (keys)
    
    Args:
        source_labels: List of source variable names
        input_labels: List of input/intermediate variable names
        
    Returns:
        Dict mapping "cos_{S}_{X}" -> (target_idx, source_idx)
    """
    cos_to_dag_map = {}
    
    for target_idx, x in enumerate(input_labels):
        for source_idx, s in enumerate(source_labels):
            cos_to_dag_map[f"cos_{s}_{x}"] = (target_idx, source_idx)
    
    return cos_to_dag_map


# =============================================================================
# Evaluation Functions
# =============================================================================

def eval_embed(experiment: str, show_plots: bool = True) -> None:
    """
    Evaluate embedding evolution during training.
    
    Analyzes how source (S) and intermediate (X) embeddings evolve during training
    by computing cosine similarities between embedding vectors across epochs.
    This is specific to SingleCausalForecaster architecture with orthogonal embeddings.
    
    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        show_plots: If True (default), display plots interactively. If False, only save to files.
        
    Returns:
        None (results are saved to files and displayed)
        
    Output Files:
        - fig/cosine_similarities_kfold_{k}_{exp_id}.pdf: Cosine similarity evolution per fold
        - fig/final_cosine_similarities_{exp_id}.pdf: Final epoch similarities across folds
        - fig/cosine_similarities_correlation_matrix_{exp_id}.pdf: Correlation heatmap
        - files/emb_df.csv: Embedding data (cached for subsequent runs)
        - files/emb_sim_corr.csv: Cosine similarity correlation matrix
        
    Notes:
        - Computes cosine similarities between source (S) and intermediate (X) embeddings
        - Results are cached in files/ directory; delete to recompute
        - Requires dataset_metadata.json in the data folder for variable information
        
    Example:
        >>> eval_embed("../experiments/single/local/my_experiment")
    """
    # Setup directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_embedding_evolution")
    
    emb_dataframe_filename = "emb_df.csv"
    emb_sim_corr_filename = "emb_sim_corr.csv"
    embedding_labels_filename = "embedding_labels.json"

    # =========================================================================
    # Load dataset metadata (NO FALLBACK - requires metadata)
    # =========================================================================
    
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    dataset_name = None
    if config_files:
        try:
            config = OmegaConf.load(join(experiment, config_files[0]))
            dataset_name = config.get("data", {}).get("dataset")
        except Exception:
            pass
    
    if not dataset_name:
        raise ValueError(
            f"No dataset specified in experiment config. "
            f"Cannot load variable mappings for embedding analysis."
        )
    
    datadir_path = join(root_path, "data")
    metadata = load_dataset_metadata(datadir_path, dataset_name)
    
    if not metadata:
        raise ValueError(
            f"Dataset metadata not found for '{dataset_name}'. "
            f"Ensure dataset_metadata.json exists in data/{dataset_name}/"
        )
    
    # Extract variable information from metadata
    source_labels = metadata["variable_info"]["source_labels"]
    input_labels = metadata["variable_info"]["input_labels"]
    n_source = metadata["variable_info"]["n_source"]
    n_input = metadata["variable_info"]["n_input"]
    
    print(f"  Loaded metadata for dataset: {dataset_name}")
    print(f"  Source variables ({n_source}): {source_labels}")
    print(f"  Input variables ({n_input}): {input_labels}")

    # =========================================================================
    # Build embedding labels from metadata
    # =========================================================================
    
    variable_mapping = _build_embedding_labels_from_metadata(metadata)
    
    embedding_labels = {
        "description": "Embedding evolution analysis - tracking cosine similarities between variable embeddings during training",
        "interpretation": {
            "high_cosine_similarity": "Variables with similar embeddings are expected to have causal relationships",
            "low_cosine_similarity": "Variables with orthogonal embeddings should not have direct causal edges",
            "evolution_over_epochs": "Embeddings typically start random and converge to reflect causal structure",
        },
        "embedding_types": {
            "embedding_S": "Source variable embeddings (OrthogonalMaskEmbedding for SingleCausal)",
            "embedding_X": "Intermediate variable embeddings (ModularEmbedding)",
        },
        "column_documentation": {
            "kfold": "Cross-validation fold identifier (k_0, k_1, ...)",
            "epoch": "Training epoch number",
            **{s: f"Embedding vector for source variable {s} (flattened)" for s in source_labels},
            **{x: f"Embedding vector for intermediate variable {x} (flattened)" for x in input_labels},
            "cos_*_*": "Cosine similarity between two variable embeddings",
        },
        "variable_mapping": variable_mapping,
        "dataset": dataset_name,
    }
    
    _save_variable_labels(eval_path_files, embedding_labels, embedding_labels_filename)

    # Save README with column documentation
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="This evaluation folder contains evaluations of the learned embeddings.",
        files_info={
            emb_dataframe_filename: "Embedding data with cosine similarities (CSV)",
            emb_sim_corr_filename: "Correlation matrix of cosine similarities (CSV)",
            embedding_labels_filename: "Variable descriptions and interpretation guide (JSON)",
        },
        column_documentation=embedding_labels["column_documentation"]
    )
    
    # Create cline notes template
    _create_cline_template(eval_path_cline, "eval_embedding_evolution", exp_id)

    print(f"Experiment ID: {exp_id}")

    # =========================================================================
    # Helper functions for embedding processing
    # =========================================================================
    
    def _compute_srow(row, num_source_vars: int):
        """Compute source embedding vectors from orthogonal mask embedding components."""
        ab = np.asarray(row["embedding_S_value_embedding_weight"]) + \
             np.asarray(row["embedding_S_value_embedding_bias"])
        c = np.asarray(row["embedding_S_binary_masks"])
        L = ab.shape[0]

        if c.size != num_source_vars * L:
            raise ValueError(f"Expected c length {num_source_vars*L}, got {c.size}")

        c_reshaped = c.reshape(num_source_vars, L)
        return c_reshaped

    def _unpack(row, label, num_vars):
        """Unpack flattened weight array into per-variable vectors."""
        arr = np.asarray(row[label])
        L = arr.size // num_vars
        return tuple(arr.reshape(num_vars, L))

    def _rowwise_cosine(df, col1, col2, eps=1e-12):
        """Compute row-wise cosine similarity between two vector columns."""
        X = np.stack(df[col1].to_numpy())
        Y = np.stack(df[col2].to_numpy())
        num = np.einsum("ij,ij->i", X, Y)
        den = (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1)).clip(min=eps)
        return num / den

    # =========================================================================
    # Load or compute embedding data
    # =========================================================================
    
    if exists(join(eval_path_files, emb_dataframe_filename)):
        df_scm = pd.read_csv(join(eval_path_files, emb_dataframe_filename))
        print("Experiment already available. Data loaded!")
    else:
        df = load_embeddings_evolution(experiment)

        group_cols = ["kfold", "epoch"]
        source_emb_name_single = [
            "embedding_S_value_embedding_weight",
            "embedding_S_value_embedding_bias",
            "embedding_S_binary_masks"
        ]
        source_emb = df.set_index("embedding_name").loc[source_emb_name_single].reset_index()

        # Process source embeddings using n_source from metadata
        gS = (
            source_emb
            .groupby(group_cols + ["embedding_name"])["weight"]
            .first()
            .unstack("embedding_name")
        )
        gS["temp_res"] = gS.apply(lambda row: _compute_srow(row, n_source), axis=1)
        
        # Dynamically assign source variable columns from metadata
        gS[source_labels] = gS.apply(
            partial(_unpack, label="temp_res", num_vars=n_source), 
            axis=1, 
            result_type="expand"
        )
        gS = gS.drop(columns=source_emb_name_single + ["temp_res"])
        df_S = gS.reset_index()

        # Process intermediate X embeddings
        # Note: The embedding name and index extraction are architecture-specific
        # For SingleCausalForecaster with shared embedding, X variables start at index n_source + 1
        # The total num_vars in the shared embedding = n_source + n_input + n_target
        n_target = metadata["variable_info"].get("n_target", 0)
        total_vars_in_embedding = n_source + n_input + n_target
        
        # X variables are located after source variables in the shared embedding
        # Indices: 0...(n_source-1) = reserved/unused, n_source...(n_source+n_input-1) = X variables
        x_indices = list(range(n_source, n_source + n_input))
        
        df_X = df.set_index("embedding_name").loc['embedding_X_var1_nn_embedding_embedding_weight'].reset_index()
        
        # Extract X embedding vectors at the correct indices
        unpacked_cols = df_X.apply(
            partial(_unpack, label="weight", num_vars=total_vars_in_embedding), 
            axis=1, 
            result_type="expand"
        )
        
        # Select only the X variable columns and rename them
        df_X[input_labels] = unpacked_cols[x_indices]
        df_X = df_X.drop(columns=["embedding_name", "weight", "type", "shape", "component"])

        df_scm = pd.concat([df_S.set_index(group_cols), df_X.set_index(group_cols)], axis=1).reset_index()
        
        # Save the embedding dataframe
        df_scm.to_csv(join(eval_path_files, emb_dataframe_filename), index=False)
        print("Embedding data saved!")

    # =========================================================================
    # Calculate cosine similarities dynamically
    # =========================================================================
    
    # Generate cosine similarity pairs from metadata
    var_pairs = _generate_cosine_pairs(source_labels, input_labels)
    
    # Compute cosine similarities for S-X pairs
    for s in source_labels:
        for x in input_labels:
            col_name = f"cos_{s}_{x}"
            if col_name not in df_scm.columns:
                df_scm[col_name] = _rowwise_cosine(df_scm, s, x)
    
    # Compute cosine similarities for X-X pairs
    for i, x1 in enumerate(input_labels):
        for x2 in input_labels[i+1:]:
            col_name = f"cos_{x1}_{x2}"
            if col_name not in df_scm.columns:
                df_scm[col_name] = _rowwise_cosine(df_scm, x1, x2)
    
    # Save correlation matrix
    df_corr = df_scm[var_pairs].corr().abs()
    ranked = df_corr.unstack().sort_values(ascending=False)
    ranked[ranked < 1].to_csv(join(eval_path_files, emb_sim_corr_filename))
    
    # =========================================================================
    # Plots
    # =========================================================================
    
    # Plot: Cosine similarity evolution per fold
    group_cols = ["kfold", "epoch"]
    df_sim_plot = df_scm.melt(
        id_vars=group_cols, 
        value_vars=var_pairs, 
        var_name="variables", 
        value_name="cosine_similarity"
    )
    
    for k in df_sim_plot["kfold"].unique():
        fig, ax = plt.subplots()
        sns.lineplot(
            df_sim_plot.set_index("kfold").loc[k], 
            x="epoch", 
            y="cosine_similarity", 
            hue="variables", 
            ax=ax
        )
        plt.title(f"Fold: {k}")
        plt.savefig(join(eval_path_fig, f"cosine_similarities_kfold_{k}_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Plot: Final epoch similarities
    fig, ax = plt.subplots()
    max_epoch = max(df_sim_plot["epoch"].unique())
    sns.scatterplot(
        df_sim_plot[df_sim_plot["epoch"] == max_epoch], 
        y="variables", 
        x="cosine_similarity", 
        hue="kfold", 
        s=200, 
        alpha=0.6, 
        ax=ax
    )
    plt.savefig(join(eval_path_fig, f"final_cosine_similarities_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Plot: Correlation matrix heatmap
    fig, ax = plt.subplots()
    sns.heatmap(df_corr, ax=ax)
    plt.savefig(join(eval_path_fig, f"cosine_similarities_correlation_matrix_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    print(f"\nEvaluation complete! Results saved to: {eval_path_root}")


def eval_embedding_dag_correlation(experiment: str, show_plots: bool = True) -> dict:
    """
    Evaluate correlation between embedding similarity and learned DAG structure.
    
    This function tests H6: whether causally-linked variables have more similar
    embeddings than non-linked variables. It combines data from:
    - eval_embed: cosine similarities between variable embeddings
    - eval_attention_scores: learned DAG structure (phi or attention)
    
    For each variable pair (e.g., S1-X1), it correlates:
    - Embedding cosine similarity (from eval_embed)
    - Learned edge probability (from eval_attention_scores phi tensor)
    - True edge indicator (from dataset DAG masks)
    
    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        show_plots: If True (default), display plots interactively. If False, only save to files.
        
    Returns:
        dict: Correlation results with keys:
            - embedding_dag_correlation: Pearson correlation between embedding sim and learned DAG
            - embedding_true_dag_correlation: Correlation between embedding sim and true DAG
            - separation_score: Difference in mean embedding sim (causal vs non-causal pairs)
            
    Output Files:
        - fig/embedding_dag_scatter_{exp_id}.pdf: Scatter plot of embedding sim vs DAG prob
        - fig/embedding_separation_{exp_id}.pdf: Box plot comparing causal vs non-causal pairs
        - files/embedding_dag_correlation.json: Detailed correlation results
        - files/embedding_dag_data.csv: Raw data for further analysis
        
    Notes:
        - Requires eval_embed and eval_attention_scores to be run first
        - Requires dataset_metadata.json for variable information
        
    Example:
        >>> results = eval_embedding_dag_correlation("../experiments/single/local/my_experiment")
        >>> print(f"Embedding-DAG correlation: {results['embedding_dag_correlation']:.3f}")
    """
    # Setup directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_embedding_dag_correlation")
    
    correlation_filename = "embedding_dag_correlation.json"
    data_filename = "embedding_dag_data.csv"
    
    # =========================================================================
    # Load dataset metadata (NO FALLBACK - requires metadata)
    # =========================================================================
    
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    if not config_files:
        raise ValueError("No config file found in experiment")
    
    config = OmegaConf.load(join(experiment, config_files[0]))
    dataset_name = config.get("data", {}).get("dataset")
    
    if not dataset_name:
        raise ValueError(
            f"No dataset specified in experiment config. "
            f"Cannot load variable mappings."
        )
    
    datadir_path = join(root_path, "data")
    metadata = load_dataset_metadata(datadir_path, dataset_name)
    
    if not metadata:
        raise ValueError(
            f"Dataset metadata not found for '{dataset_name}'. "
            f"Ensure dataset_metadata.json exists in data/{dataset_name}/"
        )
    
    # Extract variable information from metadata
    source_labels = metadata["variable_info"]["source_labels"]
    input_labels = metadata["variable_info"]["input_labels"]
    
    print(f"  Loaded metadata for dataset: {dataset_name}")
    print(f"  Source variables: {source_labels}")
    print(f"  Input variables: {input_labels}")
    
    # =========================================================================
    # Load prerequisite data
    # =========================================================================
    
    # Check if eval_embed was run
    emb_data_path = join(experiment, "eval", "eval_embedding_evolution", "files", "emb_df.csv")
    if not exists(emb_data_path):
        print(f"Error: eval_embed must be run first. Missing: {emb_data_path}")
        print("Run: eval_embed(experiment)")
        return {}
    
    # Check if eval_attention_scores was run
    dag_metrics_path = join(experiment, "eval", "eval_attention_scores", "files", "dag_metrics.json")
    att_data_path = join(experiment, "eval", "eval_attention_scores", "files", "final_scores")
    if not exists(dag_metrics_path):
        print(f"Error: eval_attention_scores must be run first. Missing: {dag_metrics_path}")
        print("Run: eval_attention_scores(experiment)")
        return {}
    
    # Load embedding cosine similarities (final epoch, mean across folds)
    df_emb = pd.read_csv(emb_data_path)
    
    # Get final epoch data
    final_epoch = df_emb["epoch"].max()
    df_final = df_emb[df_emb["epoch"] == final_epoch]
    
    # Generate cosine similarity column names dynamically from metadata
    cos_cols = [f"cos_{s}_{x}" for s in source_labels for x in input_labels]
    
    # Compute mean cosine similarities across folds for final epoch
    mean_cos_sims = {}
    for col in cos_cols:
        if col in df_final.columns:
            mean_cos_sims[col] = df_final[col].mean()
    
    if not mean_cos_sims:
        print("Error: No cosine similarity columns found in embedding data")
        return {}
    
    print(f"Loaded embedding similarities from final epoch {final_epoch}")
    
    # Load learned DAG from attention scores
    with open(dag_metrics_path, 'r') as f:
        dag_metrics = json.load(f)
    
    # Load attention data to get phi values
    attention_data = load_attention_data_from_file(att_data_path)
    
    # Get architecture type
    architecture = attention_data.architecture_type
    
    # Extract learned DAG (phi or attention) - mean across folds
    # Architecture-specific phi_key/att_key mappings
    if architecture == "SingleCausalForecaster":
        phi_key = "decoder_cross"
        att_key = "dec_cross"
    elif architecture == "NoiseAwareCausalForecaster":
        phi_key = "decoder_cross"
        att_key = "dec_cross"
    elif architecture == "StageCausalForecaster":
        phi_key = "decoder1_cross"
        att_key = "dec1_cross"
    else:
        print(f"Warning: eval_embedding_dag_correlation not yet fully tested for {architecture}")
        phi_key = "decoder_cross"
        att_key = "dec_cross"
    
    learned_dag, source = _get_learned_dag(attention_data, att_key, phi_key)
    
    if learned_dag is None:
        print("Error: Could not extract learned DAG from attention data")
        return {}
    
    print(f"Loaded learned DAG from {source}: shape={learned_dag.shape}")
    
    # Load true DAG mask
    true_dag = _load_true_dag_mask(datadir_path, dataset_name, "dec_cross")
    
    if true_dag is None:
        print("Error: Could not load true DAG mask")
        return {}
    
    print(f"Loaded true DAG: shape={true_dag.shape}")
    
    # =========================================================================
    # Map cosine similarities to DAG edges dynamically
    # =========================================================================
    
    # Build COS_TO_DAG_MAP dynamically from metadata
    COS_TO_DAG_MAP = _build_cos_to_dag_map(source_labels, input_labels)
    
    # Build direct edges set for causal relationship lookup
    edges_set = set(tuple(e) for e in metadata["causal_structure"]["direct_edges"])
    
    # Build data for correlation
    records = []
    for cos_col, (target_idx, source_idx) in COS_TO_DAG_MAP.items():
        if cos_col not in mean_cos_sims:
            continue
        
        cos_sim = mean_cos_sims[cos_col]
        learned_edge = learned_dag[target_idx, source_idx]
        true_edge = true_dag[target_idx, source_idx]
        
        # Parse variable names from column (cos_{source}_{target})
        parts = cos_col.replace("cos_", "").split("_")
        source_var = parts[0]
        target_var = parts[1]
        
        records.append({
            "pair": f"{source_var}_{target_var}",
            "source_var": source_var,
            "target_var": target_var,
            "embedding_cosine_sim": cos_sim,
            "learned_dag_prob": learned_edge,
            "true_dag_edge": int(true_edge),
            "is_causal": bool((source_var, target_var) in edges_set),
        })
    
    df_data = pd.DataFrame(records)
    df_data.to_csv(join(eval_path_files, data_filename), index=False)
    print(f"Saved: {data_filename}")
    
    # =========================================================================
    # Compute correlations
    # =========================================================================
    
    # Correlation: embedding similarity vs learned DAG probability
    corr_emb_learned, pval_emb_learned = stats.pearsonr(
        df_data["embedding_cosine_sim"], 
        df_data["learned_dag_prob"]
    )
    
    # Correlation: embedding similarity vs true DAG (binary)
    corr_emb_true, pval_emb_true = stats.pearsonr(
        df_data["embedding_cosine_sim"], 
        df_data["true_dag_edge"]
    )
    
    # Separation score: mean embedding sim for causal pairs - mean for non-causal
    causal_pairs = df_data[df_data["is_causal"]]
    non_causal_pairs = df_data[~df_data["is_causal"]]
    
    mean_causal = causal_pairs["embedding_cosine_sim"].mean() if len(causal_pairs) > 0 else 0
    mean_non_causal = non_causal_pairs["embedding_cosine_sim"].mean() if len(non_causal_pairs) > 0 else 0
    separation_score = mean_causal - mean_non_causal
    
    # Statistical test: are causal pairs significantly more similar?
    if len(causal_pairs) > 0 and len(non_causal_pairs) > 0:
        t_stat, t_pval = stats.ttest_ind(
            causal_pairs["embedding_cosine_sim"],
            non_causal_pairs["embedding_cosine_sim"]
        )
    else:
        t_stat, t_pval = None, None
    
    # =========================================================================
    # Save results
    # =========================================================================
    
    results = {
        "dataset": dataset_name,
        "architecture": architecture,
        "dag_source": source,
        "n_pairs": len(df_data),
        "n_causal_pairs": len(causal_pairs),
        "n_non_causal_pairs": len(non_causal_pairs),
        "embedding_dag_correlation": float(corr_emb_learned),
        "embedding_dag_correlation_pval": float(pval_emb_learned),
        "embedding_true_dag_correlation": float(corr_emb_true),
        "embedding_true_dag_correlation_pval": float(pval_emb_true),
        "mean_embedding_sim_causal": float(mean_causal),
        "mean_embedding_sim_non_causal": float(mean_non_causal),
        "separation_score": float(separation_score),
        "separation_ttest_stat": float(t_stat) if t_stat is not None else None,
        "separation_ttest_pval": float(t_pval) if t_pval is not None else None,
        "pair_details": records,
    }
    
    with open(join(eval_path_files, correlation_filename), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {correlation_filename}")
    
    # =========================================================================
    # Plots
    # =========================================================================
    
    # Plot 1: Scatter of embedding similarity vs learned DAG probability
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = df_data["is_causal"].map({True: "green", False: "red"})
    ax.scatter(df_data["learned_dag_prob"], df_data["embedding_cosine_sim"], 
               c=colors, s=200, alpha=0.7)
    
    # Add pair labels
    for _, row in df_data.iterrows():
        ax.annotate(row["pair"], (row["learned_dag_prob"], row["embedding_cosine_sim"]),
                   fontsize=10, ha='center', va='bottom')
    
    ax.set_xlabel("Learned DAG Probability (phi)")
    ax.set_ylabel("Embedding Cosine Similarity")
    ax.set_title(f"Embedding vs DAG Correlation\nr={corr_emb_learned:.3f}, p={pval_emb_learned:.3f}")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Legend
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Causal pair'),
        Patch(facecolor='red', alpha=0.7, label='Non-causal pair'),
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(join(eval_path_fig, f"embedding_dag_scatter_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Plot 2: Box plot comparing causal vs non-causal pairs
    fig, ax = plt.subplots(figsize=(6, 6))
    
    box_data = [
        non_causal_pairs["embedding_cosine_sim"].values,
        causal_pairs["embedding_cosine_sim"].values,
    ]
    bp = ax.boxplot(box_data, labels=["Non-causal", "Causal"], patch_artist=True)
    bp["boxes"][0].set_facecolor("red")
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("green")
    bp["boxes"][1].set_alpha(0.5)
    
    # Overlay individual points
    for i, (data, color) in enumerate([(non_causal_pairs, "red"), (causal_pairs, "green")]):
        x = np.ones(len(data)) * (i + 1) + np.random.normal(0, 0.05, len(data))
        ax.scatter(x, data["embedding_cosine_sim"], c=color, alpha=0.7, s=100)
    
    ax.set_ylabel("Embedding Cosine Similarity")
    ax.set_title(f"Embedding Separation by Causality\nSeparation={separation_score:.3f}, p={t_pval:.3f}" if t_pval else "Embedding Separation by Causality")
    
    plt.tight_layout()
    plt.savefig(join(eval_path_fig, f"embedding_separation_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
    if show_plots:
        plt.show()
    else:
        plt.close()
        
    # =========================================================================
    # Save README and cline template
    # =========================================================================
    
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="This evaluation analyzes correlation between embedding similarity and DAG structure (H6).",
        files_info={
            correlation_filename: "Correlation statistics and summary results (JSON)",
            data_filename: "Raw data: embedding similarity, learned DAG, true DAG per pair (CSV)",
        },
        column_documentation={
            "pair": "Variable pair name (e.g., 'S1_X1')",
            "embedding_cosine_sim": "Cosine similarity between variable embeddings (from eval_embed)",
            "learned_dag_prob": "Learned edge probability from phi tensor",
            "true_dag_edge": "True DAG edge indicator (0 or 1)",
            "is_causal": "Whether this is a true causal pair",
        }
    )
    
    _create_cline_template(eval_path_cline, "eval_embedding_dag_correlation", exp_id)
    
    # Print summary
    print(f"\n=== Embedding-DAG Correlation Summary ===")
    print(f"Embedding vs Learned DAG: r={corr_emb_learned:.3f} (p={pval_emb_learned:.3f})")
    print(f"Embedding vs True DAG:    r={corr_emb_true:.3f} (p={pval_emb_true:.3f})")
    print(f"Mean embedding sim (causal):     {mean_causal:.3f}")
    print(f"Mean embedding sim (non-causal): {mean_non_causal:.3f}")
    print(f"Separation score:                {separation_score:.3f}")
    if t_pval is not None:
        print(f"Separation t-test p-value:       {t_pval:.3f}")
    
    return results

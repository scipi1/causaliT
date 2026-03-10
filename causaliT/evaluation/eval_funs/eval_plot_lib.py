"""
Plotting utilities for attention score analysis and visualization.

This module provides functions to:
- Plot attention heatmaps from trained models
- Plot attention evolution over training epochs with confidence intervals

Provides visualization for CausaliT evaluation functions.
"""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import AttentionData from eval_lib (local import)
from .eval_lib import AttentionData


def plot_attention_scores(
    data: AttentionData,
    figsize: tuple = None,
    cmap: str = 'viridis',
    annotation_fontsize: int = 8,
    title_fontsize: int = 12,
    save_path: str = None,
    dpi: int = 100,
    scale_mode: str = "global",
) -> plt.Figure:
    """
    Plot attention scores from all K-folds and attention blocks using GridSpec.
    
    Creates a grid layout where:
    - Columns represent different K-folds
    - Rows represent different attention blocks (with optional phi rows below)
    - Each row has its own colorbar on the right
    - Color scale can be global (default) or per-row
    
    Args:
        data: AttentionData object from load_attention_data()
        figsize: Figure size as (width, height). If None, auto-calculated
        cmap: Colormap to use for heatmaps
        annotation_fontsize: Font size for mean±std annotations
        title_fontsize: Font size for subplot titles
        save_path: Optional path to save the figure
        dpi: DPI for the figure
        scale_mode: Color scale mode - "global" (default) for same scale across all plots,
                    or "row" for per-row scaling
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> from notebooks.eval_funs.eval_lib import load_attention_data
        >>> from notebooks.eval_funs.eval_plot_lib import plot_attention_scores
        >>> data = load_attention_data("../experiments/my_experiment")
        >>> fig = plot_attention_scores(data)  # global scale (default)
        >>> fig = plot_attention_scores(data, scale_mode="row")  # row-wise scale
        >>> plt.show()
    """
    # Determine which attention blocks to plot (non-empty ones)
    attention_blocks = []
    phi_mapping = {}  # Maps attention block name to corresponding phi key
    
    if data.architecture_type == "TransformerForecaster":
        # Check each block for non-None data
        if any(x is not None for x in data.attention_weights.get("encoder", [])):
            attention_blocks.append("encoder")
            phi_mapping["encoder"] = "encoder"
        if any(x is not None for x in data.attention_weights.get("decoder", [])):
            attention_blocks.append("decoder")
            phi_mapping["decoder"] = "decoder"
        if any(x is not None for x in data.attention_weights.get("cross", [])):
            attention_blocks.append("cross")
            phi_mapping["cross"] = "cross"  # Cross-attention DAG (for CausalCrossAttention)
    elif data.architecture_type == "StageCausalForecaster":
        if any(x is not None for x in data.attention_weights.get("dec1_self", [])):
            attention_blocks.append("dec1_self")
            phi_mapping["dec1_self"] = "decoder1"
        if any(x is not None for x in data.attention_weights.get("dec1_cross", [])):
            attention_blocks.append("dec1_cross")
            phi_mapping["dec1_cross"] = "decoder1_cross"  # Cross-attention DAG (S -> X)
        if any(x is not None for x in data.attention_weights.get("dec2_self", [])):
            attention_blocks.append("dec2_self")
            phi_mapping["dec2_self"] = "decoder2"
        if any(x is not None for x in data.attention_weights.get("dec2_cross", [])):
            attention_blocks.append("dec2_cross")
            phi_mapping["dec2_cross"] = "decoder2_cross"  # Cross-attention DAG (X -> Y)
    elif data.architecture_type in ["SingleCausalForecaster", "NoiseAwareCausalForecaster"]:
        # SingleCausalForecaster and NoiseAwareCausalForecaster have single decoder: S → X
        if any(x is not None for x in data.attention_weights.get("dec_self", [])):
            attention_blocks.append("dec_self")
            phi_mapping["dec_self"] = "decoder"  # X self-attention DAG
        if any(x is not None for x in data.attention_weights.get("dec_cross", [])):
            attention_blocks.append("dec_cross")
            phi_mapping["dec_cross"] = "decoder_cross"  # S → X cross-attention DAG
    
    if not attention_blocks:
        raise ValueError("No attention blocks with data found")
    
    # Determine number of K-folds
    n_folds = len(data.predictions)
    if n_folds == 0:
        raise ValueError("No predictions found in data")
    
    # Calculate number of rows: each attention block + optional phi row below self-attention blocks
    row_info = []  # List of (block_name, is_phi) tuples
    for block in attention_blocks:
        row_info.append((block, False))  # Attention row
        # Check if phi is available for this block
        phi_key = phi_mapping.get(block)
        if phi_key and any(x is not None for x in data.phi_tensors.get(phi_key, [])):
            row_info.append((block, True))  # Phi row
    
    n_rows = len(row_info)
    n_cols = n_folds + 1  # +1 for colorbar column
    
    # Calculate global min/max for consistent color scaling
    global_min = float('inf')
    global_max = float('-inf')
    
    for block in attention_blocks:
        for att_tensor in data.attention_weights.get(block, []):
            if att_tensor is not None:
                if len(att_tensor.shape) < 3:
                    att_tensor = np.expand_dims(att_tensor, axis=0)
                mean = att_tensor.mean(axis=0)
                global_min = min(global_min, mean.min())
                global_max = max(global_max, mean.max())
    
    # Include phi tensors in global scale
    for phi_key in phi_mapping.values():
        for phi_tensor in data.phi_tensors.get(phi_key, []):
            if phi_tensor is not None:
                global_min = min(global_min, phi_tensor.min())
                global_max = max(global_max, phi_tensor.max())
    
    print(f"Global color scale: min={global_min:.4f}, max={global_max:.4f}")
    
    # Pre-compute per-row min/max for row-wise scaling
    row_scales = {}  # Maps row_idx to (vmin, vmax)
    if scale_mode == "row":
        for row_idx, (block_name, is_phi) in enumerate(row_info):
            row_min = float('inf')
            row_max = float('-inf')
            
            if is_phi:
                phi_key = phi_mapping.get(block_name)
                phi_list = data.phi_tensors.get(phi_key, [])
                for phi_tensor in phi_list:
                    if phi_tensor is not None:
                        row_min = min(row_min, phi_tensor.min())
                        row_max = max(row_max, phi_tensor.max())
            else:
                att_list = data.attention_weights.get(block_name, [])
                for att_tensor in att_list:
                    if att_tensor is not None:
                        if len(att_tensor.shape) < 3:
                            att_tensor = np.expand_dims(att_tensor, axis=0)
                        mean = att_tensor.mean(axis=0)
                        row_min = min(row_min, mean.min())
                        row_max = max(row_max, mean.max())
            
            row_scales[row_idx] = (row_min, row_max)
            print(f"Row {row_idx} ({block_name}, phi={is_phi}) scale: min={row_min:.4f}, max={row_max:.4f}")
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        cell_width = 3.5
        cell_height = 3.0
        figsize = (cell_width * n_cols, cell_height * n_rows)
    
    # Create figure and GridSpec
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Width ratios: equal for folds, narrow for colorbar
    width_ratios = [1] * n_folds + [0.05]
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, width_ratios=width_ratios,
                           wspace=0.3, hspace=0.4)
    
    # Plot each row
    for row_idx, (block_name, is_phi) in enumerate(row_info):
        row_images = []  # Store images for colorbar
        
        # Determine vmin/vmax for this row
        if scale_mode == "row":
            vmin, vmax = row_scales[row_idx]
        else:
            vmin, vmax = global_min, global_max
        
        for col_idx in range(n_folds):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            if is_phi:
                # Plot phi tensor
                phi_key = phi_mapping.get(block_name)
                phi_list = data.phi_tensors.get(phi_key, [])
                phi_tensor = phi_list[col_idx] if col_idx < len(phi_list) else None
                
                if phi_tensor is not None:
                    im = ax.imshow(phi_tensor, vmin=vmin, vmax=vmax, cmap=cmap)
                    row_images.append(im)
                    
                    # Set title for first column only
                    if col_idx == 0:
                        ax.set_ylabel(f"φ ({block_name})", fontsize=title_fontsize)
                    
                    # Set column title (k-fold) for first row only
                    if row_idx == 0:
                        ax.set_title(f"k={col_idx}", fontsize=title_fontsize)
                    
                    # Add tick labels
                    n_queries, n_keys = phi_tensor.shape
                    ax.set_xticks(range(n_keys))
                    ax.set_yticks(range(n_queries))
                    ax.set_xlabel("Keys")
                else:
                    ax.text(0.5, 0.5, "No phi", ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                # Plot attention weights (mean ± std)
                att_list = data.attention_weights.get(block_name, [])
                att_tensor = att_list[col_idx] if col_idx < len(att_list) else None
                
                if att_tensor is not None:
                    if len(att_tensor.shape) < 3:
                        att_tensor = np.expand_dims(att_tensor, axis=0)
                    
                    mean = att_tensor.mean(axis=0)
                    std = att_tensor.std(axis=0)
                    
                    im = ax.imshow(mean, vmin=vmin, vmax=vmax, cmap=cmap)
                    row_images.append(im)
                    
                    # Annotate with mean ± std
                    for i in range(mean.shape[0]):
                        for j in range(mean.shape[1]):
                            ax.text(j, i, f"{mean[i, j]:.2f}\n±{std[i, j]:.2f}",
                                   ha="center", va="center", color="white",
                                   fontsize=annotation_fontsize,
                                   fontweight='bold')
                    
                    # Set ylabel (block name) for first column only
                    if col_idx == 0:
                        ax.set_ylabel(block_name, fontsize=title_fontsize)
                    
                    # Set column title (k-fold) for first row only
                    if row_idx == 0:
                        ax.set_title(f"k={col_idx}", fontsize=title_fontsize)
                    
                    # Add tick labels
                    n_queries, n_keys = mean.shape
                    ax.set_xticks(range(n_keys))
                    ax.set_yticks(range(n_queries))
                    ax.set_xlabel("Keys")
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        # Add colorbar for this row
        if row_images:
            cbar_ax = fig.add_subplot(gs[row_idx, n_cols - 1])
            fig.colorbar(row_images[0], cax=cbar_ax)
    
    # Add overall title
    fig.suptitle(f"Attention Scores - {data.architecture_type}", fontsize=title_fontsize + 2, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_attention_evolution(
    df: pd.DataFrame,
    columns: List[str] = None,
    include_phi: bool = True,
    aggregate_folds: bool = False,
    n_cols: int = 4,
    figsize: tuple = None,
    alpha: float = 0.3,
    cmap: str = 'tab10',
    title: str = "Attention Score Evolution",
    save_path: str = None,
    dpi: int = 100,
) -> plt.Figure:
    """
    Plot attention score evolution over epochs with confidence intervals.
    
    This function takes the DataFrame from load_attention_evolution() and creates
    a grid of subplots, one for each attention entry. The mean is plotted as a line
    and the standard deviation is shown as a shaded confidence interval.
    
    Also includes phi (learnable DAG) columns by default.
    
    Args:
        df: DataFrame from load_attention_evolution() with columns:
            - epoch: epoch number
            - kfold: fold identifier
            - {block}_{i}{j}_mean: mean attention score
            - {block}_{i}{j}_std: std of attention score
            - phi_{block}_{i}{j}: phi (DAG probability) values
        columns: List of column name prefixes to plot (e.g., ["dec_self_00", "dec_cross_01"]).
                If None, auto-detects all columns ending with "_mean" plus phi columns.
        include_phi: If True (default), also include phi columns in the plot.
        aggregate_folds: If True, aggregate across k-folds (plot mean of means with 
                        combined uncertainty). If False, plot each fold as separate line.
        n_cols: Number of columns in the subplot grid
        figsize: Figure size as (width, height). If None, auto-calculated.
        alpha: Transparency for the confidence interval shading (0-1)
        cmap: Colormap name for line colors (used when not aggregating folds)
        title: Overall figure title
        save_path: Optional path to save the figure
        dpi: DPI for the figure
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> from notebooks.eval_funs.eval_attention import load_attention_evolution
        >>> from notebooks.eval_funs.eval_plot_lib import plot_attention_evolution
        >>> 
        >>> df = load_attention_evolution("../experiments/my_experiment")
        >>> 
        >>> # Plot all attention entries + phi, separate lines per fold
        >>> fig = plot_attention_evolution(df)
        >>> plt.show()
        >>> 
        >>> # Plot with aggregated folds
        >>> fig = plot_attention_evolution(df, aggregate_folds=True)
        >>> plt.show()
        >>> 
        >>> # Plot specific columns only (no auto phi)
        >>> fig = plot_attention_evolution(df, columns=["dec_self_00", "dec_self_01"])
        >>> plt.show()
        >>>
        >>> # Exclude phi columns
        >>> fig = plot_attention_evolution(df, include_phi=False)
        >>> plt.show()
    """
    # Auto-detect columns if not provided
    if columns is None:
        # Find all columns ending with "_mean" (excluding diff columns)
        mean_cols = [c for c in df.columns if c.endswith("_mean") and "_diff_" not in c]
        # Extract the prefix (remove "_mean" suffix)
        columns = [c.rsplit("_mean", 1)[0] for c in mean_cols]
        # Remove duplicates while preserving order
        columns = list(dict.fromkeys(columns))
        
        # Also add phi columns if requested
        if include_phi:
            phi_cols = [c for c in df.columns if c.startswith("phi_") and "_diff" not in c]
            columns.extend(phi_cols)
    
    if not columns:
        raise ValueError("No columns found to plot. Check DataFrame columns or provide 'columns' argument.")
    
    # Calculate grid dimensions
    n_plots = len(columns)
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        cell_width = 4.0
        cell_height = 3.0
        figsize = (cell_width * n_cols, cell_height * n_rows)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)
    axes_flat = axes.flatten()
    
    # Get colormap for fold colors
    cmap_obj = plt.get_cmap(cmap)
    kfolds = df['kfold'].unique()
    n_folds = len(kfolds)
    fold_colors = {kfold: cmap_obj(i / max(n_folds - 1, 1)) for i, kfold in enumerate(kfolds)}
    
    # Plot each column
    for idx, col_prefix in enumerate(columns):
        ax = axes_flat[idx]
        
        # Check if this is a phi column (no _mean/_std suffix) or attention column
        is_phi_col = col_prefix.startswith("phi_") and col_prefix in df.columns
        
        if is_phi_col:
            # Phi columns are direct values (no mean/std structure)
            if aggregate_folds:
                # Aggregate across folds
                grouped = df.groupby('epoch')[col_prefix].agg(['mean', 'std']).reset_index()
                epochs = grouped['epoch'].values
                mean_values = grouped['mean'].values
                std_values = grouped['std'].values
                
                ax.plot(epochs, mean_values, 'b-', linewidth=1.5, label='Mean')
                ax.fill_between(epochs,
                               mean_values - std_values,
                               mean_values + std_values,
                               alpha=alpha, color='blue', label='±1 std')
            else:
                # Plot each fold
                for kfold in kfolds:
                    fold_df = df[df['kfold'] == kfold].sort_values('epoch')
                    epochs = fold_df['epoch'].values
                    values = fold_df[col_prefix].values
                    
                    color = fold_colors[kfold]
                    ax.plot(epochs, values, '-', color=color, linewidth=1.5, label=kfold)
            
            ax.set_ylabel('Phi Value')
            ax.set_ylim(0, 1)  # Phi values are probabilities [0, 1]
        else:
            # Attention columns have _mean/_std structure
            mean_col = f"{col_prefix}_mean"
            std_col = f"{col_prefix}_std"
            
            # Check if columns exist
            if mean_col not in df.columns:
                ax.text(0.5, 0.5, f"No data:\n{col_prefix}", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(col_prefix)
                continue
            
            has_std = std_col in df.columns
            
            if aggregate_folds:
                # Aggregate across folds: compute mean of means and propagate std
                grouped = df.groupby('epoch').agg({
                    mean_col: ['mean', 'std'],
                    **(({std_col: 'mean'}) if has_std else {})
                }).reset_index()
                
                # Flatten multi-level columns
                grouped.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                  for col in grouped.columns]
                
                epochs = grouped['epoch'].values
                mean_values = grouped[f'{mean_col}_mean'].values
                
                # Combined uncertainty: sqrt(var_between_folds + mean_var_within_samples)
                if has_std:
                    std_between = grouped[f'{mean_col}_std'].values
                    std_within = grouped[f'{std_col}_mean'].values
                    combined_std = np.sqrt(std_between**2 + std_within**2)
                else:
                    combined_std = grouped[f'{mean_col}_std'].values
                
                # Plot mean line
                ax.plot(epochs, mean_values, 'b-', linewidth=1.5, label='Mean')
                
                # Plot confidence interval
                ax.fill_between(epochs, 
                               mean_values - combined_std, 
                               mean_values + combined_std,
                               alpha=alpha, color='blue', label='±1 std')
            else:
                # Plot each fold separately
                for kfold in kfolds:
                    fold_df = df[df['kfold'] == kfold].sort_values('epoch')
                    epochs = fold_df['epoch'].values
                    mean_values = fold_df[mean_col].values
                    
                    color = fold_colors[kfold]
                    
                    # Plot mean line
                    ax.plot(epochs, mean_values, '-', color=color, linewidth=1.5, label=kfold)
                    
                    # Plot confidence interval if std is available
                    if has_std:
                        std_values = fold_df[std_col].values
                        ax.fill_between(epochs,
                                       mean_values - std_values,
                                       mean_values + std_values,
                                       alpha=alpha, color=color)
            
            ax.set_ylabel('Attention Score')
        
        ax.set_xlabel('Epoch')
        ax.set_title(col_prefix)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Add legend to first subplot
    if n_plots > 0:
        axes_flat[0].legend(loc='best', fontsize=8)
    
    # Add overall title
    fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_phi_evolution(
    df: pd.DataFrame,
    columns: List[str] = None,
    aggregate_folds: bool = False,
    n_cols: int = 4,
    figsize: tuple = None,
    alpha: float = 0.3,
    cmap: str = 'tab10',
    title: str = "Phi (DAG Probabilities) Evolution",
    save_path: str = None,
    dpi: int = 100,
) -> plt.Figure:
    """
    Plot phi tensor evolution over epochs.
    
    Similar to plot_attention_evolution but specifically for phi tensors (learned DAG
    probabilities). These don't have per-sample std, so only fold aggregation uncertainty
    is shown when aggregate_folds=True.
    
    Args:
        df: DataFrame from load_attention_evolution() with phi columns:
            - epoch: epoch number
            - kfold: fold identifier
            - phi_{block}_{i}{j}: phi value (DAG probability)
        columns: List of phi column names to plot (e.g., ["phi_decoder_00", "phi_decoder_01"]).
                If None, auto-detects all columns starting with "phi_" (excluding "_diff").
        aggregate_folds: If True, plot mean across folds with std as confidence interval.
        n_cols: Number of columns in the subplot grid
        figsize: Figure size as (width, height). If None, auto-calculated.
        alpha: Transparency for the confidence interval shading
        cmap: Colormap name for line colors
        title: Overall figure title
        save_path: Optional path to save the figure
        dpi: DPI for the figure
        
    Returns:
        matplotlib Figure object
    """
    # Auto-detect phi columns if not provided
    if columns is None:
        columns = [c for c in df.columns 
                  if c.startswith("phi_") and "_diff" not in c]
    
    if not columns:
        raise ValueError("No phi columns found. Check DataFrame or provide 'columns' argument.")
    
    # Calculate grid dimensions
    n_plots = len(columns)
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Auto-calculate figure size
    if figsize is None:
        cell_width = 4.0
        cell_height = 3.0
        figsize = (cell_width * n_cols, cell_height * n_rows)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)
    axes_flat = axes.flatten()
    
    # Get colormap for folds
    cmap_obj = plt.get_cmap(cmap)
    kfolds = df['kfold'].unique()
    n_folds = len(kfolds)
    fold_colors = {kfold: cmap_obj(i / max(n_folds - 1, 1)) for i, kfold in enumerate(kfolds)}
    
    # Plot each phi column
    for idx, col in enumerate(columns):
        ax = axes_flat[idx]
        
        if col not in df.columns:
            ax.text(0.5, 0.5, f"No data:\n{col}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(col)
            continue
        
        if aggregate_folds:
            # Aggregate across folds
            grouped = df.groupby('epoch')[col].agg(['mean', 'std']).reset_index()
            epochs = grouped['epoch'].values
            mean_values = grouped['mean'].values
            std_values = grouped['std'].values
            
            ax.plot(epochs, mean_values, 'b-', linewidth=1.5, label='Mean')
            ax.fill_between(epochs,
                           mean_values - std_values,
                           mean_values + std_values,
                           alpha=alpha, color='blue', label='±1 std')
        else:
            # Plot each fold
            for kfold in kfolds:
                fold_df = df[df['kfold'] == kfold].sort_values('epoch')
                epochs = fold_df['epoch'].values
                values = fold_df[col].values
                
                color = fold_colors[kfold]
                ax.plot(epochs, values, '-', color=color, linewidth=1.5, label=kfold)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Phi Value')
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)  # Phi values are probabilities [0, 1]
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Add legend
    if n_plots > 0:
        axes_flat[0].legend(loc='best', fontsize=8)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

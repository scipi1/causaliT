"""
Metrics evaluation module for prediction results.

This module provides utilities for computing regression metrics on model predictions,
supporting both univariate and multivariate outputs with flexible feature-wise evaluation.
"""

import torch
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union
from torchmetrics.regression import R2Score, MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

from .predictors import PredictionResult


# Available metrics registry
AVAILABLE_METRICS = {
    'r2': R2Score,
    'mse': MeanSquaredError,
    'mae': MeanAbsoluteError,
    'rmse': lambda: MeanSquaredError(squared=False),
    'mape': MeanAbsolutePercentageError,
}

DEFAULT_METRICS = ['r2', 'mse', 'mae', 'rmse']


def _create_auto_mask(targets: torch.Tensor) -> torch.Tensor:
    """
    Automatically create validity mask from targets by detecting NaN values.
    
    Args:
        targets: Target tensor (B, L, D) or (B, L)
        
    Returns:
        Boolean mask (B, L) where True = valid position
    """
    if targets.ndim == 3:
        # For multivariate, valid if any feature is not NaN
        return ~torch.isnan(targets).all(dim=-1)
    elif targets.ndim == 2:
        # For univariate
        return ~torch.isnan(targets)
    else:
        raise ValueError(f"Expected 2D or 3D targets, got shape {targets.shape}")


def _get_metric_functions(
    metrics: List[str],
    device: torch.device
) -> Dict[str, torch.nn.Module]:
    """
    Create metric function instances.
    
    Args:
        metrics: List of metric names
        device: Device to place metrics on
        
    Returns:
        Dictionary mapping metric names to initialized functions
    """
    metric_fns = {}
    for name in metrics:
        if name not in AVAILABLE_METRICS:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(AVAILABLE_METRICS.keys())}")
        metric_fns[name] = AVAILABLE_METRICS[name]().to(device)
    return metric_fns


def _align_prediction_target_shapes(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    target_feature_idx: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align prediction and target shapes by handling common mismatches.
    
    Handles:
    1. Predictions are (B,) or (B,L) but targets are (B,D) or (B,L,D) -> extract target feature
    2. Squeeze unnecessary dimensions: (B,1) -> (B), (B,L,1) -> (B,L)
    3. Ensure both tensors have compatible shapes
    
    Args:
        y_pred: Prediction tensor
        y_true: Target tensor  
        target_feature_idx: Which feature to extract from targets if needed
        
    Returns:
        Aligned (y_pred, y_true) tensors
    """
    # Squeeze single-dimensional features at the end
    if y_pred.ndim >= 2 and y_pred.shape[-1] == 1:
        y_pred = y_pred.squeeze(-1)
    if y_true.ndim >= 2 and y_true.shape[-1] == 1:
        y_true = y_true.squeeze(-1)
    
    # If shapes match now, we're done
    if y_pred.shape == y_true.shape:
        return y_pred, y_true
    
    # Determine if we need to extract a target feature
    # Case 1: pred is (B,) and target is (B, D) 
    # Case 2: pred is (B, L) and target is (B, L, D)
    need_extraction = False
    
    if y_pred.ndim == 1 and y_true.ndim == 2:
        # (B,) vs (B, D)
        need_extraction = True
    elif y_pred.ndim == 2 and y_true.ndim == 3:
        # (B, L) vs (B, L, D)
        if y_pred.shape[0] == y_true.shape[0] and y_pred.shape[1] == y_true.shape[1]:
            need_extraction = True
    
    if need_extraction:
        if target_feature_idx is None:
            raise ValueError(
                f"Predictions are univariate {y_pred.shape} but targets are multivariate {y_true.shape}. "
                f"Please provide target_feature_idx to specify which target feature to use, "
                f"or ensure the PredictionResult has 'val_idx' in metadata."
            )
        
        # Extract the specified feature from targets
        if y_true.ndim == 3:
            # (B, L, D) -> (B, L)
            y_true = y_true[:, :, target_feature_idx]
        elif y_true.ndim == 2:
            # (B, D) -> (B,)
            y_true = y_true[:, target_feature_idx]
    
    return y_pred, y_true


def _validate_inputs(y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
    """Validate input tensors."""
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: predictions {y_pred.shape} vs targets {y_true.shape}")
    
    if y_pred.ndim not in [1, 2, 3]:
        raise ValueError(f"Expected 1D (B,), 2D (B,L) or 3D (B,L,D) tensors, got shape {y_pred.shape}")


@torch.no_grad()
def compute_prediction_metrics(
    results: PredictionResult,
    feature_names: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    mask: Optional[torch.Tensor] = None,
    per_sample: bool = True,
    target_feature_idx: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute regression metrics on prediction results.
    
    Computes metrics per sample (over sequence dimension) and per feature,
    similar to eval_df_torchmetrics from notebooks but integrated with
    the PredictionResult API.
    
    Args:
        results: PredictionResult object from predict_test_from_ckpt()
        feature_names: Optional names for features. If None and multivariate,
                      uses 'f0', 'f1', etc. If univariate, column omitted.
        metrics: List of metrics to compute. Options: 'r2', 'mse', 'mae', 'rmse', 'mape'.
                Default: ['r2', 'mse', 'mae', 'rmse']
        mask: Optional (B, L) boolean mask where True = valid position.
              If None, automatically created from NaN detection in targets.
        per_sample: If True, compute per-sample metrics. If False, compute global metrics.
        target_feature_idx: If predictions are univariate but targets are multivariate,
                           specify which target feature to use. If None, tries to get
                           from results.metadata['val_idx'] or config.
        
    Returns:
        DataFrame with columns:
            - index: Sample index (if per_sample=True)
            - feature: Feature name (if multivariate)
            - [metric columns]: R2, MSE, MAE, etc.
            
    Example:
        >>> results = predict_test_from_ckpt(config, datadir, checkpoint)
        >>> metrics_df = compute_prediction_metrics(results)
        >>> print(metrics_df.head())
           index feature    R2    MSE    MAE   RMSE
        0      0      f0  0.95  0.12   0.08   0.35
        1      0      f1  0.88  0.25   0.15   0.50
        
        >>> # Aggregate across samples
        >>> summary = metrics_df.groupby('feature').agg(['mean', 'std'])
    """
    # Convert to tensors if needed
    y_pred = torch.tensor(results.outputs) if not isinstance(results.outputs, torch.Tensor) else results.outputs
    y_true = torch.tensor(results.targets) if not isinstance(results.targets, torch.Tensor) else results.targets
    
    # Ensure tensors are on same device
    device = y_true.device
    y_pred = y_pred.to(device)
    
    # Auto-detect target_feature_idx if not provided
    if target_feature_idx is None and y_pred.shape != y_true.shape:
        # Try to get from metadata
        if 'val_idx' in results.metadata:
            target_feature_idx = results.metadata['val_idx']
        elif 'config' in results.metadata and 'data' in results.metadata['config']:
            target_feature_idx = results.metadata['config']['data'].get('val_idx')
    
    # Handle dimension mismatches and squeeze operations
    y_pred, y_true = _align_prediction_target_shapes(y_pred, y_true, target_feature_idx)
    
    # Validate inputs after alignment
    _validate_inputs(y_pred, y_true)
    
    # Ensure we have 3D tensors: (B, L, D)
    if y_pred.ndim == 1:
        # (B,) -> (B, 1, 1) - single value per sample
        y_pred = y_pred.unsqueeze(-1).unsqueeze(-1)
        y_true = y_true.unsqueeze(-1).unsqueeze(-1)
    elif y_pred.ndim == 2:
        # (B, L) -> (B, L, 1) - sequence per sample
        y_pred = y_pred.unsqueeze(-1)
        y_true = y_true.unsqueeze(-1)
    # else: already 3D (B, L, D)
    
    # Now safely unpack dimensions
    if y_true.ndim != 3:
        raise ValueError(
            f"Expected 3D tensor after dimension handling. "
            f"y_pred shape: {y_pred.shape} (ndim={y_pred.ndim}), "
            f"y_true shape: {y_true.shape} (ndim={y_true.ndim})"
        )
    
    B, L, D = y_true.shape
    
    # Setup metrics
    if metrics is None:
        metrics = DEFAULT_METRICS
    metric_fns = _get_metric_functions(metrics, device)
    
    # Setup feature names
    if D > 1:
        if feature_names is not None:
            if len(feature_names) != D:
                raise ValueError(f"feature_names length {len(feature_names)} must equal D={D}")
            labels = feature_names
        else:
            labels = [f"f{d}" for d in range(D)]
    else:
        labels = None
    
    # Create or validate mask
    if mask is None:
        # Auto-detect valid positions from targets
        mask = _create_auto_mask(y_true)
    else:
        if mask.shape != (B, L):
            raise ValueError(f"mask must be (B,L); got {mask.shape}")
        mask = mask.to(device)
    
    if not per_sample:
        # Global metrics
        return _compute_global_metrics(y_pred, y_true, mask, metric_fns, labels, metrics)
    
    # Per-sample metrics
    rows = []
    for b in range(B):
        idx = mask[b]
        yt = y_true[b, idx, :]  # (L_valid, D)
        yp = y_pred[b, idx, :]  # (L_valid, D)
        
        if yt.numel() == 0:
            # No valid steps - fill with NaN
            metric_vals = {name: torch.full((D,), float('nan'), device=device) for name in metrics}
        else:
            # Compute metrics
            n_samples = yt.shape[0]  # Number of valid samples
            metric_vals = {}
            
            # Filter out R2 if insufficient samples
            available_metrics = [m for m in metrics if not (m == 'r2' and n_samples < 2)]
            
            for name in available_metrics:
                fn = metric_fns[name]
                # Reset metric for each computation
                fn.reset()
                
                # Handle multioutput for metrics that support it
                if name == 'r2':
                    fn.multioutput = 'raw_values'
                    val = fn(yp, yt)  # (D,)
                else:
                    # Compute per feature
                    val = torch.stack([fn(yp[:, d], yt[:, d]) for d in range(D)])
                
                metric_vals[name] = val.detach().cpu()
            
            # Fill R2 with NaN if it was skipped
            if 'r2' in metrics and 'r2' not in metric_vals:
                metric_vals['r2'] = torch.full((D,), float('nan'))
        
        # Convert to lists - ensure all values are lists, even for scalar tensors
        metric_lists = {}
        for name, val in metric_vals.items():
            val_list = val.tolist()
            # If tolist() returns a scalar (0D tensor), wrap it in a list
            if not isinstance(val_list, list):
                val_list = [val_list]
            metric_lists[name] = val_list
        
        # Build rows
        if D > 1 and labels is not None:
            # Multivariate with feature names
            for d, feature_name in enumerate(labels):
                row = {'index': b, 'feature': feature_name}
                for name in metrics:
                    row[name.upper()] = metric_lists[name][d]
                rows.append(row)
        else:
            # Univariate or no feature names
            row = {'index': b}
            for name in metrics:
                row[name.upper()] = metric_lists[name][0] if D == 1 else metric_lists[name]
            rows.append(row)
    
    return pd.DataFrame(rows)


def _compute_global_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    metric_fns: Dict,
    labels: Optional[List[str]],
    metrics: List[str]
) -> pd.DataFrame:
    """Compute global metrics across all samples."""
    # Safely unpack dimensions
    if y_true.ndim == 3:
        B, L, D = y_true.shape
    else:
        raise ValueError(f"Expected 3D tensor in _compute_global_metrics, got shape {y_true.shape}")
    
    # Flatten predictions and targets
    y_pred_flat = []
    y_true_flat = []
    
    for b in range(B):
        idx = mask[b]
        y_pred_flat.append(y_pred[b, idx, :])
        y_true_flat.append(y_true[b, idx, :])
    
    if len(y_pred_flat) == 0 or all(t.numel() == 0 for t in y_pred_flat):
        # No valid data
        metric_vals = {name: torch.full((D,), float('nan')) for name in metrics}
    else:
        y_pred_cat = torch.cat(y_pred_flat, dim=0)  # (N, D)
        y_true_cat = torch.cat(y_true_flat, dim=0)  # (N, D)
        n_samples = y_pred_cat.shape[0]  # Total valid samples
        
        metric_vals = {}
        
        # Filter out R2 if insufficient samples
        available_metrics = [m for m in metrics if not (m == 'r2' and n_samples < 2)]
        
        for name in available_metrics:
            fn = metric_fns[name]
            fn.reset()
            
            # Handle multioutput for metrics that support it
            if name == 'r2':
                fn.multioutput = 'raw_values'
                val = fn(y_pred_cat, y_true_cat)  # (D,)
            else:
                val = torch.stack([fn(y_pred_cat[:, d], y_true_cat[:, d]) for d in range(D)])
            metric_vals[name] = val.detach().cpu()
        
        # Fill R2 with NaN if it was skipped
        if 'r2' in metrics and 'r2' not in metric_vals:
            metric_vals['r2'] = torch.full((D,), float('nan'))
    
    # Build rows
    rows = []
    # Ensure all values are lists, even for scalar tensors
    metric_lists = {}
    for name, val in metric_vals.items():
        val_list = val.tolist()
        # If tolist() returns a scalar (0D tensor), wrap it in a list
        if not isinstance(val_list, list):
            val_list = [val_list]
        metric_lists[name] = val_list
    
    if D > 1 and labels is not None:
        for d, feature_name in enumerate(labels):
            row = {'feature': feature_name}
            for name in metrics:
                row[name.upper()] = metric_lists[name][d]
            rows.append(row)
    else:
        row = {}
        for name in metrics:
            row[name.upper()] = metric_lists[name][0] if D == 1 else metric_lists[name]
        rows.append(row)
    
    return pd.DataFrame(rows)


def compare_predictions(
    *results: PredictionResult,
    labels: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare metrics across multiple prediction results.
    
    Combines metrics from multiple models into a single DataFrame with
    a 'model' column for easy comparison and visualization.
    
    Args:
        *results: Variable number of PredictionResult objects to compare
        labels: Model names for each result. If None, uses 'Model_0', 'Model_1', etc.
        feature_names: Optional feature names passed to compute_prediction_metrics
        metrics: Which metrics to compute
        
    Returns:
        Combined DataFrame with 'model' column and all metrics
        
    Example:
        >>> results_prot = predict_test_from_ckpt(config_prot, ...)
        >>> results_lstm = predict_test_from_ckpt(config_lstm, ...)
        >>> 
        >>> comparison = compare_predictions(
        ...     results_prot, results_lstm,
        ...     labels=['proT', 'LSTM']
        ... )
        >>> 
        >>> # Analyze
        >>> print(comparison.groupby(['model', 'feature'])['MAE'].mean())
        >>> 
        >>> # Visualize
        >>> import seaborn as sns
        >>> sns.boxplot(data=comparison, x='feature', y='R2', hue='model')
    """
    if len(results) == 0:
        raise ValueError("At least one PredictionResult required")
    
    if labels is not None and len(labels) != len(results):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of results ({len(results)})")
    
    all_dfs = []
    
    for i, result in enumerate(results):
        # Compute metrics for this result
        df = compute_prediction_metrics(
            result,
            feature_names=feature_names,
            metrics=metrics
        )
        
        # Add model label at the front
        model_name = labels[i] if labels else f"Model_{i}"
        df.insert(0, 'model', model_name)
        
        all_dfs.append(df)
    
    # Concatenate all dataframes
    return pd.concat(all_dfs, ignore_index=True)


def aggregate_metrics(
    df: pd.DataFrame,
    groupby: Union[str, List[str]] = 'feature',
    agg_funcs: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregate metrics across samples or models.
    
    Args:
        df: DataFrame from compute_prediction_metrics() or compare_predictions()
        groupby: Column(s) to group by. Options:
                - 'feature': Aggregate across samples per feature
                - 'model': Aggregate across samples per model
                - ['model', 'feature']: Aggregate per model-feature combination
        agg_funcs: Aggregation functions. Default: ['mean', 'std', 'min', 'max']
        
    Returns:
        Aggregated DataFrame
        
    Example:
        >>> metrics_df = compute_prediction_metrics(results)
        >>> summary = aggregate_metrics(metrics_df, groupby='feature')
        >>> print(summary)
        
        >>> # For comparison DataFrame
        >>> comparison = compare_predictions(results1, results2, labels=['A', 'B'])
        >>> summary = aggregate_metrics(comparison, groupby=['model', 'feature'])
    """
    if agg_funcs is None:
        agg_funcs = ['mean', 'std', 'min', 'max']
    
    # Identify metric columns (uppercase names)
    metric_cols = [col for col in df.columns if col.isupper() and col not in ['INDEX', 'FEATURE', 'MODEL']]
    
    if len(metric_cols) == 0:
        raise ValueError("No metric columns found in DataFrame")
    
    # Group and aggregate
    grouped = df.groupby(groupby)[metric_cols].agg(agg_funcs)
    
    return grouped

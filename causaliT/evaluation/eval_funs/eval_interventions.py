"""
ATE (Average Treatment Effect) evaluation for CausaliT experiments.

This module computes E[X | do(S=s)] predictions and compares them to ground-truth
causal effects from the SCM.

Key function:
- eval_ate: Evaluate ATE predictions using interventions from dataset metadata

IMPORTANT: Model ATE is computed using Monte Carlo sampled S values (same as ground truth)
to ensure fair comparison. The test split is NOT used for ATE evaluation.
"""

import json
from os.path import join, exists
from os import makedirs, listdir

import numpy as np
import torch
import pandas as pd
from omegaconf import OmegaConf
from typing import Dict, List, Optional
from pathlib import Path

# Import shared utilities
from .helpers.eval_utils import root_path, load_dataset_metadata

# Import from project modules
from causaliT.evaluation.predict import create_predictor
from causaliT.training.experiment_control import update_config

# Import from the local support layer
from .helpers.eval_lib import find_config_file, find_best_or_last_checkpoint


def infer_checkpoint_type(config) -> str:
    """
    Determine which checkpoint type to use for evaluation.

    Evaluation is performed on the **last** (final-epoch) checkpoint by default.
    This is deliberate: the ``best_causal`` (min ``val_hsic_reg``) and
    ``best_reconstruction`` (min ``val_x_mae``) checkpoints can be captured at
    an epoch that does **not** reflect the end-state of the training protocol.

    In particular, for the two-stage L0 protocol the min-``val_hsic_reg`` epoch
    lands at the *start* of the structural stage — *before* the L0 penalty has
    pruned the structure gate — so a DAG evaluated on ``best_causal`` looks
    identical across ``lambda_l0`` values (the gate is still dense). Evaluating
    on the last checkpoint makes the retrieved DAG reflect the actually-pruned
    gate and keeps it consistent with the reported end-of-training ``test_*``
    sparsity metrics.

    An explicit ``evaluation.checkpoint_type`` in the config always wins, so a
    specific experiment can still opt into ``"best_causal"`` /
    ``"best_reconstruction"`` if desired.

    Args:
        config: OmegaConf or dict configuration.

    Returns:
        One of ``"best_causal"``, ``"best_reconstruction"``, ``"best"``, ``"last"``.
    """
    # Explicit override from config always wins.
    explicit = (
        config.get("evaluation", {}).get("checkpoint_type", None)
        if hasattr(config, "get") else None
    )
    if explicit is not None:
        return str(explicit)

    # Default: evaluate on the final-epoch (last) checkpoint. See docstring for
    # why best_causal / best_reconstruction are NOT used by default.
    return "last"


# =============================================================================
# SCM Registry - maps dataset names to SCM objects for MC sampling
# =============================================================================

from .helpers.datadir import resolve_datadir


def _try_random_scm_dataset(datadir_path: Optional[str], dataset_name: str):
    """
    Rebuild a randomly-sampled SCM from its persisted recipe.

    Datasets produced by the DAG sweeper (``euler_sweep.dag_provider``) are not
    in the static registry below - they are sampled on the fly and their heavy
    arrays are pruned after training.  They *do* however ship a
    ``dag_recipe.json`` holding the full ``RandomSCMConfig``, so the SCM can be
    rebuilt exactly.  This yields a live ``.scm`` for Monte-Carlo intervention
    sampling **without requiring any stored samples**.

    Returns:
        The ``SCMDataset``, or ``None`` when this is not a recipe-backed dataset.
    """
    if not datadir_path:
        return None

    recipe_path = join(datadir_path, dataset_name, "dag_recipe.json")
    if not exists(recipe_path):
        return None

    try:
        import dataclasses

        from scm_ds.random_scm import RandomSCMConfig, sample_random_scm_dataset

        with open(recipe_path, "r", encoding="utf-8") as fh:
            recipe = json.load(fh)

        fields = dict(recipe.get("random_scm_config") or {})
        known = {f.name for f in dataclasses.fields(RandomSCMConfig)}
        cfg = RandomSCMConfig(**{k: v for k, v in fields.items() if k in known})
        return sample_random_scm_dataset(cfg)
    except Exception as exc:
        print(f"  Warning: could not rebuild random SCM from {recipe_path}: {exc}")
        return None


def get_scm_for_dataset(dataset_name: str, datadir_path: Optional[str] = None):
    """
    Get the SCM object for a given dataset name.
    
    This allows us to sample fresh S values from the same noise model
    used to generate the ground truth.
    
    Args:
        dataset_name: Name of the dataset (e.g., "scm1", "scm2", "scm3")
        datadir_path: Optional data root. When given, randomly-sampled datasets
            carrying a ``dag_recipe.json`` are rebuilt from that recipe, which
            makes arbitrary sampled DAGs work without a registry entry.
        
    Returns:
        SCMDataset object
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    # Import SCM definitions (lazy import to avoid circular dependencies)
    from scm_ds.datasets import (
        ds_scm1_discrete_sampling,
        ds_scm2_discrete_sampling,
        ds_scm3_discrete_sampling,
        ds_scm1,
        ds_scm2,
        ds_scm3,
    )
    
    # Registry mapping dataset names to SCM objects
    # Discrete variants use rng.choice() for S; continuous use rng.uniform()
    SCM_REGISTRY = {
        # Discrete S (paper defaults)
        "scm1": ds_scm1_discrete_sampling,
        "scm2": ds_scm2_discrete_sampling,
        "scm3": ds_scm3_discrete_sampling,
        # Continuous S (uniform) — for HSIC kernel analysis
        "scm1_continuous": ds_scm1,
        "scm2_continuous": ds_scm2,
        "scm3_continuous": ds_scm3,
    }
    
    if dataset_name in SCM_REGISTRY:
        return SCM_REGISTRY[dataset_name]

    # Randomly-sampled DAGs (DAG sweeps) are recipe-backed, not registered.
    random_ds = _try_random_scm_dataset(datadir_path, dataset_name)
    if random_ds is not None:
        return random_ds

    raise ValueError(
        f"Dataset '{dataset_name}' not found in SCM registry and no "
        f"'dag_recipe.json' was found for it. "
        f"Available: {list(SCM_REGISTRY.keys())}."
    )


def sample_mc_source_inputs(
    scm_dataset,
    intervention: Dict[str, float],
    n_samples: int,
    norm_stats: dict,
    var_idx_map: dict,
    source_labels: List[str],
    seed: int = 42,
) -> torch.Tensor:
    """
    Sample Monte Carlo source inputs from the SCM's noise model.
    
    This generates S values from the same distribution used for ground truth
    computation, then applies the specified intervention (clamping S_j to value).
    
    Args:
        scm_dataset: SCMDataset object
        intervention: Dict mapping variable name to intervention value (raw scale)
                     E.g., {"S1": 0.5} or {} for baseline (all S values sampled freely)
        n_samples: Number of MC samples
        norm_stats: Normalization statistics from dataset
        var_idx_map: Variable name to index mapping
        source_labels: List of source variable names
        seed: Random seed
        
    Returns:
        torch.Tensor: Source input tensor of shape (n_samples, n_sources, 2)
                     where last dim is [value, variable_idx]
    """
    # Sample from intervened SCM
    if intervention:
        scm_do = scm_dataset.scm.do(intervention)
    else:
        scm_do = scm_dataset.scm
    
    # Sample n_samples from the (possibly intervened) SCM
    df_samples = scm_do.sample(n=n_samples, seed=seed)
    
    # Extract only source columns
    source_values = df_samples[source_labels].values  # Shape: (n_samples, n_sources)
    
    # Normalize source values
    if "source" in norm_stats:
        stats = norm_stats["source"]
        method = stats.get("method", "minmax")
        if method == "minmax":
            min_val = stats.get("min", 0)
            max_val = stats.get("max", 1)
            source_values = (source_values - min_val) / (max_val - min_val)
        elif method == "standardize":
            mean_val = stats.get("mean", 0)
            std_val = stats.get("std", 1)
            source_values = (source_values - mean_val) / std_val
    
    # Build tensor with [value, variable_idx] for each source
    n_sources = len(source_labels)
    source_tensor = np.zeros((n_samples, n_sources, 2), dtype=np.float32)
    
    for i, src_var in enumerate(source_labels):
        source_tensor[:, i, 0] = source_values[:, i]  # value
        source_tensor[:, i, 1] = var_idx_map[src_var]  # variable index
    
    return torch.from_numpy(source_tensor)


def run_mc_predictions(
    experiment_path: str,
    scm_dataset,
    intervention_config: Dict[str, List[float]],
    norm_stats: dict,
    source_vars_map: dict,
    input_vars_map: dict,
    source_labels: List[str],
    input_labels: List[str],
    n_samples: int = 50000,
    seed: int = 42,
    checkpoint_type: str = "best_causal",
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Run model predictions using Monte Carlo sampled S values.
    
    This ensures model ATE evaluation uses the same S distribution as ground truth.
    
    Args:
        experiment_path: Path to experiment folder
        scm_dataset: SCMDataset object for sampling
        intervention_config: Dict mapping source vars to intervention values
        norm_stats: Normalization statistics
        source_vars_map: Source variable name to index mapping (for S embedding)
        input_vars_map: Input variable name to index mapping (for X embedding)
        source_labels: List of source variable names
        input_labels: List of input variable names (model outputs)
        n_samples: Number of MC samples (should match ground truth)
        seed: Random seed
        checkpoint_type: "best_causal" (default, lowest HSIC), "best", or "last" checkpoint
        batch_size: Batch size for model inference (default 64, same as training)
        
    Returns:
        DataFrame with predictions for all interventions
    """
    from os.path import isdir

    # Load config
    config_path = find_config_file(experiment_path)
    config = OmegaConf.load(config_path)
    config_updated = update_config(config)
    
    # Find all k-fold directories
    kfold_dirs = sorted([
        d for d in listdir(experiment_path) 
        if isdir(join(experiment_path, d)) and d.startswith('k_')
    ])
    
    if not kfold_dirs:
        raise ValueError(f"No k-fold directories found in {experiment_path}")
    
    print(f"  Found {len(kfold_dirs)} k-fold directories")
    print(f"  MC samples: {n_samples}")
    
    n_inputs = len(input_labels)
    all_records = []
    
    # Process each k-fold
    for kfold_dir in kfold_dirs:
        kfold_path = join(experiment_path, kfold_dir)
        checkpoints_dir = join(kfold_path, 'checkpoints')
        
        try:
            # Find checkpoint
            checkpoint_path = find_best_or_last_checkpoint(checkpoints_dir, checkpoint_type)
            print(f"  Processing {kfold_dir}...")
            
            # Create predictor (data root may live inside the run folder for
            # DAG-sweep runs, hence the resolver rather than a fixed path)
            datadir_path = resolve_datadir(config=config_updated)
            predictor = create_predictor(config_updated, checkpoint_path, datadir_path)
            
            # Get device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            predictor.model.to(device)
            predictor.model.eval()
            
            # Build list of interventions to evaluate
            interventions_to_run = [({}, "baseline")]  # Baseline: no intervention
            
            # Add baseline for each source (do(S_j=0) individually)
            # This matches how ground truth baseline is computed
            for src_var in intervention_config.keys():
                interventions_to_run.append(({src_var: 0.0}, f"{src_var}=0_baseline"))
            
            # Add treatment interventions
            for src_var, values in intervention_config.items():
                for val_raw in values:
                    interventions_to_run.append(({src_var: val_raw}, f"{src_var}={val_raw}"))
            
            # Run predictions for each intervention
            for intervention, label in interventions_to_run:
                # Sample MC inputs (using source_vars_map for S embedding indices)
                source_tensor = sample_mc_source_inputs(
                    scm_dataset=scm_dataset,
                    intervention=intervention,
                    n_samples=n_samples,
                    norm_stats=norm_stats,
                    var_idx_map=source_vars_map,  # Use source mapping for S
                    source_labels=source_labels,
                    seed=seed,
                )
                
                # Create dummy X tensor (will be blanked by the model)
                # Shape: (n_samples, n_inputs, 2) where 2 = [value, var_idx]
                # Variable indices must be integers within embedding vocabulary range
                # Using input_vars_map for X embedding indices
                x_dummy = torch.zeros((n_samples, n_inputs, 2), dtype=torch.float32)
                for i, var_name in enumerate(input_labels):
                    # Use the actual variable index from the INPUT mapping
                    var_idx = input_vars_map.get(var_name)
                    if var_idx is None:
                        raise ValueError(f"Variable '{var_name}' not found in input_vars_map: {input_vars_map}")
                    x_dummy[:, i, 1] = float(var_idx)  # Set variable indices as floats (will be converted to int by model)
                
                # Run model forward pass in batches
                all_preds = []
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    s_batch = source_tensor[start_idx:end_idx].to(device)
                    x_batch = x_dummy[start_idx:end_idx].to(device)
                    
                    with torch.no_grad():
                        # SingleCausalForecaster.forward expects (S, X) and returns (pred_x, attn, masks, entropy)
                        # The model internally blanks X values
                        output = predictor.model.forward(
                            data_source=s_batch,
                            data_intermediate=x_batch,
                        )
                        pred_x = output[0]  # First element is pred_x
                        all_preds.append(pred_x.cpu())
                
                # Concatenate all batch predictions
                pred_tensor = torch.cat(all_preds, dim=0)  # Shape: (n_samples, n_inputs, n_features)
                
                # Extract value predictions (feature 0)
                pred_values = pred_tensor[:, :, 0].numpy()  # Shape: (n_samples, n_inputs)
                
                # Record predictions per variable
                for pos_idx, var_name in enumerate(input_labels):
                    var_preds = pred_values[:, pos_idx]
                    
                    all_records.append({
                        "intervention": label,
                        "pos_idx": pos_idx,
                        "variable": var_name,
                        "kfold": kfold_dir,
                        "pred_feat_0": float(var_preds.mean()),  # Mean prediction (normalized)
                        "pred_std": float(var_preds.std()),  # Std of predictions
                        "n_samples": n_samples,
                    })
            
        except Exception as e:
            # ASCII only: a "✗" here raises UnicodeEncodeError on Windows
            # consoles (cp1252), which MASKS the real exception `e`.
            print(f"  [FAIL] Error processing {kfold_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(all_records)


# =============================================================================
# ATE Utility Functions
# =============================================================================

def load_ate_ground_truth(datadir_path: str, dataset_name: str) -> Optional[dict]:
    """Load ATE ground truth from dataset directory."""
    ate_path = join(datadir_path, dataset_name, "ate_ground_truth.json")
    if exists(ate_path):
        with open(ate_path, 'r') as f:
            return json.load(f)
    return None


def load_normalization_stats(datadir_path: str, dataset_name: str) -> Optional[dict]:
    """Load normalization statistics from dataset directory."""
    norm_path = join(datadir_path, dataset_name, "normalization.json")
    if exists(norm_path):
        with open(norm_path, 'r') as f:
            return json.load(f)
    return None


def get_interventions_from_ground_truth(ate_ground_truth: dict) -> Dict[str, List[float]]:
    """
    Extract intervention configuration from ate_ground_truth.json.
    
    The ate_ground_truth.json contains an 'interventions' key with structure:
    {
        "S1": {"values": [0.5], "type": "in_distribution", "role": "negative_control"},
        "S2": {"values": [0.5, 1.5], "type": "mixed", "role": "positive_control", "ood_values": [1.5]},
        ...
    }
    
    Returns:
        Dict mapping source variable names to lists of intervention values.
        E.g., {"S1": [0.5], "S2": [0.5, 1.5], "S3": [-0.5, 1.5], "S5": [-0.5, 1.5]}
    """
    interventions_meta = ate_ground_truth.get("interventions", {})
    
    if not interventions_meta:
        # Fallback: infer from analytical/monte_carlo keys if no explicit interventions
        # Parse keys like "S1=0.5" -> {"S1": [0.5]}
        inferred = {}
        for method in ["analytical", "monte_carlo"]:
            if method in ate_ground_truth:
                for key in ate_ground_truth[method].keys():
                    if "=" in key:
                        var, val = key.split("=")
                        if var not in inferred:
                            inferred[var] = []
                        inferred[var].append(float(val))
        return inferred
    
    return {src: meta["values"] for src, meta in interventions_meta.items()}


def denormalize_value(normalized_value: float, norm_stats: dict, category: str = "input") -> float:
    """De-normalize a value using the normalization statistics."""
    if category not in norm_stats:
        return normalized_value
    
    stats = norm_stats[category]
    method = stats.get("method", "minmax")
    
    if method == "minmax":
        min_val = stats.get("min", 0)
        max_val = stats.get("max", 1)
        return normalized_value * (max_val - min_val) + min_val
    elif method == "standardize":
        mean_val = stats.get("mean", 0)
        std_val = stats.get("std", 1)
        return normalized_value * std_val + mean_val
    else:
        return normalized_value


def normalize_intervention_value(raw_value: float, norm_stats: dict) -> float:
    """
    Normalize an intervention value using source normalization statistics.
    
    Intervention values from ate_ground_truth.json are in RAW scale (e.g., S2=0.5).
    The model expects NORMALIZED inputs (e.g., minmax [0,1]).
    This function converts raw intervention values to normalized scale.
    
    Args:
        raw_value: Intervention value in raw/original scale
        norm_stats: Dict from normalization.json
        
    Returns:
        Normalized intervention value
    """
    if "source" not in norm_stats:
        return raw_value
    
    stats = norm_stats["source"]
    method = stats.get("method", "minmax")
    
    if method == "minmax":
        min_val = stats.get("min", 0)
        max_val = stats.get("max", 1)
        return (raw_value - min_val) / (max_val - min_val)
    elif method == "standardize":
        mean_val = stats.get("mean", 0)
        std_val = stats.get("std", 1)
        return (raw_value - mean_val) / std_val
    else:
        return raw_value


def compute_ate_metrics(
    df: pd.DataFrame,
    ate_ground_truth: dict,
    norm_stats: dict,
    input_labels: List[str],
) -> pd.DataFrame:
    """
    Compute ATE metrics: model_ATE vs true_ATE
    
    ATE = E[X | do(S=s)] - E[X | do(S=0)]  (difference, not absolute)
    
    Args:
        df: DataFrame with predictions (columns: intervention, pred_feat_0, pos_idx, kfold)
        ate_ground_truth: Dict from ate_ground_truth.json (new format with 'ate' key)
        norm_stats: Dict from normalization.json
        input_labels: List of input variable names (e.g., ["X1", "X2", ...])
        
    Returns:
        DataFrame with ATE metrics per intervention × variable × fold
        Columns include: true_ate, true_baseline, true_treatment for comparison
    """
    # Use Monte Carlo ground truth (analytical method removed due to E[f(eps)] ≠ f(0) for nonlinear SCMs)
    gt_method = "monte_carlo"
    gt_data = ate_ground_truth.get(gt_method, {})
    
    # New format: gt_data has 'ate', 'baseline', 'treated' keys
    # Old format: gt_data directly maps intervention -> {var: value}
    if "ate" in gt_data:
        ground_truth_ate = gt_data["ate"]  # New format
        ground_truth_baseline = gt_data.get("baseline", {})  # New format
        ground_truth_treated = gt_data.get("treated", {})  # New format
    else:
        ground_truth_ate = gt_data  # Old format (backward compatibility)
        ground_truth_baseline = {}
        ground_truth_treated = {}
    
    ate_records = []
    
    # Get baseline predictions
    df_baseline = df[df["intervention"] == "baseline"]
    
    # Get interventions (excluding baseline)
    interventions = [i for i in df["intervention"].unique() if i != "baseline"]
    
    for intervention in interventions:
        df_treated = df[df["intervention"] == intervention]
        
        for pos_idx in df_treated["pos_idx"].unique():
            # Map position to variable name
            var_name = input_labels[int(pos_idx)] if int(pos_idx) < len(input_labels) else f"X{pos_idx+1}"
            
            for kfold in df_treated["kfold"].unique():
                # Get treated predictions for this variable/fold
                mask_treated = (df_treated["pos_idx"] == pos_idx) & (df_treated["kfold"] == kfold)
                subset_treated = df_treated[mask_treated]["pred_feat_0"]
                
                # Get baseline predictions for this variable/fold
                mask_baseline = (df_baseline["pos_idx"] == pos_idx) & (df_baseline["kfold"] == kfold)
                subset_baseline = df_baseline[mask_baseline]["pred_feat_0"]
                
                if len(subset_treated) == 0 or len(subset_baseline) == 0:
                    continue
                
                # Compute model ATE = mean(treated) - mean(baseline)
                # De-normalize both before computing difference
                treated_mean_norm = float(subset_treated.mean())
                baseline_mean_norm = float(subset_baseline.mean())
                
                treated_mean_raw = denormalize_value(treated_mean_norm, norm_stats, "input")
                baseline_mean_raw = denormalize_value(baseline_mean_norm, norm_stats, "input")
                
                model_ate = treated_mean_raw - baseline_mean_raw
                
                # Ground truth ATE lookup
                true_ate = ground_truth_ate.get(intervention, {}).get(var_name)
                
                # Ground truth baseline and treatment lookup
                # Baseline format: {"S1": {"X1": value, ...}, "S2": {...}, ...}
                # Treatment format: {"S1=0.5": {"X1": value, ...}, ...}
                # Parse intervention to get source variable (e.g., "S1=0.5" -> "S1")
                source_var = intervention.split("=")[0] if "=" in intervention else intervention
                true_baseline = ground_truth_baseline.get(source_var, {}).get(var_name)
                true_treatment = ground_truth_treated.get(intervention, {}).get(var_name)
                
                # Compute errors
                abs_error = abs(model_ate - true_ate) if true_ate is not None else None
                rel_error = abs_error / abs(true_ate) if (true_ate and abs(true_ate) > 1e-10) else None
                
                ate_records.append({
                    "intervention": intervention,
                    "variable": var_name,
                    "kfold": kfold,
                    "model_ate": model_ate,
                    "model_treated_raw": treated_mean_raw,
                    "model_baseline_raw": baseline_mean_raw,
                    "true_ate": true_ate,
                    "true_baseline": true_baseline,
                    "true_treatment": true_treatment,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "n_samples": len(subset_treated),
                })
    
    return pd.DataFrame(ate_records)


# =============================================================================
# Main Evaluation Function - Monte Carlo Version
# =============================================================================

def eval_ate_mc(experiment: str, n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    """
    Evaluate ATE using Monte Carlo sampled S values.
    
    This ensures model ATE evaluation uses the SAME S distribution as ground truth,
    providing a fair comparison. The test split is NOT used.
    
    For each intervention do(S_j=s):
    1. Sample fresh S from noise model with S_j clamped to s
    2. Run model forward pass: S → X predictions
    3. Compute mean prediction = E[X | do(S_j=s)]
    4. Compare with ground truth
    
    Args:
        experiment: Path to the experiment folder
        n_samples: Number of MC samples (default 50000, matching ground truth)
        seed: Random seed for MC sampling
        
    Returns:
        DataFrame with ATE metrics per intervention × variable × fold
    """
    print(f"Evaluating ATE (Monte Carlo) for: {experiment}")
    
    # =========================================================================
    # Load metadata
    # =========================================================================
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    if not config_files:
        raise ValueError(f"No config file found in {experiment}")
    
    config = OmegaConf.load(join(experiment, config_files[0]))
    dataset_name = config.get("data", {}).get("dataset")
    
    # DAG-sweep runs keep their datasets under the run folder; fall back to
    # <repo>/data for classic experiments.
    datadir_path = resolve_datadir(experiment=experiment)
    metadata = load_dataset_metadata(datadir_path, dataset_name)
    if not metadata:
        raise ValueError(f"Dataset metadata not found for '{dataset_name}'")
    
    print(f"  Dataset: {dataset_name}")
    
    # =========================================================================
    # Load SCM for MC sampling
    # =========================================================================
    try:
        scm_dataset = get_scm_for_dataset(dataset_name, datadir_path=datadir_path)
        print(f"  Loaded SCM for MC sampling")
    except ValueError as e:
        # NOTE: do *not* call eval_ate()/eval_ate_mc() here. eval_ate is an
        # alias for this very function, so re-entering it recursed infinitely
        # for every dataset outside SCM_REGISTRY (e.g. sampled random DAGs).
        print(f"  Warning: {e}")
        print("  Skipping ATE evaluation (no SCM available for MC sampling).")
        return pd.DataFrame()
    
    # =========================================================================
    # Load ATE ground truth and normalization
    # =========================================================================
    ate_ground_truth = load_ate_ground_truth(datadir_path, dataset_name)
    norm_stats = load_normalization_stats(datadir_path, dataset_name)
    
    if ate_ground_truth is None:
        raise ValueError(f"ate_ground_truth.json not found for '{dataset_name}'")
    if norm_stats is None:
        raise ValueError(f"normalization.json not found for '{dataset_name}'")
    
    # =========================================================================
    # Setup output directories
    # =========================================================================
    eval_path = join(experiment, "eval", "eval_ate_mc", "files")
    makedirs(eval_path, exist_ok=True)
    
    predictions_file = join(eval_path, "predictions_mc.csv")
    ate_csv_file = join(eval_path, "ate_metrics_mc.csv")
    ate_json_file = join(eval_path, "ate_metrics_mc.json")
    
    # =========================================================================
    # Load interventions from ate_ground_truth.json
    # =========================================================================
    intervention_config = get_interventions_from_ground_truth(ate_ground_truth)
    
    if not intervention_config:
        raise ValueError(
            f"No interventions found in ate_ground_truth.json for '{dataset_name}'. "
            f"Ensure the 'interventions' key is present or regenerate the dataset."
        )
    
    source_labels = metadata["variable_info"].get("source_labels", [])
    input_labels = metadata["variable_info"].get("input_labels", [])
    
    # Load the correct variable index mappings
    # If shared_embedding was used, var_idx_map has all variables (1-10)
    # If NOT shared_embedding, source and input have SEPARATE mappings (each starting from 1)
    var_idx_map = metadata.get("variable_index_map", {})
    
    # Check if separate mappings exist (shared_embedding=False case)
    source_vars_map_path = join(datadir_path, dataset_name, "source_vars_map.json")
    input_vars_map_path = join(datadir_path, dataset_name, "input_vars_map.json")
    
    if exists(source_vars_map_path) and exists(input_vars_map_path):
        # Load separate mappings (shared_embedding=False)
        with open(source_vars_map_path, 'r') as f:
            source_vars_map = json.load(f)
        with open(input_vars_map_path, 'r') as f:
            input_vars_map = json.load(f)
        print(f"  Using separate source/input variable mappings (shared_embedding=False)")
        print(f"    Source: {source_vars_map}")
        print(f"    Input: {input_vars_map}")
    else:
        # Use shared mapping from metadata
        source_vars_map = {k: v for k, v in var_idx_map.items() if k in source_labels}
        input_vars_map = {k: v for k, v in var_idx_map.items() if k in input_labels}
        print(f"  Using shared variable mapping (shared_embedding=True)")
        print(f"    All: {var_idx_map}")
    
    print(f"  Interventions (from dataset): {list(intervention_config.keys())}")
    print(f"  MC samples: {n_samples}")
    
    # =========================================================================
    # Run MC predictions
    # =========================================================================
    # Required columns in the predictions DataFrame
    REQUIRED_COLUMNS = ["intervention", "pos_idx", "variable", "kfold", "pred_feat_0"]
    
    # Check if cached file exists AND has valid content
    cache_valid = False
    if exists(predictions_file):
        try:
            file_size = Path(predictions_file).stat().st_size
            if file_size > 100:  # File should be at least 100 bytes to have meaningful content
                df = pd.read_csv(predictions_file)
                # Verify required columns exist
                missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
                if len(df) > 0 and not missing_cols:
                    print("  Loading cached MC predictions...")
                    cache_valid = True
                elif missing_cols:
                    print(f"  Cached file missing columns: {missing_cols}, regenerating...")
                else:
                    print("  Cached file empty, regenerating...")
            else:
                print("  Cached file too small, regenerating...")
        except Exception as e:
            print(f"  Error loading cache: {e}, regenerating...")
    
    if not cache_valid:
        # Auto-detect checkpoint type: causal models → best_causal, baselines → best_reconstruction
        ckpt_type = infer_checkpoint_type(config)
        print(f"  Checkpoint type: {ckpt_type}")
        print("  Running MC predictions...")
        df = run_mc_predictions(
            experiment_path=experiment,
            scm_dataset=scm_dataset,
            intervention_config=intervention_config,
            norm_stats=norm_stats,
            source_vars_map=source_vars_map,
            input_vars_map=input_vars_map,
            source_labels=source_labels,
            input_labels=input_labels,
            n_samples=n_samples,
            seed=seed,
            checkpoint_type=ckpt_type,
        )
        df.to_csv(predictions_file, index=False)
        print(f"  Saved MC predictions to {predictions_file}")
    
    # =========================================================================
    # Compute ATE metrics
    # =========================================================================
    df_ate = compute_ate_metrics_mc(
        df=df,
        ate_ground_truth=ate_ground_truth,
        norm_stats=norm_stats,
        input_labels=input_labels,
    )
    
    # Save CSV
    df_ate.to_csv(ate_csv_file, index=False)
    
    # =========================================================================
    # Compute summary and save JSON
    # =========================================================================
    abs_errors = df_ate["abs_error"].dropna()
    rel_errors = df_ate["rel_error"].dropna()
    
    # Per-intervention-variable summary
    summary_records = []
    for intervention in df_ate["intervention"].unique():
        for variable in df_ate["variable"].unique():
            mask = (df_ate["intervention"] == intervention) & (df_ate["variable"] == variable)
            subset = df_ate[mask]
            if len(subset) == 0:
                continue
            
            abs_err = subset["abs_error"].dropna()
            rel_err = subset["rel_error"].dropna()
            
            summary_records.append({
                "intervention": intervention,
                "variable": variable,
                "true_ate": float(subset["true_ate"].iloc[0]) if subset["true_ate"].notna().any() else None,
                "true_baseline": float(subset["true_baseline"].iloc[0]) if subset["true_baseline"].notna().any() else None,
                "true_treatment": float(subset["true_treatment"].iloc[0]) if subset["true_treatment"].notna().any() else None,
                "model_ate_mean": float(subset["model_ate"].mean()),
                "model_ate_std": float(subset["model_ate"].std()),
                "abs_error_mean": float(abs_err.mean()) if len(abs_err) > 0 else None,
                "abs_error_std": float(abs_err.std()) if len(abs_err) > 0 else None,
                "rel_error_mean": float(rel_err.mean()) if len(rel_err) > 0 else None,
                "n_folds": len(subset),
            })
    
    ate_json = {
        "dataset": dataset_name,
        "interventions": list(intervention_config.keys()),
        "method": "monte_carlo",
        "n_samples": n_samples,
        "summary": {
            "mean_absolute_error": float(abs_errors.mean()) if len(abs_errors) > 0 else None,
            "std_absolute_error": float(abs_errors.std()) if len(abs_errors) > 0 else None,
            "median_absolute_error": float(abs_errors.median()) if len(abs_errors) > 0 else None,
            "mean_relative_error": float(rel_errors.mean()) if len(rel_errors) > 0 else None,
            "n_comparisons": len(df_ate),
        },
        "per_intervention_variable": summary_records,
    }
    
    with open(ate_json_file, 'w') as f:
        json.dump(ate_json, f, indent=2)
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n  === ATE Results (Monte Carlo) ===")
    print(f"  Mean Absolute Error: {abs_errors.mean():.4f} ± {abs_errors.std():.4f}" if len(abs_errors) > 0 else "  No ATE computed")
    if len(rel_errors) > 0:
        print(f"  Mean Relative Error: {rel_errors.mean():.2%}")
    print(f"  Saved: {ate_csv_file}")
    print(f"  Saved: {ate_json_file}")
    
    return df_ate


def compute_ate_metrics_mc(
    df: pd.DataFrame,
    ate_ground_truth: dict,
    norm_stats: dict,
    input_labels: List[str],
) -> pd.DataFrame:
    """
    Compute ATE metrics for MC predictions.
    
    In MC mode, baseline is computed per-source (do(S_j=0) for each S_j individually).
    
    Args:
        df: DataFrame with MC predictions
        ate_ground_truth: Dict from ate_ground_truth.json
        norm_stats: Dict from normalization.json
        input_labels: List of input variable names
        
    Returns:
        DataFrame with ATE metrics per intervention × variable × fold
    """
    gt_method = "monte_carlo"
    gt_data = ate_ground_truth.get(gt_method, {})
    
    if "ate" in gt_data:
        ground_truth_ate = gt_data["ate"]
        ground_truth_baseline = gt_data.get("baseline", {})
        ground_truth_treated = gt_data.get("treated", {})
    else:
        ground_truth_ate = gt_data
        ground_truth_baseline = {}
        ground_truth_treated = {}
    
    ate_records = []
    
    # Get treated interventions (exclude baselines)
    treated_interventions = [i for i in df["intervention"].unique() 
                            if not i.endswith("_baseline") and i != "baseline"]
    
    for intervention in treated_interventions:
        # Parse intervention to get source variable (e.g., "S1=0.5" -> "S1")
        source_var = intervention.split("=")[0] if "=" in intervention else intervention
        baseline_label = f"{source_var}=0_baseline"
        
        df_treated = df[df["intervention"] == intervention]
        df_baseline = df[df["intervention"] == baseline_label]
        
        if len(df_baseline) == 0:
            # Fallback to global baseline
            df_baseline = df[df["intervention"] == "baseline"]
        
        for pos_idx in df_treated["pos_idx"].unique():
            var_name = input_labels[int(pos_idx)] if int(pos_idx) < len(input_labels) else f"X{pos_idx+1}"
            
            for kfold in df_treated["kfold"].unique():
                # Get treated prediction for this variable/fold
                mask_treated = (df_treated["pos_idx"] == pos_idx) & (df_treated["kfold"] == kfold)
                subset_treated = df_treated[mask_treated]
                
                # Get baseline prediction for this variable/fold
                mask_baseline = (df_baseline["pos_idx"] == pos_idx) & (df_baseline["kfold"] == kfold)
                subset_baseline = df_baseline[mask_baseline]
                
                if len(subset_treated) == 0 or len(subset_baseline) == 0:
                    continue
                
                # In MC mode, pred_feat_0 is already the mean over MC samples
                treated_mean_norm = float(subset_treated["pred_feat_0"].iloc[0])
                baseline_mean_norm = float(subset_baseline["pred_feat_0"].iloc[0])
                
                treated_mean_raw = denormalize_value(treated_mean_norm, norm_stats, "input")
                baseline_mean_raw = denormalize_value(baseline_mean_norm, norm_stats, "input")
                
                model_ate = treated_mean_raw - baseline_mean_raw
                
                # Ground truth lookup
                true_ate = ground_truth_ate.get(intervention, {}).get(var_name)
                true_baseline = ground_truth_baseline.get(source_var, {}).get(var_name)
                true_treatment = ground_truth_treated.get(intervention, {}).get(var_name)
                
                # Compute errors
                abs_error = abs(model_ate - true_ate) if true_ate is not None else None
                rel_error = abs_error / abs(true_ate) if (true_ate and abs(true_ate) > 1e-10) else None
                
                ate_records.append({
                    "intervention": intervention,
                    "variable": var_name,
                    "kfold": kfold,
                    "model_ate": model_ate,
                    "model_treated_raw": treated_mean_raw,
                    "model_baseline_raw": baseline_mean_raw,
                    "true_ate": true_ate,
                    "true_baseline": true_baseline,
                    "true_treatment": true_treatment,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "n_samples": int(subset_treated["n_samples"].iloc[0]) if "n_samples" in subset_treated.columns else 0,
                })
    
    return pd.DataFrame(ate_records)


# =============================================================================
# Default Entry Point - Monte Carlo ATE Evaluation
# =============================================================================

def eval_ate(experiment: str, n_samples: int = 50000, seed: int = 42, **kwargs) -> pd.DataFrame:
    """
    Default ATE evaluation using Monte Carlo sampling.
    
    Alias for eval_ate_mc() - ensures consistency between model evaluation
    and ground truth computation.
    """
    return eval_ate_mc(experiment, n_samples=n_samples, seed=seed)


def eval_interventions(experiment: str, **kwargs) -> pd.DataFrame:
    """Backward-compatible alias for eval_ate()."""
    return eval_ate(experiment, **kwargs)

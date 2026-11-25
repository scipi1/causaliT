"""
Intervention Isolation Test for StageCausaliT

This script diagnoses where information leakage (S1→Y2) occurs in the model
by comparing intermediate values across different S1 interventions.

Expected behavior according to the DAG:
- S1 → X1 → Y1 (S1 should affect X1 and Y1)
- S2,S3 → X2 → Y2 (S1 should NOT affect X2 or Y2)
- X2 → X1 (X2 affects X1 through self-attention in decoder 1)

The test runs forward passes with different S1 values and reports which
intermediate values change, pinpointing the source of any leakage.

Usage:
    python scripts/test_intervention_isolation.py --experiment experiments/stage_SoftMax_hard_scm6_54252179
"""

import os
import sys
from os.path import dirname, abspath, join, exists
from pathlib import Path
import argparse
import re

import torch
import numpy as np
from omegaconf import OmegaConf

# Add root to path
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)

from causaliT.training.forecasters.stage_causal_forecaster import StageCausalForecaster
from causaliT.training.stage_causal_dataloader import StageCausalDataModule
from causaliT.core.utils import load_dag_masks


def find_config_file(folder_path: str) -> str:
    """Find config file in experiment folder."""
    pattern = re.compile(r'^config_.*\.yaml$')
    for filename in os.listdir(folder_path):
        if pattern.match(filename):
            return join(folder_path, filename)
    raise FileNotFoundError(f"No config_*.yaml found in {folder_path}")


def find_best_checkpoint(checkpoints_dir: str) -> str:
    """Find best or last checkpoint."""
    if not exists(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")
    
    if 'best_checkpoint.ckpt' in checkpoint_files:
        return join(checkpoints_dir, 'best_checkpoint.ckpt')
    
    # Find highest epoch
    epoch_pattern = re.compile(r'epoch=(\d+)')
    max_epoch, best_ckpt = -1, checkpoint_files[0]
    for ckpt in checkpoint_files:
        match = epoch_pattern.search(ckpt)
        if match and int(match.group(1)) > max_epoch:
            max_epoch = int(match.group(1))
            best_ckpt = ckpt
    
    return join(checkpoints_dir, best_ckpt)


def apply_intervention(S: torch.Tensor, var_id: int, value: float, 
                       var_idx: int = 1, val_idx: int = 0) -> torch.Tensor:
    """Apply intervention to source tensor S."""
    S_intervened = S.clone()
    variable_ids = S[:, :, var_idx]
    mask = (variable_ids == var_id)
    S_intervened[mask, val_idx] = value
    return S_intervened


def run_forward_with_hooks(model, S, X, Y):
    """
    Run forward pass and capture intermediate values.
    
    Returns dict with:
    - pred_x: (B, X_len, 1) - predictions for X1, X2
    - pred_y: (B, Y_len, 1) - predictions for Y1, Y2
    - s_embedded: S embeddings
    - x_blanked_embedded: Blanked X embeddings (decoder 1 input)
    - x_for_dec2: X tensor used for decoder 2 (with predicted values)
    - x_for_dec2_embedded: X embeddings for decoder 2
    - y_blanked_embedded: Blanked Y embeddings (decoder 2 input)
    """
    model.eval()
    
    # Get configuration
    val_idx = model.val_idx
    
    # Prepare blanked tensors
    x_blanked = X.clone()
    x_blanked[:, :, val_idx] = 0.0
    y_blanked = Y.clone()
    y_blanked[:, :, val_idx] = 0.0
    
    # Get embeddings using model's shared embedding
    inner_model = model.model
    
    # Stage 1: S → X
    s_embedded = inner_model.shared_embedding(X=S)
    x_blanked_embedded = inner_model.shared_embedding(X=x_blanked)
    
    # Get masks
    s_mask = inner_model.shared_embedding.get_mask(X=S)
    x_mask = inner_model.shared_embedding.get_mask(X=x_blanked)
    
    # Get positional info
    x_input_pos = inner_model.shared_embedding.pass_var(X=x_blanked)
    
    # Get hard masks
    hard_masks = model.get_hard_masks() if model.use_hard_masks else None
    dec1_cross_hard = hard_masks.get('dec1_cross') if hard_masks else None
    dec1_self_hard = hard_masks.get('dec1_self') if hard_masks else None
    dec2_cross_hard = hard_masks.get('dec2_cross') if hard_masks else None
    dec2_self_hard = hard_masks.get('dec2_self') if hard_masks else None
    
    # Run decoder 1
    dec1_out, dec1_cross_att, dec1_self_att, _, _ = inner_model.decoder1(
        X=x_blanked_embedded,
        external_context=s_embedded,
        self_mask_miss_k=x_mask,
        self_mask_miss_q=x_mask,
        cross_mask_miss_k=s_mask,
        cross_mask_miss_q=x_mask,
        dec_input_pos=x_input_pos,
        causal_mask=inner_model.dec1_causal_mask,
        cross_hard_mask=dec1_cross_hard,
        self_hard_mask=dec1_self_hard,
    )
    
    # Predict X
    pred_x = inner_model.forecaster(dec1_out)
    
    # Stage 2: X → Y
    # Construct x_for_dec2 with predicted values
    x_for_dec2 = x_blanked.clone()
    x_for_dec2[:, :, val_idx] = pred_x.squeeze(-1)
    
    # Embed x_for_dec2
    x_for_dec2_embedded = inner_model.shared_embedding(X=x_for_dec2)
    x_for_dec2_pos = inner_model.shared_embedding.pass_var(X=x_for_dec2)
    x_for_dec2_mask = inner_model.shared_embedding.get_mask(X=x_for_dec2)
    
    # Embed blanked Y
    y_blanked_embedded = inner_model.shared_embedding(X=y_blanked)
    y_input_pos = inner_model.shared_embedding.pass_var(X=y_blanked)
    y_mask = inner_model.shared_embedding.get_mask(X=y_blanked)
    
    # Run decoder 2
    dec2_out, dec2_cross_att, dec2_self_att, _, _ = inner_model.decoder2(
        X=y_blanked_embedded,
        external_context=x_for_dec2_embedded,
        self_mask_miss_k=y_mask,
        self_mask_miss_q=y_mask,
        cross_mask_miss_k=x_for_dec2_mask,
        cross_mask_miss_q=y_mask,
        dec_input_pos=y_input_pos,
        causal_mask=inner_model.dec2_causal_mask,
        cross_hard_mask=dec2_cross_hard,
        self_hard_mask=dec2_self_hard,
    )
    
    # Predict Y
    pred_y = inner_model.forecaster(dec2_out)
    
    return {
        'pred_x': pred_x,
        'pred_y': pred_y,
        's_embedded': s_embedded,
        'x_blanked_embedded': x_blanked_embedded,
        'x_for_dec2': x_for_dec2,
        'x_for_dec2_embedded': x_for_dec2_embedded,
        'y_blanked_embedded': y_blanked_embedded,
        'dec1_out': dec1_out,
        'dec2_out': dec2_out,
        'dec1_cross_att': dec1_cross_att,
        'dec1_self_att': dec1_self_att,
        'dec2_cross_att': dec2_cross_att,
        'dec2_self_att': dec2_self_att,
    }


def compute_differences(results_base, results_interv, name_base="baseline", name_interv="intervention"):
    """Compute differences between two forward pass results."""
    diffs = {}
    
    for key in results_base.keys():
        if isinstance(results_base[key], torch.Tensor):
            diff = (results_interv[key] - results_base[key]).abs().mean().item()
            diffs[key] = diff
        elif isinstance(results_base[key], list):
            # For attention weights (list of tensors per layer)
            layer_diffs = []
            for i, (base_att, interv_att) in enumerate(zip(results_base[key], results_interv[key])):
                if base_att is not None and interv_att is not None:
                    layer_diffs.append((interv_att - base_att).abs().mean().item())
            diffs[key] = layer_diffs if layer_diffs else None
    
    return diffs


def print_report(diffs, intervention_name, tolerance=1e-6):
    """Print formatted report of differences."""
    print(f"\n{'='*60}")
    print(f"Intervention: {intervention_name}")
    print('='*60)
    
    # Expected changes (should be non-zero)
    expected_changes = {
        's_embedded': "S embedding changes with S1 (EXPECTED)",
        'pred_x': "pred_X changes (X1 should change, X2 should NOT)",
        'pred_y': "pred_Y changes (Y1 should change, Y2 should NOT)",
    }
    
    # Positions that should NOT change with S1 intervention
    # According to DAG: S1 → X1 → Y1, S2,S3 → X2 → Y2
    should_not_change = {
        'x_blanked_embedded': "Blanked X embedding (S1-independent)",
        'y_blanked_embedded': "Blanked Y embedding (S1-independent)",
    }
    
    print("\n--- Stage 1 (S → X) ---")
    print(f"  s_embedded change:        {diffs.get('s_embedded', 'N/A'):.6f} (EXPECTED to change)")
    print(f"  x_blanked_embedded change: {diffs.get('x_blanked_embedded', 'N/A'):.6f} (should be 0)")
    print(f"  dec1_out change:          {diffs.get('dec1_out', 'N/A'):.6f}")
    print(f"  pred_x overall change:    {diffs.get('pred_x', 'N/A'):.6f}")
    
    print("\n--- Stage 2 (X → Y) ---")
    print(f"  x_for_dec2 change:        {diffs.get('x_for_dec2', 'N/A'):.6f}")
    print(f"  x_for_dec2_embedded change: {diffs.get('x_for_dec2_embedded', 'N/A'):.6f}")
    print(f"  y_blanked_embedded change: {diffs.get('y_blanked_embedded', 'N/A'):.6f} (should be 0)")
    print(f"  dec2_out change:          {diffs.get('dec2_out', 'N/A'):.6f}")
    print(f"  pred_y overall change:    {diffs.get('pred_y', 'N/A'):.6f}")
    
    # Print attention weight changes
    print("\n--- Attention Changes ---")
    for key in ['dec1_cross_att', 'dec1_self_att', 'dec2_cross_att', 'dec2_self_att']:
        val = diffs.get(key)
        if val is not None:
            print(f"  {key}: {val}")


def analyze_position_specific_changes(results_base, results_interv, batch_idx=0):
    """Analyze changes at specific positions (X1 vs X2, Y1 vs Y2)."""
    print("\n" + "="*60)
    print("POSITION-SPECIFIC ANALYSIS (sample 0)")
    print("="*60)
    
    # Analyze pred_x positions
    pred_x_base = results_base['pred_x'][batch_idx].squeeze()  # (X_len,)
    pred_x_interv = results_interv['pred_x'][batch_idx].squeeze()
    
    x1_change = abs(pred_x_interv[0].item() - pred_x_base[0].item())
    x2_change = abs(pred_x_interv[1].item() - pred_x_base[1].item())
    
    print("\n--- pred_X Position Analysis ---")
    print(f"  pred_X1 baseline: {pred_x_base[0].item():.6f}")
    print(f"  pred_X1 intervened: {pred_x_interv[0].item():.6f}")
    print(f"  pred_X1 change: {x1_change:.6f} (EXPECTED - X1 depends on S1)")
    print()
    print(f"  pred_X2 baseline: {pred_x_base[1].item():.6f}")
    print(f"  pred_X2 intervened: {pred_x_interv[1].item():.6f}")
    print(f"  pred_X2 change: {x2_change:.6f} ", end="")
    if x2_change > 1e-6:
        print("*** UNEXPECTED! X2 should NOT depend on S1 ***")
    else:
        print("(OK - X2 independent of S1)")
    
    # Analyze pred_y positions  
    pred_y_base = results_base['pred_y'][batch_idx].squeeze()  # (Y_len,)
    pred_y_interv = results_interv['pred_y'][batch_idx].squeeze()
    
    y1_change = abs(pred_y_interv[0].item() - pred_y_base[0].item())
    y2_change = abs(pred_y_interv[1].item() - pred_y_base[1].item())
    
    print("\n--- pred_Y Position Analysis ---")
    print(f"  pred_Y1 baseline: {pred_y_base[0].item():.6f}")
    print(f"  pred_Y1 intervened: {pred_y_interv[0].item():.6f}")
    print(f"  pred_Y1 change: {y1_change:.6f} (EXPECTED - Y1 depends on X1 → S1)")
    print()
    print(f"  pred_Y2 baseline: {pred_y_base[1].item():.6f}")
    print(f"  pred_Y2 intervened: {pred_y_interv[1].item():.6f}")
    print(f"  pred_Y2 change: {y2_change:.6f} ", end="")
    if y2_change > 1e-6:
        print("*** UNEXPECTED! Y2 should NOT depend on S1 ***")
    else:
        print("(OK - Y2 independent of S1)")
    
    # Analyze x_for_dec2 positions
    x_for_dec2_base = results_base['x_for_dec2'][batch_idx]  # (X_len, features)
    x_for_dec2_interv = results_interv['x_for_dec2'][batch_idx]
    
    print("\n--- x_for_dec2 Position Analysis (value at index 0) ---")
    x_for_dec2_x1_change = abs(x_for_dec2_interv[0, 0].item() - x_for_dec2_base[0, 0].item())
    x_for_dec2_x2_change = abs(x_for_dec2_interv[1, 0].item() - x_for_dec2_base[1, 0].item())
    
    print(f"  x_for_dec2[X1, value] baseline: {x_for_dec2_base[0, 0].item():.6f}")
    print(f"  x_for_dec2[X1, value] intervened: {x_for_dec2_interv[0, 0].item():.6f}")
    print(f"  x_for_dec2[X1, value] change: {x_for_dec2_x1_change:.6f} (EXPECTED)")
    print()
    print(f"  x_for_dec2[X2, value] baseline: {x_for_dec2_base[1, 0].item():.6f}")
    print(f"  x_for_dec2[X2, value] intervened: {x_for_dec2_interv[1, 0].item():.6f}")
    print(f"  x_for_dec2[X2, value] change: {x_for_dec2_x2_change:.6f} ", end="")
    if x_for_dec2_x2_change > 1e-6:
        print("*** UNEXPECTED! X2 value should NOT depend on S1 ***")
    else:
        print("(OK)")
    
    # Analyze x_for_dec2_embedded positions
    x_emb_base = results_base['x_for_dec2_embedded'][batch_idx]  # (X_len, d_model)
    x_emb_interv = results_interv['x_for_dec2_embedded'][batch_idx]
    
    print("\n--- x_for_dec2_embedded Position Analysis ---")
    x_emb_x1_change = (x_emb_interv[0] - x_emb_base[0]).abs().mean().item()
    x_emb_x2_change = (x_emb_interv[1] - x_emb_base[1]).abs().mean().item()
    
    print(f"  x_for_dec2_embedded[X1] mean change: {x_emb_x1_change:.6f} (EXPECTED)")
    print(f"  x_for_dec2_embedded[X2] mean change: {x_emb_x2_change:.6f} ", end="")
    if x_emb_x2_change > 1e-6:
        print("*** UNEXPECTED! X2 embedding should NOT depend on S1 ***")
    else:
        print("(OK)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    bugs_found = []
    if x2_change > 1e-6:
        bugs_found.append("pred_X2 depends on S1 (BUG in Decoder 1)")
    if x_for_dec2_x2_change > 1e-6:
        bugs_found.append("x_for_dec2[X2] depends on S1 (BUG in Stage 1→2 data flow)")
    if x_emb_x2_change > 1e-6:
        bugs_found.append("x_for_dec2_embedded[X2] depends on S1 (BUG in embedding)")
    if y2_change > 1e-6 and x2_change < 1e-6:
        bugs_found.append("pred_Y2 depends on S1 but pred_X2 doesn't (BUG in Decoder 2)")
    elif y2_change > 1e-6:
        bugs_found.append("pred_Y2 depends on S1 (caused by upstream bug)")
    
    if bugs_found:
        print("\n*** BUGS DETECTED ***")
        for i, bug in enumerate(bugs_found, 1):
            print(f"  {i}. {bug}")
    else:
        print("\n✓ No unexpected dependencies detected!")
    
    return {
        'x1_change': x1_change,
        'x2_change': x2_change,
        'y1_change': y1_change,
        'y2_change': y2_change,
        'x_for_dec2_x2_change': x_for_dec2_x2_change,
        'x_emb_x2_change': x_emb_x2_change,
    }


def main():
    parser = argparse.ArgumentParser(description='Test intervention isolation in StageCausaliT')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Path to experiment folder')
    parser.add_argument('--datadir', type=str, default='data/',
                        help='Path to data directory')
    parser.add_argument('--kfold', type=str, default='k_0',
                        help='Which k-fold to use (default: k_0)')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--intervention_var', type=int, default=1,
                        help='Variable ID to intervene on (default: 1 for S1)')
    parser.add_argument('--intervention_values', type=str, default='0,5,-5',
                        help='Comma-separated intervention values')
    
    args = parser.parse_args()
    
    # Setup paths
    experiment_path = args.experiment
    datadir_path = args.datadir
    kfold_path = join(experiment_path, args.kfold)
    checkpoints_dir = join(kfold_path, 'checkpoints')
    
    print(f"Experiment: {experiment_path}")
    print(f"K-fold: {args.kfold}")
    
    # Load config
    config_path = find_config_file(experiment_path)
    config = OmegaConf.load(config_path)
    print(f"Config: {config_path}")
    
    # Find checkpoint
    checkpoint_path = find_best_checkpoint(checkpoints_dir)
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load model
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StageCausalForecaster.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    # Load hard masks if enabled
    if model.use_hard_masks and not model._hard_masks_loaded:
        dataset_dir = join(datadir_path, config["data"]["dataset"])
        mask_files = config["training"].get("hard_mask_files", None)
        if mask_files:
            masks = load_dag_masks(dataset_dir, mask_files, device=str(device))
            if masks:
                model._hard_masks = masks
                model._hard_masks_loaded = True
                for name, mask in masks.items():
                    model.register_buffer(f'hard_mask_{name}', mask.to(device))
                print("Hard masks loaded.")
    
    print(f"use_hard_masks: {model.use_hard_masks}")
    print(f"hard_masks_loaded: {model._hard_masks_loaded}")
    
    # Create data module
    dm = StageCausalDataModule(
        data_dir=join(datadir_path, config["data"]["dataset"]),
        input_file=config["data"]["filename_input"],
        batch_size=args.n_samples,
        num_workers=1,
        data_format="float32",
        seed=config["training"]["seed"],
    )
    dm.prepare_data()
    dm.setup(stage=None)
    
    # Get one batch and move to device
    for batch in dm.test_dataloader():
        S, X, Y = batch
        S, X, Y = S.to(device), X.to(device), Y.to(device)
        break
    
    print(f"\nData shapes: S={S.shape}, X={X.shape}, Y={Y.shape}")
    print(f"Device: {device}")
    
    # Parse intervention values
    intervention_values = [float(v) for v in args.intervention_values.split(',')]
    
    # Run baseline (no intervention)
    print("\n" + "="*70)
    print("Running baseline forward pass...")
    print("="*70)
    
    with torch.no_grad():
        results_base = run_forward_with_hooks(model, S, X, Y)
    
    # Run interventions and compare
    for interv_val in intervention_values:
        print(f"\n" + "="*70)
        print(f"Running intervention: S{args.intervention_var} = {interv_val}")
        print("="*70)
        
        S_interv = apply_intervention(S, args.intervention_var, interv_val)
        
        with torch.no_grad():
            results_interv = run_forward_with_hooks(model, S_interv, X, Y)
        
        # Compute overall differences
        diffs = compute_differences(results_base, results_interv)
        print_report(diffs, f"S{args.intervention_var} = {interv_val}")
        
        # Analyze position-specific changes
        position_analysis = analyze_position_specific_changes(
            results_base, results_interv, batch_idx=0
        )
    
    print("\n" + "="*70)
    print("Test complete.")
    print("="*70)


if __name__ == "__main__":
    main()

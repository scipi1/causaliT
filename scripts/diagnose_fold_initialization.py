"""
Diagnostic Script: Cross-Validation Fold Initialization Analysis

This script diagnoses why SVFA models learn different DAGs across cross-validation folds
even with the same seed setting. The root cause is that:

1. seed_everything() is called ONCE at the beginning, not per-fold
2. nn.Embedding (and nn.Linear) weights are initialized from N(0,1)
3. By the time fold N starts, the random state has been consumed differently

This script:
1. Shows that model initialization differs across folds with current approach
2. Demonstrates the fix: reset seed before each fold's model creation
3. Verifies that data splits are different while model initialization is identical

Author: Diagnostic script for causaliT SVFA analysis
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import seed_everything
from sklearn.model_selection import KFold
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causaliT.core.modules.embedding_layers import nn_embedding
from causaliT.core.modules.orthogonal_embedding import OrthogonalMaskEmbedding


def get_rng_state_hash():
    """Get a hash of current PyTorch random state for comparison."""
    state = torch.get_rng_state()
    return hash(state.numpy().tobytes()[:100])  # First 100 bytes for quick hash


def compute_embedding_stats(embedding_module):
    """Compute statistics of embedding weights for comparison."""
    stats = {}
    for name, param in embedding_module.named_parameters():
        if param.numel() > 0:
            stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'sum': param.data.sum().item(),
                'first_5': param.data.flatten()[:5].tolist() if param.numel() >= 5 else param.data.flatten().tolist()
            }
    return stats


def compare_embedding_stats(stats1, stats2, name=""):
    """Compare two embedding statistics dictionaries."""
    all_match = True
    for key in stats1:
        if key in stats2:
            match = (abs(stats1[key]['sum'] - stats2[key]['sum']) < 1e-10)
            if not match:
                print(f"  {name}/{key}: DIFFERENT")
                print(f"    Stats1: sum={stats1[key]['sum']:.6f}, first_5={stats1[key]['first_5']}")
                print(f"    Stats2: sum={stats2[key]['sum']:.6f}, first_5={stats2[key]['first_5']}")
                all_match = False
            else:
                print(f"  {name}/{key}: IDENTICAL ✓")
    return all_match


def test_nn_embedding_initialization():
    """Test 1: Demonstrate nn.Embedding random initialization behavior."""
    print("\n" + "="*80)
    print("TEST 1: nn.Embedding Initialization Analysis")
    print("="*80)
    
    print("\n📌 PyTorch nn.Embedding initializes weights from N(0,1) by default")
    print("   This means each call to nn.Embedding() consumes random numbers from the RNG state.\n")
    
    seed = 42
    num_embeddings = 10
    embedding_dim = 8
    
    # Test 1a: Same seed, consecutive creations
    print("--- Test 1a: Same seed, multiple consecutive creations ---")
    seed_everything(seed)
    emb1 = nn.Embedding(num_embeddings, embedding_dim)
    emb2 = nn.Embedding(num_embeddings, embedding_dim)  # Will be DIFFERENT!
    
    weights_match = torch.allclose(emb1.weight, emb2.weight)
    print(f"  emb1 == emb2 after consecutive creation: {weights_match}")
    print(f"  emb1.weight[0,:3]: {emb1.weight[0,:3].tolist()}")
    print(f"  emb2.weight[0,:3]: {emb2.weight[0,:3].tolist()}")
    
    # Test 1b: Reset seed before each creation
    print("\n--- Test 1b: Reset seed before each creation ---")
    seed_everything(seed)
    emb1 = nn.Embedding(num_embeddings, embedding_dim)
    
    seed_everything(seed)  # RESET!
    emb2 = nn.Embedding(num_embeddings, embedding_dim)
    
    weights_match = torch.allclose(emb1.weight, emb2.weight)
    print(f"  emb1 == emb2 with seed reset: {weights_match} ✓")
    print(f"  emb1.weight[0,:3]: {emb1.weight[0,:3].tolist()}")
    print(f"  emb2.weight[0,:3]: {emb2.weight[0,:3].tolist()}")


def test_orthogonal_embedding_initialization():
    """Test 2: OrthogonalMaskEmbedding contains nn.Linear which is also random."""
    print("\n" + "="*80)
    print("TEST 2: OrthogonalMaskEmbedding Initialization Analysis")
    print("="*80)
    
    print("\n📌 OrthogonalMaskEmbedding contains nn.Linear(1, d_model) for value embedding")
    print("   The binary masks are deterministic, but the value embedding is random!\n")
    
    seed = 42
    num_variables = 3
    d_model = 12
    
    # Test without seed reset
    print("--- Test 2a: Without seed reset between creations ---")
    seed_everything(seed)
    
    emb1 = OrthogonalMaskEmbedding(num_variables=num_variables, d_model=d_model, freeze=False)
    emb2 = OrthogonalMaskEmbedding(num_variables=num_variables, d_model=d_model, freeze=False)
    
    # Compare value_embedding weights
    w1 = emb1.value_embedding.weight.data
    w2 = emb2.value_embedding.weight.data
    
    weights_match = torch.allclose(w1, w2)
    print(f"  value_embedding weights match: {weights_match}")
    print(f"  emb1 weight sum: {w1.sum().item():.6f}")
    print(f"  emb2 weight sum: {w2.sum().item():.6f}")
    
    # Binary masks should always match (they're deterministic)
    masks_match = torch.equal(emb1.binary_masks, emb2.binary_masks)
    print(f"  binary_masks match: {masks_match} ✓ (always, they're deterministic)")
    
    # Test with seed reset
    print("\n--- Test 2b: With seed reset between creations ---")
    seed_everything(seed)
    emb1 = OrthogonalMaskEmbedding(num_variables=num_variables, d_model=d_model, freeze=False)
    
    seed_everything(seed)  # RESET!
    emb2 = OrthogonalMaskEmbedding(num_variables=num_variables, d_model=d_model, freeze=False)
    
    w1 = emb1.value_embedding.weight.data
    w2 = emb2.value_embedding.weight.data
    
    weights_match = torch.allclose(w1, w2)
    print(f"  value_embedding weights match: {weights_match} ✓")


def test_kfold_simulation():
    """Test 3: Simulate k-fold cross-validation model initialization."""
    print("\n" + "="*80)
    print("TEST 3: K-Fold Cross-Validation Simulation")
    print("="*80)
    
    print("\n📌 This test simulates what happens in trainer.py during k-fold CV")
    print("   Current behavior: seed is set ONCE, model is recreated each fold\n")
    
    seed = 42
    k_folds = 3
    dataset_size = 100
    
    # Simulate dataset indices
    indices = np.arange(dataset_size)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    def create_simple_model():
        """Create a simple model with random initialization."""
        return nn.Sequential(
            nn.Embedding(10, 8),
            nn.Linear(8, 4)
        )
    
    # CURRENT BEHAVIOR: Seed set once
    print("--- Current Behavior (seed set ONCE at start) ---")
    seed_everything(seed)
    
    current_behavior_weights = []
    current_behavior_rng_states = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        rng_state = get_rng_state_hash()
        current_behavior_rng_states.append(rng_state)
        
        model = create_simple_model()
        emb_weight = model[0].weight.data.clone()
        current_behavior_weights.append(emb_weight)
        
        print(f"  Fold {fold}: RNG state hash = {rng_state}")
        print(f"           Embedding weight[0,:3] = {emb_weight[0,:3].tolist()}")
        print(f"           Train size = {len(train_idx)}, Val size = {len(val_idx)}")
    
    # Check if weights are identical
    weights_identical = all(
        torch.allclose(current_behavior_weights[0], w) 
        for w in current_behavior_weights[1:]
    )
    print(f"\n  All fold weights identical: {weights_identical} {'✓' if weights_identical else '✗ PROBLEM!'}")
    
    # PROPOSED FIX: Reset seed before each fold
    print("\n--- Proposed Fix (seed reset BEFORE each fold's model creation) ---")
    seed_everything(seed)  # Initial seed for KFold split
    
    fixed_behavior_weights = []
    fixed_behavior_rng_states = []
    
    # Recreate KFold to match the same splits
    kfold_fixed = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(kfold_fixed.split(indices)):
        # RESET SEED BEFORE MODEL CREATION
        seed_everything(seed)
        
        rng_state = get_rng_state_hash()
        fixed_behavior_rng_states.append(rng_state)
        
        model = create_simple_model()
        emb_weight = model[0].weight.data.clone()
        fixed_behavior_weights.append(emb_weight)
        
        print(f"  Fold {fold}: RNG state hash = {rng_state}")
        print(f"           Embedding weight[0,:3] = {emb_weight[0,:3].tolist()}")
        print(f"           Train size = {len(train_idx)}, Val size = {len(val_idx)}")
    
    # Check if weights are identical
    weights_identical = all(
        torch.allclose(fixed_behavior_weights[0], w) 
        for w in fixed_behavior_weights[1:]
    )
    print(f"\n  All fold weights identical: {weights_identical} {'✓' if weights_identical else '✗'}")
    
    # Verify data splits are DIFFERENT
    print("\n--- Verification: Data Splits Are Different ---")
    kfold_verify = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_splits = list(kfold_verify.split(indices))
    
    for i in range(k_folds):
        for j in range(i+1, k_folds):
            train_overlap = len(set(fold_splits[i][0]) & set(fold_splits[j][0]))
            val_overlap = len(set(fold_splits[i][1]) & set(fold_splits[j][1]))
            print(f"  Fold {i} vs Fold {j}: train overlap = {train_overlap}, val overlap = {val_overlap}")
    
    print("\n  ✓ Data splits are different (as expected for k-fold CV)")


def test_cosine_similarity_initialization():
    """Test 4: Show how initial cosine similarities differ across folds."""
    print("\n" + "="*80)
    print("TEST 4: Initial Cosine Similarity Analysis")
    print("="*80)
    
    print("\n📌 This demonstrates why cosine similarities start from different values")
    print("   in different folds - it's due to different embedding initialization!\n")
    
    seed = 42
    num_variables = 3
    d_model = 12
    batch_size = 2
    seq_len = 3
    
    def compute_initial_cosine_similarity(emb_module):
        """Compute cosine similarity between embeddings of different variables."""
        # Create test input: (batch, seq, features=[value, var_id])
        X = torch.randn(batch_size, seq_len, 2)
        X[:, 0, 1] = 1  # Variable 1 (S1)
        X[:, 1, 1] = 2  # Variable 2 (S2)  
        X[:, 2, 1] = 3  # Variable 3 (S3)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = emb_module(X)  # (batch, seq, d_model)
        
        # Compute cosine similarities between variables
        cos_sim = nn.CosineSimilarity(dim=-1)
        
        # Average over batch
        emb_mean = embeddings.mean(dim=0)  # (seq, d_model)
        
        sims = {}
        sims['S1_S2'] = cos_sim(emb_mean[0:1], emb_mean[1:2]).item()
        sims['S1_S3'] = cos_sim(emb_mean[0:1], emb_mean[2:3]).item()
        sims['S2_S3'] = cos_sim(emb_mean[1:2], emb_mean[2:3]).item()
        
        return sims
    
    # Current behavior (no seed reset)
    print("--- Current Behavior: Different initial cosine similarities ---")
    seed_everything(seed)
    
    for fold in range(3):
        emb = OrthogonalMaskEmbedding(num_variables=num_variables, d_model=d_model, freeze=False)
        sims = compute_initial_cosine_similarity(emb)
        print(f"  Fold {fold}: S1-S2={sims['S1_S2']:.4f}, S1-S3={sims['S1_S3']:.4f}, S2-S3={sims['S2_S3']:.4f}")
    
    # Fixed behavior (seed reset)
    print("\n--- Fixed Behavior: Identical initial cosine similarities ---")
    
    for fold in range(3):
        seed_everything(seed)  # RESET before each fold
        emb = OrthogonalMaskEmbedding(num_variables=num_variables, d_model=d_model, freeze=False)
        sims = compute_initial_cosine_similarity(emb)
        print(f"  Fold {fold}: S1-S2={sims['S1_S2']:.4f}, S1-S3={sims['S1_S3']:.4f}, S2-S3={sims['S2_S3']:.4f}")


def test_rng_state_tracking():
    """Test 5: Track RNG state consumption during training simulation."""
    print("\n" + "="*80)
    print("TEST 5: RNG State Consumption Analysis")
    print("="*80)
    
    print("\n📌 This shows how many random numbers are consumed and how")
    print("   the RNG state diverges as training progresses.\n")
    
    seed = 42
    
    # Simulate training operations that consume random numbers
    def simulate_training_epoch():
        """Simulate operations that consume random numbers during training."""
        # Dropout
        dropout = nn.Dropout(0.1)
        x = torch.randn(32, 64)
        _ = dropout(x)
        
        # Data shuffling (simulated)
        _ = torch.randperm(1000)
        
        # Batch sampling
        _ = torch.randint(0, 100, (32,))
        
        # Noise for data augmentation
        _ = torch.randn(32, 10)
    
    seed_everything(seed)
    initial_state = get_rng_state_hash()
    print(f"  Initial RNG state hash: {initial_state}")
    
    for epoch in range(3):
        simulate_training_epoch()
        current_state = get_rng_state_hash()
        print(f"  After epoch {epoch}: RNG state hash = {current_state}")
    
    print("\n  💡 Notice how the RNG state changes after each epoch!")
    print("     This is why fold 1's model initialization differs from fold 0's.")


def print_recommendation():
    """Print the recommended fix."""
    print("\n" + "="*80)
    print("RECOMMENDATION: Fix for trainer.py")
    print("="*80)
    
    code = '''
# In trainer.py, modify the k-fold loop:

for fold, (train_local_idx, val_local_idx) in enumerate(kfold.split(train_val_idx)):
    
    # ============================================================
    # RESET SEED BEFORE MODEL CREATION FOR REPRODUCIBLE INITIALIZATION
    # This ensures all folds start with identical model weights
    # while still having different train/val data splits
    # ============================================================
    seed_everything(seed)
    
    # re-initialize the model at any fold
    model = create_model_instance(config, data_dir)
    
    # ... rest of fold training code ...
'''
    print(code)
    
    print("\n💡 KEY INSIGHT:")
    print("   With this fix, all folds will have:")
    print("   ✓ Identical model initialization (embeddings, attention weights, etc.)")
    print("   ✓ Different training data (due to k-fold split)")
    print("   ✓ Fair comparison of DAG learning across different data samples")
    print("\n   The k-fold split is controlled by random_state=seed in KFold(),")
    print("   which is separate from PyTorch's RNG used for model initialization.")


def main():
    """Run all diagnostic tests."""
    print("\n" + "#"*80)
    print("# DIAGNOSTIC: Cross-Validation Fold Initialization in SVFA")
    print("#"*80)
    
    test_nn_embedding_initialization()
    test_orthogonal_embedding_initialization()
    test_kfold_simulation()
    test_cosine_similarity_initialization()
    test_rng_state_tracking()
    print_recommendation()
    
    print("\n" + "#"*80)
    print("# DIAGNOSTIC COMPLETE")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()

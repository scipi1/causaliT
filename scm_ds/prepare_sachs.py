"""
Prepare the Sachs protein signaling dataset for CausaliT.

Downloads the observational subset (853 samples, 11 proteins) used by
GraN-DAG (Lachapelle et al., 2020) and converts it into the CausaliT
data format (npz + attention masks + metadata).

Sachs Ground Truth DAG (17 edges, 11 nodes):
    S variables (roots): PKC, Plcg
    X variables (non-roots): Raf, Mek, PIP2, PIP3, Erk, Akt, PKA, P38, Jnk

References:
    - Sachs et al. (2005). Causal Protein-Signaling Networks Derived from
      Multiparameter Single-Cell Data. Science 308(5721), 523–529.
    - Lachapelle et al. (2020). Gradient-Based Neural DAG Learning. ICLR.

Usage:
    python scm_ds/prepare_sachs.py [--save_dir data/sachs_observational]
"""

import argparse
import json
from os import makedirs
from os.path import join, exists, dirname, abspath
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Sachs Ground Truth DAG
# =============================================================================
# Consensus network from Sachs et al. (2005), 17 directed edges.
# Convention: (parent, child)

SACHS_EDGES = [
    ("PKC", "Raf"),
    ("PKC", "Mek"),
    ("PKC", "P38"),
    ("PKC", "Jnk"),
    ("PKC", "PKA"),
    ("Plcg", "PIP3"),
    ("Plcg", "PIP2"),
    ("PKA", "Raf"),
    ("PKA", "Mek"),
    ("PKA", "Erk"),
    ("PKA", "Akt"),
    ("PKA", "P38"),
    ("PKA", "Jnk"),
    ("Raf", "Mek"),
    ("Mek", "Erk"),
    ("PIP3", "PIP2"),
    ("Erk", "Akt"),
]

# Source variables (roots of the DAG — no parents)
SOURCE_LABELS = ["PKC", "Plcg"]

# Input variables (non-roots, in a fixed canonical order)
INPUT_LABELS = ["Raf", "Mek", "PIP2", "PIP3", "Erk", "Akt", "PKA", "P38", "Jnk"]

# All nodes
ALL_LABELS = SOURCE_LABELS + INPUT_LABELS


# =============================================================================
# Data Download
# =============================================================================

SACHS_URL = "https://www.bnlearn.com/research/sachs05/sachs.data.txt"


def download_sachs_data(save_path: str) -> pd.DataFrame:
    """
    Download the Sachs observational dataset from bnlearn.
    
    The file at bnlearn contains 7466 samples (all experimental conditions).
    Following GraN-DAG (Lachapelle et al., 2020), we use the first 853 rows
    which correspond to the purely observational condition (condition 1: 
    anti-CD3/CD28, no additional stimulus).
    
    Args:
        save_path: Path to save the raw CSV
        
    Returns:
        pd.DataFrame with 853 rows × 11 columns
    """
    import urllib.request
    
    raw_path = save_path + ".raw.txt"
    
    if not exists(raw_path):
        print(f"Downloading Sachs data from {SACHS_URL}...")
        urllib.request.urlretrieve(SACHS_URL, raw_path)
        print(f"  Saved raw file to {raw_path}")
    else:
        print(f"  Using cached raw file: {raw_path}")
    
    # Load full dataset (tab-separated)
    df_full = pd.read_csv(raw_path, sep="\t")
    print(f"  Full dataset: {df_full.shape[0]} samples × {df_full.shape[1]} variables")
    print(f"  Variables: {list(df_full.columns)}")
    
    # Take first 853 rows (observational condition)
    # This matches GraN-DAG's usage of the Sachs dataset
    N_OBS = 853
    df = df_full.iloc[:N_OBS].copy()
    print(f"  Observational subset: {df.shape[0]} samples (first {N_OBS} rows)")
    
    return df


# =============================================================================
# CausaliT Format Conversion
# =============================================================================

def build_attention_masks() -> tuple:
    """
    Build cross-attention and self-attention DAG masks from Sachs ground truth.
    
    Convention: mask[i, j] = 1 means "column j is a parent of row i"
    
    Returns:
        cross_mask: pd.DataFrame (n_X × n_S) — S → X edges
        self_mask: pd.DataFrame (n_X × n_X) — X → X edges
    """
    n_S = len(SOURCE_LABELS)
    n_X = len(INPUT_LABELS)
    
    cross_mask = np.zeros((n_X, n_S), dtype=int)
    self_mask = np.zeros((n_X, n_X), dtype=int)
    
    for parent, child in SACHS_EDGES:
        if parent in SOURCE_LABELS and child in INPUT_LABELS:
            # S → X edge (cross-attention)
            i = INPUT_LABELS.index(child)
            j = SOURCE_LABELS.index(parent)
            cross_mask[i, j] = 1
        elif parent in INPUT_LABELS and child in INPUT_LABELS:
            # X → X edge (self-attention)
            i = INPUT_LABELS.index(child)
            j = INPUT_LABELS.index(parent)
            self_mask[i, j] = 1
    
    cross_df = pd.DataFrame(cross_mask, index=INPUT_LABELS, columns=SOURCE_LABELS)
    self_df = pd.DataFrame(self_mask, index=INPUT_LABELS, columns=INPUT_LABELS)
    
    return cross_df, self_df


def build_full_dag_mask() -> pd.DataFrame:
    """Build the full (n_S+n_X) × (n_S+n_X) DAG adjacency matrix."""
    n_total = len(ALL_LABELS)
    full_mask = np.zeros((n_total, n_total), dtype=int)
    
    for parent, child in SACHS_EDGES:
        i = ALL_LABELS.index(child)
        j = ALL_LABELS.index(parent)
        full_mask[i, j] = 1
    
    return pd.DataFrame(full_mask, index=ALL_LABELS, columns=ALL_LABELS)


def to_causalit_npz(
    df: pd.DataFrame,
    source_labels: list,
    input_labels: list,
    sv_map: dict,
    iv_map: dict,
) -> dict:
    """
    Convert a Sachs DataFrame into CausaliT npz format.
    
    CausaliT expects:
        s: (n_samples, n_S, 2) — source tensor [value, variable_id]
        x: (n_samples, n_X, 2) — input tensor [value, variable_id]
    
    Args:
        df: DataFrame with protein expression columns
        source_labels: List of source variable names
        input_labels: List of input variable names
        sv_map: Source variable name → integer ID mapping
        iv_map: Input variable name → integer ID mapping
        
    Returns:
        dict with "s" and "x" numpy arrays
    """
    n = len(df)
    n_S = len(source_labels)
    n_X = len(input_labels)
    
    # Build source tensor (n, n_S, 2)
    s = np.zeros((n, n_S, 2))
    for j, var in enumerate(source_labels):
        s[:, j, 0] = df[var].values       # value
        s[:, j, 1] = sv_map[var]           # variable ID
    
    # Build input tensor (n, n_X, 2)
    x = np.zeros((n, n_X, 2))
    for j, var in enumerate(input_labels):
        x[:, j, 0] = df[var].values       # value
        x[:, j, 1] = iv_map[var]          # variable ID
    
    return {"s": s, "x": x}


def build_dataset_metadata(sv_map: dict, iv_map: dict) -> dict:
    """Build dataset_metadata.json matching CausaliT conventions."""
    
    direct_edges = [[parent, child] for parent, child in SACHS_EDGES]
    
    # Variable index map (combined)
    variable_index_map = {}
    variable_index_map.update(sv_map)
    variable_index_map.update(iv_map)
    
    metadata = {
        "name": "sachs_observational",
        "description": (
            "Sachs protein signaling network (observational subset, 853 samples). "
            "11 nodes (2 source + 9 intermediate), 17 edges. "
            "Source: Sachs et al. (2005), Science 308(5721). "
            "Observational subset as used by GraN-DAG (Lachapelle et al., 2020)."
        ),
        "tags": ["real-world", "benchmark", "protein-signaling", "sachs"],
        "variable_info": {
            "source_labels": SOURCE_LABELS,
            "input_labels": INPUT_LABELS,
            "target_labels": [],
            "n_source": len(SOURCE_LABELS),
            "n_input": len(INPUT_LABELS),
            "n_target": 0,
        },
        "variable_descriptions": {
            "PKC": "Protein Kinase C (root, source)",
            "Plcg": "Phospholipase C-gamma (root, source)",
            "Raf": "RAF proto-oncogene serine/threonine-protein kinase",
            "Mek": "Mitogen-activated protein kinase kinase 1/2",
            "PIP2": "Phosphatidylinositol 4,5-bisphosphate",
            "PIP3": "Phosphatidylinositol (3,4,5)-trisphosphate",
            "Erk": "Extracellular signal-regulated kinase 1/2",
            "Akt": "Protein kinase B (PKB/Akt)",
            "PKA": "Protein kinase A",
            "P38": "p38 mitogen-activated protein kinase",
            "Jnk": "c-Jun N-terminal kinase",
        },
        "causal_structure": {
            "direct_edges": direct_edges,
            "n_edges": len(direct_edges),
        },
        "variable_index_map": variable_index_map,
        "feature_indices": {
            "value": 0,
            "variable": 1,
        },
        "benchmark_info": {
            "source_paper": "Sachs et al. (2005). Science 308(5721), 523-529.",
            "data_source": "bnlearn (https://www.bnlearn.com/book-crc/)",
            "n_observational": 853,
            "n_total_samples": 7466,
            "baseline_results_source": "Lachapelle et al. (2020). GraN-DAG. ICLR.",
            "published_shd_results": {
                "CAM": 12,
                "GraN-DAG": 13,
                "DAG-GNN": 16,
                "PC": 17,
                "NOTEARS": 21,
                "GES": 26,
                "RANDOM": 21,
            },
        },
    }
    
    return metadata


# =============================================================================
# Main Pipeline
# =============================================================================

def prepare_sachs_dataset(save_dir: str, train_ratio: float = 0.8, seed: int = 42):
    """
    Full pipeline: download → preprocess → save in CausaliT format.
    
    Args:
        save_dir: Directory to save the dataset (e.g., "data/sachs_observational")
        train_ratio: Train/test split ratio
        seed: Random seed for reproducibility
    """
    makedirs(save_dir, exist_ok=True)
    
    # 1. Download data
    print("=" * 60)
    print("Step 1: Download Sachs data")
    print("=" * 60)
    df = download_sachs_data(join(save_dir, "sachs"))
    
    # 2. Verify all expected columns exist
    for var in ALL_LABELS:
        if var not in df.columns:
            raise ValueError(f"Expected column '{var}' not found. Available: {list(df.columns)}")
    
    # 3. Build variable ID maps (1-indexed, 0 reserved for padding)
    sv_map = {var: i + 1 for i, var in enumerate(SOURCE_LABELS)}
    iv_map = {var: i + 1 for i, var in enumerate(INPUT_LABELS)}
    
    print(f"\n  Source variable map: {sv_map}")
    print(f"  Input variable map: {iv_map}")
    
    # 4. Standardize the data (per-variable z-score)
    print("\nStep 2: Standardize data")
    df_std = df[ALL_LABELS].copy()
    for col in ALL_LABELS:
        mu = df_std[col].mean()
        sigma = df_std[col].std()
        df_std[col] = (df_std[col] - mu) / sigma
        print(f"  {col}: mean={mu:.2f}, std={sigma:.2f}")
    
    # 5. Train/test split
    print(f"\nStep 3: Train/test split (ratio={train_ratio}, seed={seed})")
    rng = np.random.default_rng(seed)
    n_total = len(df_std)
    n_train = int(n_total * train_ratio)
    indices = rng.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    df_train = df_std.iloc[train_idx].reset_index(drop=True)
    df_test = df_std.iloc[test_idx].reset_index(drop=True)
    print(f"  Train: {len(df_train)} samples")
    print(f"  Test: {len(df_test)} samples")
    
    # 6. Convert to CausaliT npz format
    print("\nStep 4: Convert to CausaliT format")
    train_data = to_causalit_npz(df_train, SOURCE_LABELS, INPUT_LABELS, sv_map, iv_map)
    test_data = to_causalit_npz(df_test, SOURCE_LABELS, INPUT_LABELS, sv_map, iv_map)
    
    print(f"  train s: {train_data['s'].shape}, x: {train_data['x'].shape}")
    print(f"  test  s: {test_data['s'].shape}, x: {test_data['x'].shape}")
    
    # 7. Save npz files
    np.savez(join(save_dir, "ds_train.npz"), **train_data)
    np.savez(join(save_dir, "ds_test.npz"), **test_data)
    # Also save combined for single-file loading
    full_data = to_causalit_npz(df_std, SOURCE_LABELS, INPUT_LABELS, sv_map, iv_map)
    np.savez(join(save_dir, "ds.npz"), **full_data)
    print(f"  Saved: ds_train.npz, ds_test.npz, ds.npz")
    
    # 8. Save attention masks
    print("\nStep 5: Build and save attention masks")
    cross_mask, self_mask = build_attention_masks()
    full_dag = build_full_dag_mask()
    
    cross_mask.to_csv(join(save_dir, "dec1_cross_att_mask.csv"))
    self_mask.to_csv(join(save_dir, "dec1_self_att_mask.csv"))
    full_dag.to_csv(join(save_dir, "dag_adj_mask.csv"))
    
    print(f"  Cross-attention mask (S→X): {cross_mask.shape}")
    print(cross_mask.to_string())
    print(f"\n  Self-attention mask (X→X): {self_mask.shape}")
    print(self_mask.to_string())
    
    n_cross_edges = int(cross_mask.values.sum())
    n_self_edges = int(self_mask.values.sum())
    print(f"\n  Cross edges: {n_cross_edges}, Self edges: {n_self_edges}, Total: {n_cross_edges + n_self_edges}")
    
    # 9. Save metadata
    print("\nStep 6: Save dataset metadata")
    metadata = build_dataset_metadata(sv_map, iv_map)
    with open(join(save_dir, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: dataset_metadata.json")
    
    # 10. Save raw standardized CSV for inspection
    df_std.to_csv(join(save_dir, "sachs_standardized.csv"), index=False)
    print(f"  Saved: sachs_standardized.csv")
    
    # Summary
    print("\n" + "=" * 60)
    print("SACHS DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Save directory: {save_dir}")
    print(f"  Nodes: {len(ALL_LABELS)} ({len(SOURCE_LABELS)} S + {len(INPUT_LABELS)} X)")
    print(f"  Edges: {len(SACHS_EDGES)} (ground truth)")
    print(f"  Samples: {n_total} total ({len(df_train)} train + {len(df_test)} test)")
    print(f"\n  Files created:")
    print(f"    ds.npz, ds_train.npz, ds_test.npz")
    print(f"    dec1_cross_att_mask.csv, dec1_self_att_mask.csv, dag_adj_mask.csv")
    print(f"    dataset_metadata.json")
    print(f"    sachs_standardized.csv")
    print(f"\n  To use with CausaliT, set: data.dataset: 'sachs_observational'")
    print(f"  and ensure save_dir is under the data/ folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Sachs dataset for CausaliT")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the dataset. Default: data/sachs_observational",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.save_dir is None:
        root = dirname(dirname(abspath(__file__)))
        args.save_dir = join(root, "data", "sachs_observational")
    
    prepare_sachs_dataset(args.save_dir, args.train_ratio, args.seed)

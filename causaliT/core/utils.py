import torch
import logging
import os
from os.path import dirname, abspath, join, exists
from datetime import datetime
from typing import Dict, Optional, Tuple
from pytorch_lightning import seed_everything
import glob
import re
import numpy as np
import pandas as pd
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))


def set_seed(seed=42):
    """
    Sets the random seed across various libraries and enforces deterministic behavior.
    
    Parameters:
    seed (int): The random seed to use. Default is 42.
    """
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # For GPUs
    
    seed_everything(seed, workers=True)

    # Enforce deterministic operations in PyTorch
    torch.backends.cudnn.deterministic = True  # Ensures reproducible behavior in cuDNN
    torch.backends.cudnn.benchmark = False     # Disables benchmarking to avoid nondeterminism

    # Set environment variable to control other sources of randomness
    os.environ["PYTHONHASHSEED"] = str(seed)   # Controls hashing randomness in Python
    
    
    
def log_memory(stage):
    # GPU memory usage
    allocated_gpu = torch.cuda.memory_allocated() / 1e9  # GB
    reserved_gpu = torch.cuda.memory_reserved() / 1e9  # GB
    
    # # CPU memory usage
    # ram_usage = psutil.virtual_memory().used / 1e9  # GB
    # ram_total = psutil.virtual_memory().total / 1e9  # GB
    # ram_percent = psutil.virtual_memory().percent  # %

    logging.info(
        f"[{stage}] GPU Allocated: {allocated_gpu:.2f} GB | GPU Reserved: {reserved_gpu:.2f} GB | "
        # f"CPU Used: {ram_usage:.2f}/{ram_total:.2f} GB ({ram_percent}%)"
    )
    
    
def mk_fname(filename: str,label: str,suffix: str):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") # format YYYYMMDD_HHMMSS
    return filename+"_"+str(label)+f"_{timestamp}"+suffix





def find_last_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "epoch=*-train_loss=*.ckpt"))
    if not checkpoint_files:
        return None  # No checkpoints found

    # Regex to extract epoch number
    pattern = re.compile(r"epoch=(\d+)-train_loss=.*\.ckpt")

    def extract_epoch(file):
        match = pattern.search(file)
        return int(match.group(1)) if match else -1

    # Find the checkpoint with the highest epoch number
    last_checkpoint = max(checkpoint_files, key=extract_epoch, default=None)
    return last_checkpoint


def load_dag_masks(
    data_dir: str, 
    mask_files: Dict[str, str],
    device: str = 'cpu'
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load DAG adjacency masks from CSV files for StageCausaliT architecture.
    
    These masks represent the ground-truth causal structure from the SCM and can be 
    used as hard masks to enforce causal constraints in attention mechanisms.
    
    CSV format:
        - Rows = query variables, Columns = key variables
        - Values in [0, 1], where 1 = attention is allowed
    
    Args:
        data_dir: Path to directory containing mask CSV files
        mask_files: Dictionary mapping mask names to filenames, e.g.:
            {
                'dec1_cross': 'dec1_cross_att_mask.csv',
                'dec1_self': 'dec1_self_att_mask.csv',
                'dec2_cross': 'dec2_cross_att_mask.csv',
                'dec2_self': 'dec2_self_att_mask.csv',
            }
        device: Device to place tensors on ('cpu', 'cuda', etc.)
        
    Returns:
        Dictionary mapping mask names to tensors, or None if no masks found.
        Keys match the keys in mask_files parameter.
        Tensor shapes: (query_len, key_len) e.g., dec1_cross is (X_len, S_len)
    """
    masks = {}
    for key, filename in mask_files.items():
        if filename is None:
            continue
        path = join(data_dir, filename)
        if exists(path):
            df = pd.read_csv(path, index_col=0)
            # Convert to float tensor: shape (query_len, key_len)
            mask_tensor = torch.tensor(df.values, dtype=torch.float32, device=device)
            masks[key] = mask_tensor
            print(f"  Loaded mask '{key}': shape {mask_tensor.shape} from {filename}")
        else:
            print(f"  Warning: Mask file not found: {path}")
    
    if masks:
        print(f"✓ Loaded {len(masks)} DAG masks from {data_dir}")
        return masks
    else:
        print(f"✗ No DAG mask files found in {data_dir}")
        return None


def _self_mask_has_cycles(adj: np.ndarray) -> bool:
    """
    Kahn-style topological-sort cycle check on a square binary adjacency
    matrix interpreted as `adj[i, j] == 1  ⇔  edge j → i`. Self-loops on the
    diagonal are ignored (they would always count as 1-cycles).
    """
    A = (adj > 0).astype(np.int64)
    np.fill_diagonal(A, 0)
    n = A.shape[0]
    in_deg = A.sum(axis=1).astype(np.int64).tolist()
    queue = [i for i in range(n) if in_deg[i] == 0]
    visited = 0
    while queue:
        u = queue.pop()
        visited += 1
        for v in np.where(A[:, u] == 1)[0]:
            in_deg[int(v)] -= 1
            if in_deg[int(v)] == 0:
                queue.append(int(v))
    return visited != n


def corrupt_dag_masks(
    masks: Dict[str, torch.Tensor],
    *,
    seed: Optional[int],
    cross_shd: int = 0,
    self_shd: int = 0,
    X_len: int,
    preserve_sparsity: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, dict]]:
    """
    Apply controlled SHD corruption to DAG hard masks.

    For each mask, exactly `shd` entries (or as many as the eligible pool
    allows under the fallback rule) are flipped 0↔1, chosen uniformly at
    random under a deterministic seed. This produces a mask whose Hamming
    distance to the ground truth equals `shd_realised`.

    SHD semantics (preserve_sparsity=False, default)
    ------------------------------------------------
    - `seed in {None, 0}`  OR  `cross_shd == 0 == self_shd`
        → ground-truth masks are returned unchanged.
    - `0 < shd ≤ num_true_edges`
        → pick `shd` distinct positions from the eligible pool uniformly at
          random and flip each (XOR with 1).
    - `shd > num_true_edges`
        → **fallback "all edges must be wrong"**: every entry where the GT
          mask was 1 in the eligible pool becomes 0 (k flips), then up to
          `shd - k` extra non-edge positions are turned ON. If `shd - k`
          exceeds the size of the non-edge pool the realised SHD will be
          smaller than the request (and `fallback_used=True` is reported).

    Sparsity-preserving mode (preserve_sparsity=True)
    -------------------------------------------------
    Corruption is restricted to *one-for-one edge swaps*: each swap removes
    one true edge (1 → 0) AND adds one non-edge (0 → 1) somewhere else in
    the eligible pool. Each swap therefore raises SHD by exactly 2 while
    leaving the total edge count unchanged. This disentangles the effect
    of "wrong edges" from "more / fewer edges to attend to".

    - Only **even** SHD values are achievable. An odd `cross_shd` /
      `self_shd` request is silently rounded **down** to the nearest even
      number, with the realised value reported via `shd_realised`.
    - The number of swaps actually performed is capped at
      `min(k_true_in_pool, n_non_edges_in_pool)`. If the request exceeds
      this cap, fewer swaps are done and `fallback_used=True` is reported.
    - Edge count is invariant: `corrupted.sum() == ground_truth.sum()`.

    Eligible pools
    --------------
    - Cross masks (`dec_cross`, `*_cross` with `shape[0] == X_len`):
        all (Lq, Lk) entries are eligible.
    - Self masks (`dec_self`, `*_self` with shape `(X_len, X_len)`):
        off-diagonal only — the corruption never introduces self-loops.
        The resulting mask may still contain **cycles** w.r.t. the original
        DAG ordering; this is detected via topological-sort and reported in
        the per-mask `corruption_info` so it can be logged / saved alongside
        the run.

    Independent randomness for cross/self
    -------------------------------------
    Cross uses RNG `seed`; self uses RNG `seed + 1`. Changing `cross_shd`
    therefore does not reshuffle the corrupted self mask, and vice-versa.

    Args:
        masks:              Dict produced by `load_dag_masks`.
        seed:               Integer seed; `None` or `0` disables corruption
                            entirely.
        cross_shd:          Requested SHD for `*_cross` masks (>= 0).
        self_shd:           Requested SHD for `*_self`  masks (>= 0).
        X_len:              Number of X variables (used for self-mask shape
                            sanity).
        preserve_sparsity:  When True, only edge-preserving swaps are
                            performed (see "Sparsity-preserving mode"
                            above). Default False keeps legacy behavior.

    Returns:
        `(corrupted_masks, corruption_info)` where `corruption_info[name]`
        is a dict with keys: `shd_requested`, `shd_realised`,
        `num_true_edges`, `eligible_pool_size`, `fallback_used`,
        `has_cycles` (bool for self masks, `None` for cross / non-square),
        `preserve_sparsity` (bool, the mode used), and `n_swaps`
        (int when `preserve_sparsity=True`, else `None`).
    """
    if cross_shd < 0 or self_shd < 0:
        raise ValueError(f"SHD values must be >= 0 (got cross={cross_shd}, self={self_shd}).")

    out_masks: Dict[str, torch.Tensor] = {}
    info: Dict[str, dict] = {}

    seed_is_off = (seed is None) or (int(seed) == 0)
    shd_is_off = (cross_shd == 0 and self_shd == 0)
    if seed_is_off or shd_is_off:
        for name, mask in masks.items():
            out_masks[name] = mask
            info[name] = {
                "shd_requested": 0,
                "shd_realised": 0,
                "num_true_edges": int(mask.sum().item()),
                "eligible_pool_size": None,
                "fallback_used": False,
                "has_cycles": None,
                "preserve_sparsity": bool(preserve_sparsity),
                "n_swaps": 0 if preserve_sparsity else None,
            }
        return out_masks, info

    seed_int = int(seed)

    for name, mask in masks.items():
        is_self = name.endswith("_self") or name == "dec_self"
        is_cross = name.endswith("_cross") or name == "dec_cross"
        shd_req = int(self_shd) if is_self else (int(cross_shd) if is_cross else 0)

        if shd_req == 0:
            out_masks[name] = mask
            info[name] = {
                "shd_requested": 0,
                "shd_realised": 0,
                "num_true_edges": int(mask.sum().item()),
                "eligible_pool_size": None,
                "fallback_used": False,
                "has_cycles": None,
                "preserve_sparsity": bool(preserve_sparsity),
                "n_swaps": 0 if preserve_sparsity else None,
            }
            continue

        # Build eligible-pool mask (off-diagonal only for self).
        H, W = mask.shape
        elig = torch.ones_like(mask, dtype=torch.bool)
        is_square_self = is_self and H == X_len and W == X_len
        if is_square_self:
            elig.fill_diagonal_(False)

        elig_np = elig.cpu().numpy().ravel()
        gt_np = (mask > 0).cpu().numpy().ravel()

        true_pool_idx = np.flatnonzero(elig_np & gt_np)
        non_edge_pool_idx = np.flatnonzero(elig_np & (~gt_np))
        k_true = int(true_pool_idx.size)
        n_non_edges = int(non_edge_pool_idx.size)
        elig_size = int(elig.sum().item())

        # Independent sub-streams for cross/self so toggles don't bleed.
        sub_seed = seed_int + (1 if is_self else 0)
        rng = np.random.default_rng(sub_seed)

        corrupted = mask.clone().contiguous()
        flat = corrupted.view(-1)

        n_swaps_used: Optional[int] = None

        if preserve_sparsity:
            # Sparsity-preserving mode: each swap = drop one true edge
            # AND add one non-edge → SHD += 2, edge count unchanged.
            # Odd SHD requests are rounded DOWN to the nearest even number.
            n_swaps_req = shd_req // 2
            n_swaps_cap = min(k_true, n_non_edges)
            n_swaps = min(n_swaps_req, n_swaps_cap)
            fallback = bool(n_swaps < n_swaps_req)

            if n_swaps > 0:
                drop_idx = rng.choice(true_pool_idx, size=n_swaps, replace=False)
                add_idx = rng.choice(non_edge_pool_idx, size=n_swaps, replace=False)
                drop_t = torch.from_numpy(drop_idx).to(device=flat.device, dtype=torch.long)
                add_t = torch.from_numpy(add_idx).to(device=flat.device, dtype=torch.long)
                flat[drop_t] = 0.0
                flat[add_t] = 1.0

            shd_realised = 2 * n_swaps
            n_swaps_used = int(n_swaps)
        elif shd_req <= k_true:
            # Standard mode: pick `shd_req` random eligible positions and flip them.
            flat_elig_idx = np.flatnonzero(elig_np)
            chosen = rng.choice(flat_elig_idx, size=shd_req, replace=False)
            chosen_t = torch.from_numpy(chosen).to(device=flat.device, dtype=torch.long)
            flat[chosen_t] = 1.0 - flat[chosen_t]
            shd_realised = shd_req
            fallback = False
        else:
            # Fallback: zero every true edge in the eligible pool (k flips),
            # then turn ON up to `shd_req - k_true` random non-edge positions.
            n_to_add = min(shd_req - k_true, n_non_edges)
            if n_to_add < n_non_edges:
                added = rng.choice(non_edge_pool_idx, size=n_to_add, replace=False)
            else:
                added = non_edge_pool_idx
            true_t = torch.from_numpy(true_pool_idx).to(device=flat.device, dtype=torch.long)
            added_t = torch.from_numpy(added).to(device=flat.device, dtype=torch.long)
            flat[true_t] = 0.0
            flat[added_t] = 1.0
            shd_realised = k_true + int(added.size)
            fallback = True

        has_cycles = None
        if is_square_self:
            has_cycles = bool(_self_mask_has_cycles(corrupted.cpu().numpy()))

        out_masks[name] = corrupted
        info[name] = {
            "shd_requested": int(shd_req),
            "shd_realised": int(shd_realised),
            "num_true_edges": k_true,
            "eligible_pool_size": elig_size,
            "fallback_used": bool(fallback),
            "has_cycles": has_cycles,
            "preserve_sparsity": bool(preserve_sparsity),
            "n_swaps": n_swaps_used,
        }

    return out_masks, info




if __name__ == "__main__":
    
    # test for find_last_checkpoint
    checkpoint_dir = r"C:\Users\ScipioneFrancesco\Documents\Projects\prochain_transformer\experiments\training\cluster\dx_250324_base_25\sweeps\sweep_enc_pos_emb_hidden\sweep_enc_pos_emb_hidden_100\k_0\checkpoints"
    last_ckpt = find_last_checkpoint(checkpoint_dir)
    print("Last checkpoint:", last_ckpt)

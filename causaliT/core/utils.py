import torch
import logging
import os
from os.path import dirname, abspath, join, exists
from datetime import datetime
from typing import Dict, Optional
from pytorch_lightning import seed_everything
import glob
import re
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




if __name__ == "__main__":
    
    # test for find_last_checkpoint
    checkpoint_dir = r"C:\Users\ScipioneFrancesco\Documents\Projects\prochain_transformer\experiments\training\cluster\dx_250324_base_25\sweeps\sweep_enc_pos_emb_hidden\sweep_enc_pos_emb_hidden_100\k_0\checkpoints"
    last_ckpt = find_last_checkpoint(checkpoint_dir)
    print("Last checkpoint:", last_ckpt)

"""
Central location for project paths.

This module provides standard project directories that work from any location
(local development, cluster, notebooks, etc.) once proT is installed via pip install -e .

Usage:
    from proT.paths import ROOT_DIR, DATA_DIR, EXPERIMENTS_DIR
    
    data_path = DATA_DIR / "example" / "ds.npz"
    config_path = EXPERIMENTS_DIR / "example" / "config.yaml"
"""

import os
from pathlib import Path

# Project root is 2 levels up from this file
# proT/paths.py -> proT/ -> causaliT/
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Standard project directories
DATA_DIR = ROOT_DIR / "data"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
LOGS_DIR = ROOT_DIR / "logs"
CONFIG_DIR = ROOT_DIR / "proT" / "config"

# Allow environment variable override for cluster flexibility
if "PROT_ROOT" in os.environ:
    ROOT_DIR = Path(os.getenv("PROT_ROOT")).resolve()
    DATA_DIR = ROOT_DIR / "data"
    EXPERIMENTS_DIR = ROOT_DIR / "experiments"
    LOGS_DIR = ROOT_DIR / "logs"
    CONFIG_DIR = ROOT_DIR / "proT" / "config"

if "PROT_DATA" in os.environ:
    DATA_DIR = Path(os.getenv("PROT_DATA")).resolve()


def get_dirs(root: str = None):
    """
    Legacy function for backward compatibility.
    
    Returns standard directories tuple: (INPUT_DIR, OUTPUT_DIR, INTERMEDIATE_DIR, EXPERIMENTS_DIR)
    
    Args:
        root: Optional root directory. If None, uses ROOT_DIR.
        
    Returns:
        Tuple of Path objects for input, output, intermediate, and experiments directories.
    """
    root_path = Path(root) if root else ROOT_DIR
    return (
        root_path / "data" / "input",
        root_path / "data" / "output",
        root_path / "data" / "intermediate",
        root_path / "experiments"
    )


# Export all paths for convenience
__all__ = [
    'ROOT_DIR',
    'DATA_DIR',
    'EXPERIMENTS_DIR',
    'LOGS_DIR',
    'CONFIG_DIR',
    'get_dirs',
]

"""
ProT - Process Transformer Package

A transformer-based model for sequence prediction in process chains.
"""

# Import version info
__version__ = "0.1.0"

# Export commonly used paths for convenience
from causaliT.paths import (
    ROOT_DIR,
    DATA_DIR,
    EXPERIMENTS_DIR,
    LOGS_DIR,
    CONFIG_DIR,
    get_dirs,
)

# Export main components
from causaliT.core.model import ProT
from causaliT.core.architectures.stage_causal import StageCausaliT
from causaliT.training.forecasters import TransformerForecaster, StageCausalForecaster
from causaliT.training.dataloader import ProcessDataModule

__all__ = [
    # Paths
    'ROOT_DIR',
    'DATA_DIR',
    'EXPERIMENTS_DIR',
    'LOGS_DIR',
    'CONFIG_DIR',
    'get_dirs',
    # Core components
    'ProT',
    'StageCausaliT',
    'TransformerForecaster',
    'StageCausalForecaster',
    'ProcessDataModule',
]

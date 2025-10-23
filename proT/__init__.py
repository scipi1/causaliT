"""
ProT - Process Transformer Package

A transformer-based model for sequence prediction in process chains.
"""

# Import version info
__version__ = "0.1.0"

# Export commonly used paths for convenience
from proT.paths import (
    ROOT_DIR,
    DATA_DIR,
    EXPERIMENTS_DIR,
    LOGS_DIR,
    CONFIG_DIR,
    get_dirs,
)

# Export main components
from proT.core.model import ProT
from proT.training.forecasters import TransformerForecaster
from proT.training.dataloader import ProcessDataModule

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
    'TransformerForecaster',
    'ProcessDataModule',
]

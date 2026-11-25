"""
causaliT - causal discovery with attention selectors.
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
from causaliT.core.architectures import AttentionSelectorLayer
from causaliT.training.forecasters import AttentionSelectorForecaster
from causaliT.training.dataloader_sx import SXDataModule

__all__ = [
    # Paths
    'ROOT_DIR',
    'DATA_DIR',
    'EXPERIMENTS_DIR',
    'LOGS_DIR',
    'CONFIG_DIR',
    'get_dirs',
    # Core components
    'AttentionSelectorLayer',
    'AttentionSelectorForecaster',
    'SXDataModule',
]

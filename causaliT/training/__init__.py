"""
ProT Training Infrastructure Package

This package contains all training-related components:
- forecasters: Lightning model wrappers (TransformerForecaster, StageCausalForecaster)
- callbacks: Training and model monitoring callbacks
- dataloader: Data loading utilities (ProcessDataModule, StageCausalDataModule)
- trainer: Main training orchestration
- experiment_control: Experiment management and sweeps
"""

from .forecasters import TransformerForecaster, StageCausalForecaster
from .dataloader import ProcessDataModule
from .stage_causal_dataloader import StageCausalDataModule
from .trainer import trainer, get_model_class, create_model_instance, get_dataloader
from .experiment_control import combination_sweep, update_config

__all__ = [
    'TransformerForecaster',
    'StageCausalForecaster',
    'ProcessDataModule',
    'StageCausalDataModule',
    'trainer',
    'get_model_class',
    'create_model_instance',
    'get_dataloader',
    'combination_sweep',
    'update_config',
]

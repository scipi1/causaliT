"""
ProT Training Infrastructure Package

This package contains all training-related components:
- forecasters: Lightning model wrappers (TransformerForecaster, StageCausalForecaster)
- callbacks: Training and model monitoring callbacks
- dataloader: Data loading utilities (ProcessDataModule, StageCausalDataModule)
- trainer: Main training orchestration
- experiment_control: Experiment management and sweeps
- staged_trainer: Staged training pipeline (calibration → causal_init → training)
- calibration: Stage 0 - calibrate group L1 for gradient balance
- causal_initialization: Stage 1 - HSIC-dominated pre-training
"""

from .forecasters import TransformerForecaster, StageCausalForecaster
from .dataloader import ProcessDataModule
from .stage_causal_dataloader import StageCausalDataModule
from .trainer import trainer, get_model_class, create_model_instance, get_dataloader
from .experiment_control import combination_sweep, update_config
from .staged_trainer import staged_trainer, run_staged_training_from_config, check_staged_training_config
from .calibration import calibrate_group_l1
from .causal_initialization import run_causal_initialization

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
    # Staged training
    'staged_trainer',
    'run_staged_training_from_config',
    'check_staged_training_config',
    'calibrate_group_l1',
    'run_causal_initialization',
]

"""
ProT Training Infrastructure Package

This package contains all training-related components:
- forecasters: Lightning model wrappers (TransformerForecaster, StageCausalForecaster)
- callbacks: Training and model monitoring callbacks
- dataloader: Data loading utilities (ProcessDataModule, StageCausalDataModule)
- trainer: Main training orchestration + train_single_fold primitive
- experiment_control: Experiment management and sweeps
- staged_trainer: Staged training pipeline (calibration -> causal_init -> training)
- calibration: Stage 0 - calibrate group L1 for gradient balance
- causal_initialization: Stage 1 - HSIC-dominated pre-training
- config_operations: Pure config transform utilities (apply_calibration_to_config, etc.)
"""

from .forecasters import TransformerForecaster, StageCausalForecaster
from .dataloader import ProcessDataModule
from .stage_causal_dataloader import StageCausalDataModule
from .trainer import (
    trainer,
    train_single_fold,
    get_model_class,
    create_model_instance,
    get_dataloader,
    _make_fold_splits,
    _count_trainable_params,
)
from .experiment_control import combination_sweep, update_config
from .staged_trainer import staged_trainer, run_staged_training_from_config, check_staged_training_config
from .calibration import calibrate_group_l1, GradientNormTracker
from .causal_initialization import run_causal_initialization, CausalInitProgressLogger
from .config_operations import (
    apply_calibration_to_config,
    apply_score_sparsity_to_config,
    configure_main_training_from_staged,
    apply_seed_to_config,
)

__all__ = [
    # Models
    'TransformerForecaster',
    'StageCausalForecaster',
    # Data
    'ProcessDataModule',
    'StageCausalDataModule',
    # Training primitives
    'trainer',
    'train_single_fold',
    'get_model_class',
    'create_model_instance',
    'get_dataloader',
    '_make_fold_splits',
    '_count_trainable_params',
    # Experiment control
    'combination_sweep',
    'update_config',
    # Staged training pipeline
    'staged_trainer',
    'run_staged_training_from_config',
    'check_staged_training_config',
    # Calibration
    'calibrate_group_l1',
    'GradientNormTracker',
    # Causal initialization
    'run_causal_initialization',
    'CausalInitProgressLogger',
    # Config operations (pure transforms)
    'apply_calibration_to_config',
    'apply_score_sparsity_to_config',
    'configure_main_training_from_staged',
    'apply_seed_to_config',
]

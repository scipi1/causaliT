"""
causaliT training.

Entry points:
    trainer.py           k-fold trainer + ``train_single_fold`` (shared primitive)
    adaptive_trainer.py  metric-driven reconstruct/structure schedule

Packages:
    forecasters/         Lightning wrapper (AttentionSelectorForecaster)
    structural_losses/   acyclicity / independence losses
    callbacks/           training and monitoring callbacks
    core/                helpers shared by both trainers
    forecaster_utils/    helpers used only by the forecaster
    adaptive_utils/      helpers used only by adaptive_trainer
    dataloader_sx.py     SXDataModule ((S, X, [Y]) npz datasets)

Coding Assistant: the staged trainer (calibration / causal init) and the
deprecated forecasters are in ``_OLD/`` for reference only; do not import.
"""

from .forecasters import AttentionSelectorForecaster
from .dataloader_sx import SXDataModule
from .trainer import (
    trainer,
    train_single_fold,
    get_model_class,
    create_model_instance,
    get_dataloader,
)
from .core.experiment_control import combination_sweep, update_config

__all__ = [
    'AttentionSelectorForecaster',
    'SXDataModule',
    'trainer',
    'train_single_fold',
    'get_model_class',
    'create_model_instance',
    'get_dataloader',
    'combination_sweep',
    'update_config',
]

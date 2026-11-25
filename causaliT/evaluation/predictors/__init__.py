"""
Checkpoint -> prediction wrappers (deprecated predictors are in ``_OLD/``).
"""

from .base_predictor import BasePredictor, PredictionResult
from .attention_selector_predictor import AttentionSelectorPredictor

__all__ = [
    'BasePredictor',
    'PredictionResult',
    'AttentionSelectorPredictor',
]

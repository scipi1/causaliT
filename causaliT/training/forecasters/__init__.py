"""
Lightning forecaster of the attention-selector architecture.

Coding Assistant: deprecated forecasters (proT, Stage/Single/NoiseAware/
Variance causal) live in ``_OLD/`` for reference only; do not import them.
"""

from .attention_selector_forecaster import AttentionSelectorForecaster

__all__ = ['AttentionSelectorForecaster']



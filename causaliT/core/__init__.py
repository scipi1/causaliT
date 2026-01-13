"""
ProT Core Model Package

This package contains the pure transformer model architecture,
independent of training infrastructure.
"""

from .model import ProT
from .architectures.stage_causal import StageCausaliT

__all__ = ['ProT', 'StageCausaliT']

"""
SelfSelectorLayer package — homogeneous N-node causal discovery.

Whole-graph directional variable selection: a single ``GatedSelfAttention``
block over all N = L_S + L_X nodes (no assumed S -> X direction).  See
``model.py`` for the full design rationale.
"""

from .model import SelfSelectorLayer

__all__ = ["SelfSelectorLayer"]

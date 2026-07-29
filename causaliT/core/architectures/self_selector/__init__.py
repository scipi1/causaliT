"""
SelfSelectorLayer package — homogeneous N-node causal discovery.

Whole-graph directional variable selection: a single ``GatedSelfAttention``
block over all N = L_S + L_X nodes (no assumed S -> X direction).  See
``model.py`` for the full design rationale.

.. deprecated::
    Superseded by ``AttentionSelectorLayer(homogeneous_nodes=True)``, which
    implements the identical topology and is the variant wired end-to-end
    through training and evaluation.  Constructing ``SelfSelectorLayer`` emits a
    ``DeprecationWarning``; the package is kept only for backward compatibility.
"""

from .model import SelfSelectorLayer

__all__ = ["SelfSelectorLayer"]

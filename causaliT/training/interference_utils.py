"""
Gradient-interference diagnostics for the L0 <-> HSIC objectives.

Purpose
=======
When training the AttentionSelector with ``HardConcreteCrossAttention`` and an
active L0 penalty (``lambda_l0 > 0``) alongside HSIC minimisation
(``lambda_hsic > 0``), both objectives back-propagate through the **same
structural pathway**:

* the L0 penalty is a function of ``log_alpha = QK^T / sqrt(E)`` -- so its
  gradient flows into the query/key projections and, through the query/key
  *inputs*, into the structural (variable-ID) embeddings;
* the HSIC term flows through the attention output back into the same Q/K
  projections and structural embeddings (and also into the value pathway).

Because they share these parameters, the two gradients can point in opposing
directions and cancel each other -- *gradient interference*.  This module
measures that interference **per module block** via cosine similarity so you
can localise where the conflict happens:

    cos(g_hsic, g_l0) = (g_hsic . g_l0) / (||g_hsic|| * ||g_l0||)

* cos ~ +1  -> the two objectives are aligned in that block
* cos ~  0  -> orthogonal (no interaction)
* cos ~ -1  -> direct conflict (maximal interference)

The gradients are obtained with ``torch.autograd.grad(..., retain_graph=True)``,
which returns the gradients as tensors **without** writing to ``.grad``.  So
calling these utilities before the real ``backward()`` (or the gradient-routing
dual backward) leaves the subsequent optimisation completely untouched.

Which blocks are meaningful?
============================
Blocks are built over **all** trainable parameters (grouped by readable module
label), NOT a pre-filtered "structural" subset.  This is deliberate and robust:

* The **L0 penalty only reaches the structural pathway** (Q/K projections and
  the embeddings that feed them).  Pure reconstruction blocks (value/out
  projection, FFN, forecaster, value-path norms) therefore receive a
  **zero L0 gradient**, so their cosine is ``NaN`` -- a clear "no L0 signal
  here" marker that the caller can simply skip.  The non-NaN blocks are exactly
  the structural pathway, auto-discovered without hard-coding parameter names.

* For **mixed** blocks such as the embeddings (a structural variable-ID
  ``nn.Embedding`` plus a value ``nn.Linear`` living under the same module),
  the value sub-parameters get a zero L0 gradient.  The block dot product then
  picks up **only** the shared (structural) components, so the *sign* of the
  cosine -- aligned vs. conflicting -- is preserved; only its magnitude is
  attenuated by the extra HSIC-only value norm.  Sign is what matters for
  localising interference, so this is a faithful module-level readout.

Block labels (module-level granularity)
=======================================
* ``query_projection``          -- attention Q projection
* ``key_projection``            -- attention K projection
* ``value_projection``          -- attention V projection (reconstruction)
* ``out_projection``            -- attention output projection (reconstruction)
* ``value_projection_struct``   -- SVFA structural value projection (dual value)
* ``out_projection_struct``     -- SVFA structural output projection
* ``attention_internals``       -- ``inner_attention.*`` learnable params
* ``embedding_S`` / ``embedding_X`` -- S / X embeddings (structural + value)
* ``query_embedding_X``         -- decoupled free X query embedding (Q only)
* ``ffn``                       -- position-wise feed-forward (linear1/linear2)
* ``forecaster``                -- output MLP head
* ``norm``                      -- layer norms
* ``other``                     -- anything unmatched

An additional ``overall`` entry concatenates every gradient into a single
vector for a global interference summary.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _block_for_param(name: str) -> str:
    """Map a ``model.named_parameters()`` name to a readable module block."""
    # --- Attention projections (check *_struct before the generic form) ---
    if "query_projection" in name:
        return "query_projection"
    if "key_projection" in name:
        return "key_projection"
    if "value_projection_struct" in name:
        return "value_projection_struct"
    if "out_projection_struct" in name:
        return "out_projection_struct"
    if "value_projection" in name:
        return "value_projection"
    if "out_projection" in name:
        return "out_projection"
    if "inner_attention" in name:
        return "attention_internals"
    # --- Decoupled free X query embedding (feeds Q only) ---
    # Checked BEFORE embedding_X so it gets its own block instead of "other".
    if name.startswith("query_embed_X"):
        return "query_embedding_X"
    # --- Embeddings (structural variable-ID + value sub-modules) ---
    if name.startswith("orth_embed_S") or name.startswith("embedding_S"):
        return "embedding_S"
    if name.startswith("orth_embed_X") or name.startswith("embedding_X"):
        return "embedding_X"
    # --- Reconstruction blocks ---
    if name.startswith("linear1") or name.startswith("linear2"):
        return "ffn"
    if name.startswith("forecaster"):
        return "forecaster"
    if name.startswith("norm"):
        return "norm"
    return "other"


# Canonical block ordering so the logged metric columns are stable.
_BLOCK_ORDER = [
    "query_projection",
    "key_projection",
    "value_projection_struct",
    "out_projection_struct",
    "attention_internals",
    "embedding_S",
    "embedding_X",
    "query_embedding_X",
    "value_projection",
    "out_projection",
    "ffn",
    "forecaster",
    "norm",
    "other",
]


def build_interference_blocks(model: nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    """Group a model's trainable parameters into readable module blocks.

    All parameters with ``requires_grad=True`` are included.  Blocks that turn
    out to receive no L0 gradient (pure reconstruction) simply yield a NaN
    cosine at compute time and can be skipped by the caller.

    Args:
        model: The inner model (e.g. ``AttentionSelectorLayer``), NOT the
            Lightning wrapper.

    Returns:
        Ordered dict ``{block_name: [param, ...]}`` following ``_BLOCK_ORDER``.
    """
    buckets: Dict[str, List[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        block = _block_for_param(name)
        buckets.setdefault(block, []).append(param)

    ordered = [k for k in _BLOCK_ORDER if buckets.get(k)]
    ordered += [k for k in buckets if k not in _BLOCK_ORDER]
    return {k: buckets[k] for k in ordered}


def _flatten_grads(
    grads: Tuple[torch.Tensor, ...],
    params: List[torch.nn.Parameter],
) -> torch.Tensor:
    """Flatten a tuple of grads into one 1-D vector.

    ``torch.autograd.grad(..., allow_unused=True)`` returns ``None`` for
    parameters that do not participate in the loss; those are replaced with
    zeros so the two objective gradients stay shape-aligned.
    """
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            flat.append(g.reshape(-1))
    if len(flat) == 0:
        return torch.zeros(0)
    return torch.cat(flat)


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Cosine similarity between two 1-D vectors.

    Returns NaN when either vector has (near-)zero norm, so the metric clearly
    signals "no gradient here" rather than a misleading 0.0.
    """
    na = a.norm()
    nb = b.norm()
    if na < eps or nb < eps:
        return float("nan")
    return float((a @ b) / (na * nb))


def compute_l0_hsic_interference(
    model: nn.Module,
    hsic_reg: torch.Tensor,
    l0_reg: torch.Tensor,
    blocks: Dict[str, List[torch.nn.Parameter]],
) -> Dict[str, float]:
    """Compute per-block cosine similarity between HSIC and L0 gradients.

    Both gradients are obtained via ``torch.autograd.grad`` with
    ``retain_graph=True`` and ``allow_unused=True`` so that:

    * the autograd graph survives for the subsequent real ``backward()``;
    * ``.grad`` buffers are never touched (this is a pure read-out).

    Args:
        model:    Inner model whose parameters are being probed (accepted for
                  API symmetry / future use).
        hsic_reg: Scalar HSIC regularisation term (``lambda_hsic * hsic``),
                  still attached to the graph.
        l0_reg:   Scalar L0 regularisation term (``lambda_l0 * l0_penalty``),
                  still attached to the graph.
        blocks:   Mapping produced by :func:`build_interference_blocks`.

    Returns:
        ``{block: cosine}`` for each block plus ``"overall"``.  A block whose
        gradient is entirely zero for either objective yields ``NaN`` (e.g.
        pure reconstruction blocks, which receive no L0 gradient).
    """
    ordered_blocks = list(blocks.items())
    all_params: List[torch.nn.Parameter] = [
        p for _, plist in ordered_blocks for p in plist
    ]
    if len(all_params) == 0:
        return {}

    # One autograd.grad per objective over ALL params at once.
    g_hsic = torch.autograd.grad(
        hsic_reg, all_params,
        retain_graph=True, allow_unused=True, create_graph=False,
    )
    g_l0 = torch.autograd.grad(
        l0_reg, all_params,
        retain_graph=True, allow_unused=True, create_graph=False,
    )

    results: Dict[str, float] = {}

    idx = 0
    hsic_all_parts: List[torch.Tensor] = []
    l0_all_parts: List[torch.Tensor] = []
    for block_name, plist in ordered_blocks:
        n = len(plist)
        g_h = g_hsic[idx: idx + n]
        g_z = g_l0[idx: idx + n]
        idx += n

        v_h = _flatten_grads(g_h, plist)
        v_z = _flatten_grads(g_z, plist)
        results[block_name] = _cosine(v_h, v_z)

        hsic_all_parts.append(v_h)
        l0_all_parts.append(v_z)

    if hsic_all_parts:
        v_h_all = torch.cat(hsic_all_parts)
        v_z_all = torch.cat(l0_all_parts)
        results["overall"] = _cosine(v_h_all, v_z_all)

    return results

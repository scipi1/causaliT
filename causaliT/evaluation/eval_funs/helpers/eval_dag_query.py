"""
Architecture-agnostic DAG query helper.

Every causal architecture in this project exposes its learned structure as one
or more attention posteriors, but the *layout* differs:

===========================  ==================================================
Architecture                 Attention layout
===========================  ==================================================
AttentionSelectorLayer        ONE combined block, ``(B, L_X, L_S + L_X)``:
(single mode)                 columns ``0..L_S-1`` are S->X, the rest are X->X.
AttentionSelectorLayer        TWO modules (``attention`` + ``self_attention``),
(split mode, ``split_xx``)    but the layer re-concatenates them into the same
                              ``(B, L_X, L_S + L_X)`` layout.
SingleCausal / NoiseAware /   SEPARATE keys per block and per layer
StageCausal / proT            (``dec_cross``, ``dec_self``, ``dec_cross_L0``...).
===========================  ==================================================

Rather than hard-coding one of those layouts (which silently dropped folds when
the shape did not match), this module *classifies every attention tensor the
model returns by its shape* and assembles the canonical DAG blocks:

    ``cross``  (L_X, L_S)  -- S->X edges, compared to ``dec1_cross_att_mask.csv``
    ``self``   (L_X, L_X)  -- X->X edges, compared to ``dec1_self_att_mask.csv``

Multi-layer models get a ``_L{i}`` suffix (e.g. ``cross_L0``).  Blocks whose
attention module does not exist are simply absent from the result, so callers
can skip (rather than crash on) metrics that need both blocks, such as MEC.

Given ``L_S`` and ``L_X`` (read from the dataset metadata), each returned tensor
identifies itself:

    ``(L_X, L_S)``        -> ``cross``
    ``(L_X, L_X)``        -> ``self``
    ``(L_X, L_S + L_X)``  -> combined, split at ``L_S`` into ``cross`` + ``self``

When ``L_S == L_X`` makes the first two ambiguous, the attention key name
(``..._cross...`` / ``..._self...``) breaks the tie.  Keys that are not part of
the S->X / X->X DAG at all (``encoder``) are dropped by name, since an encoder
posterior is ``(L_S, L_S)`` and would otherwise masquerade as a ``self`` block
whenever ``L_S == L_X``.

No live model is required: the classification depends only on the attention
tensors and the two dataset dimensions.

Typical use::

    blocks   = query_dag_blocks(results.attention_weights, L_S, L_X)
    print(describe_topology(blocks, L_S, L_X))
    full_dag = assemble_full_dag(blocks, L_S, L_X)   # None if 'self' is absent
"""

from typing import Any, Dict, Optional, Tuple
import re

import numpy as np


# =============================================================================
# Canonical block naming
# =============================================================================

CROSS = "cross"   # S -> X
SELF = "self"     # X -> X

#: Canonical block name -> ``_load_true_dag_mask`` mask type.
BLOCK_MASK_TYPE = {
    CROSS: "dec_cross",
    SELF: "dec_self",
}

#: Attention keys that never carry S->X / X->X edges (see module docstring).
NON_DAG_KEYS = frozenset({"encoder", "enc", "enc_self", "enc_att"})


def block_mask_type(block_name: str) -> Optional[str]:
    """
    Map a canonical block name (``cross``, ``self``, ``cross_L2``, ...) to the
    ``mask_type`` understood by ``eval_utils._load_true_dag_mask``.
    """
    base = block_name.split("_L")[0]
    return BLOCK_MASK_TYPE.get(base)


def block_layer_index(block_name: str) -> Optional[int]:
    """Return the layer index encoded in a block name, or None if unsuffixed."""
    match = re.search(r"_L(\d+)$", block_name)
    return int(match.group(1)) if match else None


def canonical_block_name(attention_key: str) -> Optional[str]:
    """
    Translate an architecture-specific attention key into a canonical block name.

    This is a *name-only* hint: it says which block a key claims to be, without
    looking at the tensor.  ``query_dag_blocks`` lets the observed shape decide
    and falls back to this hint only when ``L_S == L_X``.

    Examples::

        dec_cross      -> cross
        dec_cross_L2   -> cross_L2
        decoder1_self  -> self
        cross          -> cross          (proT)
        decoder        -> self           (proT self-attention over X)
        encoder        -> None           (not part of the S->X / X->X DAG)
        att_combined   -> None           (needs splitting, not renaming)
    """
    key = attention_key
    layer = block_layer_index(key)
    suffix = f"_L{layer}" if layer is not None else ""
    base = re.sub(r"_L\d+$", "", key)

    if base in ("att_combined", "combined"):
        return None
    if base in NON_DAG_KEYS:
        return None
    if "cross" in base:
        return f"{CROSS}{suffix}"
    if "self" in base or base in ("decoder", "decoder1", "decoder2"):
        return f"{SELF}{suffix}"
    return None


def _name_hint(attention_key: str) -> Optional[str]:
    """Canonical block *base* name implied by a key, ignoring any layer suffix."""
    canonical = canonical_block_name(attention_key)
    return canonical.split("_L")[0] if canonical else None


# =============================================================================
# Attention -> DAG blocks
# =============================================================================

def reduce_attention(att: Any) -> Optional[np.ndarray]:
    """
    Reduce an attention tensor to a 2-D ``(n_targets, n_sources)`` matrix.

    Averages over the leading sample dimension and, when present, over the head
    dimension (``shared_dag_across_heads=False`` yields
    ``(B, H, L_X, L_S + L_X)``).

    Returns None when *att* is None or cannot be interpreted as a matrix.
    """
    if att is None:
        return None

    arr = att.detach().cpu().numpy() if hasattr(att, "detach") else np.asarray(att)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:            # (B, R, C)
        return arr.mean(axis=0)
    if arr.ndim == 4:            # (B, H, R, C)
        return arr.mean(axis=(0, 1))
    return None


def split_combined_attention(
    matrix: np.ndarray,
    L_S: Optional[int] = None,
    L_X: Optional[int] = None,
    verbose: bool = True,
    key_hint: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Classify one 2-D attention matrix into canonical DAG blocks by its shape.

    The observed width decides the layout:

    - width ``L_S + L_X`` -> ``{cross, self}`` (split at ``L_S``)
    - width ``L_S``       -> ``{cross}``
    - width ``L_X``       -> ``{self}``

    Args:
        matrix: 2-D attention matrix, rows = query tokens (X).
        L_S, L_X: Dataset dimensions.
        verbose: Print diagnostics for unexpected shapes.
        key_hint: Attention key the matrix came from; used only to disambiguate
            ``cross`` from ``self`` when ``L_S == L_X``.

    Returns:
        Dict of canonical block base names to matrices.  An empty dict means the
        matrix could not be interpreted, which lets the caller skip it instead of
        comparing a mis-shaped matrix against the ground truth.
    """
    if L_S is None or L_X is None:
        if verbose:
            print(
                "  [dag_query] Cannot classify attention: L_S/L_X unknown "
                f"(matrix shape {matrix.shape})."
            )
        return {}

    n_rows, n_cols = matrix.shape

    if n_rows != L_X:
        if verbose:
            print(
                f"  [dag_query] Attention has {n_rows} query rows but L_X={L_X} "
                f"(shape {matrix.shape}) - not a DAG block, skipped."
            )
        return {}

    if n_cols == L_S + L_X:
        return {CROSS: matrix[:, :L_S], SELF: matrix[:, L_S:]}

    if n_cols == L_S and n_cols == L_X:
        # Ambiguous: (L_X, L_S) and (L_X, L_X) have the same shape.
        hint = _name_hint(key_hint) if key_hint else None
        if hint is None:
            if verbose:
                print(
                    f"  [dag_query] Ambiguous attention block (L_S == L_X == {L_S}) "
                    f"and key '{key_hint}' does not say whether it is S->X or "
                    "X->X - skipped."
                )
            return {}
        return {hint: matrix}

    if n_cols == L_S:
        return {CROSS: matrix}
    if n_cols == L_X:
        return {SELF: matrix}

    if verbose:
        print(
            f"  [dag_query] Unexpected attention width {n_cols}; expected "
            f"{L_S + L_X} (S|X), {L_S} (S only) or {L_X} (X only). "
            f"Full shape {matrix.shape}."
        )
    return {}


def query_dag_blocks(
    attention: Any,
    L_S: Optional[int] = None,
    L_X: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Query the learned DAG from attention weights, in canonical block form.

    Args:
        attention: Any of
            - a dict of ``{attention_key: tensor}`` (predictor output). Each
              tensor is classified by shape; the key contributes only its
              ``_L{i}`` layer suffix, the ``encoder`` veto, and the ``L_S == L_X``
              tie-break.
            - a single tensor/array, treated as one unnamed block.
            Tensors may be 2-D, 3-D ``(B, R, C)`` or 4-D ``(B, H, R, C)``.
        L_S: Number of source tokens (columns of the ``cross`` block).
        L_X: Number of intermediate tokens (rows of both blocks).
        verbose: Print diagnostics for unexpected shapes.

    Returns:
        Dict mapping canonical block names (``cross``, ``self``, ``cross_L0``...)
        to 2-D probability matrices.  Blocks that the model does not provide are
        absent - callers must therefore use ``.get()`` and skip metrics that
        require a missing block.

    Example:
        >>> blocks = query_dag_blocks({"att_combined": att}, L_S=3, L_X=4)
        >>> sorted(blocks)
        ['cross', 'self']
    """
    if not isinstance(attention, dict):
        attention = {"att_combined": attention}

    blocks: Dict[str, np.ndarray] = {}

    for key, att in attention.items():
        key = str(key)
        base = re.sub(r"_L\d+$", "", key)
        if base in NON_DAG_KEYS:
            continue

        matrix = reduce_attention(att)
        if matrix is None:
            continue

        layer = block_layer_index(key)
        suffix = f"_L{layer}" if layer is not None else ""

        classified = split_combined_attention(
            matrix, L_S=L_S, L_X=L_X, verbose=verbose, key_hint=key
        )
        for name, block in classified.items():
            # Keep the first occurrence: per-key architectures may expose both a
            # per-layer key and a backward-compatible alias for the same tensor.
            blocks.setdefault(f"{name}{suffix}", block)

    return blocks


def describe_topology(
    blocks: Dict[str, np.ndarray],
    L_S: Optional[int] = None,
    L_X: Optional[int] = None,
) -> str:
    """
    One-line, human-readable summary of the blocks a fold produced (logging only).

    Example:
        >>> describe_topology({"cross": np.zeros((4, 3))}, L_S=3, L_X=4)
        'blocks=[cross] (no X->X block: MEC will be skipped) L_S=3 L_X=4'
    """
    names = sorted(blocks)
    has_cross = any(n.split("_L")[0] == CROSS for n in names)
    has_self = any(n.split("_L")[0] == SELF for n in names)

    if has_cross and has_self:
        note = ""
    elif has_cross:
        note = " (no X->X block: MEC will be skipped)"
    elif has_self:
        note = " (no S->X block: MEC will be skipped)"
    else:
        note = " (no DAG block could be classified)"

    return f"blocks={names or ['<none>']}{note} L_S={L_S} L_X={L_X}"


def assemble_full_dag(
    blocks: Dict[str, np.ndarray],
    L_S: Optional[int] = None,
    L_X: Optional[int] = None,
    layer_suffix: str = "",
) -> Optional[np.ndarray]:
    """
    Assemble the full ``(L_S + L_X, L_S + L_X)`` DAG from canonical blocks.

    Both a ``cross`` and a ``self`` block are required; returns None otherwise
    (e.g. a cross-attention-only model), which is the signal for callers to skip
    MEC / skeleton / v-structure metrics.

    Args:
        blocks: Output of ``query_dag_blocks``.
        L_S, L_X: Dimensions; inferred from the block shapes when omitted.
        layer_suffix: Evaluate a specific layer, e.g. ``"_L0"``.

    Returns:
        Continuous adjacency matrix with sources first, or None.
    """
    # Local import: eval_utils imports matplotlib, keep this module light.
    from .eval_utils import _combine_attention_to_full_dag

    cross = blocks.get(f"{CROSS}{layer_suffix}")
    self_adj = blocks.get(f"{SELF}{layer_suffix}")

    if cross is None or self_adj is None:
        return None

    L_X = int(L_X) if L_X is not None else int(cross.shape[0])
    L_S = int(L_S) if L_S is not None else int(cross.shape[1])

    if cross.shape != (L_X, L_S) or self_adj.shape != (L_X, L_X):
        print(
            f"  [dag_query] Cannot assemble full DAG: cross={cross.shape} "
            f"self={self_adj.shape} vs L_S={L_S}, L_X={L_X}."
        )
        return None

    return _combine_attention_to_full_dag(
        cross_adj=cross,
        self_adj=self_adj,
        n_source=L_S,
        n_intermediate=L_X,
    )


def block_axis_labels(
    block_name: str,
    n_rows: int,
    n_cols: int,
    metadata: Optional[dict] = None,
) -> Tuple[list, list]:
    """
    Build row/column labels for a DAG block.

    Rows are always targets (X).  Columns are sources (S) for ``cross`` blocks
    and targets (X) for ``self`` blocks.  Dataset metadata labels are used when
    available, otherwise generic ``S1..``/``X1..`` names.
    """
    var_info = (metadata or {}).get("variable_info", {}) or {}
    src_labels = list(var_info.get("source_labels", []))
    inp_labels = list(var_info.get("input_labels", []))

    rows = inp_labels[:n_rows] if len(inp_labels) >= n_rows else [f"X{i+1}" for i in range(n_rows)]

    if block_name.split("_L")[0] == CROSS:
        cols = src_labels[:n_cols] if len(src_labels) >= n_cols else [f"S{j+1}" for j in range(n_cols)]
    else:
        cols = inp_labels[:n_cols] if len(inp_labels) >= n_cols else [f"X{j+1}" for j in range(n_cols)]

    return rows, cols

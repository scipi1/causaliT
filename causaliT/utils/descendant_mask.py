"""
Descendant-excluding HSIC pair masks derived from the learned adjacency.

Motivation
==========
The structural loss of the AttentionSelector minimises a residual-HSIC term

    hsic = mean_{i,j} HSIC(source_j, residual_i)

over **every** (child, candidate-parent) pair.  Under an additive-noise model
``X_i = f_i(PA_i) + e_i`` the correct solution has ``r_i = e_i``, and then

    * ``j`` a NON-descendant of ``i``  ->  HSIC(X_j, e_i) == 0   (valid signal)
    * ``j`` a DESCENDANT of ``i``      ->  HSIC(X_j, e_i)  > 0   (BIAS)
    * ``j == i``                       ->  HSIC(X_i, e_i)  > 0   (BIAS)

The second and third rows are *irreducible*: a descendant ``X_j`` is by
construction a function of ``e_i``, and ``X_i`` itself trivially contains
``e_i``.  Including them means the TRUE DAG is not a minimiser of the
structural loss — the optimiser can only shrink those terms by making
``r_i != e_i``, i.e. by regressing ``X_i`` on its own descendants.  That is
exactly the documented "attends to descendants instead of parents" failure
mode.

This module builds a **pair mask** that removes ``Desc(i) U {i}`` from the HSIC
average, restoring consistency: the true DAG becomes a minimiser.

Index conventions
=================
Both the attention posteriors and the HSIC matrix use the SAME layout, which is
what makes the mask a drop-in element-wise factor:

    score_tensor[i, j]  ==  P(edge j -> i)        (row = child, col = parent)
    hsic_matrix[i, j]   ==  HSIC(source_j, res_i) (row = child/target, col = source)

The standard graph adjacency ``M[u, v] = 1 iff u -> v`` is therefore the
TRANSPOSE of the score tensor.

Two node topologies (mirrors ``AttentionSelectorLayer``)
========================================================
* **Homogeneous** (``homogeneous_nodes=True``): the score tensor is the square
  ``(N, N)`` directed posterior over all nodes.  Every column is eligible for
  masking, because S nodes are queries too and can acquire parents.

* **Split** (``homogeneous_nodes=False``): the score tensor is
  ``(L_X, L_S + L_X)`` = ``[S->X cross | X->X self]``.  The graph is padded to a
  square ``(N, N)`` by prepending ``L_S`` all-zero rows (the split-mode prior:
  nothing points into an S node).  Because those rows are empty, **no S node is
  ever reachable**, so the ``L_S`` cross columns are always kept.  Masking
  therefore applies to the SELF-attention columns only — in cross-attention the
  keys are exogenous sources and can never be descendants.  This invariant is
  additionally hard-pinned in :func:`build_hsic_pair_mask` so a future change to
  the cross block cannot silently start masking S keys.

Gradient safety
===============
The returned mask is ALWAYS detached.  A differentiable mask would open a
loss-hacking channel: the model could create an edge ``i -> j`` purely to delete
the ``(i, j)`` HSIC term and lower the loss for free.  Detaching makes the mask a
pure gating decision.
"""

from typing import Optional, Tuple

import math

import torch


__all__ = [
    "harden_adjacency",
    "transitive_closure",
    "build_hsic_pair_mask",
    "soft_transitive_closure",
    "build_hsic_pair_mask_budgeted",
]


def harden_adjacency(
    score_tensor: torch.Tensor,
    threshold: float = 0.5,
    resolve_two_cycles: bool = True,
) -> torch.Tensor:
    """Threshold a directed edge posterior into a boolean *child <- parent* matrix.

    Args:
        score_tensor: ``(R, C)`` directed posterior; ``[i, j]`` is ``P(j -> i)``.
            Typically ``GatedSelfAttention.score_tensor_for_sparsity`` (the
            ``P(z_edge>0) * d`` product), values in ``(0, 1)``.
        threshold: Posterior above which an edge is considered present.
        resolve_two_cycles: When the matrix is square, keep only the STRONGER of
            each ``(i, j)`` / ``(j, i)`` pair.  ``GatedSelfAttention`` guarantees
            ``d_ij + d_ji == 1``, so a two-cycle means both directions crossed
            the threshold on the existence gate alone; keeping the argmax
            direction removes the most common source of spurious cycles (which
            would otherwise inflate the descendant closure dramatically).

    Returns:
        Boolean tensor ``(R, C)``, ``[i, j] = True`` iff edge ``j -> i`` is present.
    """
    hard = score_tensor > float(threshold)

    if resolve_two_cycles and score_tensor.dim() == 2 and score_tensor.shape[0] == score_tensor.shape[1]:
        both = hard & hard.transpose(0, 1)
        if bool(both.any()):
            # Keep the strictly stronger direction; ties (exactly equal scores)
            # are dropped in BOTH directions, which is the conservative choice
            # (no orientation information available).
            stronger = score_tensor > score_tensor.transpose(0, 1)
            hard = hard & (~both | stronger)

    return hard


def transitive_closure(
    adjacency: torch.Tensor,
    hops: Optional[int] = None,
) -> torch.Tensor:
    """Reachability closure of a boolean *child <- parent* adjacency.

    Args:
        adjacency: Boolean ``(N, N)`` matrix, ``[i, j] = True`` iff ``j -> i``.
        hops: Maximum path length to follow.  ``None`` (default) computes the
            FULL closure; ``1`` returns the input unchanged (direct children
            only); ``k`` follows paths of up to ``k`` edges.

    Returns:
        Boolean ``(N, N)`` matrix ``R`` where ``R[i, j] = True`` iff ``j`` is
        reachable FROM ``i``, i.e. ``j`` is a descendant of ``i``.  The diagonal
        is ``True`` only when ``i`` lies on a cycle.

    Notes:
        Uses repeated squaring, so the full closure costs ``ceil(log2(N))``
        boolean matmuls — negligible next to the ``O(N^2)`` HSIC kernel work.
        Cyclic graphs are handled correctly (the closure simply saturates).
    """
    if adjacency.dim() != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"transitive_closure expects a square (N, N) adjacency, got "
            f"{tuple(adjacency.shape)}."
        )

    N = adjacency.shape[0]
    # ``desc`` holds reachability FROM i.  ``adjacency[i, j]`` means ``j -> i``,
    # i.e. j is a PARENT of i; reachability from i follows the reversed arrow,
    # so we work with the transpose: ``M[i, j] = True`` iff ``i -> j``.
    M = adjacency.transpose(0, 1).clone().bool()

    if hops is not None and int(hops) <= 1:
        return M

    desc = M
    if N <= 1:
        return desc

    if hops is None:
        n_steps = max(1, math.ceil(math.log2(N)))
    else:
        n_steps = max(1, math.ceil(math.log2(float(int(hops)))))

    for _ in range(n_steps):
        # desc_{2k} = desc_k OR (desc_k @ desc_k)
        nxt = desc | (desc.float() @ desc.float() > 0)
        if bool((nxt == desc).all()):
            break
        desc = nxt

    return desc


def build_hsic_pair_mask(
    score_tensor: torch.Tensor,
    s_seq_len: int,
    homogeneous_nodes: bool,
    threshold: float = 0.5,
    hops: Optional[int] = None,
    exclude_self: bool = True,
    excluded_weight: float = 0.0,
    resolve_two_cycles: bool = True,
) -> Tuple[torch.Tensor, float, bool]:
    """Build the element-wise HSIC pair weight matrix that drops descendants.

    Args:
        score_tensor: Directed edge posterior from the attention block.
            ``(N, N)`` in homogeneous mode, ``(L_X, L_S + L_X)`` in split mode.
            ``[i, j]`` is ``P(j -> i)``.
        s_seq_len: ``L_S`` — number of leading S (source) nodes.
        homogeneous_nodes: Node topology flag (see the module docstring).
        threshold: Posterior threshold for calling an edge present.
        hops: ``None`` for the full transitive closure, ``1`` for direct
            children only, ``k`` for paths up to length ``k``.
        exclude_self: Also exclude the diagonal pair ``HSIC(X_i, r_i)``, which
            is biased for exactly the same reason as the descendant pairs.
        excluded_weight: Weight GIVEN TO excluded pairs.  ``0.0`` (default)
            removes them entirely; a small positive value keeps a gradient
            trickle (the "soft" variant).
        resolve_two_cycles: Forwarded to :func:`harden_adjacency`.

    Returns:
        ``(mask, kept_frac, is_cyclic)`` where

        * ``mask`` — detached float tensor with the SAME shape as
          ``score_tensor``; ``1.0`` on kept pairs and ``excluded_weight`` on
          excluded ones.  Directly multiplicable with the HSIC pair matrix.
        * ``kept_frac`` — purely combinatorial fraction of pairs NOT flagged as
          descendant/self, independent of ``excluded_weight``, so it means the
          same thing in the hard and soft variants.  This is the collapse
          indicator.
        * ``is_cyclic`` — whether the thresholded graph contains a cycle
          (diagnostic; a cycle inflates the closure and is worth logging).
    """
    if score_tensor.dim() != 2:
        raise ValueError(
            f"build_hsic_pair_mask expects a 2-D (target, source) score tensor, "
            f"got {tuple(score_tensor.shape)}."
        )

    score = score_tensor.detach()
    L_target, L_source = score.shape
    S = int(s_seq_len)

    hard = harden_adjacency(score, threshold=threshold, resolve_two_cycles=resolve_two_cycles)

    if homogeneous_nodes:
        if L_target != L_source:
            raise ValueError(
                f"homogeneous_nodes=True requires a square (N, N) score tensor, "
                f"got {tuple(score.shape)}."
            )
        square = hard                                        # (N, N) child <- parent
    else:
        # Split mode: pad the top L_S rows with zeros (nothing points into an S
        # node) to obtain the square (N, N) graph over all nodes.
        N = L_source
        if L_target + S != N:
            raise ValueError(
                f"split-mode score tensor shape {tuple(score.shape)} is inconsistent "
                f"with s_seq_len={S} (expected L_target + L_S == L_source)."
            )
        square = torch.zeros((N, N), dtype=torch.bool, device=score.device)
        square[S:, :] = hard

    desc = transitive_closure(square, hops=hops)             # [i, j] -> j desc of i
    is_cyclic = bool(desc.diagonal().any())

    if exclude_self:
        eye = torch.eye(desc.shape[0], dtype=torch.bool, device=desc.device)
        desc = desc | eye

    # Slice back to the (target, source) layout of the HSIC matrix.
    excluded = desc if homogeneous_nodes else desc[S:, :]

    if not homogeneous_nodes:
        # Invariant pin: the cross (S) columns can never be descendants of an X
        # node, because the padded S rows are empty.  Enforce it explicitly so a
        # future change to the cross block cannot silently mask S keys.
        excluded = excluded.clone()
        excluded[:, :S] = False

    mask = torch.where(
        excluded,
        torch.full_like(score, float(excluded_weight)),
        torch.ones_like(score),
    )

    total = excluded.numel()
    kept_frac = float((~excluded).sum().item()) / float(total) if total > 0 else 1.0

    return mask.detach(), kept_frac, is_cyclic


def soft_transitive_closure(
    pi: torch.Tensor,
    tnorm: str = "min",
    hops: Optional[int] = None,
) -> torch.Tensor:
    """Soft (fuzzy) reachability closure of a directed edge posterior.

    The continuous analogue of :func:`transitive_closure`: ``pi[i, j]`` is
    ``P(j -> i)`` (child <- parent), and the result ``desc[i, j]`` is the soft
    score that ``j`` is reachable FROM ``i`` (a descendant of ``i``), computed
    by iterating the t-norm composition ``desc <- max(desc, desc . desc)`` to
    the full closure (or the ``hops``-limited one).

    Args:
        pi: ``(N, N)`` posterior with ``pi[i, j] = P(j -> i)``, values in [0, 1].
        tnorm: ``"min"`` (Goedel — a chain keeps the confidence of its weakest
            link) or ``"prod"`` (the literal product, which deflates).
        hops: Maximum path length to follow.  ``None`` (default) = full closure;
            ``1`` = direct children only; ``k`` = paths up to length ``k``.

    Returns:
        ``(N, N)`` soft descendant scores in [0, 1].  The diagonal is > 0 only
        when the graph has a (soft) cycle.
    """
    if pi.dim() != 2 or pi.shape[0] != pi.shape[1]:
        raise ValueError(
            f"soft_transitive_closure expects a square (N, N) posterior, got "
            f"{tuple(pi.shape)}."
        )
    n = pi.shape[0]
    # ``pi[i, j]`` means ``j -> i``; reachability from i follows the reversed
    # arrow, so (as in ``transitive_closure``) we work with the transpose.
    M = pi.transpose(0, 1)
    if hops is not None and int(hops) <= 1:
        return M
    desc = M
    if n <= 1:
        return desc
    if hops is None:
        n_steps = max(1, math.ceil(math.log2(n)))
    else:
        n_steps = max(1, math.ceil(math.log2(float(int(hops)))))
    for _ in range(n_steps):
        a = desc.unsqueeze(-1)          # (i, k, 1) -> desc[i, k]
        b = desc.unsqueeze(-3)          # (1, k, j) -> desc[k, j]
        if tnorm == "prod":
            joint = a * b
        elif tnorm == "min":
            joint = torch.minimum(a.expand(*desc.shape, n),
                                  b.expand(*desc.shape, n))
        else:
            raise ValueError(f"unknown tnorm={tnorm!r} (expected 'min' or 'prod')")
        desc = torch.maximum(desc, joint.max(dim=-2).values)
    return desc


def build_hsic_pair_mask_budgeted(
    score_tensor: torch.Tensor,
    s_seq_len: int,
    homogeneous_nodes: bool,
    budget_frac: float = 0.25,
    per_row: bool = True,
    exclude_self: bool = True,
    excluded_weight: float = 0.0,
    tnorm: str = "min",
    hops: Optional[int] = None,
    confident_score: Optional[float] = None,
) -> Tuple[torch.Tensor, float, bool]:
    """Budgeted descendant-excluding HSIC pair mask: the cap IS the guard.

    The threshold-based :func:`build_hsic_pair_mask` is brittle: it hardens the
    learned posterior at a fixed threshold and hopes the result is a DAG.  When
    the learned graph is dense and cyclic (the common case early in training),
    the transitive closure saturates, every pair is flagged as a descendant,
    and the collapse guard falls back to the unmasked HSIC — so the feature
    never acts.

    This budgeted variant replaces the threshold with a CAP on how many pairs
    are removed: the pairs are ranked by a SOFT descendant score (the detached
    posterior's fuzzy transitive closure, see :func:`soft_transitive_closure`)
    and the top ``budget_frac`` are excluded — per child row (``per_row=True``,
    the default: every child keeps most of its candidates) or globally.  The
    cap makes the collapse mode impossible by construction, so the mask
    triggers every step, and it is self-normalizing across training: early on
    the diffuse posterior still ranks the most descendant-ish pairs on top, and
    the exclusion sharpens as the posterior sharpens.

    Args:
        score_tensor: Directed edge posterior from the attention block
            (``[i, j] = P(j -> i)``); same layout as :func:`build_hsic_pair_mask`.
        s_seq_len: ``L_S`` — number of leading S (source) nodes.
        homogeneous_nodes: Node topology flag (see the module docstring).
        budget_frac: Fraction of pairs to exclude (per row when ``per_row``,
            globally otherwise).  The diagonal is always ranked first when
            ``exclude_self``.
        per_row: Rank per child row (default) rather than globally.
        exclude_self: Also exclude the diagonal pair ``HSIC(X_i, r_i)``.
        excluded_weight: Weight GIVEN TO excluded pairs (0.0 = drop entirely).
        tnorm: T-norm for the soft closure (``"min"`` default, ``"prod"``).
        hops: Maximum path length for the closure; ``None`` = full.
        confident_score: Optional CONFIDENT-DESCENDANT override.  When set,
            any pair whose soft descendant score is >= ``confident_score`` is
            excluded EVEN BEYOND the ``budget_frac`` cap, so a confidently
            identified source row (all nodes downstream) empties completely --
            the row then contributes no term to the masked mean, which is the
            correct semantics for a parentless node.  ``None`` (default)
            keeps the pure budget-cap behaviour.

    Returns:
        ``(mask, kept_frac, is_cyclic)`` mirroring :func:`build_hsic_pair_mask`:
        ``mask`` is detached and directly multiplicable with the HSIC pair
        matrix; ``kept_frac`` is the combinatorial fraction of pairs kept;
        ``is_cyclic`` flags a soft cycle (a node strongly reachable from
        itself) as a diagnostic.
    """
    if score_tensor.dim() != 2:
        raise ValueError(
            f"build_hsic_pair_mask_budgeted expects a 2-D (target, source) "
            f"score tensor, got {tuple(score_tensor.shape)}."
        )
    if not (0.0 < float(budget_frac) <= 1.0):
        raise ValueError(f"budget_frac must be in (0, 1], got {budget_frac!r}.")

    if confident_score is not None and not (0.0 < float(confident_score) <= 1.0):
        raise ValueError(
            f"confident_score must be in (0, 1] or None, got {confident_score!r}."
        )

    score = score_tensor.detach()
    L_target, L_source = score.shape
    S = int(s_seq_len)

    if homogeneous_nodes:
        if L_target != L_source:
            raise ValueError(
                f"homogeneous_nodes=True requires a square (N, N) score tensor, "
                f"got {tuple(score.shape)}."
            )
        square = score
    else:
        # Split mode: pad the top L_S rows with zeros (nothing points into an S
        # node) to obtain the square (N, N) graph over all nodes.
        N = L_source
        if L_target + S != N:
            raise ValueError(
                f"split-mode score tensor shape {tuple(score.shape)} is "
                f"inconsistent with s_seq_len={S} (expected L_target + L_S == "
                f"L_source)."
            )
        square = torch.zeros((N, N), dtype=score.dtype, device=score.device)
        square[S:, :] = score

    desc = soft_transitive_closure(square, tnorm=tnorm, hops=hops)
    # Diagnostic only: a node strongly reachable from itself means a soft cycle.
    is_cyclic = bool((desc.diagonal() > 0.5).any())

    if exclude_self:
        desc = desc.clone()
        desc.fill_diagonal_(1.0)   # the self pair is always top-ranked

    # Slice back to the (target, source) layout of the HSIC matrix.
    excluded_score = desc if homogeneous_nodes else desc[S:, :]
    if not homogeneous_nodes:
        # Invariant pin (mirrors the threshold variant): the cross (S) columns
        # can never be descendants of an X node.
        excluded_score = excluded_score.clone()
        excluded_score[:, :S] = 0.0

    # Rank by the soft descendant score and exclude the top budget_frac.  Pairs
    # with a zero score (no path) are never excluded, so a sink row keeps all
    # its off-diagonal candidates.
    if per_row:
        k = max(1, int(math.ceil(budget_frac * L_source)))
        # The k-th largest score per row is the per-row exclusion threshold.
        thr = excluded_score.sort(dim=-1, descending=True).values[:, k - 1:k]
        excluded = (excluded_score >= thr) & (excluded_score > 0)
    else:
        flat = excluded_score.reshape(-1)
        k = max(1, int(math.ceil(budget_frac * flat.numel())))
        thr = flat.sort(descending=True).values[k - 1]
        excluded = (excluded_score >= thr) & (excluded_score > 0)

    if confident_score is not None:
        # Confident-descendant override: scores this high carry orientation
        # evidence regardless of the rank budget, so a true source row (every
        # column downstream) is allowed to empty entirely.
        excluded = excluded | (excluded_score >= float(confident_score))

    mask = torch.where(
        excluded,
        torch.full_like(score, float(excluded_weight)),
        torch.ones_like(score),
    )

    total = excluded.numel()
    kept_frac = float((~excluded).sum().item()) / float(total) if total > 0 else 1.0

    return mask.detach(), kept_frac, is_cyclic

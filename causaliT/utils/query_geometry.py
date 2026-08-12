"""Geometric TRANSITIVE correction of structural queries (grandparent suppression).

Problem
=======
A child that already found its true parents keeps non-zero mass on its
GRANDparents: in the homogeneous ``scm_equal`` run, X5 (true parents X1, X2, X3)
still scores S1, S2, S3 at ~0.3-0.44 through the paths ``S1 -> X1 -> X5``.
Nothing in the loss removes them (HSIC is already satisfied once the true parents
are used and ``lambda_l0`` is numerically negligible), so the fix has to be
structural, not another penalty.

Mechanism
=========
With an ORTHONORMAL key frame the cosines ``c_ij = <u_i, khat_j>`` ARE the
coordinates of the unit query ``u_i`` in the key basis, so removing the ``khat_j``
component removes exactly the logit of the edge ``j -> i`` and nothing else:

    u_i <- u_i - W_ij * (c_ij + delta) * khat_j

``W_ij`` in [0, alpha] measures how well the edge ``j -> i`` is already explained
by a MEDIATOR ``k`` (``j -> k -> i``).  Because the query is re-normalised
downstream (``normalize_query``), the coordinate removed from a grandparent is
GIVEN BACK to the true parents — the correction raises the true edges while it
lowers the indirect ones.

Three design choices, all settled by the offline probe on a trained checkpoint
(``scripts/_probe_transitive_correction.py``, findings in
``experiments/6_INVESTIGATIONS/HOMOGENEOUS/TRANSITIVE_CORRECTION.md``):

1. the weight is built from the model's own DETACHED posterior ``Pi``, not from
   raw cosines (a differentiable weight would let the model delete a real edge
   elsewhere just to raise a logit);
2. the mediation mass uses the Goedel t-norm ``max_k min(Pi_ik, Pi_kj)``, not the
   product: with soft posteriors the product DEFLATES (0.65 * 0.80 = 0.52) and
   would barely fire;
3. the target is a NEGATIVE cosine (``delta > 0``), not a projection to zero.
   A projection to ``c = 0`` only removes the edge's SUPPORT; with the default
   Hard-Concrete stretch (``gamma=-0.1``, ``zeta=1.1``) the posterior at zero
   score is still ``sigmoid(1.6) = 0.83``, i.e. firmly ON.  Suppression requires
   pushing the coordinate to the negative side, and ``delta = 0.5`` is the
   measured operating point.

``delta`` lives in COSINE space (it is a query coordinate), NOT in logit space:
it must stay inside ``[-1, 1]``, so it cannot be rescaled by a block's logit
offsets.  A block with a positive ``init_edge_offset`` (the S->X cross block
when the init balance is active) is already biased OFF, so the same cosine push
is simply more than it needs — never less, which is the safe direction.  With
the offset disabled (existence-level balance) the push is exactly enough.

The margin gate ``relu(m_ij - Pi_ij)`` makes the correction self-silencing: it is
exactly 0 while the direct edge is better supported than the mediated path, which
is what protects the all-on centroid initialisation (measured max W = 0.007 at
init) without any warmup schedule.

Assumption on the record: the correction assumes the true graph equals its own
transitive reduction.  A shielded pair (a real direct edge that ALSO has a
mediated path) is penalised by construction, hence the opt-in flag and
``alpha < 1``, which keeps this a bias the data can overrule.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "mediation_mass",
    "transitive_weights",
    "key_frame",
    "frame_offdiag",
    "assert_orthonormal_frame",
    "correct_query",
    "transitive_logit_bias",
]

# Frames looser than this are refused: the component removal would silently
# perturb the OTHER edges of the same row (the coordinates stop being separable).
FRAME_TOL: float = 1e-3


# =============================================================================
# Trigger: how strongly is the edge j -> i explained by a mediator k?
# =============================================================================

def mediation_mass(pi: torch.Tensor, tnorm: str = "min") -> torch.Tensor:
    """``m_ij = max_k T(Pi_ik, Pi_kj)`` with ``T`` a t-norm (fuzzy AND).

    Args:
        pi: ``(..., N, N)`` posterior with ``pi[i, j] = P(j -> i)``, so the path
            ``j -> k -> i`` is supported by ``T(pi[i, k], pi[k, j])``.
        tnorm: ``"min"`` (Goedel, default — a chain of confident-but-soft edges
            keeps the confidence of its weakest link) or ``"prod"`` (the literal
            product, which deflates).

    Returns:
        ``(..., N, N)`` mediation mass in [0, 1], zero diagonal.  ``k = i`` and
        ``k = j`` cannot contribute: they need ``pi_ii`` / ``pi_jj``, which the
        zeroed diagonal kills.
    """
    n = pi.shape[-1]
    if pi.shape[-2] != n:
        raise ValueError(f"mediation_mass needs a square graph, got {tuple(pi.shape)}")
    eye = torch.eye(n, dtype=torch.bool, device=pi.device)
    p = pi.masked_fill(eye, 0.0)
    a = p.unsqueeze(-1)          # (..., i, k, 1) -> pi[i, k]
    b = p.unsqueeze(-3)          # (..., 1, k, j) -> pi[k, j]
    if tnorm == "prod":
        joint = a * b
    elif tnorm == "min":
        joint = torch.minimum(a.expand(*p.shape, n), b.expand(*p.shape, n))
    else:
        raise ValueError(f"unknown tnorm={tnorm!r} (expected 'min' or 'prod')")
    return joint.max(dim=-2).values.masked_fill(eye, 0.0)


def transitive_weights(
    pi: torch.Tensor,
    alpha: float = 0.5,
    tnorm: str = "min",
    margin: bool = True,
    symmetric: bool = True,
    symmetric_span: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Detached shrink weights ``W_ij`` in ``[0, alpha]``.

    Args:
        pi: ``(..., N, N)`` posterior, ``pi[i, j] = P(j -> i)``.
        alpha: strength.  ``0.5`` is the measured safe operating point; at
            ``1.0`` with repeated application the correction starts eating true
            edges.
        tnorm: see ``mediation_mass``.
        margin: gate by ``relu(m - Pi)`` so the correction is silent while the
            direct edge beats the mediated path (init safety).
        symmetric: symmetrise ``W <- max(W, W^T)``.  A direction-gated block
            factorises the posterior into a SYMMETRIC existence part and an
            ANTISYMMETRIC direction part; editing one triangle only would leave
            existence half-attacked and drift the orientation instead.
        symmetric_span: ``(start, stop)`` limiting the symmetrisation to a square
            sub-block (split mode: only the X-X self block is square; the
            bipartite S->X block has no reverse entry to balance).  ``None``
            symmetrises the whole matrix.

    Returns:
        ``(..., N, N)`` detached weights.
    """
    pi = pi.detach()
    m = mediation_mass(pi, tnorm=tnorm)
    w = (m - pi).clamp_min(0.0) if margin else m
    w = float(alpha) * w
    if symmetric:
        if symmetric_span is None:
            w = torch.maximum(w, w.transpose(-1, -2))
        else:
            i0, i1 = symmetric_span
            blk = w[..., i0:i1, i0:i1]
            w = w.clone()
            w[..., i0:i1, i0:i1] = torch.maximum(blk, blk.transpose(-1, -2))
    return w.detach()


# =============================================================================
# Key frame — the precondition that makes the removal exact
# =============================================================================

def key_frame(key: torch.Tensor) -> torch.Tensor:
    """Row-normalise the keys: ``khat[j] = k_j / ||k_j||``."""
    return key / key.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def frame_offdiag(khat: torch.Tensor) -> torch.Tensor:
    """Max ``|off-diagonal|`` of the Gram matrix of a row-normalised frame.

    0 = perfectly orthonormal, so every cosine is an independent coordinate.
    Returned as a tensor (kept on-device, cheap: one ``(N, N)`` matmul).
    """
    n = khat.shape[-2]
    gram = khat @ khat.transpose(-1, -2)
    eye = torch.eye(n, device=khat.device, dtype=khat.dtype)
    return (gram - eye).abs().amax()


def assert_orthonormal_frame(
    key: torch.Tensor, tol: float = FRAME_TOL, context: str = "",
) -> float:
    """Raise unless the keys form an orthonormal frame.

    The transitive correction removes a single coordinate of the query.  On a
    non-orthonormal frame that removal LEAKS into the other edges of the same
    row, so the mechanism would quietly compute something else.  We fail loud
    instead: a mis-configured arm must not produce plausible-looking numbers.
    """
    off = float(frame_offdiag(key_frame(key)))
    if off > tol:
        raise ValueError(
            f"transitive_correction requires an ORTHONORMAL key frame, but the "
            f"combined key Gram has max |off-diagonal| = {off:.3e} > {tol:.1e}"
            f"{' (' + context + ')' if context else ''}. Use "
            "struct_embedding_type='orthogonal_fixed' with remove_key_projection=true "
            "(or a single SHARED orthogonal key projection: per-block key rotations "
            "keep each block orthonormal but destroy orthogonality BETWEEN the "
            "S and X keys)."
        )
    return off


# =============================================================================
# Instruments
# =============================================================================

def correct_query(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    delta: float = 0.5,
    khat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Remove the mediated key components from the query.

    ``u <- u - (W * (c + delta)) @ khat`` with ``u`` the unit query and
    ``c = u @ khat^T`` its coordinates.  ``delta = 0`` recovers a plain
    projection (target cosine 0); ``delta > 0`` pushes the coordinate to
    ``-delta`` where the gate actually suppresses.

    Args:
        query: ``(..., N, E)`` query (normalised internally, so the caller can
            pass either the raw or the already-normalised query).
        key: ``(..., M, E)`` keys.
        weights: ``(..., N, M)`` detached weights (0 = leave the edge alone).
        delta: target cosine on the negative side.
        khat: optionally pre-computed ``key_frame(key)`` to avoid recomputing.

    Returns:
        ``(..., N, E)`` corrected query, NOT re-normalised (the caller's
        ``normalize_query`` does that and thereby reallocates the freed budget).
    """
    kh = key_frame(key) if khat is None else khat
    u = F.normalize(query, p=2.0, dim=-1, eps=1e-8)
    c = torch.einsum("...ne,...me->...nm", u, kh)
    coeff = weights * (c + float(delta))
    return u - torch.einsum("...nm,...me->...ne", coeff, kh)


def transitive_logit_bias(weights: torch.Tensor, eta: float = 4.0) -> torch.Tensor:
    """Score-space ablation: the bias ``-eta * W`` to add to the edge logits.

    Kept for comparison only.  It removes MORE false mass than the query-space
    instrument but LOWERS the true edges too, because it bypasses the query-norm
    budget and therefore cannot reallocate anything.
    """
    return -float(eta) * weights

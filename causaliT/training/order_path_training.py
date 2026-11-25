"""Independent forward policies and weight-protected order-score surgery."""
from contextlib import contextmanager

import torch


@contextmanager
def forward_policy(model, structural, order_sampling='soft'):
    """Switch order/BKD policy temporarily, retaining this forward's caches."""
    if order_sampling not in ('soft', 'hard_st'):
        raise ValueError('Separate structure forward supports soft or hard_st.')
    oa = model.order_attention()
    policy = (oa.stage, oa.gate_mode, oa.struct_gate_mode, oa.order_sampling)
    modules = [m for m in model.modules() if hasattr(m, 'set_bkd_phase_active')]
    states = [(m, getattr(m, '_bkd_phase_active', True),
               (m._bkd_step.clone() if torch.is_tensor(getattr(m, '_bkd_step', None))
                else getattr(m, '_bkd_step', None))) for m in modules]
    try:
        # Hard-ST evaluation uses the noiseless hard order, not the relaxed
        # fallback in the attention module. Loss pair weights remain soft.
        sampling = ('hard' if order_sampling == 'hard_st' and not model.training
                    else order_sampling) if structural else 'none'
        oa.set_stage_policy('structure' if structural else 'recon',
                            gate_sampling='hard',
                            order_sampling=sampling)
        for m, _, _ in states:
            m.set_bkd_phase_active(not structural)
        yield
    finally:
        oa.stage, oa.gate_mode, oa.struct_gate_mode, oa.order_sampling = policy
        for m, active, clock in states:
            m.set_bkd_phase_active(active)
            if structural and clock is not None:
                m._bkd_step = clock


@torch.no_grad()
def graph_supports(edges):
    """Detached loss supports from a hard edge matrix (source -> target).

    Returns ``(unconnected, selected, has_cycle)``, each ``(N, N)``:

    * ``unconnected[i, j]`` = 1 if no directed path joins i and j in either
      direction (full transitive closure).  Both are non-descendants of each
      other, so both directed C tests are valid and order-free.
    * ``selected[i, j]`` = 1 if the pair carries a selected edge in either
      direction (support of the anti-causal term).
    * ``has_cycle``: True if the closure has a non-zero diagonal (possible
      for directional_free draws); cycle members count as connected.
    """
    n = edges.shape[0]
    off = 1.0 - torch.eye(n, device=edges.device, dtype=edges.dtype)
    e = (edges.detach() > 0.5).to(edges.dtype) * off
    reach = e.clone()
    for _ in range(max(1, (n - 1).bit_length())):
        nxt = ((reach + reach @ reach) > 0).to(edges.dtype)
        if torch.equal(nxt, reach):
            break
        reach = nxt
    has_cycle = bool((torch.diagonal(reach) > 0).any())
    connected = ((reach + reach.T) > 0).to(edges.dtype)
    unconnected = (1.0 - connected) * off
    selected = ((e + e.T) > 0).to(edges.dtype)
    return unconnected, selected, has_cycle


def protect_weight(weight, residual):
    """Remove only the residual projection opposing the weight gradient."""
    dot = (weight * residual).sum()
    norm2 = weight.square().sum()
    if dot < 0 and norm2 > 1e-24:
        return residual - dot / norm2 * weight
    return residual
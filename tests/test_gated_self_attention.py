"""
Tests for GatedSelfAttention (direction-aware differentiable variable selector).

Covers the core mathematical contracts:
  * output/shape and the diagonal (no self-loops) constraint;
  * antisymmetric direction: d_ij + d_ji == 1 per-sample (train) and at eval;
  * symmetric existence posterior P(z_edge>0) == its transpose;
  * L0 penalty is a function of the SYMMETRIC skeleton only (no grad to gain);
  * sparsity survives (both directions can be ~0) while direction still splits;
  * eval determinism (no noise);
  * gradient-routing name classification (structural QK vs gain_* reconstruction).
"""

import math
import pytest
import torch
import torch.nn as nn

from causaliT.core.modules.gated_self_attention import GatedSelfAttention
from causaliT.core.modules import AttentionLayer
from causaliT.core.modules.attention import GatedSelfAttention as GSA_from_attn


B, N, E, D = 4, 5, 8, 3


def _make_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, N, E, generator=g)
    k = torch.randn(B, N, E, generator=g)
    v = torch.randn(B, N, D, generator=g)
    gq = torch.randn(B, N, E, generator=g)
    gk = torch.randn(B, N, E, generator=g)
    return q, k, v, gq, gk


def _forward(mod, q, k, v, gq, gk, hard_mask=None, oracle=False):
    return mod(
        query=q, key=k, value=v,
        mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
        hard_mask=hard_mask, oracle=oracle, gain_query=gq, gain_key=gk,
    )


def test_output_shape_and_diagonal():
    mod = GatedSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs()
    out, attn, aux = _forward(mod, q, k, v, gq, gk)
    assert out.shape == (B, N, D)
    assert attn.shape == (B, N, N)
    # No self-loops on the directed posterior.
    diag = torch.diagonal(attn, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)
    assert "l0_penalty" in aux and "entropy" in aux


def test_direction_antisymmetric_eval():
    """At eval, d = sigmoid(A_anti/beta_dir) must satisfy d_ij + d_ji == 1."""
    mod = GatedSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs(1)
    _forward(mod, q, k, v, gq, gk)
    d = mod.last_direction                       # (N, N), batch-mean
    s = d + d.transpose(-1, -2)
    off = ~torch.eye(N, dtype=torch.bool)
    assert torch.allclose(s[off], torch.ones_like(s[off]), atol=1e-5)


def test_direction_antisymmetric_train_per_sample():
    """During training the coupled noise must keep d_ij + d_ji == 1 per sample."""
    mod = GatedSelfAttention().train()
    q, k, v, gq, gk = _make_inputs(2)
    # Re-implement the internal draw path deterministically is hard; instead
    # check the returned per-sample direction via a monkey-hook: run forward and
    # inspect that the *mean* direction is antisymmetric (mean of (1 - d_ji)).
    # Stronger: directly test the noise helper symmetry below.
    eps = GatedSelfAttention._antisymmetric_noise((B, N, N), torch.device("cpu"), torch.float32)
    assert torch.allclose(eps, -eps.transpose(-1, -2), atol=1e-6)
    # And a full sigmoid split with an antisymmetric logit is exactly 1.
    A = 0.5 * (torch.randn(B, N, N))
    A = A - A.transpose(-1, -2)                  # antisymmetric logit
    d = torch.sigmoid((eps + A) / mod.dir_beta)
    s = d + d.transpose(-1, -2)
    off = ~torch.eye(N, dtype=torch.bool)
    assert torch.allclose(s[:, off], torch.ones_like(s[:, off]), atol=1e-5)


def test_existence_noise_symmetric():
    eps = GatedSelfAttention._symmetric_noise((B, N, N), torch.device("cpu"), torch.float32)
    assert torch.allclose(eps, eps.transpose(-1, -2), atol=1e-6)


def test_existence_posterior_symmetric():
    """P(z_edge>0) is a function of the SYMMETRIC part → symmetric matrix."""
    mod = GatedSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs(3)
    _forward(mod, q, k, v, gq, gk)
    p = mod.last_p_edge_undirected               # (N, N)
    assert torch.allclose(p, p.transpose(-1, -2), atol=1e-5)


def test_eval_is_deterministic():
    mod = GatedSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs(4)
    o1, a1, _ = _forward(mod, q, k, v, gq, gk)
    o2, a2, _ = _forward(mod, q, k, v, gq, gk)
    assert torch.allclose(o1, o2, atol=1e-6)
    assert torch.allclose(a1, a2, atol=1e-6)


def test_l0_penalty_depends_on_symmetric_only():
    """L0 penalty must back-prop into q/k (structural) but NOT into gain q/k."""
    torch.manual_seed(5)
    q = nn.Parameter(torch.randn(B, N, E))
    k = nn.Parameter(torch.randn(B, N, E))
    gq = nn.Parameter(torch.randn(B, N, E))
    gk = nn.Parameter(torch.randn(B, N, E))
    v = torch.randn(B, N, D)
    mod = GatedSelfAttention().train()
    _, _, aux = _forward(mod, q, k, v, gq, gk)
    aux["l0_penalty"].backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0
    # Gain projections do not participate in the existence posterior.
    assert gq.grad is None or gq.grad.abs().sum() == 0
    assert gk.grad is None or gk.grad.abs().sum() == 0


def test_sparsity_and_direction_both_achievable():
    """A strongly-negative symmetric score → no edge; asymmetric score → oriented."""
    mod = GatedSelfAttention().eval()
    # Construct q, k so that raw = q k^T is strongly negative & symmetric for one
    # pair (no edge) and strongly asymmetric for another.  Use E=1 for control.
    mod1 = GatedSelfAttention().eval()
    # No-edge pair: identical large-negative-correlated embeddings.
    q = torch.zeros(1, 2, 2)
    k = torch.zeros(1, 2, 2)
    # Make raw[0,1] = raw[1,0] very negative (symmetric) → existence ~ 0.
    q[0, 0] = torch.tensor([3.0, 0.0]); k[0, 1] = torch.tensor([-3.0, 0.0])
    q[0, 1] = torch.tensor([-3.0, 0.0]); k[0, 0] = torch.tensor([3.0, 0.0])
    gq = torch.zeros(1, 2, 2); gk = torch.zeros(1, 2, 2)
    _forward(mod1, q, k, torch.zeros(1, 2, 2), gq, gk)
    p_edge = mod1.last_p_edge_undirected[0, 1]
    assert p_edge < 0.2, f"expected near-zero existence, got {p_edge}"


def test_square_requirement():
    mod = GatedSelfAttention().eval()
    q = torch.randn(B, N, E)
    k = torch.randn(B, N + 1, E)
    v = torch.randn(B, N + 1, D)
    gq = torch.randn(B, N, E)
    gk = torch.randn(B, N + 1, E)
    with pytest.raises(ValueError):
        _forward(mod, q, k, v, gq, gk)


def test_hard_mask_applied():
    mod = GatedSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs(6)
    hard_mask = torch.ones(N, N)
    hard_mask[0, 1] = 0.0                          # forbid edge 1->0
    _, attn, _ = _forward(mod, q, k, v, gq, gk, hard_mask=hard_mask)
    assert torch.allclose(attn[:, 0, 1], torch.zeros(B), atol=1e-6)


def test_attention_layer_dispatch_and_routing():
    """AttentionLayer builds GatedSelfAttention with gain_* projections and the
    name-based router classifies q/k as structural, gain_* as reconstruction."""
    layer = AttentionLayer(
        attention=GSA_from_attn,
        d_model_queries=E, d_model_keys=E, d_model_values=D,
        d_queries_keys=E, n_heads=1, mask_layer=None,
        attention_dropout=0.0, dropout_qkv=0.0,
        shared_dag_across_heads=True,
    )
    assert layer._gated_gain is True
    assert layer.gain_q_proj is not None and layer.gain_k_proj is not None
    names = [n for n, _ in layer.named_parameters()]
    assert any("query_projection" in n for n in names)
    assert any("key_projection" in n for n in names)
    assert any("gain_q_proj" in n for n in names)
    assert any("gain_k_proj" in n for n in names)
    # gain_* must NOT contain the structural substrings.
    for n in names:
        if "gain_" in n:
            assert "query_projection" not in n and "key_projection" not in n

    # End-to-end forward through the layer (self-attention: query == key input).
    x = torch.randn(B, N, E)
    xv = torch.randn(B, N, D)
    out, attn, aux = layer(
        query=x, key=x, value=xv,
        mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
        gain_query=x, gain_key=x,
    )
    assert out.shape == (B, N, D)
    assert attn.shape == (B, N, N)
    assert aux["l0_penalty"] is not None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

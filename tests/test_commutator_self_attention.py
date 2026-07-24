"""
Tests for CommutatorSelfAttention (unified-gate, direction-aware selector).

CommutatorSelfAttention keeps the antisymmetric direction machinery of
GatedSelfAttention but UNIFIES the existence gate with GatedCrossAttention: a
plain (ASYMMETRIC) Hard-Concrete L0 gate applied directly to the raw alignment
score ``raw = Q Kᵀ``.  This is the key difference from GatedSelfAttention, whose
existence gate is symmetric (on the symmetric Toeplitz part).

Covers:
  * output / attn shape and the diagonal (no self-loops) constraint;
  * antisymmetric direction: d_ij + d_ji == 1 at eval (coupled noise in train);
  * existence gate is ASYMMETRIC (P(z>0) != its transpose in general);
  * L0 penalty counts DIRECTED edges (all off-diagonal), matching GCA;
  * ``use_gain=False`` bypasses the gain: A == structure;
  * eval determinism (no noise);
  * hard_mask zeros forbidden edges and the L0 penalty;
  * oracle mode uses the hard_mask as the structure gate;
  * gradient routing name classification (structural QK vs gain_* reconstruction).
"""

import math
import pytest
import torch

from causaliT.core.modules.commutator_self_attention import CommutatorSelfAttention
from causaliT.core.modules import CommutatorSelfAttention as CSA_from_pkg
from causaliT.core.modules.attention import AttentionLayer


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


# ---------------------------------------------------------------------------
# Registration / import
# ---------------------------------------------------------------------------


def test_package_export_is_same_class():
    assert CSA_from_pkg is CommutatorSelfAttention


# ---------------------------------------------------------------------------
# Shapes / diagonal
# ---------------------------------------------------------------------------


def test_output_shape_and_diagonal():
    mod = CommutatorSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs()
    out, attn, aux = _forward(mod, q, k, v, gq, gk)
    assert out.shape == (B, N, D)
    assert attn.shape == (B, N, N)
    diag = torch.diagonal(attn, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)
    assert "l0_penalty" in aux and "entropy" in aux


def test_multihead_value():
    """4-D value (B, N, H, d) should aggregate to (B, N, H, d)."""
    mod = CommutatorSelfAttention().eval()
    q, k, _, gq, gk = _make_inputs()
    H = 2
    v = torch.randn(B, N, H, D)
    out, attn, _ = _forward(mod, q, k, v, gq, gk)
    assert out.shape == (B, N, H, D)
    assert attn.shape == (B, N, N)


# ---------------------------------------------------------------------------
# Direction (antisymmetric) — d_ij + d_ji == 1
# ---------------------------------------------------------------------------


def test_direction_antisymmetric_eval():
    mod = CommutatorSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs(1)
    _forward(mod, q, k, v, gq, gk)
    d = mod.last_direction                        # (N, N), batch-mean
    s = d + d.transpose(-1, -2)
    off = ~torch.eye(N, dtype=torch.bool)
    assert torch.allclose(s[off], torch.ones_like(s[off]), atol=1e-5)


def test_antisymmetric_noise_helper():
    eps = CommutatorSelfAttention._antisymmetric_noise(
        (B, N, N), torch.device("cpu"), torch.float32
    )
    assert torch.allclose(eps, -eps.transpose(-1, -2), atol=1e-6)
    # zero diagonal
    diag = torch.diagonal(eps, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)


# ---------------------------------------------------------------------------
# Existence gate is ASYMMETRIC (this is the CSA-vs-GSA distinction)
# ---------------------------------------------------------------------------


def test_existence_gate_is_asymmetric():
    """P(z_edge>0) = sigmoid(raw - offset) uses the RAW (asymmetric) score, so
    it is NOT symmetric in general (unlike GatedSelfAttention)."""
    mod = CommutatorSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs(3)
    _forward(mod, q, k, v, gq, gk)
    p = mod.last_p_edge_undirected                # (N, N), existence posterior
    off = ~torch.eye(N, dtype=torch.bool)
    # Assert it is NOT (numerically) symmetric — the raw scores differ i<->j.
    assert not torch.allclose(p[off], p.transpose(-1, -2)[off], atol=1e-3)


# ---------------------------------------------------------------------------
# L0 penalty counts directed edges
# ---------------------------------------------------------------------------


def test_l0_penalty_counts_directed_edges():
    mod = CommutatorSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs(4)
    _, _, aux = _forward(mod, q, k, v, gq, gk)
    l0 = aux["l0_penalty"]
    # Scalar, non-negative, and at most the number of directed off-diagonal edges.
    assert l0.dim() == 0
    assert l0.item() >= 0.0
    assert l0.item() <= N * (N - 1) + 1e-4


def test_hard_mask_zeros_edges_and_penalty():
    mod = CommutatorSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs(5)
    # Allow nothing → all edges zero, L0 penalty zero.
    hard_mask = torch.zeros(N, N)
    out, attn, aux = _forward(mod, q, k, v, gq, gk, hard_mask=hard_mask)
    assert torch.allclose(attn, torch.zeros_like(attn), atol=1e-6)
    assert aux["l0_penalty"].item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# use_gain=False bypasses the gain
# ---------------------------------------------------------------------------


def test_use_gain_false_bypasses_gain():
    mod = CommutatorSelfAttention(use_gain=False).eval()
    q, k, v, gq, gk = _make_inputs(6)
    # No gain q/k needed when use_gain=False.
    out, attn, _ = mod(
        query=q, key=k, value=v,
        mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
    )
    assert out.shape == (B, N, D)
    assert mod.last_gain is None


def test_use_gain_true_requires_gain_inputs():
    mod = CommutatorSelfAttention(use_gain=True).eval()
    q, k, v, _, _ = _make_inputs(7)
    with pytest.raises(ValueError):
        mod(query=q, key=k, value=v,
            mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False)


# ---------------------------------------------------------------------------
# Eval determinism
# ---------------------------------------------------------------------------


def test_eval_deterministic():
    mod = CommutatorSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs(8)
    _, a1, _ = _forward(mod, q, k, v, gq, gk)
    _, a2, _ = _forward(mod, q, k, v, gq, gk)
    assert torch.allclose(a1, a2, atol=1e-6)


# ---------------------------------------------------------------------------
# Oracle mode
# ---------------------------------------------------------------------------


def test_oracle_uses_hard_mask_as_gate():
    mod = CommutatorSelfAttention(use_gain=False).eval()
    q, k, v, _, _ = _make_inputs(9)
    hard_mask = (1.0 - torch.eye(N))              # allow all off-diagonal
    out, attn, _ = mod(
        query=q, key=k, value=v,
        mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
        hard_mask=hard_mask, oracle=True,
    )
    # With use_gain=False and oracle, structure == hard_mask → attn == mask.
    expected = hard_mask.unsqueeze(0).expand(B, N, N)
    assert torch.allclose(attn, expected, atol=1e-6)


def test_causal_mask_not_supported():
    mod = CommutatorSelfAttention().eval()
    q, k, v, gq, gk = _make_inputs(10)
    with pytest.raises(NotImplementedError):
        mod(query=q, key=k, value=v,
            mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=True,
            gain_query=gq, gain_key=gk)


# ---------------------------------------------------------------------------
# Backward smoke
# ---------------------------------------------------------------------------


def test_backward_reaches_inputs():
    mod = CommutatorSelfAttention().train()
    q, k, v, gq, gk = _make_inputs(11)
    for t in (q, k, v, gq, gk):
        t.requires_grad_(True)
    out, _, aux = _forward(mod, q, k, v, gq, gk)
    (out.sum() + aux["l0_penalty"]).backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert gq.grad is not None and gq.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Wiring through AttentionLayer
# ---------------------------------------------------------------------------


def test_attention_layer_wiring():
    layer = AttentionLayer(
        attention=CommutatorSelfAttention,
        d_model_queries=E,
        d_model_keys=E,
        d_model_values=D,
        d_queries_keys=E,
        n_heads=1,
        mask_layer=None,
        attention_dropout=0.0,
        dropout_qkv=0.0,
        shared_dag_across_heads=True,
    ).eval()
    q = torch.randn(B, N, E)
    k = torch.randn(B, N, E)
    v = torch.randn(B, N, D)
    out, attn, aux = layer(
        query=q, key=k, value=v,
        mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
    )
    assert out.shape == (B, N, D)
    assert attn.shape == (B, N, N)
    assert "l0_penalty" in aux


# ---------------------------------------------------------------------------
# direction_mode="skew_query" — learnable so(d) commutator on the QUERY alone
# ---------------------------------------------------------------------------


def _make_skew_mod(seed=0, **kw):
    torch.manual_seed(seed)
    return CommutatorSelfAttention(
        direction_mode="skew_query", direction_dim=E, **kw
    )


def test_skew_query_requires_direction_dim():
    with pytest.raises(ValueError):
        CommutatorSelfAttention(direction_mode="skew_query")


def test_skew_query_invalid_direction_mode():
    with pytest.raises(ValueError):
        CommutatorSelfAttention(direction_mode="bogus")


def test_skew_query_output_shape_and_diagonal():
    mod = _make_skew_mod().eval()
    q, k, v, gq, gk = _make_inputs()
    out, attn, aux = _forward(mod, q, k, v, gq, gk)
    assert out.shape == (B, N, D)
    assert attn.shape == (B, N, N)
    diag = torch.diagonal(attn, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)


def test_skew_query_direction_antisymmetric_eval():
    """A_anti_ij = q_iᵀ Ω q_j with Ω antisymmetric ⇒ d_ij + d_ji == 1."""
    # Give the skew generators a non-trivial init so A_anti != 0.
    mod = _make_skew_mod(1)
    with torch.no_grad():
        mod.direction_proj_a.weight.normal_(std=1.0)
        mod.direction_proj_b.weight.normal_(std=1.0)
    mod.eval()
    q, k, v, gq, gk = _make_inputs(1)
    _forward(mod, q, k, v, gq, gk)
    d = mod.last_direction
    s = d + d.transpose(-1, -2)
    off = ~torch.eye(N, dtype=torch.bool)
    assert torch.allclose(s[off], torch.ones_like(s[off]), atol=1e-5)


def test_skew_query_identical_queries_give_undecided_direction():
    """q_i == q_j ⇒ A_anti_ij = 0 ⇒ direction → 0.5 (undecided)."""
    mod = _make_skew_mod(2)
    with torch.no_grad():
        mod.direction_proj_a.weight.normal_(std=1.0)
        mod.direction_proj_b.weight.normal_(std=1.0)
    mod.eval()
    # All query rows identical.
    q = torch.randn(B, 1, E).expand(B, N, E).contiguous()
    k = torch.randn(B, N, E)
    v = torch.randn(B, N, D)
    gq = torch.randn(B, N, E)
    gk = torch.randn(B, N, E)
    _forward(mod, q, k, v, gq, gk)
    d = mod.last_direction
    off = ~torch.eye(N, dtype=torch.bool)
    assert torch.allclose(d[off], torch.full_like(d[off], 0.5), atol=1e-5)


def test_skew_query_direction_decoupled_from_key():
    """The skew-query direction depends ONLY on the query: changing the key
    must not change the direction gate (it does change the existence gate)."""
    mod = _make_skew_mod(3)
    with torch.no_grad():
        mod.direction_proj_a.weight.normal_(std=1.0)
        mod.direction_proj_b.weight.normal_(std=1.0)
    mod.eval()
    q, k, v, gq, gk = _make_inputs(3)
    _forward(mod, q, k, v, gq, gk)
    d1 = mod.last_direction.clone()
    k2 = torch.randn(B, N, E)                      # different key
    _forward(mod, q, k2, v, gq, gk)
    d2 = mod.last_direction.clone()
    assert torch.allclose(d1, d2, atol=1e-6)


def test_skew_query_backward_reaches_generators():
    mod = _make_skew_mod(4).train()
    q, k, v, gq, gk = _make_inputs(4)
    out, _, aux = _forward(mod, q, k, v, gq, gk)
    (out.sum() + aux["l0_penalty"]).backward()
    assert mod.direction_proj_a.weight.grad is not None
    assert mod.direction_proj_a.weight.grad.abs().sum() > 0
    assert mod.direction_proj_b.weight.grad is not None


def test_skew_query_direction_rank():
    mod = CommutatorSelfAttention(
        direction_mode="skew_query", direction_dim=E, direction_rank=2
    ).eval()
    assert mod.direction_proj_a.weight.shape == (2, E)
    q, k, v, gq, gk = _make_inputs(5)
    out, attn, _ = _forward(mod, q, k, v, gq, gk)
    assert out.shape == (B, N, D)


def test_skew_query_attention_layer_wiring():
    layer = AttentionLayer(
        attention=CommutatorSelfAttention,
        d_model_queries=E,
        d_model_keys=E,
        d_model_values=D,
        d_queries_keys=E,
        n_heads=1,
        mask_layer=None,
        attention_dropout=0.0,
        dropout_qkv=0.0,
        shared_dag_across_heads=True,
        direction_mode="skew_query",
    ).eval()
    # The direction generator was built with direction_dim = d_qk * n_heads_struct.
    assert layer.inner_attention.direction_mode == "skew_query"
    assert layer.inner_attention.direction_proj_a.weight.shape[1] == E
    q = torch.randn(B, N, E)
    k = torch.randn(B, N, E)
    v = torch.randn(B, N, D)
    out, attn, aux = layer(
        query=q, key=k, value=v,
        mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
    )
    assert out.shape == (B, N, D)
    assert attn.shape == (B, N, N)


def test_skew_query_gradient_routing_classifies_generators_as_structural():
    """direction_proj_a/b must be routed to the STRUCTURAL group."""
    from causaliT.training.gradient_routing import classify_parameters

    layer = AttentionLayer(
        attention=CommutatorSelfAttention,
        d_model_queries=E,
        d_model_keys=E,
        d_model_values=D,
        d_queries_keys=E,
        n_heads=1,
        mask_layer=None,
        attention_dropout=0.0,
        dropout_qkv=0.0,
        shared_dag_across_heads=True,
        direction_mode="skew_query",
    )
    structural, reconstruction = classify_parameters(layer)
    struct_ids = {id(p) for p in structural}
    a = layer.inner_attention.direction_proj_a.weight
    b = layer.inner_attention.direction_proj_b.weight
    assert id(a) in struct_ids
    assert id(b) in struct_ids


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])



"""Geometric transitive correction (grandparent suppression).

Covers
------
1. ``causaliT/utils/query_geometry.py`` — the trigger algebra, the init-safety
   property that replaces a warmup schedule, and the exactness of the component
   removal on an orthonormal frame.
2. The attention modules — the correction is applied where the probe applied it
   (on the unit query, before the norm budget) and is a strict no-op when off.
3. ``AttentionSelectorLayer`` — the two-pass orchestration in BOTH topologies
   (homogeneous single square block, split cross + self) and the fail-loud guard
   on a non-orthonormal key frame.

Reference numbers come from ``scripts/_probe_transitive_correction.py`` on
``baseline_equal_ds_centroid_init_9494866`` (see
``experiments/6_INVESTIGATIONS/HOMOGENEOUS/TRANSITIVE_CORRECTION.md``).
"""

import pytest
import torch

from causaliT.utils.query_geometry import (
    assert_orthonormal_frame,
    correct_query,
    frame_offdiag,
    key_frame,
    mediation_mass,
    transitive_weights,
)


# =============================================================================
# Fixtures: a 3-node chain  j -> k -> i  with a leaked indirect edge j -> i
# =============================================================================

@pytest.fixture
def chain_pi():
    """``pi[i, j] = P(j -> i)`` for the chain ``0 -> 1 -> 2`` (+ leak 0 -> 2).

    Values taken from the real X5 row: the two true edges sit at 0.80 / 0.65 and
    the indirect one leaked to 0.44.
    """
    pi = torch.zeros(3, 3)
    pi[1, 0] = 0.80        # 0 -> 1   true
    pi[2, 1] = 0.65        # 1 -> 2   true
    pi[2, 0] = 0.44        # 0 -> 2   INDIRECT (to be suppressed)
    return pi


# =============================================================================
# 1. Trigger algebra
# =============================================================================

def test_mediation_mass_min_keeps_the_weakest_link(chain_pi):
    """Goedel t-norm: the mediated pair keeps the confidence of its weakest edge."""
    m = mediation_mass(chain_pi, tnorm="min")
    assert m[2, 0] == pytest.approx(min(0.65, 0.80), abs=1e-6)   # 0.65
    # No mediator exists for the two true edges of a chain.
    assert m[1, 0] == pytest.approx(0.0, abs=1e-6)
    assert m[2, 1] == pytest.approx(0.0, abs=1e-6)


def test_mediation_mass_prod_deflates(chain_pi):
    """The literal product is smaller than either edge -> a much weaker trigger."""
    m_prod = mediation_mass(chain_pi, tnorm="prod")
    m_min = mediation_mass(chain_pi, tnorm="min")
    assert m_prod[2, 0] == pytest.approx(0.65 * 0.80, abs=1e-6)   # 0.52
    assert m_prod[2, 0] < m_min[2, 0]
    # ... and the margin over the direct edge is what the correction acts on:
    assert (m_min[2, 0] - chain_pi[2, 0]) > 2 * (m_prod[2, 0] - chain_pi[2, 0])


def test_mediation_mass_has_zero_diagonal_and_rejects_non_square(chain_pi):
    m = mediation_mass(chain_pi)
    assert torch.allclose(m.diagonal(), torch.zeros(3))
    with pytest.raises(ValueError, match="square"):
        mediation_mass(torch.rand(3, 4))


def test_mediation_mass_batched_matches_per_sample(chain_pi):
    batch = torch.stack([chain_pi, chain_pi.roll(1, dims=0)])
    m_batch = mediation_mass(batch)
    for b in range(2):
        assert torch.allclose(m_batch[b], mediation_mass(batch[b]), atol=1e-6)


def test_transitive_weights_target_only_the_indirect_edge(chain_pi):
    w = transitive_weights(chain_pi, alpha=1.0, tnorm="min", margin=True,
                           symmetric=False)
    assert w[2, 0] == pytest.approx(0.65 - 0.44, abs=1e-6)   # the margin
    # The true edges are untouched: no mediator, hence no weight.
    assert w[1, 0] == pytest.approx(0.0, abs=1e-6)
    assert w[2, 1] == pytest.approx(0.0, abs=1e-6)
    assert w.max() <= 1.0


def test_transitive_weights_alpha_scales_and_detaches(chain_pi):
    pi = chain_pi.clone().requires_grad_(True)
    w_half = transitive_weights(pi, alpha=0.5)
    w_full = transitive_weights(pi, alpha=1.0)
    assert torch.allclose(w_half, 0.5 * w_full, atol=1e-6)
    # A differentiable weight would let the model delete a real edge elsewhere
    # just to raise a logit -> the weights MUST be detached.
    assert not w_half.requires_grad


def test_transitive_weights_margin_is_silent_at_the_all_on_init():
    """Init safety (replaces a warmup schedule).

    At the centroid initialisation every edge sits at the same value, so no
    mediated path can beat a direct edge and the correction must stay silent.
    Without the margin gate the same state yields a LARGE weight.
    """
    pi = torch.full((6, 6), 0.418)
    pi.fill_diagonal_(0.0)
    w_margin = transitive_weights(pi, alpha=1.0, tnorm="min", margin=True)
    w_plain = transitive_weights(pi, alpha=1.0, tnorm="min", margin=False)
    assert float(w_margin.max()) < 0.01          # probe measured 0.0074
    assert float(w_plain.max()) > 0.4            # probe measured 0.4304


def test_transitive_weights_symmetric_span_limits_symmetrisation():
    """Split mode: only the square self block may be symmetrised."""
    pi = torch.zeros(4, 4)
    pi[3, 2], pi[2, 1], pi[3, 1] = 0.8, 0.7, 0.4      # mediated inside [1:4]
    full = transitive_weights(pi, alpha=1.0, symmetric=True)
    span = transitive_weights(pi, alpha=1.0, symmetric=True, symmetric_span=(2, 4))
    none = transitive_weights(pi, alpha=1.0, symmetric=False)
    # Inside the span the transpose is mirrored; outside it is left alone.
    assert span[2, 3] == pytest.approx(float(none[3, 2]), abs=1e-6)
    assert span[1, 3] == pytest.approx(float(none[1, 3]), abs=1e-6)
    assert full[1, 3] == pytest.approx(float(none[3, 1]), abs=1e-6)


# =============================================================================
# 2. Geometry: the removal is exact on an orthonormal frame
# =============================================================================

def _orthonormal_keys(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, n, generator=g))
    return q.T.contiguous()          # (n, d), mutually orthonormal rows


def test_correct_query_moves_only_the_targeted_coordinate():
    keys = _orthonormal_keys(5, 16)
    query = torch.randn(5, 16)
    w = torch.zeros(5, 5)
    w[4, 0] = 1.0                                     # kill edge 0 -> 4 only
    u0 = torch.nn.functional.normalize(query, dim=-1)
    c0 = u0 @ key_frame(keys).T
    c1 = correct_query(query, keys, w, delta=0.5) @ key_frame(keys).T
    # The targeted coordinate lands on -delta ...
    assert float(c1[4, 0]) == pytest.approx(-0.5, abs=1e-5)
    # ... and every other coordinate of that row is untouched (exactness).
    for j in range(1, 5):
        assert float(c1[4, j]) == pytest.approx(float(c0[4, j]), abs=1e-6)
    # Other rows are untouched entirely.
    assert torch.allclose(c1[:4], c0[:4], atol=1e-6)


def test_correct_query_delta_zero_is_the_plain_projection():
    keys = _orthonormal_keys(4, 12)
    query = torch.randn(4, 12)
    w = torch.zeros(4, 4)
    w[3, 1] = 1.0
    c = correct_query(query, keys, w, delta=0.0) @ key_frame(keys).T
    # Projection reaches 0 -> exactly the P(z>0) = 0.5 THRESHOLD, which is why
    # the default instrument pushes past zero instead.
    assert float(c[3, 1]) == pytest.approx(0.0, abs=1e-6)


def test_correct_query_zero_weights_is_a_noop_up_to_normalisation():
    keys = _orthonormal_keys(4, 12)
    query = torch.randn(4, 12)
    out = correct_query(query, keys, torch.zeros(4, 4), delta=0.5)
    assert torch.allclose(out, torch.nn.functional.normalize(query, dim=-1),
                          atol=1e-6)


def test_correct_query_frees_budget_for_the_true_parents():
    """The renormalisation reallocation: suppressing one edge RAISES the others."""
    keys = _orthonormal_keys(4, 8)
    kh = key_frame(keys)
    query = (0.6 * kh[0] + 0.6 * kh[1] + 0.5 * kh[2]).unsqueeze(0)
    w = torch.zeros(1, 4)
    w[0, 0] = 1.0
    c0 = torch.nn.functional.normalize(query, dim=-1) @ kh.T
    corrected = correct_query(query, keys, w, delta=0.5)
    c1 = torch.nn.functional.normalize(corrected, dim=-1) @ kh.T
    assert float(c1[0, 0]) < 0.0                       # suppressed
    assert float(c1[0, 1]) > float(c0[0, 1])           # true parents go UP
    assert float(c1[0, 2]) > float(c0[0, 2])


# =============================================================================
# 3. The frame guard
# =============================================================================

def test_frame_guard_accepts_an_orthonormal_frame():
    keys = _orthonormal_keys(6, 20)
    assert float(frame_offdiag(key_frame(keys))) < 1e-5
    assert assert_orthonormal_frame(keys) < 1e-5


def test_frame_guard_raises_on_a_correlated_frame():
    keys = _orthonormal_keys(4, 10)
    keys[1] = keys[0] + 0.05 * keys[1]        # break mutual orthogonality
    with pytest.raises(ValueError, match="ORTHONORMAL key frame"):
        assert_orthonormal_frame(keys)


# =============================================================================
# 4. delta lives in COSINE space
# =============================================================================

def test_delta_is_a_cosine_and_stays_reachable():
    """Guard against rescaling ``delta`` by a LOGIT offset.

    The push target is a query coordinate, so it must stay inside [-1, 1].  The
    default Hard-Concrete stretch would suggest a +1.6 logit shift; interpreting
    that as a cosine drives the query PAST the antipode and flips the sign of the
    other coordinates, which is exactly the bug this test pins.
    """
    keys = _orthonormal_keys(3, 8)
    kh = key_frame(keys)
    query = (0.7 * kh[0] + 0.7 * kh[1]).unsqueeze(0)
    w = torch.zeros(1, 3)
    w[0, 0] = 1.0

    sane = correct_query(query, keys, w, delta=0.5)
    c_sane = torch.nn.functional.normalize(sane, dim=-1) @ kh.T
    assert -1.0 <= float(c_sane[0, 0]) <= 1.0
    assert float(c_sane[0, 1]) > 0.0          # the surviving parent stays positive

    absurd = correct_query(query, keys, w, delta=2.1)   # a logit masquerading as a cosine
    c_absurd = torch.nn.functional.normalize(absurd, dim=-1) @ kh.T
    # Overshoot: the removed axis now DOMINATES the query (|c| > the true parent),
    # i.e. the query points away from every key instead of at the true parents.
    assert abs(float(c_absurd[0, 0])) > abs(float(c_absurd[0, 1]))


# =============================================================================
# 5. Layer orchestration — BOTH topologies
# =============================================================================
# ``AttentionSelectorLayer`` drives the two passes differently per topology:
#   * homogeneous: the single square (N, N) block probes its own posterior
#     inside ``AttentionLayer.forward`` (no "W" key in the cfg);
#   * split: the LAYER probes BOTH blocks (``transitive_probe``), fuses the
#     (L_X, L_S+L_X) posterior, pads it square with zero S-rows, computes the
#     weights once (``symmetric_span`` limits symmetrisation to the X-X block)
#     and hands each block its slice via ``transitive_cfg["W"]``.
#
# Column convention (production): value at column 0, variable-ID at column 1.

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer

_L_D_MODEL = 16
_L_D_QK = 16            # == d_model, required by the removed projections
_L_D_FF = 32
_L_S = 3
_L_X = 4
_L_N = _L_S + _L_X
_L_VOCAB_S = _L_S + 1
_L_VOCAB_X = _L_X + 1
_L_VALUE_COL = 0
_L_VAR_COL = 1


def _l_embed_cfg(vocab: int) -> dict:
    return {
        "setting": {"d_model": _L_D_MODEL},
        "modules": [
            {
                "idx": _L_VALUE_COL,
                "embed": "linear",
                "label": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": _L_D_MODEL},
            },
            {
                "idx": _L_VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": _L_D_MODEL},
            },
        ],
    }


def _l_var_input(num_vars: int, batch: int = 2) -> torch.Tensor:
    x = torch.zeros(batch, num_vars, 2)
    x[:, :, _L_VALUE_COL] = torch.randn(batch, num_vars)
    x[:, :, _L_VAR_COL] = (
        torch.arange(1, num_vars + 1).float().unsqueeze(0).repeat(batch, 1)
    )
    return x


def _l_make_layer(
    transitive_correction: bool = True,
    homogeneous_nodes: bool = False,
    seed: int = 0,
    **overrides,
) -> AttentionSelectorLayer:
    """Split-mode layer on the orthonormal fixed frame (the correction's
    precondition), projections removed, shared free query/key — the exact
    configuration of the ``nl_value_emb_larger_mlp_noparent`` arm."""
    torch.manual_seed(seed)
    kwargs = dict(
        model="test_transitive",
        ds_embed_S=_l_embed_cfg(_L_VOCAB_S),
        ds_embed_X=_l_embed_cfg(_L_VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="summation",
        attention_type="GatedCrossAttention",
        self_attention_type="GatedSelfAttention",
        homogeneous_nodes=homogeneous_nodes,
        n_heads=1,
        dropout_emb=0.0,
        dropout_attn_out=0.0,
        dropout_ff=0.0,
        dropout_qkv=0.0,
        attention_dropout=0.0,
        activation="relu",
        norm="layer",
        use_final_norm=False,
        device="cpu",
        out_dim=1,
        d_ff=_L_D_FF,
        d_model=_L_D_MODEL,
        d_qk=_L_D_QK,
        S_seq_len=_L_S,
        X_seq_len=_L_X,
        struct_embedding_type="orthogonal_fixed",
        orthogonal_fixed_frame_type="random",
        remove_query_projection=True,
        remove_key_projection=True,
        shared_query=True,
        shared_key=True,
        normalize_query=True,
        shared_dag_across_heads=True,
        init_edge_offset=1.1,
        use_gain=False,
        transitive_correction=transitive_correction,
    )
    if homogeneous_nodes:   # a single block forbids the sharing flags
        kwargs["shared_query"] = False
        kwargs["shared_key"] = False
    kwargs.update(overrides)
    return AttentionSelectorLayer(**kwargs)


def _l_inputs():
    source = _l_var_input(_L_S)
    x_actual = _l_var_input(_L_X)
    x_blanked = x_actual.clone()
    x_blanked[:, :, _L_VALUE_COL] = 0.0
    s_blanked = source.clone()
    s_blanked[:, :, _L_VALUE_COL] = 0.0
    return source, x_blanked, x_actual, s_blanked


def _l_forward(model):
    source, x_blanked, x_actual, s_blanked = _l_inputs()
    return model.forward_with_actual(
        source, x_blanked, x_actual, s_blanked=s_blanked
    )


def test_layer_split_mode_runs_and_scatters_weights():
    """Regression: split + transitive_correction used to raise at the ctor."""
    m = _l_make_layer(transitive_correction=True, homogeneous_nodes=False)
    m.eval()
    pred, attn, _ = _l_forward(m)
    assert pred.shape == (2, _L_X, 1)
    assert attn.shape == (2, _L_X, _L_N)
    # The layer fused the per-block slices; each wrapper holds its own.
    assert m.last_transitive_W.shape == (_L_X, _L_N)
    assert m.attention.last_transitive_W.shape == (_L_X, _L_S)
    assert m.self_attention.last_transitive_W.shape == (_L_X, _L_X)


def test_layer_split_weights_match_the_padded_fused_posterior():
    """The layer's W equals the reference computation on the re-fused graph."""
    m = _l_make_layer(transitive_correction=True)
    m.eval()
    _l_forward(m)

    source, x_blanked, x_actual, _ = _l_inputs()
    # Re-derive the two posteriors EXACTLY as the layer's probes do (Identity
    # projections, dropout-free, value-independent fixed frame).
    q = m.query_embed_X(x_blanked) if m.free_query_embedding else None
    if q is None:
        q = m.orth_embed_X(x_blanked)
    k_s = m.orth_embed_S(source)
    k_x = m.orth_embed_X(x_actual)
    pi_cross = m.attention.inner_attention.structure_posterior(q, k_s, m.cross_mask)
    pi_self = m.self_attention.inner_attention.structure_posterior(q, k_x, m.self_mask)

    pi_sq = torch.zeros(_L_N, _L_N)
    pi_sq[_L_S:, :] = torch.cat([pi_cross, pi_self], dim=-1)
    w_ref = transitive_weights(
        pi_sq, alpha=0.5, tnorm="min", margin=True,
        symmetric=True, symmetric_span=(_L_S, _L_N),
    )[_L_S:, :]
    # Each wrapper gates its slice by its own hard mask.
    w_ref = w_ref * torch.cat([m.cross_mask, m.self_mask], dim=-1)

    assert torch.allclose(m.last_transitive_W, w_ref, atol=1e-6)


def test_layer_split_init_silence():
    """The margin gate keeps the correction silent at the all-on init."""
    m = _l_make_layer(transitive_correction=True)
    m.eval()
    _l_forward(m)
    assert float(m.last_transitive_W.max()) < 0.01    # probe measured 0.0074


def test_layer_split_correction_is_a_noop_when_disabled():
    """transitive_correction=False: no probe, no stash, bit-identical output."""
    m_off = _l_make_layer(transitive_correction=False)
    m_off.eval()
    pred_off, attn_off, _ = _l_forward(m_off)
    assert m_off.last_transitive_W is None
    assert m_off.attention.last_transitive_W is None

    # At init the margin gate silences the correction, so an identically-seeded
    # correction-ON model produces (numerically) the same output.
    m_on = _l_make_layer(transitive_correction=True)
    m_on.eval()
    pred_on, attn_on, _ = _l_forward(m_on)
    assert torch.allclose(pred_on, pred_off, atol=1e-5)
    assert torch.allclose(attn_on, attn_off, atol=1e-5)


def test_layer_split_oracle_skips_the_correction():
    """oracle=True bypasses the probe entirely (no W is computed)."""
    m = _l_make_layer(transitive_correction=True)
    m.eval()
    source, x_blanked, x_actual, s_blanked = _l_inputs()
    m.forward_with_actual(
        source, x_blanked, x_actual, oracle=True, s_blanked=s_blanked
    )
    assert m.last_transitive_W is None


def test_layer_cross_only_topology_is_refused():
    """Cross-only has no X->X self block: no mediator axis exists at all."""
    with pytest.raises(ValueError, match="requires self_attention_type"):
        _l_make_layer(transitive_correction=True, self_attention_type=None)


def test_layer_split_non_orthonormal_frame_fails_loud():
    """The guard must fire on the COMBINED [S ; X] frame, not per block."""
    m = _l_make_layer(
        transitive_correction=True, struct_embedding_type="standard_learnable"
    )
    m.eval()
    with pytest.raises(ValueError, match="ORTHONORMAL key frame"):
        _l_forward(m)


def test_layer_split_suppresses_the_mediated_cross_edge():
    """End-to-end: a crafted ``S1 -> X1 -> X2`` chain with a leaked ``S1 -> X2``.

    The free query table is written directly: X1's query IS the S1 key (so
    ``Pi(X1<-S1)`` is high), X2's query mixes the S1 and X1 keys (direct edge
    PLUS a mediated path).  The margin trigger must fire on the cross entry
    ``(X2, S1)`` and the correction must push that posterior DOWN while the
    true mediator edge ``X2 <- X1`` goes UP (the renormalisation reallocation).
    """
    m_on = _l_make_layer(
        transitive_correction=True, query_fanin_scale=4.0,
        free_query_embedding=True,
    )
    m_off = _l_make_layer(
        transitive_correction=False, query_fanin_scale=4.0,
        free_query_embedding=True,
    )
    assert m_on.free_query_embedding and m_on.query_embed_X is not None

    with torch.no_grad():
        for m in (m_on, m_off):
            k_s1 = m.orth_embed_S.frame[0]              # S1 key (unit row)
            k_x1 = m.orth_embed_X.frame[0]              # X1 key
            w = m.query_embed_X.embedding.weight        # row 0 = padding
            w[1] = k_s1                                 # X1 asks for S1 only
            u = 0.30 * k_s1 + 1.2 * k_x1                # X2: direct + mediated
            w[2] = u / u.norm()

    m_on.eval()
    m_off.eval()
    _l_forward(m_on)
    _l_forward(m_off)

    # The trigger fired on the mediated cross edge (row X2=1, col S1=0).
    w_cross = m_on.last_transitive_W[1, 0]
    assert float(w_cross) > 0.05
    # ... and the correction pushed the S1 -> X2 posterior DOWN ...
    p_on = m_on.attention.inner_attention.score_tensor_for_sparsity[1, 0]
    p_off = m_off.attention.inner_attention.score_tensor_for_sparsity[1, 0]
    assert float(p_off - p_on) > 0.01
    # ... while the true mediator X1 -> X2 was handed the freed budget (UP):
    # the cross correction is chained into the shared query the self block
    # scores, so the S-freed norm budget renormalises onto the X keys.
    d_on = m_on.self_attention.inner_attention.score_tensor_for_sparsity[1, 0]
    d_off = m_off.self_attention.inner_attention.score_tensor_for_sparsity[1, 0]
    assert float(d_on) > float(d_off)


def test_layer_homogeneous_path_unchanged():
    """The single square block still probes its own posterior (no "W" key)."""
    m = _l_make_layer(transitive_correction=True, homogeneous_nodes=True)
    m.eval()
    pred, attn, _ = _l_forward(m)
    assert pred.shape == (2, _L_N, 1)          # S rows are reconstructed too
    assert attn.shape == (2, _L_N, _L_N)
    w = m.attention.last_transitive_W
    assert w.shape == (_L_N, _L_N)
    assert float(w.max()) < 0.01               # init-silent here as well

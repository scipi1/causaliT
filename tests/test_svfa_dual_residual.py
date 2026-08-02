"""Tests for the SVFA dual-residual variant (``SingleCausalLayerRes``).

Run with:  pytest tests/test_svfa_dual_residual.py -v

Covers four layers of the implementation:

1. ``AttentionLayer`` with ``dual_value=True`` returns a 4-tuple and exposes
   the dedicated structural value/out projection parameters.
2. ``DualResidualDecoderLayer`` updates BOTH ``X_struct`` and ``X_val``
   (in contrast to the standard SVFA ``ReversedDecoderLayer`` which only
   updates ``X_val``).
3. ``DualResidualDecoder`` accepts an SVFA tuple and propagates updates
   layer-by-layer.
4. The ``STRUCTURAL_PATTERNS`` registry in ``gradient_routing.py`` correctly
   classifies the new structural parameters when running on a
   ``SingleCausalLayerRes`` instance built end-to-end.
"""

import sys
from pathlib import Path

import pytest
import torch

# Make project root importable.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.modules.attention import AttentionLayer, ScaledDotSoftmax
from causaliT.core.modules.extra_layers import UniformAttentionMask
from causaliT.core.architectures.single_causal_res.decoder import (
    DualResidualDecoder,
    DualResidualDecoderLayer,
)
from causaliT.training.gradient_routing import (
    STRUCTURAL_PATTERNS,
    _is_structural_param as is_structural_parameter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def d_model():
    return 16


@pytest.fixture
def d_qk():
    return 16


@pytest.fixture
def d_ff():
    return 32


@pytest.fixture
def batch_size():
    return 3


@pytest.fixture
def dec_seq_len():
    return 4


@pytest.fixture
def ext_seq_len():
    return 5


def _make_attention(d_model, d_qk, n_heads=1, dual_value=False):
    """Helper: build a fresh AttentionLayer with our defaults."""
    return AttentionLayer(
        attention=ScaledDotSoftmax,
        d_model_queries=d_model,
        d_model_keys=d_model,
        d_model_values=d_model,
        d_queries_keys=d_qk,
        n_heads=n_heads,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0.0,
        dropout_qkv=0.0,
        dual_value=dual_value,
    )


# ---------------------------------------------------------------------------
# 1. AttentionLayer with dual_value=True
# ---------------------------------------------------------------------------


class TestAttentionLayerDualValue:
    def test_dual_value_attribute(self, d_model, d_qk):
        att = _make_attention(d_model, d_qk, dual_value=True)
        assert att.dual_value is True
        assert att.value_projection_struct is not None

    def test_single_value_no_struct_projection(self, d_model, d_qk):
        att = _make_attention(d_model, d_qk, dual_value=False)
        assert att.dual_value is False
        assert att.value_projection_struct is None
        assert att.out_projection_struct is None

    def test_forward_returns_four_tuple(
        self, d_model, d_qk, batch_size, dec_seq_len, ext_seq_len
    ):
        att = _make_attention(d_model, d_qk, dual_value=True)
        Q = torch.randn(batch_size, dec_seq_len, d_model)
        K = torch.randn(batch_size, ext_seq_len, d_model)
        V = torch.randn(batch_size, ext_seq_len, d_model)

        out = att(
            query=Q,
            key=K,
            value=V,
            mask_miss_k=None,
            mask_miss_q=None,
            pos=None,
            causal_mask=False,
        )
        assert isinstance(out, tuple) and len(out) == 4, (
            "dual_value=True should return (out_val, out_struct, attn, ent)"
        )
        out_val, out_struct, attn, _ = out
        assert out_val.shape == (batch_size, dec_seq_len, d_model)
        assert out_struct.shape == (batch_size, dec_seq_len, d_model)
        # Same Q/K → same attention pattern → out_val and out_struct differ
        # only because they come from different value projections.
        assert not torch.allclose(out_val, out_struct)

    def test_key_value_routes_to_struct_path(
        self, d_model, d_qk, batch_size, dec_seq_len, ext_seq_len
    ):
        """When ``key_value`` differs from ``key``, only ``out_struct`` should change."""
        att = _make_attention(d_model, d_qk, dual_value=True)
        Q = torch.randn(batch_size, dec_seq_len, d_model)
        K = torch.randn(batch_size, ext_seq_len, d_model)
        V = torch.randn(batch_size, ext_seq_len, d_model)
        KV_a = torch.randn(batch_size, ext_seq_len, d_model)
        KV_b = torch.randn(batch_size, ext_seq_len, d_model)

        out_val_a, out_struct_a, _, _ = att(
            query=Q, key=K, value=V, key_value=KV_a,
            mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
        )
        out_val_b, out_struct_b, _, _ = att(
            query=Q, key=K, value=V, key_value=KV_b,
            mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
        )

        # out_val depends on V only (which is shared) — should match exactly.
        assert torch.allclose(out_val_a, out_val_b)
        # out_struct depends on key_value — should differ.
        assert not torch.allclose(out_struct_a, out_struct_b)

    def test_multi_head_creates_out_projection_struct(self, d_model, d_qk):
        att = _make_attention(d_model, d_qk, n_heads=2, dual_value=True)
        assert att.out_projection_struct is not None

    def test_single_head_no_out_projection_struct(self, d_model, d_qk):
        att = _make_attention(d_model, d_qk, n_heads=1, dual_value=True)
        assert att.out_projection_struct is None


# ---------------------------------------------------------------------------
# 2. DualResidualDecoderLayer
# ---------------------------------------------------------------------------


def _make_dual_layer(d_model, d_qk, d_ff):
    cross = _make_attention(d_model, d_qk, dual_value=True)
    self_ = _make_attention(d_model, d_qk, dual_value=True)
    return DualResidualDecoderLayer(
        global_cross_attention=cross,
        global_self_attention=self_,
        d_model_dec=d_model,
        activation="gelu",
        norm="layer",
        d_ff=d_ff,
        dropout_ff=0.0,
        dropout_attn_out=0.0,
    )


class TestDualResidualDecoderLayer:
    def test_rejects_attention_without_dual_value(self, d_model, d_qk, d_ff):
        bad_cross = _make_attention(d_model, d_qk, dual_value=False)
        good_self = _make_attention(d_model, d_qk, dual_value=True)
        with pytest.raises(ValueError, match="dual_value=True"):
            DualResidualDecoderLayer(
                global_cross_attention=bad_cross,
                global_self_attention=good_self,
                d_model_dec=d_model,
                activation="gelu",
                norm="layer",
                d_ff=d_ff,
                dropout_ff=0.0,
                dropout_attn_out=0.0,
            )

    def test_rejects_non_tuple_input(
        self, d_model, d_qk, d_ff, batch_size, dec_seq_len, ext_seq_len
    ):
        layer = _make_dual_layer(d_model, d_qk, d_ff)
        bad_X = torch.randn(batch_size, dec_seq_len, d_model)  # not a tuple
        ext = torch.randn(batch_size, ext_seq_len, d_model)
        with pytest.raises(ValueError, match="SVFA input"):
            layer(
                X=bad_X,
                external_context=ext,
                self_mask_miss_k=None, self_mask_miss_q=None,
                cross_mask_miss_k=None, cross_mask_miss_q=None,
                dec_input_pos=None, causal_mask=False,
            )

    def test_returns_tuple(
        self, d_model, d_qk, d_ff, batch_size, dec_seq_len, ext_seq_len
    ):
        layer = _make_dual_layer(d_model, d_qk, d_ff)
        X = (
            torch.randn(batch_size, dec_seq_len, d_model),
            torch.randn(batch_size, dec_seq_len, d_model),
        )
        ext = torch.randn(batch_size, ext_seq_len, d_model)
        out, *_ = layer(
            X=X, external_context=ext,
            self_mask_miss_k=None, self_mask_miss_q=None,
            cross_mask_miss_k=None, cross_mask_miss_q=None,
            dec_input_pos=None, causal_mask=False,
        )
        assert isinstance(out, tuple) and len(out) == 2

    def test_both_streams_are_updated(
        self, d_model, d_qk, d_ff, batch_size, dec_seq_len, ext_seq_len
    ):
        """Key invariant: BOTH X_struct and X_val change layer-to-layer."""
        layer = _make_dual_layer(d_model, d_qk, d_ff)
        X_struct_in = torch.randn(batch_size, dec_seq_len, d_model)
        X_val_in = torch.randn(batch_size, dec_seq_len, d_model)
        ext = torch.randn(batch_size, ext_seq_len, d_model)

        out, *_ = layer(
            X=(X_struct_in, X_val_in), external_context=ext,
            self_mask_miss_k=None, self_mask_miss_q=None,
            cross_mask_miss_k=None, cross_mask_miss_q=None,
            dec_input_pos=None, causal_mask=False,
        )
        X_struct_out, X_val_out = out

        # Structure stream MUST change — this is the whole point of the variant.
        assert not torch.allclose(X_struct_out, X_struct_in), (
            "Dual-residual layer must update X_struct (vs. SVFA which keeps it fixed)."
        )
        # Value stream must change too (same as SVFA).
        assert not torch.allclose(X_val_out, X_val_in)

    def test_tuple_external_context_supported(
        self, d_model, d_qk, d_ff, batch_size, dec_seq_len, ext_seq_len
    ):
        layer = _make_dual_layer(d_model, d_qk, d_ff)
        X = (
            torch.randn(batch_size, dec_seq_len, d_model),
            torch.randn(batch_size, dec_seq_len, d_model),
        )
        ext = (
            torch.randn(batch_size, ext_seq_len, d_model),
            torch.randn(batch_size, ext_seq_len, d_model),
        )
        out, *_ = layer(
            X=X, external_context=ext,
            self_mask_miss_k=None, self_mask_miss_q=None,
            cross_mask_miss_k=None, cross_mask_miss_q=None,
            dec_input_pos=None, causal_mask=False,
        )
        assert isinstance(out, tuple) and len(out) == 2

    def test_gradients_reach_struct_value_projection(
        self, d_model, d_qk, d_ff, batch_size, dec_seq_len, ext_seq_len
    ):
        """A loss on X_struct (downstream) must produce a non-zero grad on
        ``value_projection_struct`` — this is the parameter the dual residual
        is meant to train."""
        layer = _make_dual_layer(d_model, d_qk, d_ff)

        X_struct = torch.randn(batch_size, dec_seq_len, d_model, requires_grad=True)
        X_val = torch.randn(batch_size, dec_seq_len, d_model, requires_grad=True)
        ext = torch.randn(batch_size, ext_seq_len, d_model, requires_grad=True)

        out, *_ = layer(
            X=(X_struct, X_val), external_context=ext,
            self_mask_miss_k=None, self_mask_miss_q=None,
            cross_mask_miss_k=None, cross_mask_miss_q=None,
            dec_input_pos=None, causal_mask=False,
        )
        out_struct, _ = out
        out_struct.sum().backward()

        cross_struct_w = layer.global_cross_attention.value_projection_struct.weight
        self_struct_w = layer.global_self_attention.value_projection_struct.weight
        assert cross_struct_w.grad is not None and cross_struct_w.grad.abs().sum() > 0
        assert self_struct_w.grad is not None and self_struct_w.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 3. DualResidualDecoder (stack)
# ---------------------------------------------------------------------------


class TestDualResidualDecoder:
    def test_stack_propagates_tuple(
        self, d_model, d_qk, d_ff, batch_size, dec_seq_len, ext_seq_len
    ):
        layers = [
            _make_dual_layer(d_model, d_qk, d_ff),
            _make_dual_layer(d_model, d_qk, d_ff),
        ]
        decoder = DualResidualDecoder(
            decoder_layers=layers,
            norm_layer=None,
            emb_dropout=0.0,
        )
        X = (
            torch.randn(batch_size, dec_seq_len, d_model),
            torch.randn(batch_size, dec_seq_len, d_model),
        )
        ext = torch.randn(batch_size, ext_seq_len, d_model)
        out, cross_atts, self_atts, *_ = decoder(
            X=X, external_context=ext,
            self_mask_miss_k=None, self_mask_miss_q=None,
            cross_mask_miss_k=None, cross_mask_miss_q=None,
            dec_input_pos=None, causal_mask=False,
        )
        assert isinstance(out, tuple) and len(out) == 2
        assert len(cross_atts) == 2 and len(self_atts) == 2

    def test_stack_rejects_non_tuple_input(
        self, d_model, d_qk, d_ff, batch_size, dec_seq_len, ext_seq_len
    ):
        decoder = DualResidualDecoder(
            decoder_layers=[_make_dual_layer(d_model, d_qk, d_ff)],
            norm_layer=None,
            emb_dropout=0.0,
        )
        bad_X = torch.randn(batch_size, dec_seq_len, d_model)
        ext = torch.randn(batch_size, ext_seq_len, d_model)
        with pytest.raises(ValueError, match="SVFA input"):
            decoder(
                X=bad_X, external_context=ext,
                self_mask_miss_k=None, self_mask_miss_q=None,
                cross_mask_miss_k=None, cross_mask_miss_q=None,
                dec_input_pos=None, causal_mask=False,
            )


# ---------------------------------------------------------------------------
# 4. Gradient-routing classification
# ---------------------------------------------------------------------------


class TestGradientRoutingPatterns:
    def test_struct_projection_patterns_are_registered(self):
        """The two new projection names must be in STRUCTURAL_PATTERNS."""
        assert "value_projection_struct" in STRUCTURAL_PATTERNS
        assert "out_projection_struct" in STRUCTURAL_PATTERNS

    def test_is_structural_parameter_matches_dual_value_params(
        self, d_model, d_qk
    ):
        att = _make_attention(d_model, d_qk, n_heads=2, dual_value=True)
        names = dict(att.named_parameters()).keys()

        struct_names = [n for n in names if "value_projection_struct" in n]
        assert struct_names, "expected value_projection_struct.* parameters"
        for n in struct_names:
            assert is_structural_parameter(n), (
                f"{n!r} should be classified as structural"
            )

        out_struct_names = [n for n in names if "out_projection_struct" in n]
        # n_heads > 1 → out_projection_struct created.
        assert out_struct_names, "expected out_projection_struct.* parameters"
        for n in out_struct_names:
            assert is_structural_parameter(n)

    def test_value_projection_is_NOT_structural(self, d_model, d_qk):
        """Sanity: the standard ``value_projection`` (recon) must NOT be
        classified as structural just because the substring ``value_projection``
        occurs in ``value_projection_struct``."""
        att = _make_attention(d_model, d_qk, dual_value=True)
        names = dict(att.named_parameters()).keys()

        # Pick the standard value_projection (without `_struct` suffix).
        vp_names = [
            n for n in names
            if "value_projection" in n and "value_projection_struct" not in n
        ]
        assert vp_names, "expected standard value_projection.* parameters"
        for n in vp_names:
            assert not is_structural_parameter(n), (
                f"{n!r} is the recon value projection and must NOT be structural"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for the shared structural key projection (``shared_key``).

Run with:  pytest tests/test_atsel_shared_key.py -v

Motivation
----------
In split mode (``self_attention_type`` set) the S→X cross block and the X→X
self block each own an independent structural key projection ``W_K``.  With
``shared_key=True`` the self block owns NO ``W_K``; instead the cross block's
``W_K`` is applied to the X structural identity and fed to the self block as a
PRE-PROJECTED key (``key_external=True``).

The point (see CI_shared_qk): with a fixed orthonormal struct embedding the
cross ``W_K`` is an isometry, so S keys and X keys projected through the SAME
``W_K`` stay mutually orthogonal.  The shared free query then aligns on
genuinely orthonormal key axes for BOTH subspaces, removing the cheap spurious
X–X edges that flexible per-block non-orthonormal self keys made possible.

These tests verify:

1. Validation — ``shared_key=True`` requires ``self_attention_type`` set.
2. The self block does NOT build its own ``key_projection`` when shared.
3. The non-shared self block DOES own its own ``key_projection``.
4. Forward shapes + combined round-trip are unchanged in shared mode.
5. Gradient routing still classifies the single shared ``W_K`` as STRUCTURAL,
   and the self block exposes no orphan key-projection parameter.
6. Backward smoke: gradient reaches the shared ``W_K`` from the self path.
7. Orthogonality — under an orthogonal_fixed frame + orthogonal key projection,
   S keys and X keys projected through the shared ``W_K`` stay mutually
   orthonormal.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.training.gradient_routing import classify_parameters


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 5
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

VALUE_COL = 0
VAR_COL = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svfa_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": VALUE_COL,
                "embed": "linear",
                "label": "value",
                "role": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
            {
                "idx": VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "role": "structure",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
        ],
    }


def _make_model(
    self_attention_type="CommutatorSelfAttention",
    shared_query: bool = True,
    shared_key: bool = False,
    free_query_embedding: bool = True,
    gain_stream_source: str = "separate",
    struct_embedding_type: str = "standard_learnable",
    key_projection_type: str = "linear",
    orthogonal_key_scale: bool = True,
) -> AttentionSelectorLayer:
    extra = {}
    if struct_embedding_type == "orthogonal_fixed":
        extra.update(
            orthogonal_fixed_frame_type="random",
            orthogonal_fixed_scale=1.0,
        )
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_svfa_embed_cfg(VOCAB_S),
        ds_embed_X=_svfa_embed_cfg(VOCAB_X),
        comps_embed_S="svfa",
        comps_embed_X="svfa",
        attention_type="GatedCrossAttention",
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
        d_ff=D_FF,
        d_model=D_MODEL,
        d_qk=D_QK,
        S_seq_len=S_SEQ_LEN,
        X_seq_len=X_SEQ_LEN,
        shared_dag_across_heads=True,
        struct_embedding_type=struct_embedding_type,
        key_projection_type=key_projection_type,
        orthogonal_key_scale=orthogonal_key_scale,
        free_query_embedding=free_query_embedding,
        self_attention_type=self_attention_type,
        shared_query=shared_query,
        shared_key=shared_key,
        **extra,
    )


def _make_inputs():
    source = torch.zeros(BATCH, S_SEQ_LEN, 2)
    source[:, :, VALUE_COL] = torch.randn(BATCH, S_SEQ_LEN)
    source[:, :, VAR_COL] = (
        torch.arange(1, S_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)
    )

    x_actual = torch.zeros(BATCH, X_SEQ_LEN, 2)
    x_actual[:, :, VALUE_COL] = torch.randn(BATCH, X_SEQ_LEN)
    x_actual[:, :, VAR_COL] = (
        torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)
    )

    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0
    return source, x_actual, x_blanked


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_shared_key_requires_split(self):
        with pytest.raises(ValueError):
            _make_model(self_attention_type=None, shared_query=False, shared_key=True)

    def test_shared_key_flag_recorded(self):
        m = _make_model(shared_key=True)
        assert m.shared_key is True
        m2 = _make_model(shared_key=False)
        assert m2.shared_key is False


# ---------------------------------------------------------------------------
# 2 & 3. Ownership of the shared W_K
# ---------------------------------------------------------------------------


class TestOwnership:
    def test_self_block_has_no_own_key_projection(self):
        m = _make_model(shared_key=True)
        assert m.self_attention is not None
        # In shared mode the self block is built with key_external=True and
        # therefore owns NO key_projection.
        assert m.self_attention.key_projection is None
        names = dict(m.self_attention.named_parameters())
        assert not any("key_projection" in n for n in names), (
            f"self block must not register a key_projection param: "
            f"{[n for n in names if 'key_projection' in n]}"
        )

    def test_non_shared_self_block_owns_key_projection(self):
        m = _make_model(shared_key=False)
        assert m.self_attention is not None
        assert m.self_attention.key_projection is not None

    def test_cross_block_owns_shared_wk(self):
        m = _make_model(shared_key=True)
        assert m.attention.key_projection is not None


# ---------------------------------------------------------------------------
# 4. Forward shapes + round-trip
# ---------------------------------------------------------------------------


class TestForward:
    def test_shapes_shared(self):
        model = _make_model(shared_key=True)
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, aux = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)
        assert "l0_penalty" in aux

    def test_round_trip_shared(self):
        model = _make_model(shared_key=True)
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        _, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
        att_sx, att_xx = model.split_attention(attn)
        assert att_sx.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN)
        assert att_xx.shape == (BATCH, X_SEQ_LEN, X_SEQ_LEN)
        assert torch.allclose(torch.cat([att_sx, att_xx], dim=-1), attn)

    def test_shared_qk_orthogonal_fixed_forward(self):
        model = _make_model(
            shared_query=True,
            shared_key=True,
            struct_embedding_type="orthogonal_fixed",
            key_projection_type="orthogonal",
        )
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)


# ---------------------------------------------------------------------------
# 5. Gradient routing — single shared W_K classified structural
# ---------------------------------------------------------------------------


class TestParameterClassification:
    def test_shared_wk_is_structural_no_orphan(self):
        model = _make_model(shared_key=True)
        structural, reconstruction = classify_parameters(model)
        struct_ids = {id(p) for p in structural}

        for name, p in model.attention.named_parameters():
            if "key_projection" in name:
                assert id(p) in struct_ids, f"{name} (shared W_K) must be structural"

        self_names = dict(model.self_attention.named_parameters())
        assert not any("key_projection" in n for n in self_names)


# ---------------------------------------------------------------------------
# 6. Backward smoke — gradient reaches the shared W_K
# ---------------------------------------------------------------------------


class TestBackward:
    def test_shared_wk_receives_gradient(self):
        # Use a linear (learnable) key projection so W_K carries a gradient.
        model = _make_model(shared_key=True, key_projection_type="linear")
        model.train()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, _, _ = model.forward_with_actual(source, x_blanked, x_actual)
        pred_x.sum().backward()

        got_grad = False
        for name, p in model.attention.named_parameters():
            if "key_projection" in name and p.grad is not None:
                if p.grad.abs().sum() > 0:
                    got_grad = True
        assert got_grad, "Shared W_K must receive a non-zero gradient."


# ---------------------------------------------------------------------------
# 7. Orthogonality — S keys ⊥ X keys through the shared isometric W_K
# ---------------------------------------------------------------------------


class TestOrthogonality:
    def test_shared_keys_remain_orthonormal(self):
        model = _make_model(
            shared_query=True,
            shared_key=True,
            struct_embedding_type="orthogonal_fixed",
            key_projection_type="orthogonal",
            orthogonal_key_scale=False,  # pure isometry (no learnable scale)
        )
        model.eval()
        source, x_actual, x_blanked = _make_inputs()

        with torch.no_grad():
            s_struct = model.orth_embed_S(source)          # (B, L_S, d)
            xk_struct = model.orth_embed_X(x_actual)       # (B, L_X, d)
            sx = torch.cat([s_struct, xk_struct], dim=1)   # (B, L_S+L_X, d)
            keys = model.attention.key_projection(sx)      # shared W_K

        # Take one batch element; distinct variable indices should map to a
        # mutually orthonormal key set (Gram ≈ identity).
        k = keys[0]                                        # (L_S+L_X, d_qk)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gram = k @ k.t()
        eye = torch.eye(gram.shape[0])
        off_diag = (gram - eye).abs()
        # Cross S/X orthogonality is the block we care about.
        assert off_diag.max().item() < 1e-4, (
            f"S and X keys should stay mutually orthonormal under the shared "
            f"isometric W_K; max off-diagonal={off_diag.max().item():.2e}"
        )


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

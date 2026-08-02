"""Tests for the value-structure injection options.

Run with:  pytest tests/test_value_structure_injection.py -v

Standard Structure-Value Factorized attention makes the value projection depend
on DATA ONLY, so the shared W_V cannot learn a per-node functional.  Two
complementary options condition the value on the structural identity:

``value_structure_injection`` (SOURCE / parent-node keying)
    CONCATENATES a per-source-node identity code onto the value before W_V
    (V_j = W_V([v_j ; e_j])), widening W_V by d_model.

``value_structure_query_injection`` (QUERY / child-node keying)
    ADDS a key-independent term ``(sum_j A_ij) * W_V^q(e_i^q)`` computed from a
    dedicated bias-free ``value_query_proj``, so the SAME parent can be mapped
    to a DIFFERENT functional depending on which child it feeds (e.g. X2 -> X4
    vs X2 -> X5).  This does NOT widen W_V.  The additive term is applied INSIDE
    the gated inner attentions (GatedCrossAttention / GatedSelfAttention), which
    alone know the TRUE applied weights A.

Both share the same option set:

    "none"            -- disabled (default; data-only value, backward compatible).
    "separate"        -- dedicated reconstruction-routed identity table(s).
    "struct_detached" -- reuse the (detached) structural identity, no new params.

Covered here for BOTH ``AttentionSelectorLayer`` and ``SelfSelectorLayer``:

1. Backward-compat: default "none" leaves W_V input width == d_model, adds no
   identity tables / W_V^q, and forward shapes are unchanged.
2. "separate" / "struct_detached" widen W_V (key injection) / add W_V^q (query
   injection) and run forward.
3. "separate" adds identity tables; "struct_detached" adds no new tables.
4. Requiring SVFA: summation mode raises ValueError.
5. Gradient routing: the injected identity table(s) and W_V^q land in the
   RECONSTRUCTION group (name-based for AttentionSelector, module-ref for
   SelfSelector).
6. The prediction actually depends on the injected identity (perturbation test).
7. Key and query injection can be combined.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.architectures.self_selector.model import SelfSelectorLayer
from causaliT.training.gradient_routing import _is_structural_param


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 2
VOCAB_S = 5
VOCAB_X = 6
VOCAB_SHARED = S_SEQ_LEN + X_SEQ_LEN + 1


# ---------------------------------------------------------------------------
# Embedding configs
# ---------------------------------------------------------------------------


# Production feature layout (matches OrthogonalMaskEmbedding / FreeQueryEmbedding,
# which read var_idx=1): value at column 0, variable-ID at column 1.
def _svfa_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": 0,
                "embed": "linear",
                "label": "value",
                "role": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
            {
                "idx": 1,
                "embed": "nn_embedding",
                "label": "variable",
                "role": "structure",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
        ],
    }


def _summation_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": 0,
                "embed": "linear",
                "label": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
            {
                "idx": 1,
                "embed": "nn_embedding",
                "label": "variable",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
        ],
    }



# ---------------------------------------------------------------------------
# AttentionSelectorLayer helpers
# ---------------------------------------------------------------------------


def _make_atsel(
    value_structure_injection: str = "none",
    comps_embed_X: str = "svfa",
    value_structure_query_injection: str = "none",
    attention_type: str = "ScaledDotSoftmax",
):
    ds_embed_X = (
        _svfa_embed_cfg(VOCAB_X)
        if comps_embed_X == "svfa"
        else _summation_embed_cfg(VOCAB_X)
    )
    # ``ScaledDotSoftmax`` does not implement the additive query term, so
    # EFFECT-dependent checks build the layer with ``GatedCrossAttention``
    # (which genuinely applies ``(sum_j A_ij) * W_V^q(e_i)``).  The structural
    # wiring checks (projection presence/width, tables, routing) are inner-
    # attention agnostic and use the default ScaledDotSoftmax.
    extra = {}
    if attention_type == "GatedCrossAttention":
        extra["gain_stream_source"] = "separate"
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_svfa_embed_cfg(VOCAB_S),
        ds_embed_X=ds_embed_X,
        comps_embed_S="svfa" if comps_embed_X == "svfa" else "summation",
        comps_embed_X=comps_embed_X,
        attention_type=attention_type,
        # MANDATORY since the legacy cross-only variant was removed: the X→X
        # posterior now always comes from a dedicated direction-aware block.
        self_attention_type="GatedSelfAttention",

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
        value_structure_injection=value_structure_injection,
        value_structure_query_injection=value_structure_query_injection,
        **extra,
    )


def _atsel_inputs():
    # Column convention: value at col 0, 1-indexed variable-ID at col 1
    # (0 = padding).  The injected FreeQueryEmbedding tables have
    # ``seq_len + 1`` rows and are indexed by the variable-ID column, so IDs
    # must live in [1, seq_len].
    source = torch.zeros(BATCH, S_SEQ_LEN, 2)
    source[:, :, 0] = torch.randn(BATCH, S_SEQ_LEN)
    source[:, :, 1] = torch.arange(1, S_SEQ_LEN + 1).float().unsqueeze(0)

    x_actual = torch.zeros(BATCH, X_SEQ_LEN, 2)
    x_actual[:, :, 0] = torch.randn(BATCH, X_SEQ_LEN)
    x_actual[:, :, 1] = torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0)

    x_blanked = x_actual.clone()
    x_blanked[:, :, 0] = 0.0  # zero the VALUE column (col 0)
    return source, x_actual, x_blanked




# ---------------------------------------------------------------------------
# SelfSelectorLayer helpers
# ---------------------------------------------------------------------------


def _make_self(
    value_structure_injection: str = "none",
    comps_embed: str = "svfa",
    value_structure_query_injection: str = "none",
):
    ds_embed = (
        _svfa_embed_cfg(VOCAB_SHARED)
        if comps_embed == "svfa"
        else _summation_embed_cfg(VOCAB_SHARED)
    )
    return SelfSelectorLayer(
        model="test_self",
        ds_embed=ds_embed,
        comps_embed=comps_embed,
        attention_type="GatedSelfAttention",
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
        gain_stream_source="separate",
        value_structure_injection=value_structure_injection,
        value_structure_query_injection=value_structure_query_injection,
    )


def _self_inputs():
    # Column convention: value at col 0, 1-indexed variable-ID at col 1.
    N = S_SEQ_LEN + X_SEQ_LEN
    all_actual = torch.zeros(BATCH, N, 2)
    all_actual[:, :, 0] = torch.randn(BATCH, N)
    all_actual[:, :, 1] = torch.arange(1, N + 1).float().unsqueeze(0)

    all_blanked = all_actual.clone()
    all_blanked[:, :, 0] = 0.0  # zero the VALUE column (col 0)
    return all_blanked, all_actual



# ===========================================================================
# AttentionSelectorLayer -- value_structure_injection (SOURCE / key keying)
# ===========================================================================


class TestAtselBackwardCompat:
    def test_default_is_none(self):
        model = _make_atsel()
        assert model.value_structure_injection == "none"
        assert model.inject_value_structure is False

    def test_none_value_projection_width_unchanged(self):
        model = _make_atsel("none")
        # W_V accepts d_model only (no concatenated identity).
        assert model.attention.value_projection.in_features == D_MODEL


    def test_none_no_identity_tables(self):
        model = _make_atsel("none")
        assert model.val_id_embed_S is None
        assert model.val_id_embed_X is None

    def test_none_forward_shapes(self):
        model = _make_atsel("none")
        source, x_actual, x_blanked = _atsel_inputs()
        pred, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)


class TestAtselInjectionModes:
    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_value_projection_widened(self, mode):
        model = _make_atsel(mode)
        assert model.attention.value_projection.in_features == 2 * D_MODEL


    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_forward_shapes(self, mode):
        model = _make_atsel(mode)
        source, x_actual, x_blanked = _atsel_inputs()
        pred, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)

    def test_separate_adds_identity_tables(self):
        model = _make_atsel("separate")
        assert model.val_id_embed_S is not None
        assert model.val_id_embed_X is not None

    def test_struct_detached_adds_no_params(self):
        base = sum(p.numel() for p in _make_atsel("none").parameters())
        detached = _make_atsel("struct_detached")
        # struct_detached adds the widened W_V (extra columns) but NO identity
        # tables; separate adds identity tables on top.
        assert detached.val_id_embed_S is None
        assert detached.val_id_embed_X is None
        n_detached = sum(p.numel() for p in detached.parameters())
        n_separate = sum(p.numel() for p in _make_atsel("separate").parameters())
        # separate must have strictly more params than struct_detached.
        assert n_separate > n_detached > base


class TestAtselSvfaRequirement:
    def test_summation_raises(self):
        with pytest.raises(ValueError, match="requires SVFA"):
            _make_atsel("separate", comps_embed_X="summation")


class TestAtselGradientRouting:
    def test_identity_tables_are_reconstruction(self):
        model = _make_atsel("separate")
        val_id_names = [
            n for n, _ in model.named_parameters() if "val_id_embed" in n
        ]
        assert len(val_id_names) > 0, "expected val_id_embed_* parameters"
        for name in val_id_names:
            assert not _is_structural_param(name), (
                f"{name} must be classified as a RECONSTRUCTION parameter"
            )


class TestAtselOutputDependsOnIdentity:
    def test_perturbing_identity_changes_prediction(self):
        model_a = _make_atsel("separate")
        model_b = _make_atsel("separate")
        model_a.eval(); model_b.eval()
        model_b.load_state_dict(model_a.state_dict())

        perturbed = False
        for name, param in model_b.named_parameters():
            if "val_id_embed_X" in name:
                param.data += torch.randn_like(param) * 2.0
                perturbed = True
        assert perturbed

        source, x_actual, x_blanked = _atsel_inputs()
        with torch.no_grad():
            pred_a, _, _ = model_a.forward_with_actual(source, x_blanked, x_actual)
            pred_b, _, _ = model_b.forward_with_actual(source, x_blanked, x_actual)
        assert not torch.allclose(pred_a, pred_b), (
            "Perturbing the value-structure identity must change pred_x."
        )


# ===========================================================================
# AttentionSelectorLayer -- value_structure_query_injection (QUERY / child keying)
# ===========================================================================


class TestAtselQueryBackwardCompat:
    def test_default_is_none(self):
        model = _make_atsel()
        assert model.value_structure_query_injection == "none"
        assert model.inject_value_structure_query is False

    def test_none_no_value_query_proj(self):
        model = _make_atsel(value_structure_query_injection="none")
        # No additive query term -> no W_V^q projection.
        assert model.attention.value_query_proj is None
        assert model.val_q_id_embed_X is None

    def test_none_value_projection_width_unchanged(self):
        # Query injection is ADDITIVE (not concatenated) -> W_V width unchanged.
        model = _make_atsel(value_structure_query_injection="none")
        assert model.attention.value_projection.in_features == D_MODEL


class TestAtselQueryInjectionModes:
    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_value_query_proj_created(self, mode):
        model = _make_atsel(value_structure_query_injection=mode)
        assert model.inject_value_structure_query is True
        # A bias-free W_V^q: d_model -> d_model_values * n_heads (n_heads=1).
        assert model.attention.value_query_proj is not None
        assert model.attention.value_query_proj.in_features == D_MODEL
        assert model.attention.value_query_proj.bias is None

    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_value_projection_not_widened(self, mode):
        # Query-only injection must NOT widen the key/value W_V.
        model = _make_atsel(value_structure_query_injection=mode)
        assert model.attention.value_projection.in_features == D_MODEL

    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_forward_shapes(self, mode):
        model = _make_atsel(value_structure_query_injection=mode)
        source, x_actual, x_blanked = _atsel_inputs()
        pred, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)

    def test_separate_adds_query_identity_table(self):
        model = _make_atsel(value_structure_query_injection="separate")
        assert model.val_q_id_embed_X is not None

    def test_struct_detached_no_query_table(self):
        model = _make_atsel(value_structure_query_injection="struct_detached")
        assert model.val_q_id_embed_X is None


class TestAtselQuerySvfaRequirement:
    def test_summation_raises(self):
        with pytest.raises(ValueError, match="requires SVFA"):
            _make_atsel(
                comps_embed_X="summation",
                value_structure_query_injection="separate",
            )


class TestAtselQueryGradientRouting:
    def test_query_identity_table_is_reconstruction(self):
        model = _make_atsel(value_structure_query_injection="separate")
        names = [n for n, _ in model.named_parameters() if "val_q_id_embed" in n]
        assert len(names) > 0, "expected val_q_id_embed_* parameters"
        for name in names:
            assert not _is_structural_param(name), (
                f"{name} must be classified as a RECONSTRUCTION parameter"
            )

    def test_value_query_proj_is_reconstruction(self):
        model = _make_atsel(value_structure_query_injection="separate")
        names = [n for n, _ in model.named_parameters() if "value_query_proj" in n]
        assert len(names) > 0, "expected value_query_proj parameters"
        for name in names:
            assert not _is_structural_param(name), (
                f"{name} must be classified as a RECONSTRUCTION parameter"
            )


class TestAtselQueryOutputDependsOnIdentity:
    def test_perturbing_query_identity_changes_prediction(self):
        # Use GatedCrossAttention: only the gated inner attentions apply the
        # additive ``(sum_j A_ij) * W_V^q(e_i)`` term.
        model_a = _make_atsel(
            value_structure_query_injection="separate",
            attention_type="GatedCrossAttention",
        )
        model_b = _make_atsel(
            value_structure_query_injection="separate",
            attention_type="GatedCrossAttention",
        )
        model_a.eval(); model_b.eval()
        model_b.load_state_dict(model_a.state_dict())

        perturbed = False
        for name, param in model_b.named_parameters():
            if "val_q_id_embed_X" in name:
                param.data += torch.randn_like(param) * 5.0
                perturbed = True
        assert perturbed

        source, x_actual, x_blanked = _atsel_inputs()
        with torch.no_grad():
            pred_a, _, _ = model_a.forward_with_actual(source, x_blanked, x_actual)
            pred_b, _, _ = model_b.forward_with_actual(source, x_blanked, x_actual)
        assert not torch.allclose(pred_a, pred_b), (
            "Perturbing the value-structure QUERY identity must change pred_x."
        )


class TestAtselKeyAndQueryCombined:
    def test_both_injections_together(self):
        model = _make_atsel(
            value_structure_injection="separate",
            value_structure_query_injection="separate",
        )
        # Key injection widens W_V; query injection adds W_V^q.
        assert model.attention.value_projection.in_features == 2 * D_MODEL
        assert model.attention.value_query_proj is not None
        assert model.val_id_embed_X is not None
        assert model.val_q_id_embed_X is not None

        source, x_actual, x_blanked = _atsel_inputs()
        pred, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)


# ===========================================================================
# SelfSelectorLayer -- value_structure_injection (source / key keying)
# ===========================================================================


class TestSelfBackwardCompat:
    def test_default_is_none(self):
        model = _make_self()
        assert model.value_structure_injection == "none"
        assert model.inject_value_structure is False
        assert model.val_id_embed is None
        assert model.attention.value_projection.in_features == D_MODEL

    def test_none_forward_shapes(self):
        model = _make_self("none")
        all_blanked, all_actual = _self_inputs()
        pred, attn, _ = model.forward_with_actual(all_blanked, all_actual)
        N = S_SEQ_LEN + X_SEQ_LEN
        assert pred.shape == (BATCH, N, 1)
        assert attn.shape == (BATCH, N, N)


class TestSelfInjectionModes:
    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_value_projection_widened(self, mode):
        model = _make_self(mode)
        assert model.attention.value_projection.in_features == 2 * D_MODEL

    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_forward_shapes(self, mode):
        model = _make_self(mode)
        all_blanked, all_actual = _self_inputs()
        pred, attn, _ = model.forward_with_actual(all_blanked, all_actual)
        N = S_SEQ_LEN + X_SEQ_LEN
        assert pred.shape == (BATCH, N, 1)
        assert attn.shape == (BATCH, N, N)

    def test_separate_adds_table(self):
        assert _make_self("separate").val_id_embed is not None

    def test_struct_detached_no_table(self):
        assert _make_self("struct_detached").val_id_embed is None


class TestSelfSvfaRequirement:
    def test_summation_raises(self):
        with pytest.raises(ValueError, match="requires SVFA"):
            _make_self("separate", comps_embed="summation")


class TestSelfGradientRouting:
    def test_identity_table_is_reconstruction(self):
        model = _make_self("separate")
        structural, reconstruction = model.parameter_groups()
        struct_ids = {id(p) for p in structural}
        val_id_params = list(model.val_id_embed.parameters())
        assert len(val_id_params) > 0
        for p in val_id_params:
            assert id(p) not in struct_ids, (
                "val_id_embed must be in the RECONSTRUCTION group"
            )


# ===========================================================================
# SelfSelectorLayer -- value_structure_query_injection (query / child keying)
# ===========================================================================


class TestSelfQueryBackwardCompat:
    def test_default_is_none(self):
        model = _make_self()
        assert model.value_structure_query_injection == "none"
        assert model.inject_value_structure_query is False
        assert model.val_q_id_embed is None
        assert model.attention.value_query_proj is None


class TestSelfQueryInjectionModes:
    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_value_query_proj_created(self, mode):
        model = _make_self(value_structure_query_injection=mode)
        assert model.inject_value_structure_query is True
        assert model.attention.value_query_proj is not None
        assert model.attention.value_query_proj.in_features == D_MODEL
        assert model.attention.value_query_proj.bias is None

    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_value_projection_not_widened(self, mode):
        model = _make_self(value_structure_query_injection=mode)
        assert model.attention.value_projection.in_features == D_MODEL

    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_forward_shapes(self, mode):
        model = _make_self(value_structure_query_injection=mode)
        all_blanked, all_actual = _self_inputs()
        pred, attn, _ = model.forward_with_actual(all_blanked, all_actual)
        N = S_SEQ_LEN + X_SEQ_LEN
        assert pred.shape == (BATCH, N, 1)
        assert attn.shape == (BATCH, N, N)

    def test_separate_adds_query_table(self):
        assert _make_self(value_structure_query_injection="separate").val_q_id_embed is not None

    def test_struct_detached_no_query_table(self):
        assert _make_self(value_structure_query_injection="struct_detached").val_q_id_embed is None


class TestSelfQuerySvfaRequirement:
    def test_summation_raises(self):
        with pytest.raises(ValueError, match="requires SVFA"):
            _make_self(
                comps_embed="summation",
                value_structure_query_injection="separate",
            )


class TestSelfQueryGradientRouting:
    def test_query_identity_table_is_reconstruction(self):
        model = _make_self(value_structure_query_injection="separate")
        structural, reconstruction = model.parameter_groups()
        struct_ids = {id(p) for p in structural}
        val_q_params = list(model.val_q_id_embed.parameters())
        assert len(val_q_params) > 0
        for p in val_q_params:
            assert id(p) not in struct_ids, (
                "val_q_id_embed must be in the RECONSTRUCTION group"
            )

    def test_value_query_proj_is_reconstruction(self):
        model = _make_self(value_structure_query_injection="separate")
        structural, reconstruction = model.parameter_groups()
        struct_ids = {id(p) for p in structural}
        vq_params = list(model.attention.value_query_proj.parameters())
        assert len(vq_params) > 0
        for p in vq_params:
            assert id(p) not in struct_ids, (
                "value_query_proj must be in the RECONSTRUCTION group"
            )


class TestSelfQueryOutputDependsOnIdentity:
    def test_perturbing_query_identity_changes_prediction(self):
        # GatedSelfAttention (used by _make_self) applies the additive query term.
        model_a = _make_self(value_structure_query_injection="separate")
        model_b = _make_self(value_structure_query_injection="separate")
        model_a.eval(); model_b.eval()
        model_b.load_state_dict(model_a.state_dict())

        perturbed = False
        for name, param in model_b.named_parameters():
            if "val_q_id_embed" in name:
                param.data += torch.randn_like(param) * 5.0
                perturbed = True
        assert perturbed

        all_blanked, all_actual = _self_inputs()
        with torch.no_grad():
            pred_a, _, _ = model_a.forward_with_actual(all_blanked, all_actual)
            pred_b, _, _ = model_b.forward_with_actual(all_blanked, all_actual)
        assert not torch.allclose(pred_a, pred_b), (
            "Perturbing the value-structure QUERY identity must change pred."
        )


class TestSelfKeyAndQueryCombined:
    def test_both_injections_together(self):
        model = _make_self(
            value_structure_injection="separate",
            value_structure_query_injection="separate",
        )
        assert model.attention.value_projection.in_features == 2 * D_MODEL
        assert model.attention.value_query_proj is not None
        assert model.val_id_embed is not None
        assert model.val_q_id_embed is not None

        all_blanked, all_actual = _self_inputs()
        pred, attn, _ = model.forward_with_actual(all_blanked, all_actual)
        N = S_SEQ_LEN + X_SEQ_LEN
        assert pred.shape == (BATCH, N, 1)
        assert attn.shape == (BATCH, N, N)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

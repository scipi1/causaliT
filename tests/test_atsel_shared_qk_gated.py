"""Verification: ``GatedSelfAttention`` accepts ``shared_query=True`` AND
``shared_key=True`` and, when it does, its ANTISYMMETRIC direction term is
computed from BOTH the provided query and key.

Run with:  pytest tests/test_atsel_shared_qk_gated.py -v

Background
----------
In split mode (``self_attention_type="GatedSelfAttention"``) the X→X self block
can be fed a PRE-PROJECTED query (``shared_query=True`` → cross ``W_q`` applied
to the shared free query) and a PRE-PROJECTED key (``shared_key=True`` → cross
``W_K`` applied to the X structural identity).

Unlike ``CommutatorSelfAttention`` in ``skew_query`` mode (which resolves edge
direction from the query alone), ``GatedSelfAttention`` ALWAYS builds its
antisymmetric direction logit from the Toeplitz split of the full score::

    raw    = <q_i, k_j>
    A_anti = 0.5 * (raw - raw^T)

so BOTH the (shared) query and the (shared) key participate in the direction.

These tests verify:

1. Construction — the combination builds; the self block owns no structural
   query/key projection (both externally supplied).
2. Forward — shapes + combined round-trip are correct in shared_qk mode.
3. Antisymmetry (unit) — ``A_anti`` genuinely depends on BOTH inputs: perturbing
   ONLY the key changes the directed posterior, and perturbing ONLY the query
   changes it too.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.modules.gated_self_attention import GatedSelfAttention
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
    shared_query: bool = True,
    shared_key: bool = True,
    struct_embedding_type: str = "standard_learnable",
    key_projection_type: str = "linear",
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
        free_query_embedding=True,
        self_attention_type="GatedSelfAttention",
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
# 1. Construction — GatedSelfAttention accepts shared_query AND shared_key
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_both_flags_recorded_and_no_orphan_projections(self):
        m = _make_model(shared_query=True, shared_key=True)
        assert m.shared_query is True and m.shared_key is True
        assert m.self_attention is not None
        # Self block owns NO structural query/key projection (both external).
        assert m.self_attention.query_projection is None
        assert m.self_attention.key_projection is None
        self_names = dict(m.self_attention.named_parameters())
        assert not any("query_projection" in n for n in self_names)
        assert not any("key_projection" in n for n in self_names)
        # The inner attention really is GatedSelfAttention.
        assert isinstance(m.self_attention.inner_attention, GatedSelfAttention)

    def test_shared_projections_classified_structural(self):
        m = _make_model(shared_query=True, shared_key=True)
        structural, _ = classify_parameters(m)
        struct_ids = {id(p) for p in structural}
        for name, p in m.attention.named_parameters():
            if "query_projection" in name or "key_projection" in name:
                assert id(p) in struct_ids, f"{name} (shared W) must be structural"


# ---------------------------------------------------------------------------
# 2. Forward — shapes + combined round-trip in shared_qk mode
# ---------------------------------------------------------------------------


class TestForward:
    def test_shapes_shared_qk(self):
        model = _make_model(shared_query=True, shared_key=True)
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, aux = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)
        assert "l0_penalty" in aux

    def test_round_trip_shared_qk(self):
        model = _make_model(shared_query=True, shared_key=True)
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        _, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
        att_sx, att_xx = model.split_attention(attn)
        assert att_sx.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN)
        assert att_xx.shape == (BATCH, X_SEQ_LEN, X_SEQ_LEN)
        assert torch.allclose(torch.cat([att_sx, att_xx], dim=-1), attn)

    def test_orthogonal_fixed_shared_qk_forward(self):
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
# 3. Antisymmetry (unit) — A_anti depends on BOTH the query and the key
# ---------------------------------------------------------------------------


class TestAntisymmetryUsesBothInputs:
    """Feed a bare GatedSelfAttention the SAME pre-projected q / k the shared-qk
    self block would supply, then confirm the directed posterior (built from the
    antisymmetric part) reacts to a change in EITHER input.
    """

    B, N, E, D = 2, 5, 8, 4

    def _mod_and_inputs(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        mod = GatedSelfAttention().eval()  # A = structure gate (gain removed)
        q = torch.randn(self.B, self.N, self.E, generator=g)
        k = torch.randn(self.B, self.N, self.E, generator=g)
        v = torch.randn(self.B, self.N, self.D, generator=g)
        return mod, q, k, v

    def _directed_posterior(self, mod, q, k, v):
        # attn == directed posterior P(z_edge>0) * d  (masked).
        _, attn, _ = mod(
            query=q, key=k, value=v,
            mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
            hard_mask=None, oracle=False,
        )
        return attn

    def test_antisymmetric_part_formula_uses_q_and_k(self):
        """Direct algebra: A_anti = 0.5 (q k^T - (q k^T)^T) — non-zero and it
        changes when EITHER q or k changes."""
        mod, q, k, v = self._mod_and_inputs(1)
        import math

        def a_anti(q_, k_):
            raw = torch.einsum("bne,bme->bnm", q_, k_) / math.sqrt(q_.shape[-1])
            return 0.5 * (raw - raw.transpose(-1, -2))

        base = a_anti(q, k)
        # It is genuinely antisymmetric and not identically zero.
        assert torch.allclose(base, -base.transpose(-1, -2), atol=1e-6)
        assert base.abs().sum() > 0

        # Perturb ONLY the key → antisymmetric part changes.
        k2 = k.clone()
        k2[:, 0] = k2[:, 0] + 3.0
        assert not torch.allclose(a_anti(q, k2), base, atol=1e-4)

        # Perturb ONLY the query → antisymmetric part changes.
        q2 = q.clone()
        q2[:, 0] = q2[:, 0] + 3.0
        assert not torch.allclose(a_anti(q2, k), base, atol=1e-4)

    def test_directed_posterior_reacts_to_key_change(self):
        mod, q, k, v = self._mod_and_inputs(2)
        base = self._directed_posterior(mod, q, k, v)
        k2 = k.clone()
        k2[:, 1] = k2[:, 1] + 2.5
        changed = self._directed_posterior(mod, q, k2, v)
        assert not torch.allclose(base, changed, atol=1e-4), (
            "directed posterior must react to a key change (key feeds A_anti)"
        )

    def test_directed_posterior_reacts_to_query_change(self):
        mod, q, k, v = self._mod_and_inputs(3)
        base = self._directed_posterior(mod, q, k, v)
        q2 = q.clone()
        q2[:, 1] = q2[:, 1] + 2.5
        changed = self._directed_posterior(mod, q2, k, v)
        assert not torch.allclose(base, changed, atol=1e-4), (
            "directed posterior must react to a query change (query feeds A_anti)"
        )


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

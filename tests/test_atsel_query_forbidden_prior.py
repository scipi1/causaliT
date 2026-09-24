"""Tests for AttentionSelectorLayer ``query_forbidden_prior``.

Run with:  pytest tests/test_atsel_query_forbidden_prior.py -v

The forbidden-parents prior keeps the query of the listed nodes ORTHOGONAL to
their forbidden parents' fixed structural keys (e.g. parents observed in the
future): on every forward the constrained query rows are replaced by
``q @ P`` with ``P = I - K_f (K_f^T K_f)^{-1} K_f^T`` built from the fixed
orthonormal frame (direct null-space projection, no penalty term), and — for
specs with ``mask: true`` (the default) — the forbidden parent columns are
additionally zeroed in the hard mask (a zero score is not a zero posterior).

Node IDs are GLOBAL and 1-based over the concatenated dataset layout: S nodes
are 1..L_S, X nodes are L_S+1..L_S+L_X.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer

D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 2
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1
L = S_SEQ_LEN + X_SEQ_LEN   # global ids: S=1..3, X=4..7

VALUE_COL = 0
VAR_COL = 1


def _summation_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
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
    forbidden=None,
    remove_projections: bool = True,
    free_query_embedding: bool = True,
    struct_embedding_type: str = "orthogonal_fixed",
    self_attention_type="GatedSelfAttention",
    homogeneous_nodes: bool = False,
    query_exclude_self: bool = False,
) -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_summation_embed_cfg(VOCAB_S),
        ds_embed_X=_summation_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="summation",
        attention_type="ScaledDotSoftmax",
        self_attention_type=self_attention_type,
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
        free_query_embedding=free_query_embedding,
        query_centroid_init=False,
        remove_query_projection=remove_projections,
        remove_key_projection=remove_projections,
        homogeneous_nodes=homogeneous_nodes,
        query_forbidden_prior=forbidden,
        query_exclude_self=query_exclude_self,
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
    s_blanked = source.clone()
    s_blanked[:, :, VALUE_COL] = 0.0
    return source, x_actual, x_blanked, s_blanked


def _frame(model) -> torch.Tensor:
    """The shared fixed orthonormal frame, GLOBAL 0-based row order [S ; X]."""
    return torch.cat(
        [model.orth_embed_S.frame, model.orth_embed_X.frame], dim=0
    ).detach()


def _x_row(child_global: int) -> int:
    """Row of a global X child id inside the X query stream / table."""
    return child_global - S_SEQ_LEN  # 1-based var id == embedding table row


# ---------------------------------------------------------------------------
# 1. Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_is_none(self):
        model = _make_model()
        assert model.query_forbidden_prior is None
        assert model.qfp_hard_mask is None
        assert not any(
            n.startswith("qfp_proj_") for n, _ in model.named_buffers()
        )

    def test_spec_normalised_and_stored(self):
        model = _make_model(forbidden={5: {"forbidden": [1, "2"], "mask": False}})
        assert model.query_forbidden_prior == {
            5: {"forbidden": [1, 2], "mask": False}
        }

    def test_mask_defaults_true(self):
        model = _make_model(forbidden={5: {"forbidden": [1]}})
        assert model.query_forbidden_prior[5]["mask"] is True

    def test_requires_free_query_embedding(self):
        with pytest.raises(ValueError, match="free_query_embedding"):
            _make_model(
                forbidden={5: {"forbidden": [1]}}, free_query_embedding=False
            )

    def test_requires_orthogonal_fixed(self):
        with pytest.raises(ValueError, match="orthogonal_fixed"):
            _make_model(
                forbidden={5: {"forbidden": [1]}},
                struct_embedding_type="standard_learnable",
            )

    def test_requires_removed_projections(self):
        with pytest.raises(ValueError, match="remove_query_projection"):
            _make_model(
                forbidden={5: {"forbidden": [1]}}, remove_projections=False
            )

    def test_child_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            _make_model(forbidden={L + 1: {"forbidden": [1]}})

    def test_s_child_rejected_in_split_mode(self):
        with pytest.raises(ValueError, match="S node"):
            _make_model(forbidden={2: {"forbidden": [1]}})

    def test_s_child_allowed_in_homogeneous_mode(self):
        model = _make_model(
            forbidden={2: {"forbidden": [5]}}, homogeneous_nodes=True
        )
        assert model.query_forbidden_prior[2]["forbidden"] == [5]

    def test_empty_forbidden_rejected(self):
        with pytest.raises(ValueError, match="no forbidden parents"):
            _make_model(forbidden={5: {"forbidden": []}})

    def test_forbidden_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            _make_model(forbidden={5: {"forbidden": [L + 1]}})

    def test_self_forbidden_rejected(self):
        with pytest.raises(ValueError, match="cannot forbid itself"):
            _make_model(forbidden={5: {"forbidden": [5]}})

    def test_projector_is_exact_null_space_projector(self):
        model = _make_model(forbidden={5: {"forbidden": [1, 2]}})
        P = model.qfp_proj_5
        K = _frame(model)[[0, 1]]                      # (2, d) forbidden rows
        I = torch.eye(D_MODEL)
        assert torch.allclose(P, I - K.T @ K, atol=1e-6)
        assert torch.allclose(P, P @ P, atol=1e-6)     # idempotent
        assert torch.allclose(P, P.T, atol=1e-6)       # symmetric

    def test_projector_buffers_are_persistent(self):
        model = _make_model(forbidden={5: {"forbidden": [1]}})
        assert "qfp_proj_5" in model.state_dict()


# ---------------------------------------------------------------------------
# 2. Mask buffer
# ---------------------------------------------------------------------------


class TestMaskBuffer:
    def test_mask_true_zeroes_forbidden_columns_split_layout(self):
        model = _make_model(
            forbidden={5: {"forbidden": [1, 6], "mask": True}}
        )
        m = model.qfp_hard_mask
        assert m.shape == (X_SEQ_LEN, L)
        row = _x_row(5) - 1                            # 0-based query row
        assert m[row, 0] == 0.0                        # node 1 (S1)
        assert m[row, 5] == 0.0                        # node 6 (X3)
        assert m[row, 1] == 1.0                        # allowed column kept
        for r in range(X_SEQ_LEN):
            if r != row:
                assert torch.all(m[r] == 1.0)

    def test_mask_false_children_do_not_touch_the_mask(self):
        model = _make_model(
            forbidden={
                5: {"forbidden": [1], "mask": False},
                6: {"forbidden": [2], "mask": True},
            }
        )
        m = model.qfp_hard_mask
        assert torch.all(m[_x_row(5) - 1] == 1.0)
        assert m[_x_row(6) - 1, 1] == 0.0

    def test_all_mask_false_gives_none(self):
        model = _make_model(
            forbidden={5: {"forbidden": [1], "mask": False}}
        )
        assert model.qfp_hard_mask is None

    def test_homogeneous_mask_layout(self):
        model = _make_model(
            forbidden={2: {"forbidden": [5], "mask": True}},
            homogeneous_nodes=True,
        )
        m = model.qfp_hard_mask
        assert m.shape == (L, L)
        assert m[1, 4] == 0.0                          # child 2, parent 5



# ---------------------------------------------------------------------------
# 3. Orthogonal projection (geometry + gradients)
# ---------------------------------------------------------------------------


class TestProjection:
    def test_projected_rows_are_orthogonal_to_forbidden_keys(self):
        model = _make_model(forbidden={5: {"forbidden": [1, 2], "mask": False}})
        q = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
        out = model._project_forbidden_query_rows(q.clone(), row_offset=S_SEQ_LEN)
        K = _frame(model)[[0, 1]]                      # forbidden keys
        row = _x_row(5) - 1
        scores = out[:, row, :] @ K.T                  # (B, F)
        assert scores.abs().max() < 1e-5

    def test_allowed_components_and_other_rows_are_unchanged(self):
        model = _make_model(forbidden={5: {"forbidden": [1, 2], "mask": False}})
        q = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
        out = model._project_forbidden_query_rows(q.clone(), row_offset=S_SEQ_LEN)
        frame = _frame(model)
        row = _x_row(5) - 1
        # Orthonormal frame: the projection removes ONLY the forbidden
        # components, so scores against every allowed key are unchanged.
        allowed = [j for j in range(L) if j not in (0, 1)]
        assert torch.allclose(
            out[:, row, :] @ frame[allowed].T,
            q[:, row, :] @ frame[allowed].T,
            atol=1e-5,
        )
        # Rows without a constraint are untouched.
        for r in range(X_SEQ_LEN):
            if r != row:
                assert torch.equal(out[:, r, :], q[:, r, :])

    def test_projection_is_idempotent(self):
        model = _make_model(forbidden={5: {"forbidden": [1, 2], "mask": False}})
        q = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
        once = model._project_forbidden_query_rows(q.clone(), row_offset=S_SEQ_LEN)
        twice = model._project_forbidden_query_rows(
            once.clone(), row_offset=S_SEQ_LEN
        )
        assert torch.allclose(once, twice, atol=1e-6)

    def test_gradient_lies_in_allowed_subspace(self):
        """The chain rule through the fixed projector projects the embedding-
        row gradient: K_f grad == 0 for constrained rows (GPM/INLP-style)."""
        model = _make_model(forbidden={5: {"forbidden": [1, 2], "mask": True}})
        source, x_actual, x_blanked, _ = _make_inputs()
        pred, att, _ = model.forward_with_actual(source, x_blanked, x_actual)
        loss = pred.sum() + att.sum()
        loss.backward()
        grad = model.query_embed_X.embedding.weight.grad[_x_row(5)]
        assert grad is not None and grad.abs().max() > 0
        K = _frame(model)[[0, 1]]
        assert (K @ grad).abs().max() < 1e-4
        # Unconstrained rows are NOT restricted to the allowed subspace.
        other_grad = model.query_embed_X.embedding.weight.grad[_x_row(6)]
        assert other_grad.abs().max() > 0


# ---------------------------------------------------------------------------
# 4. Forward masking behaviour
# ---------------------------------------------------------------------------


class TestForwardMasking:
    def test_mask_true_zeroes_forbidden_columns(self):
        model = _make_model(
            forbidden={5: {"forbidden": [1, 6], "mask": True}},
            self_attention_type=None,                  # cross-only softmax arm
        )
        model.eval()
        source, x_actual, x_blanked, _ = _make_inputs()
        _, att, _ = model.forward_with_actual(source, x_blanked, x_actual)
        assert att.shape == (BATCH, X_SEQ_LEN, L)
        row = _x_row(5) - 1
        assert torch.all(att[:, row, 0] == 0.0)        # node 1 forbidden
        assert torch.all(att[:, row, 5] == 0.0)        # node 6 forbidden
        assert torch.all(att[:, row, 1] > 0.0)         # allowed column live

    def test_mask_false_leaves_posterior_alive(self):
        model = _make_model(
            forbidden={5: {"forbidden": [1, 6], "mask": False}},
            self_attention_type=None,
        )
        model.eval()
        source, x_actual, x_blanked, _ = _make_inputs()
        _, att, _ = model.forward_with_actual(source, x_blanked, x_actual)
        row = _x_row(5) - 1
        # No mask: dense softmax keeps the forbidden columns alive (the pure-
        # geometry arm; the DE-ALIGNMENT itself is checked in TestProjection).
        assert torch.all(att[:, row, 0] > 0.0)
        assert torch.all(att[:, row, 5] > 0.0)

    def test_split_mode_masks_both_blocks(self):
        model = _make_model(
            forbidden={5: {"forbidden": [1, 6], "mask": True}},
            self_attention_type="GatedSelfAttention",  # split mode
        )
        model.eval()
        source, x_actual, x_blanked, _ = _make_inputs()
        _, att, _ = model.forward_with_actual(source, x_blanked, x_actual)
        assert att.shape == (BATCH, X_SEQ_LEN, L)
        row = _x_row(5) - 1
        assert torch.all(att[:, row, 0] == 0.0)        # S1 via the cross block
        assert torch.all(att[:, row, 5] == 0.0)        # X3 via the self block

    def test_homogeneous_forward_mask(self):
        model = _make_model(
            forbidden={2: {"forbidden": [5], "mask": True}},
            homogeneous_nodes=True,
        )
        model.eval()
        source, x_actual, x_blanked, s_blanked = _make_inputs()
        _, att, _ = model.forward_with_actual(
            source, x_blanked, x_actual, s_blanked=s_blanked
        )
        assert att.shape == (BATCH, L, L)
        assert torch.all(att[:, 1, 4] == 0.0)          # child 2, parent 5


# ---------------------------------------------------------------------------
# 5. Composition with the known-parents prior
# ---------------------------------------------------------------------------


class TestComposition:
    def test_parents_prior_target_is_projected_into_allowed_subspace(self):
        parents = {5: {"parents": [4, 6], "fixed": False}}
        forbidden = {5: {"forbidden": [1], "mask": False}}
        model = _make_model(forbidden=forbidden)
        source, x_actual, _, _ = _make_inputs()
        model.init_queries_from_parents(parents, source, x_actual)
        frame = _frame(model)
        target = (frame[3] + frame[5]) / 2.0           # mean of parents 4, 6
        expected = model.qfp_proj_5 @ target
        got = model.query_embed_X.embedding.weight[_x_row(5)].detach()
        assert torch.allclose(got, expected, atol=1e-5)
        # And the written row is orthogonal to the forbidden key.
        assert abs(frame[0] @ got) < 1e-5


# ---------------------------------------------------------------------------
# 6. Config-template wiring
# ---------------------------------------------------------------------------


class TestConfigTemplate:
    def test_default_null_and_interpolation(self):
        from omegaconf import OmegaConf

        tmpl = (
            project_root
            / "causaliT"
            / "config"
            / "templates"
            / "config_attention_selector.yaml"
        )
        cfg = OmegaConf.load(str(tmpl))

        assert cfg.experiment.query_forbidden_prior is None
        assert cfg.model.kwargs.query_forbidden_prior is None

        cfg.experiment.query_forbidden_prior = {
            5: {"forbidden": [1, 2], "mask": True}
        }
        assert cfg.model.kwargs.query_forbidden_prior[5]["forbidden"] == [1, 2]
        assert cfg.model.kwargs.query_forbidden_prior[5]["mask"] is True


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# 7. Automatic self-key exclusion (query_exclude_self)
# ---------------------------------------------------------------------------


class TestExcludeSelf:
    def test_ctor_default_false(self):
        model = _make_model()
        assert model.query_exclude_self is False
        assert model._qfp_children == []
        assert not any(
            n.startswith("qfp_proj_") for n, _ in model.named_buffers()
        )

    def test_requires_free_query_embedding(self):
        with pytest.raises(ValueError, match="free_query_embedding"):
            _make_model(free_query_embedding=False, query_exclude_self=True)

    def test_requires_orthogonal_fixed(self):
        with pytest.raises(ValueError, match="orthogonal_fixed"):
            _make_model(
                struct_embedding_type="standard_learnable",
                query_exclude_self=True,
            )

    def test_requires_removed_projections(self):
        with pytest.raises(ValueError, match="remove_query_projection"):
            _make_model(remove_projections=False, query_exclude_self=True)

    def test_all_x_children_constrained_in_split_mode(self):
        model = _make_model(query_exclude_self=True)
        assert model._qfp_children == [4, 5, 6, 7]
        for c in (4, 5, 6, 7):
            assert hasattr(model, f"qfp_proj_{c}")

    def test_all_children_constrained_in_homogeneous_mode(self):
        model = _make_model(query_exclude_self=True, homogeneous_nodes=True)
        assert model._qfp_children == [1, 2, 3, 4, 5, 6, 7]

    def test_projector_excludes_self_row(self):
        model = _make_model(query_exclude_self=True)
        k_self = _frame(model)[4]                      # child 5 self key
        I = torch.eye(D_MODEL)
        expected = I - torch.outer(k_self, k_self) / (k_self @ k_self)
        assert torch.allclose(model.qfp_proj_5, expected, atol=1e-6)

    def test_union_with_forbidden_prior(self):
        model = _make_model(
            forbidden={5: {"forbidden": [1], "mask": False}},
            query_exclude_self=True,
        )
        K = _frame(model)[[0, 4]]                      # forbidden + self
        I = torch.eye(D_MODEL)
        assert torch.allclose(model.qfp_proj_5, I - K.T @ K, atol=1e-6)
        # Children without a forbidden prior still get self-exclusion only.
        K6 = _frame(model)[5]
        expected6 = I - torch.outer(K6, K6) / (K6 @ K6)
        assert torch.allclose(model.qfp_proj_6, expected6, atol=1e-6)

    def test_projected_query_orthogonal_to_self_key(self):
        model = _make_model(query_exclude_self=True)
        q = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
        out = model._project_forbidden_query_rows(q.clone(), row_offset=S_SEQ_LEN)
        frame = _frame(model)
        for c in (4, 5, 6, 7):
            k_self = frame[c - 1]
            scores = out[:, c - S_SEQ_LEN - 1, :] @ k_self
            assert scores.abs().max() < 1e-5

    def test_gradient_orthogonal_to_self_key(self):
        model = _make_model(query_exclude_self=True)
        source, x_actual, x_blanked, _ = _make_inputs()
        pred, att, _ = model.forward_with_actual(source, x_blanked, x_actual)
        loss = pred.sum() + att.sum()
        loss.backward()
        k_self = _frame(model)[4]                      # child 5 self key
        grad = model.query_embed_X.embedding.weight.grad[_x_row(5)]
        assert grad is not None and grad.abs().max() > 0
        assert abs(k_self @ grad) < 1e-4

    def test_no_mask_is_created(self):
        # The diagonal is already zero in the structural masks; the self
        # exclusion must NOT create a qfp mask on its own.
        model = _make_model(query_exclude_self=True)
        assert model.qfp_hard_mask is None

    def test_template_default_true_and_interpolation(self):
        from omegaconf import OmegaConf

        tmpl = (
            project_root
            / "causaliT"
            / "config"
            / "templates"
            / "config_attention_selector.yaml"
        )
        cfg = OmegaConf.load(str(tmpl))
        assert cfg.experiment.query_exclude_self is True
        assert cfg.model.kwargs.query_exclude_self is True
        cfg.experiment.query_exclude_self = False
        assert cfg.model.kwargs.query_exclude_self is False

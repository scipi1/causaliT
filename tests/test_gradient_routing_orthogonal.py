"""Gradient-routing classification tests for the orthogonal structural embeddings.

Run with:  pytest tests/test_gradient_routing_orthogonal.py -v

Background
----------
``struct_embedding_type`` selects the structural (Q/K) embedding scheme for the
``AttentionSelectorLayer``.  Two orthogonal schemes populate ``orth_embed_S`` /
``orth_embed_X``:

* ``"orthogonal_learnable"`` -> ``OrthogonalMaskEmbedding``: its ``value_embedding``
  (``nn.Linear``, ``freeze=False``) is a **learnable** embedding that builds the
  structural KEYS feeding the gate score ``log_alpha = QK^T``.  It is therefore a
  **structural** parameter and must be updated in the structure phase (frozen in
  the reconstruct phase).
* ``"orthogonal_fixed"`` -> ``FixedOrthonormalEmbedding``: a frozen **buffer-only**
  frame with NO trainable parameters, so it contributes nothing to either group.

Regression guarded here
-----------------------
The learnable orthogonal key embeddings were previously misclassified as
RECONSTRUCTION parameters (no ``orth_embed_*`` entry in ``STRUCTURAL_PATTERNS``),
which inverted the gradient-routing contract: they were frozen in the structure
phase and trained by HSIC/L0 in the reconstruct phase.  These tests assert the
corrected routing and that ``orthogonal_fixed`` gets the same (no-op) treatment.

Column convention (production): value at column 0, variable-ID at column 1.
"""

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.training.gradient_routing import classify_parameters


# ---------------------------------------------------------------------------
# Constants / helpers  (mirrors tests/test_atsel_orthonormal_frame.py)
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
D_QK = 16          # >= d_model for the orthogonal (isometric) key projection
S_SEQ_LEN = 3
X_SEQ_LEN = 4
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

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
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
            {
                "idx": VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
        ],
    }


def _make_model(
    struct_embedding_type: str,
    key_projection_type: str = "linear",
    free_query_embedding: bool = False,
    d_qk: int = D_QK,
    homogeneous_nodes: bool = False,
) -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_summation_embed_cfg(VOCAB_S),
        ds_embed_X=_summation_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="summation",
        attention_type="ScaledDotProduct",
        # MANDATORY since the legacy cross-only variant was removed.
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
        d_qk=d_qk,
        S_seq_len=S_SEQ_LEN,
        X_seq_len=X_SEQ_LEN,
        struct_embedding_type=struct_embedding_type,
        key_projection_type=key_projection_type,
        free_query_embedding=free_query_embedding,
        homogeneous_nodes=homogeneous_nodes,
    )


def _named_groups(model):
    """Return (structural_names, reconstruction_names) as sets of param names."""
    struct_params, recon_params = classify_parameters(model)
    struct_ids = {id(p) for p in struct_params}
    recon_ids = {id(p) for p in recon_params}
    struct_names, recon_names = set(), set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in struct_ids:
            struct_names.add(name)
        elif id(param) in recon_ids:
            recon_names.add(name)
    return struct_names, recon_names


# ---------------------------------------------------------------------------
# orthogonal_learnable: the learnable key embeddings must be STRUCTURAL
# ---------------------------------------------------------------------------


class TestOrthogonalLearnableRouting:
    def test_orth_embeds_are_trainable(self):
        m = _make_model("orthogonal_learnable")
        orth_params = [
            n for n, p in m.named_parameters()
            if p.requires_grad and (n.startswith("orth_embed_S") or n.startswith("orth_embed_X"))
        ]
        assert orth_params, (
            "orthogonal_learnable must expose trainable orth_embed_{S,X} params "
            "(OrthogonalMaskEmbedding.value_embedding, freeze=False)."
        )

    def test_orth_embeds_routed_structural_not_reconstruction(self):
        m = _make_model("orthogonal_learnable")
        struct_names, recon_names = _named_groups(m)

        orth = {n for n in (struct_names | recon_names)
                if n.startswith("orth_embed_S") or n.startswith("orth_embed_X")}
        assert orth, "Expected orth_embed_{S,X} trainable params to classify."

        # Every orth_embed_* param must be structural, none reconstruction.
        assert orth <= struct_names, (
            f"orth_embed_* params must be STRUCTURAL, but these were not: "
            f"{orth - struct_names}"
        )
        assert not (orth & recon_names), (
            f"orth_embed_* params leaked into RECONSTRUCTION: {orth & recon_names}"
        )

    def test_qk_projections_still_structural(self):
        m = _make_model("orthogonal_learnable")
        struct_names, _ = _named_groups(m)
        assert any("query_projection" in n for n in struct_names)
        assert any("key_projection" in n for n in struct_names)


# ---------------------------------------------------------------------------
# orthogonal_fixed: same treatment -> no trainable orth_embed params at all
# ---------------------------------------------------------------------------


class TestOrthogonalFixedRouting:
    def test_no_trainable_orth_embed_params(self):
        m = _make_model("orthogonal_fixed")
        orth_params = [
            n for n, p in m.named_parameters()
            if p.requires_grad and (n.startswith("orth_embed_S") or n.startswith("orth_embed_X"))
        ]
        assert orth_params == [], (
            "orthogonal_fixed uses a frozen buffer-only frame -> it must contribute "
            f"no trainable parameters, but found: {orth_params}"
        )

    def test_orth_embed_absent_from_both_groups(self):
        m = _make_model("orthogonal_fixed")
        struct_names, recon_names = _named_groups(m)
        leaked = {n for n in (struct_names | recon_names)
                  if n.startswith("orth_embed_S") or n.startswith("orth_embed_X")}
        assert not leaked, f"orthogonal_fixed frame should not classify: {leaked}"

    def test_qk_projections_still_structural(self):
        m = _make_model("orthogonal_fixed", key_projection_type="orthogonal")
        struct_names, _ = _named_groups(m)
        assert any("query_projection" in n for n in struct_names)
        assert any("key_projection" in n for n in struct_names)


# ---------------------------------------------------------------------------
# Phase behaviour: freezing the reconstruction group must NOT freeze the
# learnable orthogonal key embeddings (they belong to the structure group).
# ---------------------------------------------------------------------------


class TestPhaseFreezingConsistency:
    def test_freezing_reconstruction_keeps_orth_embeds_trainable(self):
        m = _make_model("orthogonal_learnable")
        struct_params, recon_params = classify_parameters(m)

        # Simulate entering the STRUCTURE phase: freeze reconstruction params.
        for p in recon_params:
            p.requires_grad_(False)

        still_trainable = {
            n for n, p in m.named_parameters()
            if p.requires_grad and (n.startswith("orth_embed_S") or n.startswith("orth_embed_X"))
        }
        assert still_trainable, (
            "In the structure phase (reconstruction frozen) the learnable "
            "orthogonal key embeddings must remain trainable."
        )
        # And they are exactly the structural orth_embed params.
        struct_ids = {id(p) for p in struct_params}
        for n, p in m.named_parameters():
            if n.startswith("orth_embed_S") or n.startswith("orth_embed_X"):
                assert id(p) in struct_ids


# ---------------------------------------------------------------------------
# Free query embeddings: the router keys on the ``query_embed`` PREFIX, so the
# S-side table introduced by ``homogeneous_nodes=True`` (where S nodes are
# children too) must be routed STRUCTURAL exactly like the X-side one.  Keying
# on the old exact name ``query_embed_X`` would have silently dropped
# ``query_embed_S`` into the reconstruction group.
# ---------------------------------------------------------------------------


class TestFreeQueryEmbeddingRouting:
    def test_query_embed_x_is_structural_in_split_mode(self):
        m = _make_model("standard_learnable", free_query_embedding=True)
        struct_names, recon_names = _named_groups(m)

        assert any(n.startswith("query_embed_X") for n in struct_names)
        assert not any(n.startswith("query_embed") for n in recon_names)

    def test_both_query_tables_are_structural_in_homogeneous_mode(self):
        m = _make_model(
            "standard_learnable",
            free_query_embedding=True,
            homogeneous_nodes=True,
        )
        assert m.query_embed_S is not None, (
            "homogeneous_nodes must build an S-side query table (S is a child)."
        )
        struct_names, recon_names = _named_groups(m)

        query_names = {
            n for n in (struct_names | recon_names) if n.startswith("query_embed")
        }
        assert any(n.startswith("query_embed_S") for n in query_names)
        assert any(n.startswith("query_embed_X") for n in query_names)
        assert query_names <= struct_names, (
            "All query_embed_* params must be STRUCTURAL; these were not: "
            f"{query_names - struct_names}"
        )
        assert not (query_names & recon_names), (
            f"query_embed_* leaked into RECONSTRUCTION: {query_names & recon_names}"
        )

    def test_query_embed_s_survives_the_structure_phase(self):
        """Freezing the reconstruction group must not freeze the S query table."""
        m = _make_model(
            "standard_learnable",
            free_query_embedding=True,
            homogeneous_nodes=True,
        )
        _, recon_params = classify_parameters(m)
        for p in recon_params:
            p.requires_grad_(False)

        assert m.query_embed_S.embedding.weight.requires_grad
        assert m.query_embed_X.embedding.weight.requires_grad


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

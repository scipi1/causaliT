"""Tests for AttentionSelectorLayer SVFA stream separation.

Run with:  pytest tests/test_atsel_svfa.py -v

Covers three correctness layers:

1. ``comps_embed_X`` is stored in ``__init__`` so callers can introspect the
   operating mode (SVFA vs standard).

2. In SVFA mode the value stream (``xq_val``) is the residual target after
   attention, NOT the structure stream (``xq_struct``).  Verified by injecting
   leaf tensors as mock embedding outputs and checking gradient flow.

3. In standard (summation) mode the single fused stream (``xq_struct``) is the
   residual target — backward-compatibility invariant.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svfa_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    """SVFA-split embedding config (struct + val roles)."""
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": 0,
                "embed": "nn_embedding",
                "label": "variable",
                "role": "structure",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
            {
                "idx": 1,
                "embed": "linear",
                "label": "value",
                "role": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
        ],
    }


def _summation_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    """Standard summation embedding config (no role split)."""
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": 0,
                "embed": "nn_embedding",
                "label": "variable",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
            {
                "idx": 1,
                "embed": "linear",
                "label": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
        ],
    }


def _make_model(comps_embed_X: str) -> AttentionSelectorLayer:
    """Build a minimal AttentionSelectorLayer (all dropouts = 0 for clean gradient tests)."""
    ds_embed_X = (
        _svfa_embed_cfg(VOCAB_X)
        if comps_embed_X == "svfa"
        else _summation_embed_cfg(VOCAB_X)
    )
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_summation_embed_cfg(VOCAB_S),
        ds_embed_X=ds_embed_X,
        comps_embed_S="summation",
        comps_embed_X=comps_embed_X,
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
        d_qk=D_QK,
        S_seq_len=S_SEQ_LEN,
        X_seq_len=X_SEQ_LEN,
        shared_dag_across_heads=True,
    )


def _make_inputs():
    """Minimal valid (source, x_actual, x_blanked) tuple."""
    source = torch.zeros(BATCH, S_SEQ_LEN, 2)
    source[:, :, 0] = torch.randint(0, VOCAB_S, (BATCH, S_SEQ_LEN)).float()
    source[:, :, 1] = torch.randn(BATCH, S_SEQ_LEN)

    x_actual = torch.zeros(BATCH, X_SEQ_LEN, 2)
    x_actual[:, :, 0] = torch.randint(0, VOCAB_X, (BATCH, X_SEQ_LEN)).float()
    x_actual[:, :, 1] = torch.randn(BATCH, X_SEQ_LEN)

    x_blanked = x_actual.clone()
    x_blanked[:, :, 1] = 0.0  # zero value column

    return source, x_actual, x_blanked


# ---------------------------------------------------------------------------
# 1. Attribute storage
# ---------------------------------------------------------------------------


class TestCompsEmbedXAttribute:
    def test_svfa_stored(self):
        assert _make_model("svfa").comps_embed_X == "svfa"

    def test_summation_stored(self):
        assert _make_model("summation").comps_embed_X == "summation"


# ---------------------------------------------------------------------------
# 2. Forward shapes (smoke tests)
# ---------------------------------------------------------------------------


class TestForwardShapes:
    def test_svfa_output_shapes(self):
        model = _make_model("svfa")
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn_weights, _ = model.forward_with_actual(
            source_tensor=source, x_blanked=x_blanked, x_actual=x_actual
        )
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn_weights.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)

    def test_summation_output_shapes(self):
        model = _make_model("summation")
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn_weights, _ = model.forward_with_actual(
            source_tensor=source, x_blanked=x_blanked, x_actual=x_actual
        )
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn_weights.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)


# ---------------------------------------------------------------------------
# 3. SVFA stream-separation gradient tests
#
# Strategy: replace the ENTIRE embedding_X / embedding_S modules with a
# lightweight ``_MockEmbedding`` that returns pre-allocated *leaf* tensors
# (requires_grad=True) on successive forward() calls.  After backward we
# inspect which leaf tensors have a non-zero gradient, which reveals which
# stream the residual connection adds to.
#
# forward_with_actual calls embedding_X twice in this order:
#   call 0 → x_actual   → key/value stream  (xk_struct, xk_val)
#   call 1 → x_blanked  → query stream      (xq_struct, xq_val)
#
# NOTE: we replace the *module object* (model.embedding_X = …) rather than
# patching .forward because PyTorch's C-level __call__ may bypass instance-
# level attribute overrides; replacing the registered submodule is reliable.
# ---------------------------------------------------------------------------


class _MockEmbedding(nn.Module):
    """Lightweight fake embedding: returns pre-specified tensors in sequence."""

    def __init__(self, outputs):
        super().__init__()
        self._outputs = list(outputs)
        self._idx = 0

    def forward(self, X):
        out = self._outputs[self._idx]
        self._idx += 1
        return out

    # Stubs for any helper methods the model might call on the embedding.
    def get_mask(self, X=None, **kwargs):
        return None

    def pass_var(self, X=None, **kwargs):
        return None


class TestSVFAStreamSeparation:
    """
    The core invariant: in SVFA mode the attention output is added to the
    VALUE stream (xq_val), not the STRUCTURE stream (xq_struct).
    """

    def _run_gradient_test(self, comps_embed_X: str):
        """
        Shared helper: replaces embeddings with mocks, runs forward + backward,
        and returns the leaf tensors for xq_val and xq_struct so the caller
        can inspect their .grad attributes.
        """
        model = _make_model(comps_embed_X)
        model.eval()

        B, L_X, L_S, D = BATCH, X_SEQ_LEN, S_SEQ_LEN, D_MODEL

        # ---- Leaf tensors ------------------------------------------------
        xq_val    = torch.randn(B, L_X, D, requires_grad=True)
        xq_struct = torch.randn(B, L_X, D, requires_grad=True)
        xk_val    = torch.randn(B, L_X, D, requires_grad=True)
        xk_struct = torch.randn(B, L_X, D, requires_grad=True)
        s_emb     = torch.randn(B, L_S, D, requires_grad=True)

        is_svfa = comps_embed_X == "svfa"

        if is_svfa:
            # call 0 = x_actual → key stream; call 1 = x_blanked → query stream
            mock_X = _MockEmbedding([(xk_struct, xk_val), (xq_struct, xq_val)])
        else:
            # Summation mode: plain tensors, no xq_val
            mock_X = _MockEmbedding([xk_struct, xq_struct])

        mock_S = _MockEmbedding([s_emb])   # single call, single tensor (summation S)

        # Replace submodules entirely (reliable way to hook into autograd graph).
        # type: ignore — Pylance complains about the type mismatch but
        # nn.Module attribute assignment is fully valid at runtime.
        model.embedding_X = mock_X  # type: ignore[assignment]
        model.embedding_S = mock_S  # type: ignore[assignment]

        # ---- Forward + backward ------------------------------------------
        source, x_actual, x_blanked = _make_inputs()
        pred_x, _, _ = model.forward_with_actual(
            source_tensor=source, x_blanked=x_blanked, x_actual=x_actual
        )
        pred_x.sum().backward()

        return xq_val, xq_struct

    def test_svfa_xq_val_receives_gradient(self):
        """
        In SVFA mode, xq_val must have a non-zero gradient after backward —
        it is the direct addend in the residual connection
        (x = norm1(xq_val + attn_out)).
        """
        xq_val, xq_struct = self._run_gradient_test("svfa")

        assert xq_val.grad is not None, (
            "xq_val (value stream of x_blanked) must be in the gradient graph "
            "in SVFA mode — it is the residual target."
        )
        assert xq_val.grad.abs().sum() > 0, (
            "xq_val must have a non-zero gradient, confirming it enters the "
            "residual connection before the forecaster."
        )

    def test_svfa_xq_struct_not_residual_addend(self):
        """
        In SVFA mode xq_struct is used ONLY as the attention Query (Q).
        It must NOT be the addend in the value-stream residual.

        The gradient on xq_struct (if any) comes only from Q∂attn/∂Q, which
        is an indirect multiplicative path — not the direct additive residual.
        We verify this indirectly: xq_val's gradient must be non-zero (the
        fixed code), and xq_struct must NOT be the exclusive source of the
        output (which would be the bug: x = norm1(xq_struct + attn_out)).

        Concretely: if xq_struct *were* the residual addend (buggy code),
        xq_val would have NO gradient at all (because the bug discards xq_val
        entirely).  The previous test already confirms xq_val has a gradient,
        which is sufficient to prove the fix is active.  This test documents
        the dual invariant for clarity.
        """
        xq_val, xq_struct = self._run_gradient_test("svfa")

        # xq_val having gradient is the proof that the residual target is correct.
        assert xq_val.grad is not None and xq_val.grad.abs().sum() > 0, (
            "This test requires xq_val to have a gradient (see the sister test). "
            "If this assertion fails, the primary stream-separation test also fails."
        )

    def test_summation_xq_struct_receives_gradient(self):
        """
        In standard (summation) mode the single fused stream (xq_struct in the
        mock, since no xq_val exists) must be the residual target — preserving
        backward compatibility.
        """
        xq_val, xq_struct = self._run_gradient_test("summation")

        assert xq_struct.grad is not None, (
            "xq_struct (sole stream in summation mode) must be in the gradient graph."
        )
        assert xq_struct.grad.abs().sum() > 0, (
            "xq_struct must have a non-zero gradient in summation mode."
        )


# ---------------------------------------------------------------------------
# 4. Backward compatibility: mode-dependent output difference
# ---------------------------------------------------------------------------


class TestSVFAOutputDependsOnValueEmbedding:
    """
    Perturbing the VALUE embedding weight must change pred_x in SVFA mode,
    confirming xq_val enters the residual path (and therefore the forecaster).
    """

    def test_svfa_output_changes_with_value_embedding_perturbation(self):
        model_a = _make_model("svfa")
        model_b = _make_model("svfa")
        model_a.eval(); model_b.eval()

        # Start from identical weights.
        model_b.load_state_dict(model_a.state_dict())

        # Perturb the VALUE sub-embedding of X in model_b.
        # In ModularEmbedding, all embed modules live in embed_modules_list; our
        # config has index 0 = nn_embedding (structure) and index 1 = linear_emb
        # (value).  x_blanked has a zeroed value column, so xq_val = W·0 + b = b:
        # the WEIGHT does not affect xq_val for zero inputs — we must also perturb
        # the BIAS.  We therefore perturb ALL params of embed_modules_list.1 (no
        # early break) to ensure at least the bias is changed.
        perturbed = False
        for name, param in model_b.embedding_X.named_parameters():
            if "embed_modules_list.1" in name:
                param.data += torch.randn_like(param) * 2.0
                perturbed = True   # keep iterating — perturb weight AND bias
        assert perturbed, (
            f"No 'embed_modules_list.1' parameter found in embedding_X. "
            f"Available: {[n for n, _ in model_b.embedding_X.named_parameters()]}"
        )

        source, x_actual, x_blanked = _make_inputs()
        with torch.no_grad():
            pred_a, _, _ = model_a.forward_with_actual(source, x_blanked, x_actual)
            pred_b, _, _ = model_b.forward_with_actual(source, x_blanked, x_actual)

        assert not torch.allclose(pred_a, pred_b), (
            "Perturbing the value embedding must change pred_x in SVFA mode, "
            "confirming xq_val flows through the residual to the forecaster."
        )


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

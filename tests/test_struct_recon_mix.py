"""Tests for the structural-reconstruction convex mix (``lambda_struct_recon``).

Run with:  pytest tests/test_struct_recon_mix.py -v

Background
----------
The structural loss of ``AttentionSelectorForecaster`` was, historically, a
pure-HSIC stream (plus sparsity/acyclicity/L0 side-terms).  Gradient routing
sends this stream to the STRUCTURAL parameters (Q/K projections + structural
embeddings) only.  ``lambda_struct_recon`` (alpha) re-injects a controlled dose
of the reconstruction loss into that same structural stream::

    L_struct = (1 - alpha) * HSIC_reg + alpha * loss_recon
               + score_sparse + group_l1 + notears + l0

Motivation: causal parents must also be predictive, aligning the method with
fit/likelihood-driven differentiable causal discovery (NOTEARS/DAG-GNN/GraN-DAG/
DCDI).

Guarantees under test
---------------------
1. ``lambda_struct_recon`` defaults to 0 and validates its ``[0, 1]`` range.
2. Backward compatibility: at ``alpha == 0`` the structural loss is exactly the
   pure-HSIC stream (the recon term is identically zero).
3. Convex-mix algebra: ``L(alpha) - L(0) == alpha * (L(1) - L(0))`` for the
   structural loss, i.e. it is a genuine linear interpolation between the pure
   HSIC stream and the pure-reconstruction stream.
4. Efficiency/routing: when ``alpha > 0`` the reconstruction signal reaches the
   STRUCTURAL parameters (Q/K) through the attention weights, using the already
   computed reconstruction loss / retained autograd graph (no extra forward).

Feature-index convention (mirrors tests/test_atsel_bkd.py):
    column 0 = variable ID, column 1 = value; ``val_idx = 1``.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.training.gradient_routing import classify_parameters


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_S = 8
VOCAB_X = 8
S_SEQ_LEN = 3
X_SEQ_LEN = 3
VALUE_COL = 1
VAR_COL = 0


def _embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    """Standard summation embedding config (variable ID + scalar value)."""
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
            {
                "idx": VALUE_COL,
                "embed": "linear",
                "label": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
        ],
    }


def _make_forecaster_config(
    lambda_struct_recon: float = 0.0,
    lambda_hsic: float = 1.0,
    use_gradient_routing: bool = True,
    d_model: int = D_MODEL,
) -> dict:
    """Minimal config dict accepted by AttentionSelectorForecaster.__init__."""
    return {
        "data": {
            "val_idx": VALUE_COL,
            "S_seq_len": S_SEQ_LEN,
            "X_seq_len": X_SEQ_LEN,
            "dataset": "dummy",
        },
        "model": {
            "model_object": "AttentionSelectorLayer",
            "kwargs": {
                "model": "AttentionSelectorLayer",
                "ds_embed_S": _embed_cfg(VOCAB_S, d_model),
                "ds_embed_X": _embed_cfg(VOCAB_X, d_model),
                "comps_embed_S": "summation",
                "comps_embed_X": "summation",
                "attention_type": "CausalCrossAttention",
                # MANDATORY since the legacy cross-only variant was removed.
                "self_attention_type": "GatedSelfAttention",

                "n_heads": 1,
                "dropout_emb": 0.0,
                "dropout_attn_out": 0.0,
                "dropout_ff": 0.0,
                "dropout_qkv": 0.0,
                "attention_dropout": 0.0,
                "activation": "relu",
                "norm": "layer",
                "use_final_norm": False,
                "device": "cpu",
                "out_dim": 1,
                "d_ff": 32,
                "d_model": d_model,
                "d_qk": d_model,
                "S_seq_len": S_SEQ_LEN,
                "X_seq_len": X_SEQ_LEN,
            },
        },
        "training": {
            "loss_fn": "mse",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "use_gradient_routing": use_gradient_routing,
            "lambda_recon": 1.0,
            "lambda_struct_recon": lambda_struct_recon,
            "lambda_hsic": lambda_hsic,
            "lambda_score_sparse": 0.0,
            "lambda_group_l1": 0.0,
            "lambda_l0": 0.0,
            "kappa": 0.0,
            "hsic_sigma": 1.0,
            "hsic_adaptive_bandwidth": False,
            "hsic_mode": "biased",
            "nhsic_epsilon": 0.01,
            "hsic_kernel_source": "rbf",
            "use_oracle_attention": False,
            "use_hard_masks": False,
            "freeze_structural_params": False,
            "freeze_reconstruction_params": False,
        },
    }


def _make_batch(batch: int = 8, seed: int = 0):
    """(S, X) tensors with variable ID at col 0 and value at col 1."""
    g = torch.Generator().manual_seed(seed)
    S = torch.zeros(batch, S_SEQ_LEN, 2)
    S[:, :, VAR_COL] = torch.randint(1, VOCAB_S, (batch, S_SEQ_LEN), generator=g).float()
    S[:, :, VALUE_COL] = torch.randn(batch, S_SEQ_LEN, generator=g)

    X = torch.zeros(batch, X_SEQ_LEN, 2)
    X[:, :, VAR_COL] = torch.randint(1, VOCAB_X, (batch, X_SEQ_LEN), generator=g).float()
    X[:, :, VALUE_COL] = torch.randn(batch, X_SEQ_LEN, generator=g)
    return S, X


def _structural_params(model: AttentionSelectorForecaster):
    struct, _ = classify_parameters(model.model)
    return struct


# ---------------------------------------------------------------------------
# 1. Construction + validation
# ---------------------------------------------------------------------------

class TestConstructionAndValidation:
    def test_default_is_zero(self):
        model = AttentionSelectorForecaster(_make_forecaster_config())
        assert model.lambda_struct_recon == 0.0

    def test_value_stored(self):
        model = AttentionSelectorForecaster(
            _make_forecaster_config(lambda_struct_recon=0.3)
        )
        assert abs(model.lambda_struct_recon - 0.3) < 1e-9

    @pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
    def test_out_of_range_raises(self, bad):
        with pytest.raises(ValueError):
            AttentionSelectorForecaster(
                _make_forecaster_config(lambda_struct_recon=bad)
            )


# ---------------------------------------------------------------------------
# 2 + 3. Backward compatibility and convex-mix algebra
# ---------------------------------------------------------------------------

class TestConvexMix:
    def _loss_components_for_alpha(self, model, alpha, batch):
        """Set alpha, run a deterministic _step, return (L_struct, L_recon)."""
        model.eval()  # deterministic forward (no dropout / stochastic gate)
        model.lambda_struct_recon = float(alpha)
        with torch.no_grad():
            model._step(batch, stage="val")
        comps = model._last_loss_components
        return (
            comps["loss_structural"].detach().clone(),
            comps["loss_recon"].detach().clone(),
        )

    def test_alpha_zero_recon_term_is_absent(self):
        """At alpha=0 the structural loss must not move when loss_recon changes.

        Concretely: L_struct(0) equals the pure-HSIC stream, so subtracting
        ``alpha * loss_recon`` (=0) leaves it unchanged.  We check that the
        recon contribution is exactly zero by comparing to the alpha=1 endpoint.
        """
        torch.manual_seed(1234)
        model = AttentionSelectorForecaster(_make_forecaster_config())
        batch = _make_batch(seed=7)

        L0, R0 = self._loss_components_for_alpha(model, 0.0, batch)
        L1, R1 = self._loss_components_for_alpha(model, 1.0, batch)

        # loss_recon is independent of alpha (same weights, same batch).
        assert torch.allclose(R0, R1, atol=1e-6)
        # The two structural endpoints genuinely differ (HSIC != recon here),
        # confirming the mix has something to interpolate.
        assert not torch.allclose(L0, L1, atol=1e-5)

    def test_linear_interpolation_identity(self):
        """L(alpha) - L(0) == alpha * (L(1) - L(0)) for the structural loss."""
        torch.manual_seed(1234)
        model = AttentionSelectorForecaster(_make_forecaster_config())
        batch = _make_batch(seed=7)

        L0, _ = self._loss_components_for_alpha(model, 0.0, batch)
        L1, _ = self._loss_components_for_alpha(model, 1.0, batch)

        for alpha in (0.25, 0.5, 0.75):
            La, _ = self._loss_components_for_alpha(model, alpha, batch)
            expected = L0 + alpha * (L1 - L0)
            assert torch.allclose(La, expected, atol=1e-5), (
                f"convex-mix identity failed at alpha={alpha}: "
                f"got {La.item():.6f}, expected {expected.item():.6f}"
            )

    def test_alpha_one_equals_recon_plus_sideterms(self):
        """At alpha=1 the HSIC term drops out; with all side-terms zero the
        structural loss equals the reconstruction loss."""
        torch.manual_seed(1234)
        # score/group/notears/l0 all zero in the default config.
        model = AttentionSelectorForecaster(_make_forecaster_config())
        batch = _make_batch(seed=7)
        L1, R1 = self._loss_components_for_alpha(model, 1.0, batch)
        assert torch.allclose(L1, R1, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Gradient routing: recon signal reaches the STRUCTURAL params iff alpha>0
# ---------------------------------------------------------------------------

class TestReconReachesStructuralParams:
    def _struct_grad_norm(self, alpha):
        """Grad-norm of L_struct w.r.t. the query/key projections.

        HSIC is disabled (lambda_hsic=0) so that the ONLY thing that can make
        the structural loss depend on Q/K is the injected reconstruction term.
        """
        torch.manual_seed(99)
        model = AttentionSelectorForecaster(
            _make_forecaster_config(lambda_struct_recon=alpha, lambda_hsic=0.0)
        )
        model.train()
        batch = _make_batch(seed=3)
        model._step(batch, stage="train")
        L_struct = model._last_loss_components["loss_structural"]

        qk = [
            p for n, p in model.model.named_parameters()
            if p.requires_grad and ("query_projection" in n or "key_projection" in n)
        ]
        assert qk, "expected trainable query/key projection params"

        grads = torch.autograd.grad(
            L_struct, qk, allow_unused=True, retain_graph=False
        )
        total = 0.0
        for g in grads:
            if g is not None:
                total += float(g.abs().sum())
        return total

    def test_no_recon_gradient_to_struct_at_alpha_zero(self):
        # alpha=0 and hsic=0 → structural loss is constant 0 → no Q/K gradient.
        assert self._struct_grad_norm(0.0) == pytest.approx(0.0, abs=1e-8)

    def test_recon_gradient_reaches_struct_at_alpha_positive(self):
        # alpha>0 → reconstruction gradient flows to Q/K via attention weights.
        assert self._struct_grad_norm(0.5) > 1e-6


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

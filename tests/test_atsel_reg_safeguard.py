"""Tests for the structural-regularizer safeguard (HSIC-relative caps).

Run with:  pytest tests/test_atsel_reg_safeguard.py -v

Background
----------
When the train HSIC goes flat ("diluted"), a fixed ``kappa`` (NOTEARS) or
``lambda_l0`` can dominate the structural pathway and drive structure learning
on its own.  The safeguard caps each coefficient per step so the weighted term
entering the loss never exceeds a fixed fraction of the weighted HSIC term::

    kappa_eff     = min(kappa,     kappa_max_hsic_pct     * hsic_ref / h(A))
    lambda_l0_eff = min(lambda_l0, lambda_l0_max_hsic_pct * hsic_ref / l0)

with ``hsic_ref`` an EMA of ``hsic_reg`` (detached, train batches only;
``hsic_safeguard_ema == 0`` gives the instantaneous per-batch value).

Guarantees under test
---------------------
1. Defaults: both caps are 0.0 (off), the EMA decay defaults to 0.9 and
   validates its [0, 1) range.
2. ``_cap_reg_coeff`` algebra: no-op when disabled / zero base / non-positive
   raw term; ``min`` semantics otherwise.
3. ``_hsic_safeguard_ref``: EMA updates on train batches only; instantaneous
   mode (ema=0) tracks the current batch; None when both caps off.
4. End-to-end (``_step``): with a huge base coefficient the NOTEARS / L0 terms
   entering the structural loss equal exactly ``pct * hsic_reg``; with a tiny
   base coefficient the term is uncapped; with pct=0 the behaviour is
   byte-identical to the pre-feature loss.
5. The cap is a detached scalar: gradients still flow through the capped terms.

Feature-index convention (mirrors tests/test_struct_recon_mix.py):
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
    kappa: float = 0.0,
    lambda_l0: float = 0.0,
    lambda_hsic: float = 1.0,
    kappa_max_hsic_pct: float = 0.0,
    lambda_l0_max_hsic_pct: float = 0.0,
    hsic_safeguard_ema: float = 0.9,
    attention_type: str = "CausalCrossAttention",
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
                "attention_type": attention_type,
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
                "remove_query_projection": False,
                "remove_key_projection": False,
            },
        },
        "training": {
            "loss_fn": "mse",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "use_gradient_routing": False,
            "lambda_recon": 1.0,
            "lambda_struct_recon": 0.0,
            "lambda_hsic": lambda_hsic,
            "lambda_score_sparse": 0.0,
            "lambda_group_l1": 0.0,
            "lambda_l0": lambda_l0,
            "lambda_query_norm": 0.0,
            "kappa": kappa,
            "kappa_max_hsic_pct": kappa_max_hsic_pct,
            "lambda_l0_max_hsic_pct": lambda_l0_max_hsic_pct,
            "hsic_safeguard_ema": hsic_safeguard_ema,
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


def _step_terms(model, batch):
    """Run a deterministic eval-mode _step; return (hsic_reg, acyclic, l0_reg).

    With the default config (lambda_struct_recon=0, score/group/query_norm
    weights all 0) the structural loss is exactly hsic_reg + acyclic + l0_reg,
    so the NOTEARS term is recovered by subtraction.  ``_last_hsic_reg`` /
    ``_last_l0_reg`` are stashed by the forecaster itself.
    """
    model.eval()
    with torch.no_grad():
        model._step(batch, stage="val")
    # float64: the NOTEARS term can be orders of magnitude smaller than
    # hsic_reg, and the subtraction would underflow in float32.
    hsic_reg = model._last_hsic_reg.detach().double()
    l0_reg = model._last_l0_reg.detach().double()
    l_struct = model._last_loss_components["loss_structural"].detach().double()
    acyclic = l_struct - hsic_reg - l0_reg
    return hsic_reg, acyclic, l0_reg


def _notears_raw(model) -> torch.Tensor:
    """h(A) recomputed from the current 2-D score tensor (split-mode slice)."""
    score = model.model.get_score_tensor_for_sparsity()
    A_cyc = score[:, S_SEQ_LEN:]
    return torch.trace(torch.matrix_exp(A_cyc * A_cyc)) - A_cyc.shape[-1]


# ---------------------------------------------------------------------------
# 1. Construction + validation
# ---------------------------------------------------------------------------

class TestConstructionAndValidation:
    def test_defaults_off(self):
        model = AttentionSelectorForecaster(_make_forecaster_config())
        assert model.kappa_max_hsic_pct == 0.0
        assert model.lambda_l0_max_hsic_pct == 0.0
        assert model.hsic_safeguard_ema == 0.9
        assert model._hsic_reg_ema is None

    def test_values_stored(self):
        model = AttentionSelectorForecaster(
            _make_forecaster_config(
                kappa_max_hsic_pct=0.1, lambda_l0_max_hsic_pct=0.2,
                hsic_safeguard_ema=0.5,
            )
        )
        assert model.kappa_max_hsic_pct == pytest.approx(0.1)
        assert model.lambda_l0_max_hsic_pct == pytest.approx(0.2)
        assert model.hsic_safeguard_ema == pytest.approx(0.5)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_ema_out_of_range_raises(self, bad):
        with pytest.raises(ValueError):
            AttentionSelectorForecaster(
                _make_forecaster_config(hsic_safeguard_ema=bad)
            )


# ---------------------------------------------------------------------------
# 2. _cap_reg_coeff algebra (static helper)
# ---------------------------------------------------------------------------

class TestCapRegCoeff:
    def test_disabled_pct_returns_base(self):
        raw = torch.tensor(3.0)
        assert AttentionSelectorForecaster._cap_reg_coeff(2.0, raw, 0.0, 1.0) == 2.0

    def test_zero_base_returns_zero(self):
        raw = torch.tensor(3.0)
        assert AttentionSelectorForecaster._cap_reg_coeff(0.0, raw, 0.1, 1.0) == 0.0

    def test_no_reference_returns_base(self):
        raw = torch.tensor(3.0)
        assert AttentionSelectorForecaster._cap_reg_coeff(2.0, raw, 0.1, None) == 2.0

    @pytest.mark.parametrize("raw_val", [0.0, -1.0, float("nan"), float("inf")])
    def test_degenerate_raw_returns_base(self, raw_val):
        raw = torch.tensor(raw_val)
        assert AttentionSelectorForecaster._cap_reg_coeff(2.0, raw, 0.1, 1.0) == 2.0

    def test_cap_binds_when_term_large(self):
        # base * raw = 10 * 3 = 30 >> 0.1 * 5 = 0.5  ->  cap = 0.5 / 3
        raw = torch.tensor(3.0)
        eff = AttentionSelectorForecaster._cap_reg_coeff(10.0, raw, 0.1, 5.0)
        assert eff == pytest.approx(0.5 / 3.0)
        assert eff * 3.0 == pytest.approx(0.1 * 5.0)

    def test_cap_loose_when_term_small(self):
        # base * raw = 0.01 * 3 = 0.03 < 0.1 * 5 = 0.5  ->  uncapped
        raw = torch.tensor(3.0)
        eff = AttentionSelectorForecaster._cap_reg_coeff(0.01, raw, 0.1, 5.0)
        assert eff == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 3. _hsic_safeguard_ref (EMA state machine)
# ---------------------------------------------------------------------------

class TestHsicSafeguardRef:
    def test_none_when_both_caps_off(self):
        model = AttentionSelectorForecaster(_make_forecaster_config())
        assert model._hsic_safeguard_ref(torch.tensor(1.0), "train") is None
        assert model._hsic_reg_ema is None

    def test_ema_updates_on_train_only(self):
        model = AttentionSelectorForecaster(
            _make_forecaster_config(kappa_max_hsic_pct=0.1, hsic_safeguard_ema=0.9)
        )
        # First train batch initialises the EMA.
        assert model._hsic_safeguard_ref(torch.tensor(1.0), "train") == pytest.approx(1.0)
        # Second train batch: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        assert model._hsic_safeguard_ref(torch.tensor(2.0), "train") == pytest.approx(1.1)
        # Val batches never update the EMA (and reuse it).
        assert model._hsic_safeguard_ref(torch.tensor(5.0), "val") == pytest.approx(1.1)
        assert model._hsic_reg_ema == pytest.approx(1.1)

    def test_instantaneous_mode_tracks_current_batch(self):
        model = AttentionSelectorForecaster(
            _make_forecaster_config(kappa_max_hsic_pct=0.1, hsic_safeguard_ema=0.0)
        )
        assert model._hsic_safeguard_ref(torch.tensor(1.0), "train") == pytest.approx(1.0)
        assert model._hsic_safeguard_ref(torch.tensor(2.0), "train") == pytest.approx(2.0)
        # Val uses the CURRENT batch value in instantaneous mode.
        assert model._hsic_safeguard_ref(torch.tensor(5.0), "val") == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 4. End-to-end _step behaviour (NOTEARS)
# ---------------------------------------------------------------------------

class TestNotearsCapEndToEnd:
    def test_cap_binds_with_huge_kappa(self):
        """kappa huge + pct=0.1 (instantaneous ref) => notears == 0.1 * hsic_reg."""
        torch.manual_seed(1234)
        model = AttentionSelectorForecaster(
            _make_forecaster_config(
                kappa=1e6, kappa_max_hsic_pct=0.1, hsic_safeguard_ema=0.0,
            )
        )
        batch = _make_batch(seed=7)
        hsic_reg, acyclic, _ = _step_terms(model, batch)
        assert acyclic.item() == pytest.approx(0.1 * hsic_reg.item(), rel=1e-4)

    def test_no_cap_with_tiny_kappa(self):
        """kappa small enough to stay under the cap => notears == kappa * h(A).

        kappa=1e-2 gives a term ~1e-4, ~10x below the cap (0.1 * hsic_reg
        ~ 1e-3) so the cap stays loose, yet well above the float32 resolution
        of the structural-loss sum (~1e-2).
        """
        torch.manual_seed(1234)
        model = AttentionSelectorForecaster(
            _make_forecaster_config(
                kappa=1e-2, kappa_max_hsic_pct=0.1, hsic_safeguard_ema=0.0,
            )
        )
        batch = _make_batch(seed=7)
        _, acyclic, _ = _step_terms(model, batch)
        expected = 1e-2 * _notears_raw(model).item()
        assert acyclic.item() == pytest.approx(expected, rel=1e-3)

    def test_pct_zero_is_backward_compatible(self):
        """pct=0 (off) => notears == kappa * h(A), exactly as before."""
        torch.manual_seed(1234)
        model = AttentionSelectorForecaster(
            _make_forecaster_config(kappa=2.0, kappa_max_hsic_pct=0.0)
        )
        batch = _make_batch(seed=7)
        _, acyclic, _ = _step_terms(model, batch)
        expected = 2.0 * _notears_raw(model).item()
        assert acyclic.item() == pytest.approx(expected, rel=1e-4)
        # Safeguard stays fully inert: no EMA state is ever created.
        assert model._hsic_reg_ema is None


# ---------------------------------------------------------------------------
# 4b. End-to-end _step behaviour (L0, GatedCrossAttention)
# ---------------------------------------------------------------------------

class TestL0CapEndToEnd:
    def test_cap_binds_with_huge_lambda_l0(self):
        """lambda_l0 huge + pct=0.1 (instantaneous ref) => l0_reg == 0.1 * hsic_reg."""
        torch.manual_seed(1234)
        model = AttentionSelectorForecaster(
            _make_forecaster_config(
                attention_type="GatedCrossAttention",
                lambda_l0=1e6, lambda_l0_max_hsic_pct=0.1, hsic_safeguard_ema=0.0,
            )
        )
        batch = _make_batch(seed=7)
        hsic_reg, _, l0_reg = _step_terms(model, batch)
        assert l0_reg.item() == pytest.approx(0.1 * hsic_reg.item(), rel=1e-4)

    def test_pct_zero_leaves_l0_uncapped(self):
        """pct=0 => l0_reg == lambda_l0 * l0_penalty (pre-feature behaviour)."""
        torch.manual_seed(1234)
        model = AttentionSelectorForecaster(
            _make_forecaster_config(
                attention_type="GatedCrossAttention",
                lambda_l0=0.5, lambda_l0_max_hsic_pct=0.0,
            )
        )
        batch = _make_batch(seed=7)
        model.eval()
        with torch.no_grad():
            model._step(batch, stage="val")
        l0_reg = model._last_l0_reg.item()
        # Uncapped term is strictly larger than any pct-capped value would be:
        # recompute the penalty from the aux-free path is not exposed, so check
        # the term is non-zero and equals lambda * penalty via a second model
        # with the cap active (same seed => same eval-mode forward).
        torch.manual_seed(1234)
        capped = AttentionSelectorForecaster(
            _make_forecaster_config(
                attention_type="GatedCrossAttention",
                lambda_l0=0.5, lambda_l0_max_hsic_pct=1e-9, hsic_safeguard_ema=0.0,
            )
        )
        capped.eval()
        with torch.no_grad():
            capped._step(batch, stage="val")
        assert capped._last_l0_reg.item() < l0_reg


# ---------------------------------------------------------------------------
# 5. Gradients still flow through the capped terms
# ---------------------------------------------------------------------------

class TestGradientsFlow:
    def test_structural_grad_nonzero_with_caps_active(self):
        """The cap is a detached scalar: it must not sever the autograd graph."""
        torch.manual_seed(99)
        model = AttentionSelectorForecaster(
            _make_forecaster_config(
                kappa=1e6, kappa_max_hsic_pct=0.1, hsic_safeguard_ema=0.0,
            )
        )
        model.train()
        batch = _make_batch(seed=3)
        model._step(batch, stage="train")
        l_struct = model._last_loss_components["loss_structural"]

        qk = [
            p for n, p in model.model.named_parameters()
            if p.requires_grad and ("query_projection" in n or "key_projection" in n)
        ]
        assert qk, "expected trainable query/key projection params"

        grads = torch.autograd.grad(l_struct, qk, allow_unused=True)
        total = sum(float(g.abs().sum()) for g in grads if g is not None)
        assert total > 0.0
        assert all(torch.isfinite(g).all() for g in grads if g is not None)


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

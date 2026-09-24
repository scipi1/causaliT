"""Tests for the MSE-vs-acyclicity cap (``training.mse_max_acyclic_pct``).

Run with:  pytest tests/test_mse_acyclic_cap.py -v

Background
----------
Without gradient routing the plain MSE can be minimised by predicting a
source from its DESCENDANTS (anti-causal edges also reduce the MSE),
fighting the acyclicity term.  The cap throttles the effective
reconstruction weight per step so the weighted MSE entering the loss never
exceeds ``mse_max_acyclic_pct`` times the EMA of the WEIGHTED acyclic
term::

    lambda_recon_eff = min(lambda_recon, pct * acy_ref / (mse + eps))

with ``acy_ref`` an EMA of ``acyclic_reg`` (detached, train batches only;
decay reuses ``hsic_safeguard_ema``).  ``mse_cap_release_tol`` releases the
cap once the acyclic EMA falls at/below the tolerance.

Guarantees under test
---------------------
1. Construction/validation: defaults off (backward compatible); negative
   values raise; mutually exclusive with use_gradient_routing and
   gradient_surgery.
2. ``_acyclic_safeguard_ref``: None when disabled; EMA updates on train
   batches only; instantaneous mode (ema=0) tracks the current batch.
3. End-to-end (``_step``): with a binding cap the reconstruction term
   entering total_loss equals exactly ``pct * acy_ref``; the release
   tolerance restores the full lambda_recon.
4. The cap is a detached scalar: gradients still flow through the loss.
5. Checkpoint roundtrip of the reference EMA.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from test_atsel_reg_safeguard import (  # noqa: E402
    _make_batch,
    _make_forecaster_config,
)

from causaliT.training.forecasters.attention_selector_forecaster import (  # noqa: E402
    AttentionSelectorForecaster,
)


def _cap_config(
    pct: float = 0.5,
    release_tol: float = 0.0,
    kappa: float = 1.0,
    ema: float = 0.0,
    **overrides,
):
    """Config with the MSE cap enabled (instantaneous reference by default)."""
    cfg = _make_forecaster_config(kappa=kappa, hsic_safeguard_ema=ema)
    cfg["training"]["mse_max_acyclic_pct"] = pct
    cfg["training"]["mse_cap_release_tol"] = release_tol
    cfg["training"].update(overrides)
    return cfg


def _step_losses(model, batch, stage="val"):
    """Run a deterministic _step; return (total_loss, loss_x, acyclic_reg)."""
    model.eval()
    with torch.no_grad():
        total, _, _ = model._step(batch, stage=stage)
    return (
        total.detach().double(),
        model._last_loss_components["loss_recon"].detach().double(),
        model._last_acyclic_reg.detach().double(),
    )


class TestConstruction:
    def test_defaults_off(self):
        model = AttentionSelectorForecaster(_make_forecaster_config())
        assert model.mse_max_acyclic_pct == 0.0
        assert model.mse_cap_release_tol == 0.0
        assert model._acyclic_reg_ema is None

    def test_values_stored(self):
        model = AttentionSelectorForecaster(_cap_config(pct=0.3, release_tol=1e-4))
        assert model.mse_max_acyclic_pct == pytest.approx(0.3)
        assert model.mse_cap_release_tol == pytest.approx(1e-4)

    @pytest.mark.parametrize(
        "key", ["mse_max_acyclic_pct", "mse_cap_release_tol"]
    )
    def test_negative_raises(self, key):
        cfg = _make_forecaster_config()
        cfg["training"][key] = -0.1
        with pytest.raises(ValueError, match=key):
            AttentionSelectorForecaster(cfg)

    def test_gradient_routing_conflict_raises(self):
        cfg = _cap_config(use_gradient_routing=True)
        with pytest.raises(ValueError, match="mse_max_acyclic_pct"):
            AttentionSelectorForecaster(cfg)

    def test_gradient_surgery_conflict_raises(self):
        cfg = _cap_config(gradient_surgery=True)
        with pytest.raises(ValueError, match="mse_max_acyclic_pct"):
            AttentionSelectorForecaster(cfg)



class TestSafeguardRef:
    def test_none_when_disabled(self):
        model = AttentionSelectorForecaster(_make_forecaster_config())
        ref = model._acyclic_safeguard_ref(torch.tensor(1.0), "train")
        assert ref is None
        assert model._acyclic_reg_ema is None

    def test_instantaneous_mode_tracks_batch(self):
        model = AttentionSelectorForecaster(_cap_config(ema=0.0))
        ref = model._acyclic_safeguard_ref(torch.tensor(2.0), "train")
        assert ref == pytest.approx(2.0)
        ref = model._acyclic_safeguard_ref(torch.tensor(4.0), "train")
        assert ref == pytest.approx(4.0)

    def test_ema_updates_train_only(self):
        model = AttentionSelectorForecaster(_cap_config(ema=0.5))
        ref = model._acyclic_safeguard_ref(torch.tensor(2.0), "train")
        assert ref == pytest.approx(2.0)
        ref = model._acyclic_safeguard_ref(torch.tensor(4.0), "train")
        assert ref == pytest.approx(0.5 * 2.0 + 0.5 * 4.0)
        # val batches never update the running reference.
        ref_val = model._acyclic_safeguard_ref(torch.tensor(100.0), "val")
        assert ref_val == pytest.approx(3.0)


class TestEndToEnd:
    def test_binding_cap_matches_algebra(self):
        """The recon term entering total_loss equals exactly pct * acy_ref."""
        torch.manual_seed(1234)
        cfg_off = _make_forecaster_config(kappa=1.0)
        model_off = AttentionSelectorForecaster(cfg_off)
        torch.manual_seed(1234)
        model_on = AttentionSelectorForecaster(_cap_config(pct=0.1))

        batch = _make_batch(seed=7)
        total_off, loss_x, acyclic = _step_losses(model_off, batch)
        total_on, loss_x_on, acyclic_on = _step_losses(model_on, batch)

        assert loss_x_on.item() == pytest.approx(loss_x.item(), rel=1e-6)
        assert acyclic_on.item() == pytest.approx(acyclic.item(), rel=1e-6)

        # Cap must actually bind for the check to be meaningful.
        assert 0.1 * acyclic.item() < loss_x.item()
        # total_on = total_off - loss_x + pct * acyclic (instantaneous ref).
        expected = total_off - loss_x + 0.1 * acyclic
        assert total_on.item() == pytest.approx(expected.item(), rel=1e-6)

    def test_release_tolerance_restores_lambda_recon(self):
        """acy_ref <= release_tol -> cap released, loss identical to uncapped."""
        torch.manual_seed(1234)
        model_off = AttentionSelectorForecaster(_make_forecaster_config(kappa=1.0))
        torch.manual_seed(1234)
        model_rel = AttentionSelectorForecaster(
            _cap_config(pct=0.1, release_tol=1.0e12)
        )
        batch = _make_batch(seed=7)
        total_off, _, _ = _step_losses(model_off, batch)
        total_rel, _, _ = _step_losses(model_rel, batch)
        assert total_rel.item() == pytest.approx(total_off.item(), rel=1e-6)

    def test_disabled_matches_no_feature(self):
        """pct=0.0 (default) is byte-identical to the pre-feature loss."""
        batch = _make_batch(seed=7)
        totals = []
        for explicit in (False, True):
            torch.manual_seed(1234)
            cfg = _make_forecaster_config(kappa=1.0)
            if explicit:
                cfg["training"]["mse_max_acyclic_pct"] = 0.0
                cfg["training"]["mse_cap_release_tol"] = 0.0
            model = AttentionSelectorForecaster(cfg)
            totals.append(_step_losses(model, batch)[0])
        assert totals[0].item() == pytest.approx(totals[1].item(), rel=1e-12)

    def test_cap_is_detached_scalar_gradients_flow(self):
        torch.manual_seed(99)
        model = AttentionSelectorForecaster(_cap_config(pct=0.1))
        model.train()
        total, _, _ = model._step(_make_batch(seed=3), stage="train")
        total.backward()
        grads = [
            p.grad for p in model.model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0.0


class TestCheckpointRoundtrip:
    def test_ema_survives_save_load(self):
        torch.manual_seed(0)
        model = AttentionSelectorForecaster(_cap_config())
        model._acyclic_reg_ema = 0.42
        ckpt = {"state_dict": model.state_dict()}
        model.on_save_checkpoint(ckpt)
        assert "mse_acyclic_cap" in ckpt

        fresh = AttentionSelectorForecaster(_cap_config())
        fresh.on_load_checkpoint(ckpt)
        assert fresh._acyclic_reg_ema == pytest.approx(0.42)

    def test_missing_key_keeps_init(self):
        fresh = AttentionSelectorForecaster(_cap_config())
        fresh.on_load_checkpoint({"state_dict": fresh.state_dict()})
        assert fresh._acyclic_reg_ema is None

    def test_not_saved_when_disabled(self):
        model = AttentionSelectorForecaster(_make_forecaster_config())
        ckpt = {"state_dict": model.state_dict()}
        model.on_save_checkpoint(ckpt)
        assert "mse_acyclic_cap" not in ckpt


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

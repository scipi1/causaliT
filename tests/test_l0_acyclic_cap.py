"""Tests for the L0-vs-acyclicity cap (``training.l0_max_acyclic_pct``).

Run with:  pytest tests/test_l0_acyclic_cap.py -v

Background
----------
In HSIC-free arms (``lambda_hsic=0``, e.g. the MSE+DAGMA investigations) a
FIXED ``lambda_l0`` is unopposed: every open HardConcrete gate receives
constant closing pressure and, integrated over a long run, the posterior
collapses to the empty graph (``train_l0_penalty`` decays monotonically to
~0).  The cap mirrors the MSE-vs-acyclicity cap with the roles reassigned::

    lambda_l0_eff = min(lambda_l0, pct * acy_ref / (l0 + eps))

with ``acy_ref`` the EMA of the WEIGHTED acyclic term (detached, train
batches only) -- the SAME reference the MSE cap uses, including the
``mse_cap_release_tol`` release.  As h -> 0 the L0 pressure fades with it,
so the gates freeze on an acyclic graph instead of being ground down.

Guarantees under test
---------------------
1. Construction/validation: defaults off (backward compatible); negative
   values raise; mutually exclusive with ``lambda_l0_max_hsic_pct`` (one
   anchor per coefficient) and with routing/surgery.
2. The acyclic EMA is maintained when ONLY the L0 cap is enabled.
3. End-to-end (``_step``): with a binding cap the L0 term equals exactly
   ``pct * acy_ref``; the release tolerance restores the full lambda_l0;
   disabled is byte-identical to the pre-feature loss.
4. The acyclic EMA round-trips through checkpoints with only the L0 cap on.
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
    _step_terms,
)

from causaliT.training.forecasters.attention_selector_forecaster import (  # noqa: E402
    AttentionSelectorForecaster,
)


def _l0cap_config(pct: float = 0.5, ema: float = 0.0, **overrides):
    """GatedCrossAttention config with the L0 acyclic cap enabled."""
    cfg = _make_forecaster_config(
        attention_type="GatedCrossAttention",
        kappa=1.0,
        lambda_l0=0.5,
        hsic_safeguard_ema=ema,
    )
    cfg["training"]["l0_max_acyclic_pct"] = pct
    cfg["training"].update(overrides)
    return cfg


class TestConstruction:
    def test_default_off(self):
        model = AttentionSelectorForecaster(_make_forecaster_config())
        assert model.l0_max_acyclic_pct == 0.0

    def test_value_stored(self):
        model = AttentionSelectorForecaster(_l0cap_config(pct=0.3))
        assert model.l0_max_acyclic_pct == pytest.approx(0.3)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="l0_max_acyclic_pct"):
            AttentionSelectorForecaster(_l0cap_config(pct=-0.1))

    def test_mutually_exclusive_with_hsic_anchor(self):
        cfg = _l0cap_config(pct=0.5, lambda_l0_max_hsic_pct=0.2)
        with pytest.raises(ValueError, match="mutually exclusive"):
            AttentionSelectorForecaster(cfg)

    @pytest.mark.parametrize(
        "flag", ["use_gradient_routing", "gradient_surgery"]
    )
    def test_mutually_exclusive_with_routing_and_surgery(self, flag):
        cfg = _l0cap_config(pct=0.5)
        cfg["training"][flag] = True
        with pytest.raises(ValueError, match="mutually exclusive"):
            AttentionSelectorForecaster(cfg)


class TestAcyclicEma:
    def test_ema_maintained_with_only_l0_cap(self):
        """The acyclic reference must exist even when the MSE cap is off."""
        model = AttentionSelectorForecaster(_l0cap_config(pct=0.5, ema=0.9))
        model.train()
        model._step(_make_batch(seed=3), stage="train")
        assert model._acyclic_reg_ema is not None

    def test_no_ema_when_all_caps_off(self):
        model = AttentionSelectorForecaster(
            _make_forecaster_config(attention_type="GatedCrossAttention")
        )
        model.train()
        model._step(_make_batch(seed=3), stage="train")
        assert model._acyclic_reg_ema is None


class TestEndToEnd:
    def test_binding_cap_makes_l0_reg_equal_pct_times_acy_ref(self):
        """Huge lambda_l0 + pct=0.1 => l0_reg == 0.1 * acyclic (inst. ref)."""
        torch.manual_seed(1234)
        model = AttentionSelectorForecaster(
            _l0cap_config(pct=0.1, lambda_l0=1e6)
        )
        _, acyclic, l0_reg = _step_terms(model, _make_batch(seed=7))
        assert acyclic.item() > 0.0, "cap must have a positive reference"
        assert l0_reg.item() == pytest.approx(0.1 * acyclic.item(), rel=1e-4)

    def test_pct_zero_leaves_l0_uncapped(self):
        """pct=0 => l0_reg == lambda_l0 * l0_penalty (pre-feature behaviour)."""
        batch = _make_batch(seed=7)
        torch.manual_seed(1234)
        uncapped = AttentionSelectorForecaster(_l0cap_config(pct=0.0))
        _, _, l0_uncapped = _step_terms(uncapped, batch)

        torch.manual_seed(1234)
        capped = AttentionSelectorForecaster(_l0cap_config(pct=1e-9))
        _, _, l0_capped = _step_terms(capped, batch)
        assert l0_uncapped.item() > 0.0
        assert l0_capped.item() < l0_uncapped.item()

    def test_release_tolerance_restores_lambda_l0(self):
        """acy_ref <= release_tol -> cap released, l0_reg == uncapped."""
        batch = _make_batch(seed=7)
        torch.manual_seed(1234)
        uncapped = AttentionSelectorForecaster(_l0cap_config(pct=0.0))
        _, _, l0_uncapped = _step_terms(uncapped, batch)

        torch.manual_seed(1234)
        released = AttentionSelectorForecaster(
            _l0cap_config(pct=0.1, mse_cap_release_tol=1.0e12)
        )
        _, _, l0_released = _step_terms(released, batch)
        assert l0_released.item() == pytest.approx(l0_uncapped.item(), rel=1e-6)

    def test_disabled_matches_no_feature(self):
        """pct=0.0 (default) is byte-identical to the pre-feature loss."""
        batch = _make_batch(seed=7)
        totals = []
        for explicit in (False, True):
            torch.manual_seed(1234)
            cfg = _l0cap_config(pct=0.0)
            if not explicit:
                del cfg["training"]["l0_max_acyclic_pct"]
            model = AttentionSelectorForecaster(cfg)
            model.eval()
            with torch.no_grad():
                total, _, _ = model._step(batch, stage="val")
            totals.append(total)
        assert totals[0].item() == pytest.approx(totals[1].item(), rel=1e-12)


class TestCheckpointRoundtrip:
    def test_ema_saved_with_only_l0_cap(self):
        model = AttentionSelectorForecaster(_l0cap_config())
        model._acyclic_reg_ema = 0.42
        ckpt = {"state_dict": model.state_dict()}
        model.on_save_checkpoint(ckpt)
        assert "mse_acyclic_cap" in ckpt

        fresh = AttentionSelectorForecaster(_l0cap_config())
        fresh.on_load_checkpoint(ckpt)
        assert fresh._acyclic_reg_ema == pytest.approx(0.42)

    def test_not_saved_when_disabled(self):
        model = AttentionSelectorForecaster(
            _make_forecaster_config(attention_type="GatedCrossAttention")
        )
        ckpt = {"state_dict": model.state_dict()}
        model.on_save_checkpoint(ckpt)
        assert "mse_acyclic_cap" not in ckpt


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])


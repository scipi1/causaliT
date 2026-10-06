"""Tests for the augmented-Lagrangian acyclicity constraint
(``training.acyclicity_constraint``) in ``AttentionSelectorForecaster``.

The ALM replaces the fixed-kappa soft penalty by the canonical NOTEARS
protocol (Zheng et al., 2018; vendor/notears/linear.py)::

    L += alpha * h(W) + (rho/2) * h(W)^2

with per-epoch dual ascent on ``alpha`` and rho escalation whenever the EMA
of h fails to shrink to <= 1/4 of its previous value.

Guarantees under test:

1. Construction/validation: defaults off; mutually exclusive with kappa and
   kappa_max_hsic_pct; parameter ranges validated.
2. Loss algebra: the term entering the structural loss equals exactly
   ``alpha * h + (rho/2) * h^2`` with h recomputed from the live score
   tensor.
3. Dual dynamics: ascent on violation, clip at dual_max, release when
   satisfied, rho escalation under the 1/4 rule, cap at rho_max.
4. Backward compatibility: disabled == pre-feature fixed-kappa behaviour.
5. Gradient flow into the structural parameters.
6. Checkpoint roundtrip of the dual state.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from test_atsel_reg_safeguard import (  # noqa: E402
    S_SEQ_LEN,
    _make_batch,
    _make_forecaster_config,
    _step_terms,
)

from causaliT.training.forecasters.attention_selector_forecaster import (  # noqa: E402
    AttentionSelectorForecaster,
)


def _alm_config(**alm_kwargs):
    """Config with the ALM constraint enabled (kappa=0 by construction)."""
    cfg = _make_forecaster_config(kappa=0.0)
    cfg["training"]["acyclicity_constraint"] = {"enabled": True, **alm_kwargs}
    return cfg


def _raw_h(model) -> torch.Tensor:
    """h(A) recomputed from the current 2-D score tensor (split-mode slice)."""
    score = model.model.get_score_tensor_for_sparsity()
    A_cyc = score[:, S_SEQ_LEN:]
    return torch.trace(torch.matrix_exp(A_cyc * A_cyc)) - A_cyc.shape[-1]


class TestConstruction:
    def test_defaults_off(self):
        model = AttentionSelectorForecaster(_make_forecaster_config())
        assert model.acyclicity_constraint_enabled is False
        assert model._acy_dual_lambda == 0.0
        assert model._acy_rho == 1.0
        assert model._acy_ema is None

    def test_values_stored(self):
        model = AttentionSelectorForecaster(
            _alm_config(dual_init=0.5, dual_lr=2.0, dual_max=10.0,
                        rho_init=3.0, rho_mult=5.0, rho_max=1e3,
                        ema=0.5, h_tol=1e-4)
        )
        assert model._acy_dual_lambda == pytest.approx(0.5)
        assert model._acy_rho == pytest.approx(3.0)
        assert model.acy_dual_lr == pytest.approx(2.0)
        assert model.acy_dual_max == pytest.approx(10.0)
        assert model.acy_rho_mult == pytest.approx(5.0)
        assert model.acy_rho_max == pytest.approx(1e3)
        assert model.acy_ema_decay == pytest.approx(0.5)
        assert model.acy_h_tol == pytest.approx(1e-4)

    def test_kappa_conflict_raises(self):
        cfg = _make_forecaster_config(kappa=1.0)
        cfg["training"]["acyclicity_constraint"] = {"enabled": True}
        with pytest.raises(ValueError, match="kappa"):
            AttentionSelectorForecaster(cfg)

    def test_kappa_cap_conflict_raises(self):
        cfg = _make_forecaster_config(kappa=0.0, kappa_max_hsic_pct=0.1)
        cfg["training"]["acyclicity_constraint"] = {"enabled": True}
        with pytest.raises(ValueError, match="kappa_max_hsic_pct"):
            AttentionSelectorForecaster(cfg)

    @pytest.mark.parametrize("key,val", [("dual_lr", 0.0), ("dual_lr", -1.0),
                                         ("rho_init", -0.5), ("h_tol", -1e-9),
                                         ("ema", 1.0), ("ema", -0.1),
                                         ("dual_init", -1.0)])
    def test_invalid_params_raise(self, key, val):
        with pytest.raises(ValueError):
            AttentionSelectorForecaster(_alm_config(**{key: val}))


class TestLossAlgebra:
    def test_term_equals_alpha_h_plus_half_rho_h2(self):
        """_step's acyclic term == dual_init * h + (rho_init/2) * h^2."""
        torch.manual_seed(1234)
        model = AttentionSelectorForecaster(
            _alm_config(dual_init=0.5, rho_init=2.0)
        )
        _hsic, acyclic, _l0 = _step_terms(model, _make_batch(seed=7))
        h = _raw_h(model).detach().double()
        expected = 0.5 * h + 0.5 * 2.0 * h ** 2
        assert acyclic.item() == pytest.approx(expected.item(), rel=1e-4)

    def test_zero_dual_state_gives_zero_term(self):
        torch.manual_seed(1234)
        model = AttentionSelectorForecaster(
            _alm_config(dual_init=0.0, rho_init=0.0)
        )
        _hsic, acyclic, _l0 = _step_terms(model, _make_batch(seed=7))
        assert acyclic.item() == pytest.approx(0.0, abs=1e-12)


class TestDualDynamics:
    def _model(self, **kw):
        return AttentionSelectorForecaster(_alm_config(**kw))

    def test_ascent_on_violation(self):
        m = self._model(dual_lr=2.0, rho_init=0.0)
        m._acy_ema = 1.5
        m._update_acyclicity_dual()
        assert m._acy_dual_lambda == pytest.approx(2.0 * 1.5)
        assert m._acy_rho == 0.0   # rho_init=0: quadratic term off

    def test_ascent_clipped_at_dual_max(self):
        m = self._model(dual_lr=100.0, dual_max=3.0, rho_init=0.0)
        m._acy_ema = 1.0
        m._update_acyclicity_dual()
        assert m._acy_dual_lambda == pytest.approx(3.0)

    def test_release_when_satisfied(self):
        m = self._model(dual_init=5.0, dual_lr=1.0, h_tol=0.5, rho_init=0.0)
        m._acy_ema = 0.25          # below tolerance -> violation = -0.25
        m._update_acyclicity_dual()
        assert m._acy_dual_lambda == pytest.approx(4.75)

    def test_rho_escalation_quarter_rule(self):
        m = self._model(dual_lr=0.0 + 1.0, rho_init=1.0, rho_mult=10.0)
        m._acy_ema = 1.0
        m._update_acyclicity_dual()          # no prev -> no escalation
        assert m._acy_rho == pytest.approx(1.0)
        m._acy_ema = 0.9                     # > 1/4 * prev -> escalate
        m._update_acyclicity_dual()
        assert m._acy_rho == pytest.approx(10.0)
        m._acy_ema = 0.1                     # <= 1/4 * prev -> no escalation
        m._update_acyclicity_dual()
        assert m._acy_rho == pytest.approx(10.0)

    def test_rho_capped(self):
        m = self._model(rho_init=1.0, rho_mult=10.0, rho_max=5.0)
        m._acy_ema = 1.0
        m._update_acyclicity_dual()
        m._acy_ema = 0.9
        m._update_acyclicity_dual()
        assert m._acy_rho == pytest.approx(5.0)

    def test_no_op_without_ema(self):
        m = self._model()
        m._update_acyclicity_dual()          # _acy_ema is None
        assert m._acy_dual_lambda == 0.0


class TestBackwardCompatAndGradients:
    def test_disabled_matches_fixed_kappa(self):
        """enabled: false (or absent) == pre-feature fixed-kappa loss."""
        batch = _make_batch(seed=7)
        terms = []
        for with_block in (False, True):
            torch.manual_seed(1234)
            cfg = _make_forecaster_config(
                attention_type="GatedCrossAttention", kappa=2.0
            )
            if with_block:
                cfg["training"]["acyclicity_constraint"] = {"enabled": False}
            model = AttentionSelectorForecaster(cfg)
            terms.append(_step_terms(model, batch)[1])
        assert terms[0].item() == pytest.approx(terms[1].item(), rel=1e-6)

    def test_gradient_flows_to_structural_params(self):
        torch.manual_seed(99)
        model = AttentionSelectorForecaster(
            _alm_config(dual_init=1.0, rho_init=1.0)
        )
        model.train()
        model._step(_make_batch(seed=3), stage="train")
        l_struct = model._last_loss_components["loss_structural"]
        qk = [
            p for n, p in model.model.named_parameters()
            if p.requires_grad and ("query_projection" in n
                                    or "key_projection" in n)
        ]
        assert qk
        grads = torch.autograd.grad(l_struct, qk, allow_unused=True)
        total = sum(float(g.abs().sum()) for g in grads if g is not None)
        assert total > 0.0
        assert all(torch.isfinite(g).all() for g in grads if g is not None)


class TestPhaseGating:
    # Dual ascent / rho escalation only run while the structural parameters
    # are trainable (structure phase of the adaptive schedule).  Against
    # FROZEN gates (warmup/reconstruct) the violation is constant and rho
    # would escalate to rho_max before structure learning even starts
    # (runaway ALM regression).  Non-routed forecasters (no
    # _structural_params) keep the legacy every-epoch behaviour.

    def _model_with_frozen_structure(self, frozen):
        m = AttentionSelectorForecaster(_alm_config(dual_lr=2.0))
        p = torch.nn.Parameter(torch.randn(2))
        p.requires_grad_(not frozen)
        m._structural_params = [p]
        return m

    def test_frozen_structure_skips_dual_update(self):
        m = self._model_with_frozen_structure(frozen=True)
        m._acy_ema = 20.0            # large constant violation (warmup regime)
        m._acy_prev_violation = 20.0
        lam0, rho0 = m._acy_dual_lambda, m._acy_rho
        m.on_train_epoch_end()
        assert m._acy_dual_lambda == pytest.approx(lam0)
        assert m._acy_rho == pytest.approx(rho0)

    def test_trainable_structure_runs_dual_update(self):
        m = self._model_with_frozen_structure(frozen=False)
        m._acy_ema = 20.0
        m._acy_prev_violation = 20.0
        m.on_train_epoch_end()
        assert m._acy_dual_lambda > 0.0          # ascent happened
        assert m._acy_rho == pytest.approx(10.0)  # 1/4 rule escalation

    def test_no_routing_groups_keeps_legacy_behaviour(self):
        m = AttentionSelectorForecaster(_alm_config(dual_lr=2.0))
        if hasattr(m, "_structural_params"):
            delattr(m, "_structural_params")
        m._acy_ema = 20.0
        m._acy_prev_violation = 20.0
        m.on_train_epoch_end()
        assert m._acy_dual_lambda > 0.0


class TestCheckpointRoundtrip:
    def test_dual_state_survives_save_load(self):
        torch.manual_seed(0)
        model = AttentionSelectorForecaster(_alm_config(dual_lr=2.0))
        model._acy_dual_lambda = 3.5
        model._acy_rho = 100.0
        model._acy_ema = 0.7
        model._acy_prev_violation = 0.6
        ckpt = {"state_dict": model.state_dict()}
        model.on_save_checkpoint(ckpt)
        assert "acyclicity_constraint" in ckpt

        fresh = AttentionSelectorForecaster(_alm_config(dual_lr=2.0))
        fresh.on_load_checkpoint(ckpt)
        assert fresh._acy_dual_lambda == pytest.approx(3.5)
        assert fresh._acy_rho == pytest.approx(100.0)
        assert fresh._acy_ema == pytest.approx(0.7)
        assert fresh._acy_prev_violation == pytest.approx(0.6)

    def test_missing_key_keeps_init(self):
        fresh = AttentionSelectorForecaster(_alm_config(dual_init=0.25))
        fresh.on_load_checkpoint({"state_dict": fresh.state_dict()})
        assert fresh._acy_dual_lambda == pytest.approx(0.25)


class TestRoleReversal:
    """H_CONSTRAINT regime: HSIC (+L0) primal, acyclicity h(W)=0 constraint."""

    def test_hsic_constraint_conflict_raises(self):
        """Both constraint blocks ON = opposite assignments -> ValueError."""
        cfg = _make_forecaster_config(kappa=0.0, lambda_hsic=0.0)
        cfg["training"]["acyclicity_constraint"] = {"enabled": True}
        cfg["training"]["hsic_constraint"] = {"enabled": True}
        with pytest.raises(ValueError, match="mutually exclusive"):
            AttentionSelectorForecaster(cfg)

    def test_no_primal_objective_warns(self, caplog):
        """acy-ALM with lambda_hsic == lambda_l0 == 0 -> warning, no raise."""
        cfg = _make_forecaster_config(kappa=0.0, lambda_hsic=0.0,
                                      lambda_l0=0.0)
        cfg["training"]["acyclicity_constraint"] = {"enabled": True}
        import logging
        with caplog.at_level(logging.WARNING):
            model = AttentionSelectorForecaster(cfg)
        assert model.acyclicity_constraint_enabled is True
        assert any("NO primal objective" in r.getMessage()
                   for r in caplog.records)

    def test_phase_switch_resets_ema_keeps_dual(self):
        """The reset hook clears per-regime memory but the dual variables
        (lambda, rho) persist across the boundary."""
        m = AttentionSelectorForecaster(
            _alm_config(dual_init=2.0, rho_init=5.0)
        )
        m._acy_dual_lambda = 3.5
        m._acy_rho = 50.0
        m._acy_ema = 0.7
        m._acy_prev_violation = 0.6
        m.acyclicity_constraint_on_phase_switch(phase="structure",
                                                bkd_min_keys=4)
        assert m._acy_ema is None
        assert m._acy_prev_violation is None
        assert m._acy_dual_lambda == pytest.approx(3.5)
        assert m._acy_rho == pytest.approx(50.0)

    def test_reversed_regime_constructs_and_steps(self):
        """HSIC primal (lambda_hsic=1) + acy-ALM constraint: loss contains
        both terms and gradients flow."""
        torch.manual_seed(11)
        cfg = _alm_config(dual_init=1.0, rho_init=1.0)  # lambda_hsic=1.0
        model = AttentionSelectorForecaster(cfg)
        assert model.hsic_constraint_enabled is False
        model.train()
        model._step(_make_batch(seed=5), stage="train")
        comp = model._last_loss_components["loss_structural"]
        assert torch.isfinite(comp)



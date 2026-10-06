"""
Tests for the dense-adjacency edge-dropout override (adaptive trainer).

Semantics under test
--------------------
When a module's ``set_dense_gate_mode(True)`` is active AND it is in training
mode, the applied adjacency A is replaced by a fully dense {0,1} sample:
every (allowed, off-diagonal) edge ij is KEPT with probability equal to the
DETACHED structural gate posterior p_ij (dropped w.p. 1 - p_ij), so
E[A_ij] = p_ij.  Eval passes are unaffected (learned gates), the returned
posterior / L0 / diagnostics stay gate-based, and no structural gradient
leaks through the reconstruction loss.

Covers:
1. GatedCrossAttention: default off, {0,1} applied adjacency, hard-mask
   respected, keep frequency ~= posterior, eval unchanged, no grad leakage,
   toggle-off restores behaviour.
2. GatedSelfAttention: same contracts + zero diagonal.
3. PhaseController._apply_dense_gate_cfg: active in warmup / reconstruct /
   final_reconstruct / structure when the phase block sets
   ``dense_adjacency_edge_dropout: true``; off when unset.  The eval probe
   (``dense_gate_eval``) defaults ON in structure phases only.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from causaliT.core.modules.gated_cross_attention import GatedCrossAttention
from causaliT.core.modules.gated_self_attention import GatedSelfAttention
from causaliT.training.adaptive_trainer import PhaseController


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

D_QK = 16
D_MODEL = 8
L_X, L_S, N = 4, 3, 5


def _gca_inputs(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, L_X, D_QK, generator=g)
    k = torch.randn(B, L_S, D_QK, generator=g)
    v = torch.randn(B, L_S, D_MODEL, generator=g)
    return q, k, v


def _gsa_inputs(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, N, D_QK, generator=g)
    k = torch.randn(B, N, D_QK, generator=g)
    v = torch.randn(B, N, D_MODEL, generator=g)
    return q, k, v


def _gsa_forward(mod, q, k, v, hard_mask=None):
    return mod(
        query=q, key=k, value=v,
        mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
        hard_mask=hard_mask, oracle=False,
    )


def _assert_binary(A):
    assert torch.all((A == 0.0) | (A == 1.0)), "applied adjacency must be {0,1}"


# ===========================================================================
# 1. GatedCrossAttention
# ===========================================================================

class TestDenseGateCrossAttention:
    def test_disabled_by_default(self):
        att = GatedCrossAttention()
        assert att._dense_gate_active is False
        att.train()
        q, k, v = _gca_inputs()
        att(q, k, v)
        A = att.last_applied_A
        # Learned-gate applied weight: continuous in [0, 1], not {0,1}.
        assert not torch.all((A == 0.0) | (A == 1.0))

    def test_applied_adjacency_is_binary(self):
        att = GatedCrossAttention()
        att.set_dense_gate_mode(True)
        att.train()
        q, k, v = _gca_inputs()
        att(q, k, v)
        _assert_binary(att.last_applied_A)

    def test_hard_mask_stays_zero(self):
        att = GatedCrossAttention()
        att.set_dense_gate_mode(True)
        att.train()
        q, k, v = _gca_inputs()
        hard_mask = torch.zeros(L_X, L_S)
        hard_mask[:, 0] = 1.0                      # only key 0 allowed
        att(q, k, v, hard_mask=hard_mask)
        A = att.last_applied_A
        _assert_binary(A)
        assert torch.all(A[..., 1:] == 0.0)

    def test_keep_frequency_matches_posterior(self):
        torch.manual_seed(0)
        att = GatedCrossAttention()
        att.set_dense_gate_mode(True)
        att.train()
        B = 4096
        q, k, v = _gca_inputs(B=B, seed=1)
        _, posterior, _ = att(q, k, v)             # (B, L, S) gate posterior
        freq = att.last_applied_A.mean(dim=0)      # (L, S) keep frequency
        target = posterior.mean(dim=0)
        assert torch.allclose(freq, target, atol=0.05)

    def test_eval_uses_learned_gate(self):
        att = GatedCrossAttention()
        att.eval()
        q, k, v = _gca_inputs()
        att(q, k, v)
        ref = att.last_applied_A.clone()
        att.set_dense_gate_mode(True)
        att(q, k, v)
        assert torch.equal(att.last_applied_A, ref)

    def test_no_structural_gradient_leak(self):
        att = GatedCrossAttention()
        att.set_dense_gate_mode(True)
        att.train()
        q, k, v = _gca_inputs()
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        out, _, _ = att(q, k, v)
        out.sum().backward()
        # The value stream receives gradient; the structural q/k must not.
        assert v.grad is not None and torch.any(v.grad != 0.0)
        assert q.grad is None or torch.all(q.grad == 0.0)
        assert k.grad is None or torch.all(k.grad == 0.0)

    def test_toggle_off_restores_learned_gate(self):
        att = GatedCrossAttention()
        att.train()
        q, k, v = _gca_inputs()
        att.set_dense_gate_mode(True)
        att(q, k, v)
        _assert_binary(att.last_applied_A)
        att.set_dense_gate_mode(False)
        att(q, k, v)
        assert not torch.all(
            (att.last_applied_A == 0.0) | (att.last_applied_A == 1.0)
        )


# ===========================================================================
# 2. GatedSelfAttention
# ===========================================================================

class TestDenseGateSelfAttention:
    def test_disabled_by_default(self):
        mod = GatedSelfAttention()
        assert mod._dense_gate_active is False

    def test_applied_adjacency_is_binary_with_zero_diagonal(self):
        mod = GatedSelfAttention()
        mod.set_dense_gate_mode(True)
        mod.train()
        q, k, v = _gsa_inputs()
        _gsa_forward(mod, q, k, v)
        A = mod.last_applied_A
        _assert_binary(A)
        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        assert torch.all(diag == 0.0)

    def test_hard_mask_stays_zero(self):
        mod = GatedSelfAttention()
        mod.set_dense_gate_mode(True)
        mod.train()
        q, k, v = _gsa_inputs()
        hard_mask = torch.ones(N, N)
        hard_mask[:, 2:] = 0.0                     # only keys 0,1 allowed
        _gsa_forward(mod, q, k, v, hard_mask=hard_mask)
        A = mod.last_applied_A
        _assert_binary(A)
        assert torch.all(A[..., 2:] == 0.0)

    def test_keep_frequency_matches_posterior(self):
        torch.manual_seed(0)
        mod = GatedSelfAttention()
        mod.set_dense_gate_mode(True)
        mod.train()
        B = 4096
        q, k, v = _gsa_inputs(B=B, seed=1)
        _, posterior, _ = _gsa_forward(mod, q, k, v)   # directed posterior
        freq = mod.last_applied_A.mean(dim=0)
        target = posterior.mean(dim=0)
        assert torch.allclose(freq, target, atol=0.05)

    def test_eval_uses_learned_gate(self):
        mod = GatedSelfAttention()
        mod.eval()
        q, k, v = _gsa_inputs()
        _gsa_forward(mod, q, k, v)
        ref = mod.last_applied_A.clone()
        mod.set_dense_gate_mode(True)
        _gsa_forward(mod, q, k, v)
        assert torch.equal(mod.last_applied_A, ref)

    def test_no_structural_gradient_leak(self):
        mod = GatedSelfAttention()
        mod.set_dense_gate_mode(True)
        mod.train()
        q, k, v = _gsa_inputs()
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        out, _, _ = _gsa_forward(mod, q, k, v)
        out.sum().backward()
        # The value stream receives gradient; the structural q/k must not.
        assert v.grad is not None and torch.any(v.grad != 0.0)
        assert q.grad is None or torch.all(q.grad == 0.0)
        assert k.grad is None or torch.all(k.grad == 0.0)


# ===========================================================================
# 3. PhaseController integration
# ===========================================================================

class _DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross = GatedCrossAttention()
        self.self_att = GatedSelfAttention()
        self.logged = {}

    def log(self, name, value, on_step=False, on_epoch=True):
        self.logged[name] = value


@pytest.fixture
def work_dir():
    root = Path("_pytest_tmp")
    root.mkdir(exist_ok=True)
    d = tempfile.mkdtemp(dir=root)
    yield Path(d)


def _make_controller(work_dir, recon_cfg=None, final_cfg=None, struct_cfg=None):
    config = {
        "model": {"model_object": "AttentionSelectorLayer"},
        "adaptive_training": {
            "reconstruct": recon_cfg or {},
            "final_reconstruct": final_cfg or {},
            "structure": struct_cfg or {},
        },
    }
    return PhaseController(
        config=config,
        data_dir=str(work_dir),
        save_dir=str(work_dir / "out"),
        cluster=True,
    )


class TestDenseGatePhaseControl:
    def test_reconstruct_activates_structure_deactivates(self, work_dir):
        ctl = _make_controller(
            work_dir, recon_cfg={"dense_adjacency_edge_dropout": True}
        )
        mod = _DummyModule()

        ctl._apply_dense_gate_cfg(mod, "reconstruct")
        assert mod.cross._dense_gate_active is True
        assert mod.self_att._dense_gate_active is True
        assert mod.logged["dense_gate_active"] == 1.0

        ctl._apply_dense_gate_cfg(mod, "structure")
        assert mod.cross._dense_gate_active is False
        assert mod.self_att._dense_gate_active is False
        assert mod.logged["dense_gate_active"] == 0.0

    def test_warmup_never_activates(self, work_dir):
        ctl = _make_controller(
            work_dir, recon_cfg={"dense_adjacency_edge_dropout": True}
        )
        mod = _DummyModule()
        ctl._apply_dense_gate_cfg(mod, "warmup")
        assert mod.cross._dense_gate_active is False

    def test_warmup_activates_with_own_flag(self, work_dir):
        ctl = _make_controller(work_dir)
        ctl.warmup_cfg = {"dense_adjacency_edge_dropout": True}
        mod = _DummyModule()
        ctl._apply_dense_gate_cfg(mod, "warmup")
        assert mod.cross._dense_gate_active is True
        assert mod.self_att._dense_gate_active is True
        # Eval probe stays off in warmup (train-only dense sampling).
        assert mod.cross._dense_gate_eval is False

    def test_final_reconstruct_inherits_reconstruct_block(self, work_dir):
        ctl = _make_controller(
            work_dir, recon_cfg={"dense_adjacency_edge_dropout": True}
        )
        mod = _DummyModule()
        ctl._apply_dense_gate_cfg(mod, "final_reconstruct")
        assert mod.cross._dense_gate_active is True

    def test_final_reconstruct_can_override_inherited(self, work_dir):
        ctl = _make_controller(
            work_dir,
            recon_cfg={"dense_adjacency_edge_dropout": True},
            final_cfg={"dense_adjacency_edge_dropout": False},
        )
        mod = _DummyModule()
        ctl._apply_dense_gate_cfg(mod, "final_reconstruct")
        assert mod.cross._dense_gate_active is False

    def test_unset_key_is_off(self, work_dir):
        ctl = _make_controller(work_dir, recon_cfg={})
        mod = _DummyModule()
        ctl._apply_dense_gate_cfg(mod, "reconstruct")
        assert mod.cross._dense_gate_active is False
        assert mod.self_att._dense_gate_active is False

    def test_structure_activates_with_eval_probe_by_default(self, work_dir):
        ctl = _make_controller(
            work_dir, struct_cfg={"dense_adjacency_edge_dropout": True}
        )
        mod = _DummyModule()
        ctl._apply_dense_gate_cfg(mod, "structure")
        assert mod.cross._dense_gate_active is True
        assert mod.self_att._dense_gate_active is True
        # Structure phase: eval probe on by default (drift monitoring).
        assert mod.cross._dense_gate_eval is True
        assert mod.self_att._dense_gate_eval is True
        assert mod.logged["dense_gate_active"] == 1.0

    def test_structure_eval_probe_can_be_disabled(self, work_dir):
        ctl = _make_controller(
            work_dir,
            struct_cfg={
                "dense_adjacency_edge_dropout": True,
                "dense_gate_eval": False,
            },
        )
        mod = _DummyModule()
        ctl._apply_dense_gate_cfg(mod, "structure")
        assert mod.cross._dense_gate_active is True
        assert mod.cross._dense_gate_eval is False

    def test_structure_without_flag_trains_relaxed_but_probes_binary(self, work_dir):
        # Structure phase default: no train-time override (the HSIC residual
        # keeps its differentiable path through the relaxed gate) but the
        # eval probe is ON (drift trigger measures binary-gate damage).
        ctl = _make_controller(work_dir, struct_cfg={})
        mod = _DummyModule()
        ctl._apply_dense_gate_cfg(mod, "structure")
        assert mod.cross._dense_gate_active is False
        assert mod.cross._dense_gate_eval is True

    def test_structure_eval_probe_can_be_fully_disabled(self, work_dir):
        ctl = _make_controller(
            work_dir, struct_cfg={"dense_gate_eval": False}
        )
        mod = _DummyModule()
        ctl._apply_dense_gate_cfg(mod, "structure")
        assert mod.cross._dense_gate_active is False
        assert mod.cross._dense_gate_eval is False

    def test_reconstruct_eval_probe_off_by_default(self, work_dir):
        ctl = _make_controller(
            work_dir, recon_cfg={"dense_adjacency_edge_dropout": True}
        )
        mod = _DummyModule()
        ctl._apply_dense_gate_cfg(mod, "reconstruct")
        assert mod.cross._dense_gate_active is True
        assert mod.cross._dense_gate_eval is False

    def test_deactivation_clears_eval_flag(self, work_dir):
        ctl = _make_controller(
            work_dir, struct_cfg={"dense_adjacency_edge_dropout": True}
        )
        mod = _DummyModule()
        ctl._apply_dense_gate_cfg(mod, "structure")
        assert mod.cross._dense_gate_eval is True
        ctl._apply_dense_gate_cfg(mod, "reconstruct")
        assert mod.cross._dense_gate_active is False
        assert mod.cross._dense_gate_eval is False


# ===========================================================================
# 4. Eval-mode dense probe (module level)
# ===========================================================================

class TestDenseGateEvalProbe:
    def test_eval_uses_learned_gate_by_default(self):
        att = GatedCrossAttention()
        att.set_dense_gate_mode(True)
        att.eval()
        q, k, v = _gca_inputs()
        att(q, k, v)
        A = att.last_applied_A
        # Eval without the probe flag: continuous learned gate.
        assert not torch.all((A == 0.0) | (A == 1.0))

    def test_eval_probe_samples_binary_adjacency(self):
        att = GatedCrossAttention()
        att.set_dense_gate_mode(True, eval_mode=True)
        att.eval()
        q, k, v = _gca_inputs()
        att(q, k, v)
        _assert_binary(att.last_applied_A)

    def test_eval_probe_is_reproducible_per_step(self):
        att = GatedCrossAttention()
        att.set_dense_gate_mode(True, eval_mode=True)
        att.eval()
        q, k, v = _gca_inputs()
        att._dense_eval_step.zero_()
        att(q, k, v)
        A1 = att.last_applied_A.clone()
        att._dense_eval_step.zero_()
        att(q, k, v)
        A2 = att.last_applied_A.clone()
        assert torch.equal(A1, A2)

    def test_eval_probe_does_not_consume_training_rng(self):
        att = GatedCrossAttention()
        att.set_dense_gate_mode(True, eval_mode=True)
        att.eval()
        q, k, v = _gca_inputs()
        torch.manual_seed(123)
        att(q, k, v)
        after_probe = torch.rand(4)
        torch.manual_seed(123)
        without_probe = torch.rand(4)
        assert torch.equal(after_probe, without_probe)

    def test_eval_probe_self_attention_binary(self):
        mod = GatedSelfAttention()
        mod.set_dense_gate_mode(True, eval_mode=True)
        mod.eval()
        q, k, v = _gsa_inputs()
        _gsa_forward(mod, q, k, v)
        _assert_binary(mod.last_applied_A)

    def test_probe_without_train_override(self):
        # Structure-phase regime: relaxed-gate TRAINING, binary EVAL probe.
        att = GatedCrossAttention()
        att.set_dense_gate_mode(False, eval_mode=True)
        q, k, v = _gca_inputs()
        att.train()
        att(q, k, v)
        A_train = att.last_applied_A
        # Relaxed gate: continuous in [0, 1], not binary.
        assert not torch.all((A_train == 0.0) | (A_train == 1.0))
        att.eval()
        att(q, k, v)
        _assert_binary(att.last_applied_A)


# ===========================================================================
# 5. Entry-anchored reconstruction-drift trigger (structure phase)
# ===========================================================================

class _FakeTrainer:
    def __init__(self, epoch=0):
        self.current_epoch = epoch
        self.callback_metrics = {}
        self.sanity_checking = False
        self.max_epochs = 1000
        self.should_stop = False
        self.optimizers = []


class _FakePLModule(_DummyModule):
    def __init__(self):
        super().__init__()
        self._structural_params = [torch.nn.Parameter(torch.randn(2))]
        self._reconstruction_params = [torch.nn.Parameter(torch.randn(2))]


def _feed_structure(ctl, events, values, start_epoch=0):
    trainer = _FakeTrainer()
    module = _FakePLModule()
    ctl._apply_phase(trainer, module, "structure")
    # _apply_phase resets the entry snapshot; it is taken on the first
    # validation epoch of the phase.
    assert ctl._phase_entry_monitor is None
    for i, v in enumerate(values):
        trainer.current_epoch = start_epoch + i
        trainer.callback_metrics = {"val_loss_x": v}
        ctl.on_validation_epoch_end(trainer, module)
        struct_events = [e for e in events if e["from_phase"] == "structure"]
        if struct_events:
            return struct_events[0]
    return None


class TestStructReconDriftTrigger:
    def _ctl(self, work_dir, **overrides):
        struct = {"max_epochs": 200, "drop_pct": 0.20, "drop_patience": 2}
        struct.update(overrides)
        ctl = _make_controller(work_dir, struct_cfg=struct)
        ctl.monitor = "val_loss_x"
        ctl.transitions = []
        ctl._record_transition = (
            lambda tr, pl, reason, from_phase, to_phase, monitor_val:
            ctl.transitions.append({
                "reason": reason, "from_phase": from_phase,
                "to_phase": to_phase, "monitor": monitor_val,
            })
        )
        return ctl

    def test_drift_fires_against_entry_snapshot(self, work_dir):
        ctl = self._ctl(work_dir)
        # Entry = 1.0; sustained 1.3 (> 1.0 + 0.2) -> fires after patience 2.
        event = _feed_structure(ctl, ctl.transitions, [1.0, 1.3, 1.3, 1.3])
        assert event is not None
        assert event["reason"] == "struct_recon_drift"

    def test_mid_phase_improvement_does_not_tighten_baseline(self, work_dir):
        ctl = self._ctl(work_dir)
        # Entry = 1.0, improves to 0.5 mid-phase, then 1.15: above the old
        # running-best threshold (0.5*1.2 = 0.6) but BELOW the entry-anchored
        # one (1.2) -> must NOT trigger.
        event = _feed_structure(
            ctl, ctl.transitions, [1.0, 0.5, 0.5, 1.15, 1.15, 1.15]
        )
        assert event is None

    def test_sign_safe_negative_metric(self, work_dir):
        ctl = self._ctl(work_dir)
        # NLL-like negative monitor: entry -2.0, degrade to -1.5
        # (drift = +0.5 > 0.2*|-2.0| = 0.4) -> fires after patience 2.
        event = _feed_structure(ctl, ctl.transitions, [-2.0, -1.5, -1.5, -1.5])
        assert event is not None
        assert event["reason"] == "struct_recon_drift"

    def test_no_drift_no_switch(self, work_dir):
        ctl = self._ctl(work_dir)
        event = _feed_structure(ctl, ctl.transitions, [1.0, 1.05, 1.1, 1.1, 1.15])
        assert event is None


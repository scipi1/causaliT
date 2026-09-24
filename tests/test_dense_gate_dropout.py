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
3. PhaseController._apply_dense_gate_cfg: active in reconstruct /
   final_reconstruct when the phase block sets
   ``dense_adjacency_edge_dropout: true``; always off in structure/warmup.
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


def _make_controller(work_dir, recon_cfg=None, final_cfg=None):
    config = {
        "model": {"model_object": "AttentionSelectorLayer"},
        "adaptive_training": {
            "reconstruct": recon_cfg or {},
            "final_reconstruct": final_cfg or {},
            "structure": {},
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

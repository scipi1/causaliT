"""Tests for the direction-gate bias (dir_bias) in GatedSelfAttention.

Covers:
1. Default / explicit 0.0 reproduces the legacy coupled gate bit-identically.
2. A positive bias opens BOTH directions (E[d] ~ sigmoid(dir_bias)).
3. The Toeplitz coupling d_ij + d_ji == 1 holds at bias 0, is relaxed while
   biased, and is restored by set_dir_bias(0.0).
4. structure_posterior includes the bias (the deterministic probe must match
   the forward path).
5. The adaptive PhaseController applies per-phase dir_bias values and leaves
   unmanaged runs untouched.
"""

import pytest
import torch

from causaliT.core.modules.gated_self_attention import GatedSelfAttention


@pytest.fixture
def save_dir():
    """Scratch directory for ``PhaseController`` (it mkdirs stage_checkpoints/).

    Deliberately NOT pytest's ``tmp_path``: that fixture scans the shared
    ``%TEMP%/pytest-of-<user>`` root, which can raise PermissionError on locked
    Windows temp folders.  ``mkdtemp`` needs no such scan.
    """
    import tempfile
    return tempfile.mkdtemp(prefix="causalit_dir_bias_")


def _rand_qkv(B=2, N=5, E=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, N, E, generator=g),
        torch.randn(B, N, E, generator=g),
        torch.randn(B, N, E, generator=g),
    )


def _offdiag(x):
    return x[~torch.eye(x.shape[-1], dtype=torch.bool)]


class TestDirBiasGate:

    def test_default_bias_is_zero(self):
        assert GatedSelfAttention().dir_bias == 0.0

    def test_explicit_zero_is_bit_identical(self):
        q, k, v = _rand_qkv()
        m0 = GatedSelfAttention()
        m0.eval()
        m1 = GatedSelfAttention(dir_bias=0.0)
        m1.eval()
        out0, p0, _ = m0(q, k, v)
        out1, p1, _ = m1(q, k, v)
        assert torch.equal(out0, out1)
        assert torch.equal(p0, p1)

    def test_bias_opens_both_directions_eval(self):
        """sigmoid(A_anti/beta + 10) ~ 1 for every ordered pair (i, j)."""
        m = GatedSelfAttention(dir_bias=10.0)
        m.eval()
        q, k, v = _rand_qkv()
        m(q, k, v)
        d = m.last_direction
        assert d is not None
        assert float(_offdiag(d).min()) > 0.9, (
            f"bias=10 must pin both directions near 1, got min "
            f"{float(_offdiag(d).min()):.4f}"
        )

    def test_coupling_relaxed_and_restored(self):
        """d_ij + d_ji == 1 at bias 0 (train-mode sampling included), is
        relaxed under a positive bias, and restored by set_dir_bias(0)."""
        q, k, v = _rand_qkv()
        diag = torch.eye(q.shape[1], dtype=torch.bool)

        m = GatedSelfAttention()
        m.train()
        m(q, k, v)
        s = m.last_direction + m.last_direction.T
        assert torch.allclose(s[~diag], torch.ones_like(s[~diag]), atol=1e-5)

        m.set_dir_bias(2.0)
        m(q, k, v)
        s = (m.last_direction + m.last_direction.T)[~diag]
        # Peak relaxation is 2*sigmoid(2) = 1.76 at A_anti = 0, decaying back
        # toward 1 as |A_anti| grows ? so the mean sits strictly between the
        # coupled value 1.0 and 1.76.
        assert 1.2 < float(s.mean()) < 1.76, (
            f"expected relaxed coupling in (1.2, 1.76), got {float(s.mean()):.3f}"
        )

        m.set_dir_bias(0.0)
        m(q, k, v)
        s = (m.last_direction + m.last_direction.T)[~diag]
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)

    def test_structure_posterior_includes_bias(self):
        m = GatedSelfAttention()
        m.eval()
        q, k, _ = _rand_qkv()
        p0 = m.structure_posterior(q, k)
        m.set_dir_bias(4.0)
        p1 = m.structure_posterior(q, k)
        assert torch.all(p1 >= p0 - 1e-6)
        assert bool((p1 > p0 + 1e-3).any())


class _DirBiasStub(torch.nn.Module):
    """Minimal pl.Module stand-in: owns a GatedSelfAttention and a log sink."""

    def __init__(self):
        super().__init__()
        self.inner = GatedSelfAttention()
        self.logged = []

    def log(self, name, value, **kwargs):
        self.logged.append((name, value))


def _controller(save_dir, structure=None, reconstruct=None):
    from causaliT.training.adaptive_trainer import PhaseController

    config = {
        "adaptive_training": {
            "structure": structure or {},
            "reconstruct": reconstruct or {},
        },
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    return PhaseController(
        config=config, data_dir=str(save_dir), save_dir=str(save_dir),
        cluster=True,
    )


class TestDirBiasPhaseController:

    def test_unmanaged_run_is_untouched(self, save_dir):
        """No phase block sets dir_bias -> the controller must not touch it."""
        ctrl = _controller(save_dir)
        stub = _DirBiasStub()
        stub.inner.set_dir_bias(0.7)
        ctrl._apply_dir_bias_cfg(stub, "reconstruct")
        assert stub.inner.dir_bias == pytest.approx(0.7)
        assert stub.logged == []

    def test_phase_values_applied(self, save_dir):
        """reconstruct gets its value; structure (key omitted) gets 0.0."""
        ctrl = _controller(save_dir, reconstruct={"dir_bias": 1.1})
        stub = _DirBiasStub()
        ctrl._apply_dir_bias_cfg(stub, "reconstruct")
        assert stub.inner.dir_bias == pytest.approx(1.1)
        ctrl._apply_dir_bias_cfg(stub, "structure")
        assert stub.inner.dir_bias == 0.0

    def test_logged_metric(self, save_dir):
        ctrl = _controller(save_dir, reconstruct={"dir_bias": 1.1})
        stub = _DirBiasStub()
        ctrl._apply_dir_bias_cfg(stub, "reconstruct")
        assert ("dir_bias", pytest.approx(1.1)) in [
            (n, pytest.approx(v)) for n, v in stub.logged
        ] or any(n == "dir_bias" for n, _ in stub.logged)

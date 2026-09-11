"""Tests for the hsic_unrolled (DARTS second-order) shadow evidence.

Covers the finite-difference mixed-Hessian term against an exact double
backward, no-mutation/determinism of ``_unrolled_shadow_grads``, and the
cadence fallback in ``_shadow_evidence``.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_bilevel_probe import _model, _qbatches

from causaliT.training.gradient_routing import classify_parameters
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)


def _unrolled_model(every=1):
    """Toy SVFA forecaster (real column convention) with centroid commit and
    the unrolled shadow enabled."""
    m = _model()
    cfg = m.config
    cfg["model"]["kwargs"]["homogeneous_nodes"] = True
    cfg["training"]["centroid_commit"] = {
        "enabled": True, "evidence_lr": 10.0, "evidence_leak": 0.95,
        "shadow_source": "hsic_unrolled", "reset_m_on_commit": "one",
        "unrolled": {"inner_lr": 1e-3, "fd_epsilon": 1e-2, "every": every},
    }
    torch.manual_seed(0)
    m2 = AttentionSelectorForecaster(cfg)
    m2.log = lambda *a, **k: None     # no Trainer attached in unit tests
    return m2


def _step_once(m, batch):
    """Populate _last_loss_components / _last_hsic_reg via a train _step."""
    m.train()
    return m._step(batch=batch, stage="train")


class TestFDvsExact:
    def test_fd_matches_exact_mixed_hessian(self):
        """[grad_q L(t+e v) - grad_q L(t-e v)] / 2e  ==  d(grad_q L)/dt . v

        NOTE: with the straight-through shadow the loss VALUE does not depend
        on the shadow, so the symmetric mixed partial d/dq <grad_t L, v> is
        identically ~0 and is the WRONG reference.  The DARTS correction is
        the directional derivative of the surrogate gradient field
        g_q(theta) = grad_shadow L in the theta direction v; the exact
        reference is the Jacobian block J = d g_q / d p for ONE small
        parameter tensor p (v supported on p), built column by column.
        Float64: the FD numerator is O(eps * curvature) and float32
        subtractive cancellation swamps it at eps <= 1e-3.
        """
        m = _unrolled_model().double()
        S, X = _qbatches(1)[0]
        S, X = S.double(), X.double()
        shadows = [t.shadow for t in m._commit.tables]
        _, recon = classify_parameters(m.model, verbose=False)
        # Smallest recon tensor (a bias) as the perturbation support.
        p = min(recon, key=lambda t: t.numel())
        g = torch.Generator().manual_seed(11)
        v = torch.randn(p.shape, generator=g, dtype=torch.float64)
        v = v / v.norm()

        # Surrogate gradient field g_q(theta) at the base point.  All lean
        # forwards share one forked RNG seed: the gated attentions sample
        # hard-concrete gates in train mode, and unpaired draws would swamp
        # the FD numerator (measured: cos 1.0 paired vs -0.35 unpaired).
        def _lean():
            with torch.random.fork_rng():
                torch.manual_seed(7)
                return m._lean_recon(S, X)

        g_q = torch.autograd.grad(_lean(), shadows,
                                  create_graph=True, allow_unused=True)
        g_flat = torch.cat([torch.zeros_like(sh).flatten() if gi is None
                            else gi.flatten()
                            for gi, sh in zip(g_q, shadows)])

        # Exact J columns: d g_q[j] / d p[k], elementwise over the shadow.
        J = torch.zeros(g_flat.numel(), p.numel(), dtype=torch.float64)
        for j in range(g_flat.numel()):
            gp = torch.autograd.grad(g_flat[j], p, retain_graph=True,
                                     allow_unused=True)[0]
            if gp is not None:
                J[j] = gp.flatten()
        exact = J @ v.flatten()

        # Finite difference of g_q along v on p.
        eps = 1e-4
        saved = p.detach().clone()
        fd = []
        for sign in (+1.0, -1.0):
            with torch.no_grad():
                p.add_(v, alpha=sign * eps)
            g2 = torch.autograd.grad(_lean(), shadows, allow_unused=True)
            fd.append(torch.cat([torch.zeros_like(sh).flatten()
                                 if gi is None else gi.flatten()
                                 for gi, sh in zip(g2, shadows)]))
            with torch.no_grad():
                p.copy_(saved)
        fdv = (fd[0] - fd[1]) / (2 * eps)

        cos = float((exact * fdv).sum()
                    / (exact.norm() * fdv.norm() + 1e-30))
        rel = float((exact - fdv).norm() / (exact.norm() + 1e-30))
        assert cos > 0.99, f"cos(exact, fd) = {cos}"
        assert rel < 0.05, f"relative error = {rel}"


class TestUnrolledStep:
    def test_no_mutation_and_determinism(self):
        m = _unrolled_model()
        batch = _qbatches(1)[0]
        _step_once(m, batch)
        # One-time lazy init inside the first lean/functional_call pass makes
        # call 1 differ from call 2 by ~1e-9; from call 2 on the result is
        # bit-exact (verified: calls 2-4 identical).  Warm up first.
        m._unrolled_shadow_grads(batch)
        snap_p = {n: p.detach().clone() for n, p in m.named_parameters()}
        snap_b = {n: b.detach().clone() if b is not None else None
                  for n, b in m.named_buffers()}
        g1 = m._unrolled_shadow_grads(batch)
        g2 = m._unrolled_shadow_grads(batch)
        assert len(g1) == len(m._commit.tables)
        for a, b in zip(g1, g2):
            assert (a is None) == (b is None)
            if a is not None:
                assert torch.equal(a, b)
        for n, p in m.named_parameters():
            assert torch.equal(p, snap_p[n]), f"param mutated: {n}"
        for n, b in m.named_buffers():
            if snap_b[n] is None:
                assert b is None
            else:
                assert torch.equal(b, snap_b[n]), f"buffer mutated: {n}"

    def test_unrolled_differs_from_first_order(self):
        m = _unrolled_model()
        batch = _qbatches(1)[0]
        _step_once(m, batch)
        g_unrolled = m._unrolled_shadow_grads(batch)
        g_first = torch.autograd.grad(
            m._last_hsic_reg, [t.shadow for t in m._commit.tables],
            retain_graph=True, allow_unused=True)
        for gu, gf in zip(g_unrolled, g_first):
            if gu is None or gf is None:
                continue
            assert torch.isfinite(gu).all()
            cos = float((gu * gf).sum() / (gu.norm() * gf.norm() + 1e-30))
            assert cos < 0.999999   # the correction actually entered

    def test_training_step_end_to_end(self):
        """Full routing training_step with unrolled shadow: the retained main
        graph must survive the virtual passes (the version-counter bug), both
        optimizer steps run, and the controller consumes the evidence."""
        m = _unrolled_model()
        m.config["training"]["use_gradient_routing"] = True
        m2_cfg = m.config
        torch.manual_seed(0)
        m2 = AttentionSelectorForecaster(m2_cfg)   # routing -> dual optimizers
        m2.log = lambda *a, **k: None
        from causaliT.training.gradient_routing import classify_parameters as cp
        sp, rp = cp(m2.model, verbose=False)
        opt_r = torch.optim.SGD(rp, lr=1e-3)
        opt_s = torch.optim.SGD(sp, lr=1e-3)
        m2.optimizers = lambda: (opt_r, opt_s)
        # No Trainer attached: manual_backward requires one in this Lightning
        # version — substitute the plain backward it wraps.
        m2.manual_backward = (
            lambda loss, retain_graph=False: loss.backward(
                retain_graph=retain_graph))
        loss = m2.training_step(_qbatches(1)[0], 0)
        assert torch.isfinite(torch.as_tensor(loss))
        assert m2._unrolled_step_count == 1

    def test_structure_phase_frozen_theta_r(self):
        """Structure phase: the adaptive controller freezes theta_R
        (``requires_grad_(False)``) and unfreezes theta_S.  The unrolled
        shadow must still produce finite evidence — differentiating the MAIN
        graph w.r.t. the frozen live parameters used to raise
        ``RuntimeError: One of the differentiated Tensors does not require
        grad`` on the first structure-phase step (cluster job 12486528)."""
        m = _unrolled_model()
        batch = _qbatches(1)[0]
        struct, recon = classify_parameters(m.model, verbose=False)
        for p in recon:            # exactly what _apply_phase("structure") does
            p.requires_grad_(False)
        for p in struct:
            p.requires_grad_(True)
        # The forecaster caches the partition at construction time, as in a
        # real routed run.
        m._reconstruction_params = recon
        m._structural_params = struct
        _step_once(m, batch)
        g = m._unrolled_shadow_grads(batch)
        assert len(g) == len(m._commit.tables)
        assert any(gi is not None for gi in g)
        for gi in g:
            if gi is not None:
                assert torch.isfinite(gi).all()
        # theta_R stays frozen and untouched.
        for p in recon:
            assert not p.requires_grad
            assert p.grad is None

    def test_cadence_fallback(self):
        m = _unrolled_model(every=2)
        batch = _qbatches(1)[0]
        _step_once(m, batch)
        calls = []
        orig = m._unrolled_shadow_grads
        m._unrolled_shadow_grads = lambda b: (calls.append(1), orig(b))[1]
        m._shadow_evidence(batch)   # count 1: off-step -> first-order
        m._shadow_evidence(batch)   # count 2: unrolled
        m._shadow_evidence(batch)   # count 3: first-order
        assert len(calls) == 1

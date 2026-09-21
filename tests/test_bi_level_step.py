"""Tests for the bi-level (DARTS second-order) structural gradient.

Covers ``training.structural_grad: hsic_unrolled`` in the
AttentionSelectorForecaster: config guards, the gradient written by
``_bi_level_step`` (scaling, rest-term additivity, no-mutation of theta_R,
determinism), fold-B consumption, and the default first-order path staying
byte-identical.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_bilevel_probe import _model, _qbatches

from causaliT.training.gradient_routing import classify_parameters
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)


def _bilevel_model(**overrides):
    """Toy SVFA forecaster with routing + structural_grad='hsic_unrolled'."""
    m = _model()
    cfg = m.config
    cfg["model"]["kwargs"]["homogeneous_nodes"] = True
    cfg["training"]["use_gradient_routing"] = True
    cfg["training"]["structural_grad"] = "hsic_unrolled"
    cfg["training"]["unrolled"] = {"inner_lr": 1e-3, "fd_epsilon": 1e-2}
    cfg["training"].update(overrides)
    torch.manual_seed(0)
    m2 = AttentionSelectorForecaster(cfg)
    m2.log = lambda *a, **k: None     # no Trainer attached in unit tests
    return m2


def _attach_optimizers(m, lr=1e-3):
    sp, rp = classify_parameters(m.model, verbose=False)
    opt_r = torch.optim.SGD(rp, lr=lr)
    opt_s = torch.optim.SGD(sp, lr=lr)
    m.optimizers = lambda: (opt_r, opt_s)
    m.manual_backward = (
        lambda loss, retain_graph=False: loss.backward(
            retain_graph=retain_graph))
    return sp, rp


class TestConfigGuards:
    def test_default_is_first_order(self):
        m = _model()
        assert m.structural_grad == "hsic"

    def test_invalid_value_raises(self):
        m = _model()
        m.config["training"]["structural_grad"] = "darts"
        with pytest.raises(ValueError, match="structural_grad"):
            AttentionSelectorForecaster(m.config)

    def test_requires_gradient_routing(self):
        m = _model()
        m.config["training"]["use_gradient_routing"] = False
        m.config["training"]["structural_grad"] = "hsic_unrolled"
        with pytest.raises(ValueError, match="use_gradient_routing"):
            AttentionSelectorForecaster(m.config)


class TestBiLevelGrads:
    def test_finite_scaled_and_corrected(self):
        """_bi_level_step writes finite grads = scale * unrolled-HSIC + rest,
        and the correction actually enters (differs from first-order)."""
        m = _bilevel_model()
        batch = _qbatches(1)[0]
        m.train()
        m._step(batch=batch, stage="train")
        struct = [p for p in m._structural_params if p.requires_grad]
        # First-order reference BEFORE _bi_level_step consumes the graph.
        g_first = torch.autograd.grad(
            m._last_hsic_reg, struct, retain_graph=True, allow_unused=True)
        m._bi_level_step(batch)
        any_correction = False
        for p, gf in zip(struct, g_first):
            assert p.grad is not None and torch.isfinite(p.grad).all()
            if gf is None or float(gf.abs().sum()) == 0.0:
                continue
            cos = float((p.grad * gf).sum() / (p.grad.norm() * gf.norm()
                                               + 1e-30))
            if cos < 0.999999:
                any_correction = True
        assert any_correction

    def test_no_mutation_and_determinism(self):
        """theta_R values/version counters untouched; repeated call identical
        after the one-time warm-up (mirrors the shadow-path invariant)."""
        m = _bilevel_model()
        batch = _qbatches(1)[0]
        _, recon = classify_parameters(m.model, verbose=False)
        m.train()
        m._step(batch=batch, stage="train")
        before = [(p.detach().clone(), p._version) for p in recon]
        m._bi_level_step(batch)          # warm-up (consumes the graph)
        for p in m._structural_params:
            p.grad = None
        m._step(batch=batch, stage="train")   # fresh graph for each call
        m._bi_level_step(batch)
        g1 = [p.grad.clone() if p.grad is not None else None
              for p in m._structural_params]
        for p in m._structural_params:
            p.grad = None
        m._step(batch=batch, stage="train")
        m._bi_level_step(batch)
        g2 = [p.grad for p in m._structural_params]
        for (val, ver), p in zip(before, recon):
            assert torch.equal(val, p.detach())
            assert p._version == ver
        for a, b in zip(g1, g2):
            if a is None:
                assert b is None
            else:
                assert torch.allclose(a, b, atol=1e-6)

    def test_rest_terms_added(self):
        """With lambda_l0 > 0 the L0 gradient must ride along (live graph)."""
        m = _bilevel_model(lambda_l0=0.5)
        batch = _qbatches(1)[0]
        m.train()
        m._step(batch=batch, stage="train")
        m._bi_level_step(batch)
        struct = [p for p in m._structural_params if p.requires_grad]
        g_hsic_only = m._darts_second_order_grads(batch[0], batch[1], struct)
        scale = m.lambda_hsic * (1.0 - m.lambda_struct_recon)
        differs = any(
            p.grad is not None and gh is not None
            and not torch.allclose(p.grad, scale * gh, atol=1e-7)
            for p, gh in zip(struct, g_hsic_only)
        )
        assert differs   # the rest terms changed at least one gradient


class TestTrainingStepIntegration:
    def test_end_to_end_routed_step(self):
        """Full routing training_step in bi-level mode: both optimizers step,
        the retained main graph survives the virtual passes."""
        m = _bilevel_model()
        _attach_optimizers(m)
        loss = m.training_step(_qbatches(1)[0], 0)
        assert torch.isfinite(torch.as_tensor(loss))

    def test_fold_b_batch_is_used(self):
        """When cross-fitting yields a fold-B batch, _bi_level_step receives
        THAT batch, not the fold-A training batch."""
        m = _bilevel_model()
        _attach_optimizers(m)
        batch_a, batch_b = _qbatches(2)
        seen = {}
        orig = m._bi_level_step
        m._bi_level_step = lambda b: (seen.setdefault("batch", b), orig(b))[1]
        m._next_cross_fit_batch = lambda: [batch_b[0], batch_b[1]]
        m.training_step(batch_a, 0)
        assert seen["batch"][0] is batch_b[0]

    def test_default_path_ignores_bi_level(self):
        """structural_grad='hsic' (default) never calls _bi_level_step."""
        m = _model()
        cfg = m.config
        cfg["training"]["use_gradient_routing"] = True
        torch.manual_seed(0)
        m2 = AttentionSelectorForecaster(cfg)
        m2.log = lambda *a, **k: None
        _attach_optimizers(m2)
        calls = []
        m2._bi_level_step = lambda b: calls.append(1)
        loss = m2.training_step(_qbatches(1)[0], 0)
        assert torch.isfinite(torch.as_tensor(loss))
        assert calls == []

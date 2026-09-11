"""Tests for the paired bilevel refit probe (Phase 1, BILEVEL_CENTROID_COMMIT).

Covers ``causaliT/training/bilevel_probe.py``: state isolation of the live
forecaster, determinism, identity-candidate invariance, k_inner=0 edge, and
the inner-optimizer config inheritance/override.
"""

import torch

from test_dropout_selection import _make_batch, _make_forecaster_config

from causaliT.training.bilevel_probe import (
    node_locations,
    paired_refit_probe,
    query_tables_of,
    resolve_inner_config,
)
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)


def _model():
    """Forecaster in the REAL data convention: col 0 = value, col 1 = var id.

    The shared fixture uses VAR_COL=0/VALUE_COL=1, but FreeQueryEmbedding is
    hardcoded to read the id from column 1 AND the forecaster blanks column
    ``val_idx`` before the query lookup — with the fixture convention the
    query table reads the blanked column and is completely dead.  Flip the
    column roles so the query pathway is actually exercised.
    """
    torch.manual_seed(0)
    cfg = _make_forecaster_config()
    cfg["data"]["val_idx"] = 0
    for embed in ("ds_embed_S", "ds_embed_X"):
        for mod in cfg["model"]["kwargs"][embed]["modules"]:
            mod["idx"] = 1 if mod["label"] == "variable" else 0
            mod["role"] = "structure" if mod["label"] == "variable" else "value"
    # "summation" makes K = V = structural constants -> predictions are
    # per-node constants (pred std = 0) no matter the training.  SVFA gives a
    # genuine value pathway, as in the real d20 configs.
    cfg["model"]["kwargs"]["comps_embed_S"] = "svfa"
    cfg["model"]["kwargs"]["comps_embed_X"] = "svfa"
    return AttentionSelectorForecaster(cfg)


def _qbatches(n=3, batch=16):
    """(S, X) batches, col 0 = value, col 1 = 1-indexed var id.

    X values DEPEND on S values (X_j = S_j + noise): with independent random
    values a query-row change shifts a node's prediction by a ~constant
    offset, which the centered HSIC kernel cannot see; a real dependence
    makes attention reweighting sample-varying, as in the real datasets.
    """
    out = []
    for i in range(n):
        g = torch.Generator().manual_seed(100 + i)
        S = torch.zeros(batch, 3, 2)
        S[:, :, 0] = torch.randn(batch, 3, generator=g)
        S[:, :, 1] = torch.arange(1, 4).float()
        X = torch.zeros(batch, 3, 2)
        X[:, :, 0] = S[:, :, 0] + 0.1 * torch.randn(batch, 3, generator=g)
        X[:, :, 1] = torch.arange(1, 4).float()
        out.append((S, X))
    return out


def _trained_model(steps=30, lr=5e-3):
    """_model() + a short MSE fit on the dependent toy data.

    At init the model's predictions are CONSTANT per node (pred std = 0):
    the value pathway contributes nothing before training, so HSIC rows sit
    at the noise floor and any query write is invisible (constant-shift
    blindness).  Probes in practice run mid-training, so the fixture must be
    a (lightly) trained model.
    """
    m = _model()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for t in range(steps):
        S, X = _qbatches(1, batch=32)[0]
        m.train()
        opt.zero_grad()
        pred = m.forward(data_source=S, data_intermediate=X)[0]
        x_target = X[:, :, m.val_idx]
        loss = torch.nn.functional.mse_loss(pred.squeeze(), x_target.squeeze())
        loss.backward()
        opt.step()
    m.eval()
    return m


def _state_snapshot(model):
    params = {n: p.detach().clone()
              for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() if b is not None else None
               for n, b in model.named_buffers()}
    return params, buffers


def _assert_state_equal(snap, model):
    params, buffers = snap
    for n, p in model.named_parameters():
        assert torch.equal(p, params[n]), f"param mutated: {n}"
    for n, b in model.named_buffers():
        if buffers[n] is None:
            assert b is None
        else:
            assert torch.equal(b, buffers[n]), f"buffer mutated: {n}"


def _random_row(model, node, seed=7):
    """A unit-norm candidate vector for ``node`` (d_model,)."""
    tables = query_tables_of(model)
    t, r = node_locations(model)[node]
    d = tables[t].embedding.weight.shape[1]
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return v / v.norm()


class TestNoMutation:
    def test_live_forecaster_untouched(self):
        model = _trained_model()
        snap = _state_snapshot(model)
        paired_refit_probe(model, {2: _random_row(model, 2)},
                           _qbatches(), k_inner=2, seed=0)
        _assert_state_equal(snap, model)

    def test_no_mutation_with_zero_refit(self):
        model = _trained_model()
        snap = _state_snapshot(model)
        paired_refit_probe(model, {2: _random_row(model, 2)},
                           _qbatches(), k_inner=0, seed=0)
        _assert_state_equal(snap, model)


class TestIdentityAndDeterminism:
    def test_identity_candidate_exactly_equal(self):
        model = _trained_model()
        tables = query_tables_of(model)
        t, r = node_locations(model)[2]
        current = tables[t].embedding.weight[r].detach().clone()
        res = paired_refit_probe(model, {2: current}, _qbatches(),
                                 k_inner=3, seed=0)
        assert torch.allclose(res.rows_incumbent, res.rows_candidate,
                              equal_nan=True)
        assert res.deltas[2] == 0.0
        assert not res.accepted[2]      # delta 0 is not < -margin

    def test_determinism_same_seed(self):
        model = _trained_model()
        w = {1: _random_row(model, 1), 2: _random_row(model, 2, seed=8)}
        r1 = paired_refit_probe(model, w, _qbatches(), k_inner=2, seed=42)
        r2 = paired_refit_probe(model, w, _qbatches(), k_inner=2, seed=42)
        assert torch.allclose(r1.rows_incumbent, r2.rows_incumbent,
                              equal_nan=True)
        assert torch.allclose(r1.rows_candidate, r2.rows_candidate,
                              equal_nan=True)
        assert r1.deltas == r2.deltas and r1.accepted == r2.accepted

    def test_perturbed_candidate_changes_rows(self):
        model = _trained_model()
        res = paired_refit_probe(model, {2: _random_row(model, 2)},
                                 _qbatches(), k_inner=2, seed=0)
        assert res.deltas[2] != 0.0

    def test_k_inner_zero_runs(self):
        model = _trained_model()
        res = paired_refit_probe(model, {2: _random_row(model, 2)},
                                 _qbatches(), k_inner=0, seed=0)
        assert res.rows_incumbent.shape == res.rows_candidate.shape

    def test_empty_val_batches_raises(self):
        model = _trained_model()
        try:
            paired_refit_probe(model, {2: _random_row(model, 2)}, [],
                               k_inner=1)
        except ValueError:
            return
        raise AssertionError("expected ValueError on empty val_batches")


class TestProbeAfterTrainingStep:
    """Regression test for the cluster crash (job 12281530): after a train
    ``_step``, ``_last_loss_components`` carries graph-holding tensors (inside
    a dict), which naive deepcopy rejects.  The probe must sanitize them."""

    def test_probe_after_step_succeeds_and_restores(self):
        model = _trained_model()
        model.log = lambda *a, **k: None          # no Trainer attached
        model.train()
        model._step(batch=_qbatches(1)[0], stage="train")
        # Precondition: the crash condition is actually present.
        lc = model._last_loss_components
        assert any(torch.is_tensor(v) and v.grad_fn is not None
                   for v in lc.values())
        ref = {k: v for k, v in lc.items()}
        res = paired_refit_probe(model, {2: _random_row(model, 2)},
                                 _qbatches(), k_inner=2, seed=0)
        assert 2 in res.deltas
        # Live state restored: same tensor objects, still graph-carrying.
        for k, v in model._last_loss_components.items():
            assert v is ref[k]
            assert v.grad_fn is not None

    def test_probe_after_step_does_not_mutate_params(self):
        model = _trained_model()
        model.log = lambda *a, **k: None
        model.train()
        model._step(batch=_qbatches(1)[0], stage="train")
        snap = _state_snapshot(model)
        paired_refit_probe(model, {2: _random_row(model, 2)},
                           _qbatches(), k_inner=2, seed=0)
        _assert_state_equal(snap, model)

    def test_probe_after_dtype_move(self):
        """Buffers become non-leaf after nn.Module._apply (device/dtype move);
        the sanitizer must cover _buffers too (cluster job 12294402)."""
        model = _trained_model()
        model = model.double()   # _apply without a GPU
        model.log = lambda *a, **k: None
        model.train()
        S, X = _qbatches(1)[0]
        model._step(batch=(S.double(), X.double()), stage="train")
        res = paired_refit_probe(model, {2: _random_row(model, 2).double()},
                                 [(S.double(), X.double())], k_inner=2, seed=0)
        assert 2 in res.deltas

    def test_inherits_reconstruction_config(self):
        model = _trained_model()
        cfg = resolve_inner_config(model)
        tc = model.config["training"]
        assert cfg["optimizer_type"] == tc.get("optimizer", "adamw")
        assert cfg["lr"] == tc["lr"]
        assert cfg["weight_decay"] == tc["weight_decay"]

    def test_per_key_override(self):
        model = _trained_model()
        cfg = resolve_inner_config(model, inner_optimizer="sgd",
                                   inner_lr=0.5, inner_weight_decay=0.0)
        assert cfg["optimizer_type"] == "sgd"
        assert cfg["lr"] == 0.5
        assert cfg["weight_decay"] == 0.0

    def test_probe_with_sgd_override_runs(self):
        model = _trained_model()
        res = paired_refit_probe(model, {2: _random_row(model, 2)},
                                 _qbatches(), k_inner=2,
                                 inner_optimizer="sgd", inner_lr=1e-2,
                                 inner_weight_decay=0.0, seed=0)
        assert all(map(torch.isfinite,
                       res.rows_incumbent[~res.rows_incumbent.isnan()]))

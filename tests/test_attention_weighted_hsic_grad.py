"""Surgical gradient checks for the ATTENTION-WEIGHTED HSIC arm.

Context (experiments/1_FOUNDATIONS/2_HSIC/dense_bias): the plain per-pair HSIC
cannot distinguish the true DAG from a dense one — the pair matrices are
numerically identical, and descendant-aware WEIGHTS are what make the oracle
the minimiser.  Two weighting paths exist in the code:

* ``causaliT.utils.descendant_mask.build_hsic_pair_mask``  -> always DETACHED
  (a differentiable mask would let the model delete its own penalty term);
* ``use_attention_weighted_hsic`` -> ``hsic_attention_weighted(att_mean, ...)``
  with ``att_mean = attention_weights.mean(0)`` taken from the LIVE posterior,
  i.e. NOT detached.

The HSIC_OPT_3 arm relies on the second path actually carrying gradient into
the query/key parameters THROUGH THE WEIGHTS (so the optimiser can drive the
weight of a descendant pair to zero).  These tests pin that down:

1. gradient exists and is finite/non-zero;
2. PATH ISOLATION — with the HSIC values detached the gradient must SURVIVE
   (it can then only come from the weights), and with the weights detached it
   must change: the weight channel is live, not incidental;
3. the documented bypasses of this branch (no descendant mask, no probe pair
   mask) hold, so the arm's semantics are known;
4. the homogeneous-mode diagonal carries ~no attention weight, otherwise the
   irreducible ``HSIC(X_i, r_i)`` term enters the objective.
"""

import torch

from test_dropout_selection import _make_forecaster_config

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.hsic_utils import hsic_attention_weighted


def _model(attention_weighted: bool):
    """Forecaster in the real data convention (col 0 = value, col 1 = var id).

    Mirrors ``tests/test_bilevel_probe._model``: the shared fixture uses the
    opposite column roles, which leaves the query pathway dead.
    """
    torch.manual_seed(0)
    cfg = _make_forecaster_config()
    cfg["data"]["val_idx"] = 0
    for embed in ("ds_embed_S", "ds_embed_X"):
        for mod in cfg["model"]["kwargs"][embed]["modules"]:
            mod["idx"] = 1 if mod["label"] == "variable" else 0
            mod["role"] = "structure" if mod["label"] == "variable" else "value"
    cfg["model"]["kwargs"]["comps_embed_S"] = "svfa"
    cfg["model"]["kwargs"]["comps_embed_X"] = "svfa"
    cfg["training"]["use_attention_weighted_hsic"] = attention_weighted
    cfg["training"]["lambda_hsic"] = 1.0
    cfg["training"]["hsic_adaptive_bandwidth"] = True
    return AttentionSelectorForecaster(cfg)


def _batch(batch=48, seed=100):
    """(S, X) with X depending on S so attention reweighting is sample-varying."""
    g = torch.Generator().manual_seed(seed)
    S = torch.zeros(batch, 3, 2)
    S[:, :, 0] = torch.randn(batch, 3, generator=g)
    S[:, :, 1] = torch.arange(1, 4).float()
    X = torch.zeros(batch, 3, 2)
    X[:, :, 0] = S[:, :, 0] + 0.1 * torch.randn(batch, 3, generator=g)
    X[:, :, 1] = torch.arange(1, 4).float()
    return S, X


def _trained(model, steps=25, lr=5e-3):
    """Short MSE fit: at init predictions are constant per node (HSIC blind)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    S, X = _batch(batch=64, seed=7)
    for _ in range(steps):
        model.train()
        opt.zero_grad()
        pred = model.forward(data_source=S, data_intermediate=X)[0]
        x_val = X[:, :, model.val_idx]
        if model.homogeneous_nodes:
            x_val = torch.cat([S[:, :, model.val_idx], x_val], dim=1)
        torch.nn.functional.mse_loss(pred.squeeze(),
                                     x_val.squeeze()).backward()
        opt.step()
    model.eval()
    return model


def _query_params(model):
    return [t.embedding.weight for t in (model.model.query_embed_S,
                                         model.model.query_embed_X)
            if getattr(t, "embedding", None) is not None]


def _forward_parts(model, S, X):
    """(attention_weights, combined_source, residuals) with the live graph."""
    pred_x, attention_weights, _ = model.forward(data_source=S,
                                                 data_intermediate=X)
    x_val = X[:, :, model.val_idx]
    if model.homogeneous_nodes:
        x_val = torch.cat([S[:, :, model.val_idx], x_val], dim=1)
    x_target = torch.nan_to_num(x_val)
    residuals = x_target.squeeze() - pred_x.squeeze()
    if model.homogeneous_nodes:
        combined = x_target.squeeze()
    else:
        combined = torch.cat([S[:, :, model.val_idx].squeeze(),
                              x_target.squeeze()], dim=1)
    return attention_weights, combined, residuals


def _weighted_hsic(model, S, X, detach_values=False, detach_weights=False):
    """The arm's objective, with either channel optionally detached."""
    att, combined, residuals = _forward_parts(model, S, X)
    att_mean = att.mean(dim=0)
    if detach_weights:
        att_mean = att_mean.detach()
    if detach_values:
        residuals = residuals.detach()
        combined = combined.detach()
    return hsic_attention_weighted(
        source_values=combined, residuals=residuals,
        attention_weights=att_mean, sigma=model.hsic_sigma,
        exclude_diagonal=False,
        adaptive_bandwidth=model.hsic_adaptive_bandwidth,
        mode=model.hsic_mode, nhsic_epsilon=model.nhsic_epsilon,
        source_kernel=model.hsic_kernel_source,
        bandwidth_multipliers=getattr(model, "hsic_bandwidth_multipliers", None),
    )


class TestGradientFlows:
    def test_gradient_exists_and_is_finite(self):
        model = _trained(_model(True))
        S, X = _batch()
        hsic = _weighted_hsic(model, S, X)
        grads = torch.autograd.grad(hsic, _query_params(model),
                                    allow_unused=True)
        assert any(g is not None for g in grads), "no gradient reached queries"
        total = sum(float(g.abs().sum()) for g in grads if g is not None)
        assert torch.isfinite(torch.tensor(total)), "non-finite gradient"
        assert total > 0.0, f"gradient is exactly zero (sum={total})"

    def test_weight_channel_is_live(self):
        """PATH ISOLATION: with the HSIC VALUES detached the gradient can only
        come from the attention weights.  It must survive and be non-zero."""
        model = _trained(_model(True))
        S, X = _batch()
        hsic_w_only = _weighted_hsic(model, S, X, detach_values=True)
        grads = torch.autograd.grad(hsic_w_only, _query_params(model),
                                    allow_unused=True)
        total = sum(float(g.abs().sum()) for g in grads if g is not None)
        assert total > 0.0, (
            "attention weights carry NO gradient: the attention-weighted arm "
            "would be inert (weights effectively detached)"
        )

    def test_detaching_weights_changes_the_gradient(self):
        """The weight channel must contribute a DIFFERENT gradient than the
        value channel alone (otherwise it is numerically irrelevant)."""
        model = _trained(_model(True))
        S, X = _batch()
        params = _query_params(model)
        g_full = torch.autograd.grad(_weighted_hsic(model, S, X), params,
                                     allow_unused=True, retain_graph=False)
        g_val = torch.autograd.grad(
            _weighted_hsic(model, S, X, detach_weights=True), params,
            allow_unused=True)
        diff = sum(float((a - b).abs().sum())
                   for a, b in zip(g_full, g_val)
                   if a is not None and b is not None)
        assert diff > 0.0, "detaching the weights left the gradient unchanged"


class TestArmSemantics:
    """Pin the documented bypasses of this branch (forecaster L1294-1321)."""

    def test_probe_pair_mask_is_none_in_this_branch(self):
        model = _trained(_model(True))
        model._gate_enabled = True          # would otherwise skip the stash
        S, X = _batch()
        model.train()
        model._step((S, X), stage="train")
        assert model._last_probe_pair_mask is None, (
            "the attention-weighted branch must not stash a probe pair mask "
            "(descendant masking is bypassed: the attention weight IS the "
            "pair weight)"
        )

    def test_row_means_off_by_default_in_this_branch(self):
        """Rows are opt-in: without ``log_hsic_rows`` the attention-weighted
        path computes no pair matrix and stashes nothing."""
        model = _trained(_model(True))
        assert model.log_hsic_rows is False   # default
        S, X = _batch()
        model.train()
        model._step((S, X), stage="train")
        assert model._last_hsic_row_means is None

    def test_row_means_available_when_enabled(self):
        """Diagnostic gap CLOSED: with ``log_hsic_rows=True`` the attention-
        weighted branch now produces per-node HSIC rows (weighted by the same
        attention posterior as the scalar), one entry per target node."""
        model = _trained(_model(True))
        model.log_hsic_rows = True
        S, X = _batch()
        model.train()
        model._step((S, X), stage="train")
        rows = model._last_hsic_row_means
        assert rows is not None
        assert rows.ndim == 1
        # Rows are diagnostics only: never attached to the autograd graph.
        assert not rows.requires_grad
        finite = rows[~torch.isnan(rows)]
        assert finite.numel() > 0
        assert torch.isfinite(finite).all()

    def test_plain_branch_still_produces_rows(self):
        model = _trained(_model(False))
        model.log_hsic_rows = True
        S, X = _batch()
        model.train()
        model._step((S, X), stage="train")
        assert model._last_hsic_row_means is not None


class TestDiagonalWeight:
    """In homogeneous mode the (N, N) posterior includes the diagonal, and
    ``hsic_attention_weighted`` is called with ``exclude_diagonal=False``.
    HSIC(X_i, r_i) is irreducibly positive, so a non-zero diagonal attention
    weight injects a constant bias into the objective."""

    def test_diagonal_attention_is_negligible(self):
        model = _trained(_model(True))
        S, X = _batch()
        att, _, _ = _forward_parts(model, S, X)
        att_mean = att.mean(dim=0).detach()
        if att_mean.shape[0] != att_mean.shape[1]:
            # split mode: no diagonal concept in the cross block
            return
        diag = att_mean.diagonal().abs()
        off = att_mean.abs().sum() - diag.sum()
        share = float(diag.sum() / (diag.sum() + off + 1e-12))
        assert share < 0.05, (
            f"diagonal carries {share:.1%} of the attention mass: the "
            f"irreducible HSIC(X_i, r_i) term enters the weighted objective"
        )



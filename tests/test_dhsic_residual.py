"""NOTIME dHSIC residual-independence structural loss.

Covers the utility functions in ``causaliT.utils.hsic_utils``
(``gaussian_gram``, ``dhsic_from_kernels``, ``dhsic_residual_independence``)
against a verbatim re-implementation of the reference NOTIME code
(STAN-UAntwerp/NOTIME, hsic_utilis.py), and the forecaster integration via
``training.hsic_objective: "dhsic_residual"``.
"""

import torch
import pytest

from causaliT.utils.hsic_utils import (
    dhsic_from_kernels,
    dhsic_residual_independence,
    gaussian_gram,
)


# ----------------------------------------------------------------------
# Verbatim reference implementation (NOTIME hsic_utilis.py), double precision
# ----------------------------------------------------------------------
def _ref_centering(M):
    n = M.shape[0]
    unit = torch.ones([n, n], dtype=M.dtype)
    identity = torch.eye(n, dtype=M.dtype)
    H = identity - unit / n
    return M @ H


def _ref_gaussian_grammat(x, sigma=None):
    try:
        x.shape[1]
    except IndexError:
        x = x.view(x.shape[0], 1)
    xxT = torch.matmul(x, x.T)
    xnorm = torch.diag(xxT) - xxT + (torch.diag(xxT) - xxT).T
    if sigma is None:
        mdist = torch.median(xnorm[xnorm != 0])
        sigma = torch.sqrt(mdist * 0.5)
    if sigma == 0:
        eps = 7.0 / 3 - 4.0 / 3 - 1
        sigma += eps
    KX = -0.5 * xnorm / sigma / sigma
    return torch.exp(KX)


def _ref_dHSIC_calc(K_list):
    length = K_list[0].shape[0]
    term1 = 1.0
    term2 = 1.0
    term3 = 2.0 / length
    for K_j in K_list:
        term1 = torch.multiply(term1, K_j)
        term2 = 1.0 / length / length * term2 * torch.sum(K_j)
        term3 = 1.0 / length * term3 * K_j.sum(axis=0)
    term1 = torch.sum(term1)
    term3 = torch.sum(term3)
    return (1.0 / length) ** 2 * term1 + term2 - term3


def _ref_dHSIC(*argv):
    if len(argv) == 2:
        x, y = argv
        n = x.shape[0]
        return torch.trace(torch.matmul(
            _ref_centering(_ref_gaussian_grammat(x)),
            _ref_centering(_ref_gaussian_grammat(y)),
        )) / n / n
    K_list = [_ref_gaussian_grammat(a) for a in argv]
    return _ref_dHSIC_calc(K_list)


# ----------------------------------------------------------------------
# Utility-level tests
# ----------------------------------------------------------------------
class TestReferenceMatch:
    def test_matches_notime_reference_d3(self):
        """Our estimator reproduces the reference NOTIME dHSIC exactly."""
        g = torch.Generator().manual_seed(0)
        R = torch.randn(64, 3, generator=g, dtype=torch.float64)
        ours = dhsic_residual_independence(R)
        ref = _ref_dHSIC(R[:, 0], R[:, 1], R[:, 2])
        assert torch.allclose(ours, ref, rtol=1e-10, atol=1e-12)

    def test_matches_notime_reference_d2(self):
        g = torch.Generator().manual_seed(1)
        R = torch.randn(50, 2, generator=g, dtype=torch.float64)
        ours = dhsic_residual_independence(R)
        ref = _ref_dHSIC(R[:, 0], R[:, 1])
        assert torch.allclose(ours, ref, rtol=1e-10, atol=1e-12)

    def test_d2_equals_trace_form(self):
        """dHSIC_calc with d=2 equals tr(KH LH) / n^2 for the same kernels."""
        g = torch.Generator().manual_seed(2)
        x = torch.randn(40, generator=g, dtype=torch.float64)
        y = torch.randn(40, generator=g, dtype=torch.float64)
        K = gaussian_gram(x, sigma=0.7)
        L = gaussian_gram(y, sigma=1.3)
        n = x.shape[0]
        H = (torch.eye(n, dtype=torch.float64)
             - torch.ones(n, n, dtype=torch.float64) / n)
        trace_form = torch.trace(K @ H @ L @ H) / n / n
        assert torch.allclose(
            dhsic_from_kernels([K, L]), trace_form, rtol=1e-10, atol=1e-12
        )

    def test_fixed_and_per_column_sigma(self):
        g = torch.Generator().manual_seed(3)
        R = torch.randn(32, 4, generator=g, dtype=torch.float64)
        v_scalar = dhsic_residual_independence(R, sigma=1.0)
        v_per_col = dhsic_residual_independence(
            R, sigma=torch.ones(4, dtype=torch.float64)
        )
        assert torch.allclose(v_scalar, v_per_col)


class TestStatisticalBehaviour:
    def test_dependent_columns_score_higher_than_independent(self):
        g = torch.Generator().manual_seed(4)
        n = 200
        e = torch.randn(n, 3, generator=g)
        indep = dhsic_residual_independence(e)
        # Copy one column (plus small noise) -> mutual dependence.
        dep = e.clone()
        dep[:, 1] = e[:, 0] + 0.05 * torch.randn(n, generator=g)
        dep_val = dhsic_residual_independence(dep)
        assert dep_val > indep

    def test_independent_columns_near_zero(self):
        g = torch.Generator().manual_seed(5)
        R = torch.randn(400, 3, generator=g)
        v = float(dhsic_residual_independence(R))
        assert abs(v) < 0.05

    def test_gradient_flows_to_residuals(self):
        g = torch.Generator().manual_seed(6)
        R = torch.randn(64, 3, generator=g, requires_grad=True)
        v = dhsic_residual_independence(R)
        v.backward()
        assert R.grad is not None
        assert torch.isfinite(R.grad).all()
        assert R.grad.abs().sum() > 0

    def test_degenerate_constant_column_does_not_nan(self):
        R = torch.randn(32, 3)
        R[:, 0] = 1.0  # zero variance -> zero bandwidth, eps guard
        v = dhsic_residual_independence(R)
        assert torch.isfinite(v)


class TestValidation:
    def test_single_column_raises(self):
        with pytest.raises(ValueError, match="at least 2 residual columns"):
            dhsic_residual_independence(torch.randn(10, 1))

    def test_single_sample_raises(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            dhsic_residual_independence(torch.randn(1, 3))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            dhsic_residual_independence(torch.randn(10))

    def test_wrong_per_column_sigma_length_raises(self):
        with pytest.raises(ValueError, match="per-column sigma"):
            dhsic_residual_independence(torch.randn(10, 3), sigma=[1.0, 2.0])

    def test_single_kernel_list_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            dhsic_from_kernels([gaussian_gram(torch.randn(8))])


# ----------------------------------------------------------------------
# Forecaster integration
# ----------------------------------------------------------------------
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)

D, VOCAB, LS, LX = 16, 8, 3, 3
VAL, VAR = 1, 0


def _embed_cfg():
    return {
        "setting": {"d_model": D},
        "modules": [
            {"idx": VAR, "embed": "nn_embedding", "label": "variable",
             "role": "structure",
             "kwargs": {"num_embeddings": VOCAB, "embedding_dim": D}},
            {"idx": VAL, "embed": "linear", "label": "value", "role": "value",
             "kwargs": {"input_dim": 1, "embedding_dim": D}},
        ],
    }


def _config(hsic_objective, lambda_recon=1.0):
    return {
        "data": {"val_idx": VAL, "S_seq_len": LS, "X_seq_len": LX,
                 "dataset": "dummy"},
        "model": {
            "model_object": "AttentionSelectorLayer",
            "kwargs": {
                "model": "AttentionSelectorLayer",
                "ds_embed_S": _embed_cfg(),
                "ds_embed_X": _embed_cfg(),
                "comps_embed_S": "summation",
                "comps_embed_X": "summation",
                "attention_type": "ScaledDotSoftmax",
                "self_attention_type": None,
                "n_heads": 4,
                "dropout_emb": 0.0, "dropout_attn_out": 0.0, "dropout_ff": 0.0,
                "dropout_qkv": 0.0, "attention_dropout": 0.0,
                "activation": "gelu", "norm": "layer", "use_final_norm": True,
                "device": "cpu", "out_dim": 1, "d_ff": 32, "d_model": D,
                "d_qk": D, "S_seq_len": LS, "X_seq_len": LX,
                "struct_embedding_type": "standard_learnable",
                "value_structure_injection": "none",
                "value_structure_query_injection": "none",
            },
        },
        "training": {
            "loss_fn": "mse", "lr": 1e-3, "weight_decay": 0.0,
            "optimizer": "adamw", "use_gradient_routing": False,
            "lambda_recon": lambda_recon, "lambda_struct_recon": 0.0,
            "lambda_hsic": 1.0, "lambda_score_sparse": 0.0,
            "lambda_group_l1": 0.0, "lambda_l0": 0.0, "kappa": 0.0,
            "lambda_query_norm": 0.0,
            "hsic_objective": hsic_objective,
            "hsic_sigma": 1.0, "hsic_adaptive_bandwidth": True,
            "hsic_mode": "biased", "nhsic_epsilon": 0.01,
            "hsic_kernel_source": "rbf",
            "freeze_structural_params": False,
            "freeze_reconstruction_params": False,
        },
    }


def _batch(batch_size: int = 16):
    g = torch.Generator().manual_seed(0)
    S = torch.zeros(batch_size, LS, 2)
    S[:, :, VAR] = torch.randint(1, VOCAB, (batch_size, LS), generator=g).float()
    S[:, :, VAL] = torch.randn(batch_size, LS, generator=g)
    X = torch.zeros(batch_size, LX, 2)
    X[:, :, VAR] = torch.randint(1, VOCAB, (batch_size, LX), generator=g).float()
    X[:, :, VAL] = torch.randn(batch_size, LX, generator=g)
    return S, X


class TestForecasterIntegration:
    def test_invalid_objective_raises(self):
        cfg = _config("bogus")
        with pytest.raises(ValueError, match="hsic_objective"):
            AttentionSelectorForecaster(cfg)

    def test_default_is_pairwise(self):
        cfg = _config("pairwise")
        del cfg["training"]["hsic_objective"]
        fc = AttentionSelectorForecaster(cfg)
        assert fc.hsic_objective == "pairwise"

    def test_dhsic_training_step_backprops(self):
        """Finite, differentiable loss with the NOTIME residual objective."""
        fc = AttentionSelectorForecaster(_config("dhsic_residual"))
        S, X = _batch()
        fc.train()
        out = fc.training_step((S, X), 0)
        loss = out["loss"] if isinstance(out, dict) else out
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in fc.parameters()
                 if p.requires_grad and p.grad is not None]
        assert grads, "no parameter received a gradient"
        assert any(g.abs().sum() > 0 for g in grads)

    def test_notime_only_objective_backprops(self):
        """NOTIME-mimic: lambda_recon=0, dHSIC is the sole fitting term."""
        fc = AttentionSelectorForecaster(
            _config("dhsic_residual", lambda_recon=0.0)
        )
        S, X = _batch()
        fc.train()
        out = fc.training_step((S, X), 0)
        loss = out["loss"] if isinstance(out, dict) else out
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in fc.parameters()
                 if p.requires_grad and p.grad is not None]
        assert grads, "no parameter received a gradient"
        assert any(g.abs().sum() > 0 for g in grads)

    def test_dhsic_mode_disables_pair_diagnostics(self):
        """No pair matrix / row means are stashed in dhsic_residual mode."""
        fc = AttentionSelectorForecaster(_config("dhsic_residual"))
        S, X = _batch()
        fc.train()
        fc.training_step((S, X), 0)
        assert fc._last_hsic_pair_mat is None
        assert fc._last_hsic_row_means is None

    def test_dhsic_permutation_null_uses_column_wise_permutation(self):
        """Null replicates must permute each residual column independently.

        Row-wise permutation preserves inter-column dependence and would NOT
        be a valid null for the mutual-independence dHSIC.  On strongly
        dependent residuals the observed statistic must sit clearly ABOVE
        the permutation-null replicates.
        """
        fc = AttentionSelectorForecaster(_config("dhsic_residual"))
        g = torch.Generator().manual_seed(7)
        n = 128
        e0 = torch.randn(n, generator=g)
        # Residuals with a shared component: columns are mutually dependent.
        residuals = torch.stack(
            [e0 + 0.1 * torch.randn(n, generator=g),
             e0 - 0.1 * torch.randn(n, generator=g),
             0.5 * e0 + 0.1 * torch.randn(n, generator=g)],
            dim=1,
        )
        combined_source = residuals  # unused in dhsic mode
        reps = fc._hsic_permutation_replicates(
            combined_source=combined_source,
            residuals=residuals,
            attention_weights=None,
            bkd_keep_mask=None,
            hsic_pair_mask=None,
            n_perms=20,
            generator=torch.Generator().manual_seed(11),
        )
        assert len(reps) == 20
        assert all(torch.isfinite(torch.tensor(r)) for r in reps)
        observed = float(fc._dhsic_residual_hsic(residuals))
        null_mean = sum(reps) / len(reps)
        assert observed > null_mean, (
            f"dependent residuals must exceed the null: observed={observed}, "
            f"null_mean={null_mean}"
        )
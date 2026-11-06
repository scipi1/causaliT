"""
Non-equal-variance (NV) scores for the continuous-optimisation baselines.

Ng, Huang, Zhang (CLeaR 2024, "Structure Learning with Continuous Optimization:
A Sober Look and Beyond") show that least-squares / shared-variance scores
implicitly assume equal noise variances, an assumption broken by standardised
data.  For a fair comparison they swap each method's score for the GOLEM-NV
likelihood and keep every other hyperparameter unchanged.  This module does
the same for DAGMA (linear, MLP) and DAGuerreotype:

    S_NV = 0.5 * sum_j log( (1/n) * sum_i r_ij^2 )        r = X - f(X)

GOLEM's ``-log|det(I - W)|`` term is dropped: it is exactly zero on a DAG,
and all three methods enforce acyclicity.

The original (equal-variance) wrappers are untouched; these classes back the
separate ``*_nv`` methods.  Third-party imports are lazy, as elsewhere in
``methods/``.
"""

import typing

import numpy as np

#: Floor inside the log, guards against a perfectly fitted column.
EPS = 1e-12

#: ``dagma`` version whose ``DagmaLinear.minimize`` is mirrored below.
DAGMA_VERSION_MIRRORED = "1.1.1"


def linear_nv_score(W: np.ndarray, cov: np.ndarray) -> typing.Tuple[float, np.ndarray]:
    """
    NV score and gradient of a linear SEM, from the centred covariance.

    Args:
        W: ``(d, d)`` weights, ``W[i, j] != 0`` means ``i -> j``.
        cov: ``(d, d)`` centred covariance ``X^T X / n``.

    Returns:
        ``(loss, grad)`` with ``loss = 0.5 * sum_j log v_j`` and
        ``grad = -cov (I - W) diag(1 / v)``, ``v_j`` the residual variance of
        column j.
    """
    dif = np.eye(W.shape[0]) - W
    rhs = cov @ dif
    v = np.maximum(np.einsum("ij,ij->j", dif, rhs), EPS)   # diag(dif^T cov dif)
    return 0.5 * float(np.log(v).sum()), -rhs / v[None, :]


def make_dagma_linear_nv():
    """Build the ``DagmaLinearNV`` class (lazy: ``dagma`` is optional)."""
    import scipy.linalg as sla
    from dagma.linear import DagmaLinear

    class DagmaLinearNV(DagmaLinear):
        """
        ``DagmaLinear`` with the NV score.

        Runs as ``loss_type='l2'`` so the package keeps centring the data and
        computing ``cov``.  ``minimize`` mirrors dagma 1.1.1 line by line; the
        only change is the score gradient (marked ``NV CHANGE``), which the
        package hard-codes inline instead of calling ``_score``.
        """

        def __init__(self, verbose: bool = False, dtype: type = np.float64) -> None:
            super().__init__(loss_type="l2", verbose=verbose, dtype=dtype)

        def _score(self, W: np.ndarray) -> typing.Tuple[float, np.ndarray]:
            return linear_nv_score(W, self.cov)

        def minimize(self, W, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99,
                     beta_2=0.999, pbar=None):
            obj_prev = 1e16
            self.opt_m, self.opt_v = 0, 0
            mask_inc = np.zeros((self.d, self.d))
            if self.inc_c is not None:
                mask_inc[self.inc_r, self.inc_c] = -2 * mu * self.lambda1
            mask_exc = np.ones((self.d, self.d), dtype=self.dtype)
            if self.exc_c is not None:
                mask_exc[self.exc_r, self.exc_c] = 0.

            for iter in range(1, max_iter + 1):
                M = sla.inv(s * self.Id - W * W) + 1e-16
                while np.any(M < 0):  # sI - W o W is not an M-matrix
                    if iter == 1 or s <= 0.9:
                        return W, False
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, True
                    W -= lr * grad
                    M = sla.inv(s * self.Id - W * W) + 1e-16

                # NV CHANGE: was ``G_score = -mu * self.cov @ (self.Id - W)``
                G_score = mu * self._score(W)[1]

                Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T \
                    + mask_inc * np.sign(W)
                grad = self._adam_update(Gobj, iter, beta_1, beta_2)
                W -= lr * grad
                W *= mask_exc

                if iter % self.checkpoint == 0 or iter == max_iter:
                    obj_new, _, _ = self._func(W, mu, s)
                    if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                        if pbar is not None:
                            pbar.update(max_iter - iter + 1)
                        break
                    obj_prev = obj_new
                if pbar is not None:
                    pbar.update(1)
            return W, True

    return DagmaLinearNV


def make_dagma_nonlinear_nv():
    """Build the ``DagmaNonlinearNV`` class (lazy: torch / ``dagma`` optional)."""
    import torch
    from dagma.nonlinear import DagmaNonlinear

    class DagmaNonlinearNV(DagmaNonlinear):
        """``DagmaNonlinear`` with per-variable log-MSE instead of a pooled one."""

        def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            mse = torch.mean((output - target) ** 2, dim=0)
            return 0.5 * torch.sum(torch.log(torch.clamp(mse, min=EPS)))

    return DagmaNonlinearNV


def nll_nv(output, target, dim=(-2, -1)):
    """
    DAGuerreotype NLL with non-equal variances (counterpart of ``nll_ev``).

    Same tensor layout as ``daguerreo.utils.nll_ev``: the leading dimension
    runs over DAGs, ``dim`` holds ``(samples, variables)``.  Uses sums like
    ``nll_ev``; on equal residual variances both differ only by the constant
    ``d/2 * log(d)``.
    """
    import torch

    sq = (output - target).square().sum(dim=dim[0], keepdim=True)  # per variable
    return 0.5 * torch.log(torch.clamp(sq, min=EPS)).sum(dim=dim)

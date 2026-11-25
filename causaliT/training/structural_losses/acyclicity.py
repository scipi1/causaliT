"""
Acyclicity penalties h(A) >= 0, zero iff A induces a DAG.

All take a square, NONNEGATIVE score matrix A (the gate posterior) of
shape (d, d) and return a scalar tensor.  Config (``training``):

    acyclicity_fn: notears | logdet | nilpotent      (default notears)
    acyclicity_s:  "adaptive" | positive float       (logdet only)

Coding Assistant:
    * The forecaster builds ONE penalty with :func:`build_acyclicity_penalty`
      and calls it; the kappa / ALM weighting stays in the forecaster.
    * Deprecated forecasters still carry their own NOTEARS copy; do not add
      new callers there.
"""

from dataclasses import dataclass
from typing import Union

import torch

ACYCLICITY_FNS = ("notears", "logdet", "nilpotent")


def notears(A: torch.Tensor) -> torch.Tensor:
    """NOTEARS h(A) = tr(exp(A * A)) - d (Zheng et al., 2018)."""
    return torch.trace(torch.matrix_exp(A * A)) - A.shape[-1]


def logdet(A: torch.Tensor, s: Union[str, float] = "adaptive") -> torch.Tensor:
    """
    DAGMA log-det h(A) = -log det(sI - A) + d log s (Bello et al., 2022).

    ``s="adaptive"`` uses max row sum (stop-grad) + 1e-4, so sI - A is always
    nonsingular.  A fixed float raises ``FloatingPointError`` if not.
    No elementwise square: A is already nonnegative.
    """
    d = A.shape[-1]
    if isinstance(s, str):
        assert s == "adaptive"
        s_val = A.detach().sum(dim=-1).max() + 1e-4
    else:
        s_val = torch.as_tensor(float(s), dtype=A.dtype, device=A.device)
    eye = torch.eye(d, dtype=A.dtype, device=A.device)
    sign, logabsdet = torch.linalg.slogdet(s_val * eye - A)
    sign_f = float(sign.detach())
    if not torch.isfinite(logabsdet.detach()) or sign_f <= 0.0:
        raise FloatingPointError(
            f"log-det acyclicity: det(sI - A) non-positive (sign={sign_f}); "
            f"rho(A) >= s.  Use acyclicity_s='adaptive' or a larger fixed shift."
        )
    return -logabsdet + d * torch.log(s_val)


def nilpotent(A: torch.Tensor) -> torch.Tensor:
    """Exact h(A) = sum_{k=1..d} tr(A^k): zero iff no closed walk (A >= 0)."""
    h = torch.zeros((), dtype=A.dtype, device=A.device)
    Ak = A
    for _ in range(A.shape[-1]):
        h = h + torch.trace(Ak)
        Ak = Ak @ A
    return h


@dataclass(frozen=True)
class AcyclicityPenalty:
    """Configured acyclicity penalty; call it on a (d, d) score matrix."""

    name: str = "notears"
    s: Union[str, float] = "adaptive"

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        if self.name == "logdet":
            return logdet(A, self.s)
        if self.name == "nilpotent":
            return nilpotent(A)
        return notears(A)


def build_acyclicity_penalty(training_cfg: dict) -> AcyclicityPenalty:
    """Validate ``acyclicity_fn`` / ``acyclicity_s`` and build the penalty."""
    name = str(training_cfg.get("acyclicity_fn", "notears"))
    if name not in ACYCLICITY_FNS:
        raise ValueError(
            f"acyclicity_fn must be one of 'notears' | 'logdet' | 'nilpotent', got {name!r}"
        )
    s = training_cfg.get("acyclicity_s", "adaptive")
    if isinstance(s, str):
        if s != "adaptive":
            raise ValueError(f"acyclicity_s must be 'adaptive' or a positive float, got {s!r}")
    else:
        s = float(s)
        if s <= 0.0:
            raise ValueError(f"acyclicity_s must be positive, got {s}")
    return AcyclicityPenalty(name=name, s=s)

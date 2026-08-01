"""
Benchmark method contract and registry.

Every benchmark is a callable with one fixed signature::

    fit(X: np.ndarray, **params) -> BenchmarkResult

where ``X`` is the ``(n_samples, N)`` design matrix from ``data.py`` and the
result carries the estimated weighted adjacency in **paper orientation**
(``W[i, j] != 0`` means ``i -> j``).  The single conversion to causaliT's
canonical orientation happens in ``postprocess.py``, so no method module has to
think about it.

Two deliberate design choices:

1. **Paper hyperparameters are frozen per method** (``DEFAULT_PARAMS``).  The
   nonlinear variants are per-node MLPs with ``dims=[d, H, 1]``, and both papers
   use ``H = 10`` for every graph size they report, so the architecture is
   size-independent by construction and nothing needs to be tuned or derived
   from ``N``.  ``benchmark.params`` in the config can still override any value,
   but the sweeps never search over them.
2. **Optional dependencies are imported lazily** inside each method module, so a
   missing ``dagma`` or ``causal-learn`` install disables exactly one method
   instead of breaking the package import.  ``available_methods()`` reports what
   can currently run.
"""

from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class BenchmarkResult:
    """
    Output of one benchmark fit.

    Attributes:
        W: ``(N, N)`` estimated weighted adjacency in **paper orientation**
            (row = parent, column = child).  Unthresholded whenever the method
            allows it, so any ``w_threshold`` can be re-scored offline.
        seconds: Wall-clock fit time.
        params: The hyperparameters actually used (defaults + overrides).
        is_binary: True when the method yields a discrete graph rather than
            weights (PC), which tells ``postprocess`` not to threshold.
        extra: Free-form diagnostics stored in ``benchmark_run.json``.
    """

    W: np.ndarray
    seconds: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    is_binary: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


#: Method name -> ``<module>:<callable>`` inside ``causaliT.benchmarks.methods``.
#: Kept as strings so importing this module never imports torch, dagma or
#: causal-learn.
METHOD_SPECS: Dict[str, str] = {
    "notears_linear": "notears_linear:fit",
    "notears_mlp": "notears_mlp:fit",
    "dagma_linear": "dagma_linear:fit",
    "dagma_mlp": "dagma_mlp:fit",
    "pc": "pc:fit",
}

#: Short human-readable description per method, used by the CLI listing.
METHOD_DESCRIPTIONS: Dict[str, str] = {
    "notears_linear": "NOTEARS, linear SEM (vendored paper code)",
    "notears_mlp": "NOTEARS, per-node MLP dims=[d,10,1] (vendored paper code)",
    "dagma_linear": "DAGMA, linear SEM (dagma package)",
    "dagma_mlp": "DAGMA, per-node MLP dims=[d,10,1] (dagma package)",
    "pc": "PC constraint-based, Fisher-z (causal-learn package)",
}

#: Third-party package each method needs, or None when vendored/stdlib only.
METHOD_REQUIREMENTS: Dict[str, Optional[str]] = {
    "notears_linear": None,          # vendored + scipy
    "notears_mlp": "torch",          # vendored + torch
    "dagma_linear": "dagma",
    "dagma_mlp": "dagma",
    "pc": "causallearn",
}


def method_names() -> List[str]:
    """All registered method names, in a stable order."""
    return list(METHOD_SPECS)


def resolve_method(name: str) -> Callable[..., BenchmarkResult]:
    """
    Import and return the ``fit`` callable of a registered method.

    Args:
        name: Method name, e.g. ``dagma_mlp``.

    Returns:
        The method's ``fit(X, **params) -> BenchmarkResult``.

    Raises:
        KeyError: unknown method name (message lists the valid ones).
        ImportError: the method's optional dependency is not installed; the
            message names the missing package and the pip install line.
    """
    if name not in METHOD_SPECS:
        raise KeyError(
            f"Unknown benchmark method '{name}'. Available: {', '.join(method_names())}."
        )
    module_name, attr = METHOD_SPECS[name].split(":")
    try:
        module = import_module(f"causaliT.benchmarks.methods.{module_name}")
    except ImportError as exc:
        requirement = METHOD_REQUIREMENTS.get(name)
        hint = f" Install it with: pip install {requirement}" if requirement else ""
        raise ImportError(
            f"Benchmark '{name}' is unavailable: {exc}.{hint}"
        ) from exc
    return getattr(module, attr)


def is_available(name: str) -> bool:
    """
    Whether *name* can actually run right now.

    Importing the wrapper module is not enough: the third-party imports inside
    ``fit`` are deliberately lazy, so a wrapper for a missing package imports
    fine.  The probe therefore also checks ``METHOD_REQUIREMENTS[name]`` with
    ``importlib.util.find_spec``, which locates the package without executing it.
    """
    try:
        resolve_method(name)
    except Exception:  # noqa: BLE001 - availability probe, any failure means "no"
        return False

    requirement = METHOD_REQUIREMENTS.get(name)
    if requirement is None:
        return True
    try:
        return find_spec(requirement) is not None
    except Exception:  # noqa: BLE001 - broken/partial install counts as missing
        return False


def available_methods() -> Dict[str, bool]:
    """Map every registered method to its current availability."""
    return {name: is_available(name) for name in method_names()}


def default_params(name: str) -> Dict[str, Any]:
    """
    Paper default hyperparameters of a method (a fresh copy).

    Raises:
        KeyError: unknown method.
        ImportError: optional dependency missing.
    """
    module_name = METHOD_SPECS[name].split(":")[0] if name in METHOD_SPECS else None
    if module_name is None:
        raise KeyError(
            f"Unknown benchmark method '{name}'. Available: {', '.join(method_names())}."
        )
    module = import_module(f"causaliT.benchmarks.methods.{module_name}")
    return dict(getattr(module, "DEFAULT_PARAMS", {}))


def merge_params(name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Paper defaults updated with *overrides* (``benchmark.params`` in the config).

    Unknown keys are kept: each method validates its own arguments, and passing
    an unexpected one should fail loudly rather than be silently dropped.
    """
    params = default_params(name)
    for key, value in (overrides or {}).items():
        params[key] = value
    return params


__all__ = [
    "BenchmarkResult",
    "METHOD_SPECS",
    "METHOD_DESCRIPTIONS",
    "METHOD_REQUIREMENTS",
    "method_names",
    "resolve_method",
    "is_available",
    "available_methods",
    "default_params",
    "merge_params",
]

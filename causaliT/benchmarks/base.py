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
            weights (PC, order-based methods), which tells ``postprocess`` not
            to threshold.
        extra: Free-form diagnostics stored in ``benchmark_run.json``.
        output_type: ``"weighted"``, ``"dag"``, ``"cpdag"`` or ``"samples"``.
            Derived from ``is_binary`` when left as None.
        order: Estimated causal order (node indices, most upstream first), or
            None for methods without an explicit order.
        graph_samples: ``(S, N, N)`` sampled binary graphs (paper orientation)
            for probabilistic methods, else None.
        edge_probs: ``(N, N)`` edge probabilities (paper orientation); filled
            from ``graph_samples`` when omitted.

    The optional fields drive which metrics apply (``metrics.py``); a method
    only fills what it genuinely produces.
    """

    W: np.ndarray
    seconds: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    is_binary: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
    output_type: Optional[str] = None
    order: Optional[List[int]] = None
    graph_samples: Optional[np.ndarray] = None
    edge_probs: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.output_type is None:
            if self.graph_samples is not None:
                self.output_type = "samples"
            else:
                self.output_type = "dag" if self.is_binary else "weighted"
        if self.output_type not in OUTPUT_TYPES:
            raise ValueError(
                f"output_type must be one of {OUTPUT_TYPES}, got '{self.output_type}'."
            )
        if self.graph_samples is not None and self.edge_probs is None:
            self.edge_probs = (np.asarray(self.graph_samples) != 0).mean(axis=0)
        if self.order is not None:
            self.order = [int(i) for i in self.order]


#: Allowed values of ``BenchmarkResult.output_type``.
OUTPUT_TYPES = ("weighted", "dag", "cpdag", "samples")


#: Method name -> ``<module>:<callable>`` inside ``causaliT.benchmarks.methods``.
#: Kept as strings so importing this module never imports torch, dagma or
#: causal-learn.
METHOD_SPECS: Dict[str, str] = {
    "dagma_linear": "dagma_linear:fit",
    "dagma_mlp": "dagma_mlp:fit",
    "pc": "pc:fit",
    "resit": "resit:fit",
    "direct_lingam": "direct_lingam:fit",
    "cam": "toporder:fit_cam",
    "score": "toporder:fit_score",
    "das": "toporder:fit_das",
    "nogam": "toporder:fit_nogam",
    "caps": "caps:fit",
    "daguerreo": "daguerreo:fit",
    "var_sort": "sortnregress:fit_var",
    "r2_sort": "sortnregress:fit_r2",
}

#: Short human-readable description per method, used by the CLI listing.
METHOD_DESCRIPTIONS: Dict[str, str] = {
    "dagma_linear": "DAGMA, linear SEM (dagma package)",
    "dagma_mlp": "DAGMA, per-node MLP dims=[d,10,1] (dagma package)",
    "pc": "PC constraint-based, Fisher-z (causal-learn package)",
    "resit": "RESIT, residual-independence order search (lingam package)",
    "direct_lingam": "DirectLiNGAM, linear non-Gaussian order + adaptive Lasso (lingam package)",
    "cam": "CAM, greedy additive-model order + CAM pruning (dodiscover)",
    "score": "SCORE, Stein-Hessian order + CAM pruning (dodiscover)",
    "das": "DAS, SCORE order + Hessian edge selection (dodiscover)",
    "nogam": "NoGAM, score-residual order + CAM pruning (dodiscover)",
    "caps": "CaPS, Hessian order + parent score (vendored, Python pruning)",
    "daguerreo": "DAGuerreotype, permutahedron order learning (external repo)",
    "var_sort": "Var-sortnregress, trivial variance-order baseline (CausalDisco)",
    "r2_sort": "R2-sortnregress, trivial R2-order baseline (CausalDisco)",
}

#: Third-party package each method needs, or None when vendored/stdlib only.
METHOD_REQUIREMENTS: Dict[str, Optional[str]] = {
    "dagma_linear": "dagma",
    "dagma_mlp": "dagma",
    "pc": "causallearn",
    "resit": "lingam",
    "direct_lingam": "lingam",
    "cam": "dodiscover",
    "score": "dodiscover",
    "das": "dodiscover",
    "nogam": "dodiscover",
    "caps": "dodiscover",            # vendored order search + dodiscover pruning
    "daguerreo": "daguerreo",        # compiled extensions; see BENCHMARKS.md
    "var_sort": "CausalDisco",
    "r2_sort": "CausalDisco",
}

#: Default number of init seeds per method (seed policy).  Deterministic
#: methods are refitted once; stochastic ones (random init) three times.
#: Used when ``benchmark.seeds`` is omitted from the config.
METHOD_DEFAULT_N_SEEDS: Dict[str, int] = {
    "dagma_linear": 1,
    "dagma_mlp": 3,
    "pc": 1,
    "resit": 1,
    "direct_lingam": 1,
    "cam": 1,
    "score": 1,
    "das": 1,
    "nogam": 1,
    "caps": 1,
    "daguerreo": 3,
    "var_sort": 1,
    "r2_sort": 1,
}


def default_seeds(name: str) -> List[int]:
    """Default init seeds ``[0, ..., k-1]`` for *name* (see METHOD_DEFAULT_N_SEEDS)."""
    return list(range(METHOD_DEFAULT_N_SEEDS.get(name, 1)))


#: Methods considered and deliberately NOT included, with the reason.  Kept in
#: code so the CLI and the docs report the same list.
EXCLUDED_METHODS: Dict[str, str] = {
    "notears_linear": "dropped: superseded by DAGMA (same continuous-penalty family)",
    "notears_mlp": "dropped: superseded by DAGMA-MLP",
    "vi_dp_dag": "no licence in the authors' repository (Charpentier et al., 2022)",
    "bcd_nets": "no licence in the authors' repository (Cundy et al., 2021)",
    "diffintersort": "needs interventional data; no public code (Chevalley et al., 2024)",
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
    if name not in METHOD_SPECS:
        raise KeyError(
            f"Unknown benchmark method '{name}'. Available: {', '.join(method_names())}."
        )
    module_name, attr = METHOD_SPECS[name].split(":")
    module = import_module(f"causaliT.benchmarks.methods.{module_name}")
    # Modules hosting several methods keep one defaults dict per fit callable.
    by_fit = getattr(module, "DEFAULT_PARAMS_BY_FIT", None)
    if by_fit is not None:
        return dict(by_fit.get(attr, {}))
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
    "OUTPUT_TYPES",
    "EXCLUDED_METHODS",
    "METHOD_SPECS",
    "METHOD_DESCRIPTIONS",
    "METHOD_REQUIREMENTS",
    "METHOD_DEFAULT_N_SEEDS",
    "default_seeds",
    "method_names",
    "resolve_method",
    "is_available",
    "available_methods",
    "default_params",
    "merge_params",
]

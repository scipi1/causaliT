"""
Device-aware activation budget for the size-derived batch size.

The batch size cannot be a constant in a scaling benchmark: attention activations
grow like ``B * N * H * (N + d)``, so a batch that is comfortable on a 6-node DAG
OOMs on a 400-node one.  ``search_space.activation_batch_size`` therefore solves

    B = C / (N * H * (N + d))

for the batch, where ``C`` is a single number describing *how much activation the
device can hold*.  This module is where ``C`` comes from.

Two ways to obtain it, in order of preference:

1. MEASURED (``calibrate_activation_budget``): read the real total memory of the
   visible accelerator and convert it into an activation budget.  The result is
   cached per device name, so every later sweep on that machine reuses it and the
   YAML stays portable.
2. DECLARED: ``C`` given explicitly in ``dagsweep.yaml``.  Useful to pin a value
   for reproducibility, or to shrink it after an OOM.

If neither is available we fall back to ``DEFAULT_ACTIVATION_BUDGET`` and say so
in the log - an unnoticed default is exactly how a benchmark ends up with
different effective batch sizes on different nodes.

Note on fairness: whatever ``C`` is, it must be IDENTICAL for the Optuna search
and for the evaluation runs of the same group, otherwise the tuned learning rate
belongs to a batch size that is never used again.  Both paths call
``resolve_budget`` with the same ``size_derived`` spec, which guarantees that.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Fallback used when nothing is declared and no calibration is cached.
#: Corresponds to roughly a 24 GB card at fp32 (see estimate_budget defaults).
DEFAULT_ACTIVATION_BUDGET = 4.9e8

#: Fraction of device memory that may hold activations.  The rest pays for
#: parameters, optimiser state, gradients, the CUDA context and fragmentation.
DEFAULT_SAFETY = 0.35

#: Number of activation tensors of the dominant shape kept alive by autograd
#: (embeddings, Q/K/V, attention maps, FF, residuals, ...).  Empirical.
DEFAULT_MULTIPLICITY = 12


def cache_path() -> Path:
    """Location of the per-machine calibration cache."""
    root = os.environ.get("CAUSALIT_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".causalit"
    return base / "activation_budget.json"


def device_key() -> str:
    """Stable identifier of the current accelerator (or ``cpu``)."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # torch missing or driver error -> treat as CPU
        pass
    return "cpu"


def load_cache() -> Dict[str, float]:
    path = cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r") as fh:
            return {str(k): float(v) for k, v in json.load(fh).items()}
    except Exception as exc:  # a corrupt cache must not break a sweep
        logger.warning("Ignoring unreadable activation-budget cache %s: %s",
                       path, exc)
        return {}


def save_cache(cache: Dict[str, float]) -> Path:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
    return path


def device_memory_bytes() -> Optional[int]:
    """Total memory of the visible accelerator, or ``None`` on CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    return None


def estimate_budget(total_bytes: int, dtype_bytes: int = 4,
                    multiplicity: int = DEFAULT_MULTIPLICITY,
                    safety: float = DEFAULT_SAFETY) -> float:
    """
    Convert device memory into an activation-element budget ``C``.

    ``C`` counts ELEMENTS of the dominant activation shape, so it is independent
    of dtype and of the model width::

        C = safety * total_bytes / (dtype_bytes * multiplicity)

    ``multiplicity`` is how many such tensors autograd keeps alive at once; it is
    the one empirical number here.  Halve ``safety`` after an OOM rather than
    guessing a batch size by hand: the rule then scales down at every DAG size at
    once, which keeps the comparison across sizes consistent.
    """
    if total_bytes <= 0:
        raise ValueError(f"total_bytes must be positive, got {total_bytes}")
    return float(safety) * float(total_bytes) / (float(dtype_bytes) * float(multiplicity))


def calibrate_activation_budget(dtype_bytes: int = 4,
                                multiplicity: int = DEFAULT_MULTIPLICITY,
                                safety: float = DEFAULT_SAFETY,
                                write_cache: bool = True) -> Dict[str, Any]:
    """
    Measure and (optionally) cache the activation budget of this machine.

    Returns a report dict; on CPU-only machines it falls back to the default so
    that a local smoke test still runs.
    """
    key = device_key()
    total = device_memory_bytes()

    if total is None:
        budget = DEFAULT_ACTIVATION_BUDGET
        logger.warning("No CUDA device visible: using the default budget %.3g", budget)
    else:
        budget = estimate_budget(total, dtype_bytes, multiplicity, safety)

    report = {
        "device": key,
        "total_bytes": total,
        "dtype_bytes": dtype_bytes,
        "multiplicity": multiplicity,
        "safety": safety,
        "C": budget,
    }

    if write_cache:
        cache = load_cache()
        cache[key] = float(budget)
        report["cache_path"] = str(save_cache(cache))

    return report


def resolve_budget(declared: Optional[Any] = None) -> float:
    """
    Resolve ``C``: declared value > cached calibration > default.

    Accepts ``None`` / ``"auto"`` for "use the cache, else the default", so a
    ``dagsweep.yaml`` can stay device-agnostic.
    """
    if declared is not None and str(declared).lower() != "auto":
        return float(declared)

    cache = load_cache()
    key = device_key()
    if key in cache:
        return float(cache[key])

    logger.warning(
        "No activation budget calibrated for device '%s'; using the default "
        "C=%.3g. Run 'cli calibrate-batch-budget' once per machine to measure it.",
        key, DEFAULT_ACTIVATION_BUDGET,
    )
    return DEFAULT_ACTIVATION_BUDGET


__all__ = [
    "DEFAULT_ACTIVATION_BUDGET",
    "DEFAULT_MULTIPLICITY",
    "DEFAULT_SAFETY",
    "cache_path",
    "calibrate_activation_budget",
    "device_key",
    "device_memory_bytes",
    "estimate_budget",
    "load_cache",
    "resolve_budget",
    "save_cache",
]

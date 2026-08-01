"""
Unified training summary.

One file format, written by every path that fits a model:

    <run_dir>/training_summary.json

Both the causal-attention trainers and the benchmark methods (NOTEARS, DAGMA,
PC, ...) produce this file, so the training record of a run is readable without
knowing which method produced it.

Scope
-----
This file holds the TRAINING record only:

  - runtime (seconds per fit, epochs, whether the budget was exhausted),
  - final training/validation/test metrics (loss, MAE, R2, HSIC, ...),
  - model size and the environment the fit ran on.

Structural (DAG) evaluation belongs to ``eval/`` and is produced by the
post-training evaluation functions; it must NOT be written here.

The "fit" abstraction
---------------------
A *fit* is one optimisation run that produced one estimate:

  - causal-attention models: one cross-validation fold  (``k_0``, ``k_1``, ...)
  - benchmark methods:       one seed                   (``seed_0``, ...)

These are the same object for reporting purposes, which is what lets runtime
and structural metrics be averaged the same way across methods (mean/std over
the repetition axis).

Schema (schema_version 1)
-------------------------
    {
      "schema_version": 1,
      "run": {"kind": "model"|"benchmark", "method": str, "dataset": str|None,
              "save_dir": str, "timestamp": str},
      "environment": {"device": str, "n_threads": int, "python": str,
                      "torch": str|None},
      "fits": [ {"id": str, "method": str, "seconds": float|None,
                 "epochs_run": int|None, "max_epochs": int|None,
                 "stopped_early": bool|None, "avg_time_per_epoch": float|None,
                 "trainable_params": int|None, "converged": bool|None,
                 "iterations": int|None, "checkpoint": str|None,
                 "metrics": {..}} ],
      "statistics": {"<method>": {"<field>": {"mean","std","min","max","n"}}},
      "best_fit": {"id","method","selection_criterion","selection_value",
                   "checkpoint","metrics"}
    }

Conventions
-----------
  - Missing quantities are OMITTED, never zero-filled.  A benchmark has no
    ``epochs_run`` and no ``test_r2``; a consumer must see the absence rather
    than a fake ``0.0``.
  - Everything is converted to JSON-native types AT WRITE TIME (no
    ``"tensor(0.0005)"`` strings), so no post-hoc repair pass is needed.
  - ``statistics`` is computed per method, so a benchmark run that fits several
    methods in one folder does not average them together.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from datetime import datetime
from os.path import exists, join
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Canonical file name written by every training path.
TRAINING_SUMMARY_FILE = "training_summary.json"

#: Legacy file name, still present in previously finished experiments.  Read
#: (via :func:`load_training_summary`) but never written.
LEGACY_SUMMARY_FILE = "kfold_summary.json"

#: Markers identifying a folder as a completed run, newest first.
TRAINED_RUN_MARKERS = (TRAINING_SUMMARY_FILE, LEGACY_SUMMARY_FILE)

#: Metrics used to pick the best fit, in order of preference.  HSIC first:
#: within a run the folds cluster together and minimum validation HSIC is a
#: reliable proxy for structural quality when ground truth is unavailable.
BEST_FIT_PRIORITY = ("val_hsic_reg", "val_hsic_cross", "val_hsic", "val_x_mae")

#: Runtime fields promoted out of ``metrics`` onto the fit itself.  Keeping
#: them structural (rather than just another metric key) is what makes the
#: runtime comparable across methods.
RUNTIME_FIELDS = (
    "seconds",
    "epochs_run",
    "max_epochs",
    "stopped_early",
    "avg_time_per_epoch",
    "trainable_params",
    "converged",
    "iterations",
)


# ---------------------------------------------------------------------------
# JSON coercion
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    """
    Convert a value to something ``json.dump`` handles natively.

    Tensors and numpy scalars become floats, arrays become lists.  This is the
    reason the summary needs no post-hoc repair: the conversion happens once,
    at write time, instead of leaving ``"tensor(0.0005)"`` strings behind.
    """
    # torch is optional here: the benchmark path must not require it.
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
    except ImportError:
        pass

    if isinstance(value, np.ndarray):
        return float(value) if value.ndim == 0 else value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        # JSON has no NaN/Infinity; null round-trips to None -> NaN downstream.
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_number(value: Any) -> bool:
    """True for a finite real scalar (bools excluded: they are flags, not data)."""
    if isinstance(value, bool) or value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    return not (np.isnan(value) or np.isinf(value))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def describe_environment(device: Optional[str] = None) -> Dict[str, Any]:
    """
    Capture what the fit ran on.

    Runtime is only interpretable next to the hardware that produced it: the
    causal-attention models train on GPU while the benchmark methods are
    single-process CPU fits, so a runtime comparison without this block is
    partly a hardware comparison presented as a method claim.
    """
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    n_threads = None
    torch_version = None
    try:
        import torch

        torch_version = torch.__version__
        n_threads = torch.get_num_threads()
        if device is None:
            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.get_device_name(0)}"
            else:
                device = "cpu"
    except ImportError:
        if device is None:
            device = "cpu"

    if n_threads is None:
        n_threads = os.cpu_count()

    env["device"] = device
    env["n_threads"] = n_threads
    if torch_version is not None:
        env["torch"] = torch_version
    return env


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class TrainingSummaryWriter:
    """
    Accumulate fits and write ``training_summary.json``.

    The file is rewritten after every :meth:`add_fit`, so a run interrupted
    half-way still leaves a valid summary of the fits that completed.

    Example
    -------
        writer = TrainingSummaryWriter(save_dir, kind="model", method="atsel")
        writer.add_fit("k_0", seconds=812.4, epochs_run=140, max_epochs=200,
                       stopped_early=True, metrics={"val_x_mae": 0.031})
        writer.finalize()
    """

    def __init__(
        self,
        save_dir: str,
        kind: str,
        method: str,
        dataset: Optional[str] = None,
        device: Optional[str] = None,
        extra_run: Optional[Dict[str, Any]] = None,
    ):
        if kind not in ("model", "benchmark"):
            raise ValueError(f"kind must be 'model' or 'benchmark', got {kind!r}")

        self.save_dir = str(save_dir)
        self.path = join(self.save_dir, TRAINING_SUMMARY_FILE)
        self.run: Dict[str, Any] = {
            "kind": kind,
            "method": method,
            "dataset": dataset,
            "save_dir": self.save_dir,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if extra_run:
            self.run.update(_json_safe(extra_run))
        self.environment = describe_environment(device)
        self.fits: List[Dict[str, Any]] = []

    # -- fits ---------------------------------------------------------------

    def add_fit(
        self,
        fit_id: str,
        metrics: Optional[Dict[str, Any]] = None,
        method: Optional[str] = None,
        checkpoint: Optional[str] = None,
        **runtime_fields: Any,
    ) -> Dict[str, Any]:
        """
        Record one fit (a fold for models, a seed for benchmarks).

        Unknown keyword arguments are rejected rather than silently stored, so
        a typo in a runtime field surfaces immediately instead of producing a
        column that is quietly always missing.
        """
        unknown = set(runtime_fields) - set(RUNTIME_FIELDS)
        if unknown:
            raise TypeError(
                f"unknown runtime field(s) {sorted(unknown)}; "
                f"expected any of {list(RUNTIME_FIELDS)}"
            )

        fit: Dict[str, Any] = {
            "id": str(fit_id),
            "method": method or self.run["method"],
        }
        # Omit rather than zero-fill: absence is information.
        for key in RUNTIME_FIELDS:
            if key in runtime_fields and runtime_fields[key] is not None:
                fit[key] = _json_safe(runtime_fields[key])

        # Derive the per-epoch cost when both parts are known, so callers that
        # only time the whole fit still get the size-normalised number.
        if (
            "avg_time_per_epoch" not in fit
            and _is_number(fit.get("seconds"))
            and _is_number(fit.get("epochs_run"))
            and fit["epochs_run"] > 0
        ):
            fit["avg_time_per_epoch"] = fit["seconds"] / fit["epochs_run"]

        if checkpoint is not None:
            fit["checkpoint"] = str(checkpoint)

        cleaned = {k: _json_safe(v) for k, v in (metrics or {}).items()}
        # Runtime lives on the fit, not among the metrics; drop duplicates so
        # there is exactly one place to read each quantity from.
        fit["metrics"] = {k: v for k, v in cleaned.items() if k not in RUNTIME_FIELDS}
        for key in RUNTIME_FIELDS:
            if key not in fit and cleaned.get(key) is not None:
                fit[key] = cleaned[key]

        self.fits.append(fit)
        self._write()
        return fit

    def add_fits(self, fits: List[Dict[str, Any]]) -> None:
        """Record several fits at once (``fit_id`` given as key ``id``)."""
        for entry in fits:
            payload = dict(entry)
            fit_id = payload.pop("id")
            self.add_fit(fit_id, **payload)

    # -- aggregation --------------------------------------------------------

    def _statistics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Aggregate every numeric field per method.

        Fields are aggregated over whichever fits report them; ``n`` records
        how many did, so partial coverage is visible instead of being averaged
        away.
        """
        stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        for method in sorted({f["method"] for f in self.fits}):
            group = [f for f in self.fits if f["method"] == method]
            values: Dict[str, List[float]] = {}
            for fit in group:
                for key in RUNTIME_FIELDS:
                    if _is_number(fit.get(key)):
                        values.setdefault(key, []).append(float(fit[key]))
                for key, value in fit.get("metrics", {}).items():
                    if _is_number(value):
                        values.setdefault(key, []).append(float(value))

            stats[method] = {
                key: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n": len(vals),
                }
                for key, vals in sorted(values.items())
            }
        return stats

    def _best_fit(self) -> Optional[Dict[str, Any]]:
        """Select the best fit by the first available criterion (lower is better)."""
        for criterion in BEST_FIT_PRIORITY:
            candidates = [
                f for f in self.fits if _is_number(f.get("metrics", {}).get(criterion))
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda f: f["metrics"][criterion])
            return {
                "id": best["id"],
                "method": best["method"],
                "selection_criterion": criterion,
                "selection_value": best["metrics"][criterion],
                "checkpoint": best.get("checkpoint"),
                "metrics": best.get("metrics", {}),
            }

        if not self.fits:
            return None
        first = self.fits[0]
        return {
            "id": first["id"],
            "method": first["method"],
            "selection_criterion": "first_available",
            "selection_value": None,
            "checkpoint": first.get("checkpoint"),
            "metrics": first.get("metrics", {}),
        }

    # -- output -------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run": self.run,
            "environment": self.environment,
            "n_fits": len(self.fits),
            "fits": self.fits,
            "statistics": self._statistics(),
        }
        best = self._best_fit()
        if best is not None:
            payload["best_fit"] = best
        return payload

    def _write(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def finalize(self) -> str:
        """Write the summary and return its path."""
        self._write()
        return self.path


# ---------------------------------------------------------------------------
# Reader (with legacy fallback)
# ---------------------------------------------------------------------------

def _from_legacy(payload: Dict[str, Any], run_dir: str) -> Dict[str, Any]:
    """
    Translate a legacy ``kfold_summary.json`` into the current schema.

    Only what the old file actually contains is filled in; the fields it never
    had (device, epochs_run, stopped_early, ...) stay absent rather than being
    invented, so an old run is visibly less informative instead of silently
    looking complete.
    """
    fits: List[Dict[str, Any]] = []
    for fold, entry in (payload.get("fold_results") or {}).items():
        metrics = dict(entry.get("metrics") or {})
        fit: Dict[str, Any] = {"id": f"k_{fold}", "method": "unknown"}
        # The old format stored timings among the metrics.
        if _is_number(metrics.get("total_training_time")):
            fit["seconds"] = float(metrics["total_training_time"])
        if _is_number(metrics.get("avg_time_per_epoch")):
            fit["avg_time_per_epoch"] = float(metrics["avg_time_per_epoch"])
        if _is_number(metrics.get("trainable_params")):
            fit["trainable_params"] = int(metrics["trainable_params"])
        checkpoint = entry.get("best_checkpoint_path")
        if checkpoint:
            fit["checkpoint"] = checkpoint
        fit["metrics"] = metrics
        fits.append(fit)

    best = payload.get("best_fold") or {}
    out: Dict[str, Any] = {
        "schema_version": 0,
        "legacy": True,
        "run": {"kind": "model", "method": "unknown", "save_dir": str(run_dir)},
        "environment": {},
        "n_fits": len(fits),
        "fits": fits,
        "statistics": {"unknown": payload.get("statistics") or {}},
    }
    if best:
        out["best_fit"] = {
            "id": f"k_{best.get('fold_number')}",
            "method": "unknown",
            "selection_criterion": best.get("selection_criterion"),
            "selection_value": best.get("selection_value"),
            "checkpoint": best.get("checkpoint_path"),
            "metrics": best.get("metrics", {}),
        }
    return out


def load_training_summary(run_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load a run's training summary, falling back to the legacy format.

    Returns ``None`` when the folder holds neither file, so callers can tell
    "not a run" from "a run with no metrics".
    """
    new_path = join(run_dir, TRAINING_SUMMARY_FILE)
    if exists(new_path):
        try:
            with open(new_path, "r") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", new_path, exc)
            return None

    legacy_path = join(run_dir, LEGACY_SUMMARY_FILE)
    if exists(legacy_path):
        try:
            with open(legacy_path, "r") as fh:
                return _from_legacy(json.load(fh), run_dir)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", legacy_path, exc)
            return None

    return None


def get_statistic(
    summary: Optional[Dict[str, Any]],
    field: str,
    stat: str = "mean",
    method: Optional[str] = None,
) -> Optional[float]:
    """
    Read one aggregated number, e.g. ``get_statistic(s, "seconds")``.

    Returns ``None`` when the field was never recorded, which callers should
    carry through as ``NaN`` rather than substituting a zero.
    """
    if not summary:
        return None
    stats = summary.get("statistics") or {}
    groups = [stats[method]] if method is not None and method in stats else list(stats.values())
    for group in groups:
        entry = (group or {}).get(field)
        if isinstance(entry, dict) and stat in entry:
            return entry[stat]
    return None


def iter_fits(summary: Optional[Dict[str, Any]]):
    """Yield the fits of a summary (empty when absent)."""
    if not summary:
        return []
    return summary.get("fits") or []

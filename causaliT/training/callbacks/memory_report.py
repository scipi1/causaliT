"""
One-shot GPU memory report of the first training step.

The first train batch of every fit is profiled (forward + loss + backward +
optimizer step) and the result is written to ``memory_report.json`` in the
fold's save directory, plus a compact summary in the training log (which lands
in the SLURM log on cluster runs).

Why: memory failures (see the size-derived batch rule in
``euler_sweep/search_space.py``) are otherwise only visible as a bare CUDA OOM
after the fact.  The report pins down, at essentially zero cost (one profiled
step per fit):

* the process-wide counters before/after the step (allocated / reserved / peak),
* the per-operator self device memory of the step (torch.profiler), so the
  expensive terms (e.g. the per-pair HSIC kernel matrices) are identifiable by
  name instead of by guesswork.

The callback is a strict no-op without CUDA and never interrupts training: any
profiling failure degrades to a counters-only report (or a warning).
"""

import json
import logging
import os
from typing import Any, Dict, List

import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)

#: Default number of operators listed in the per-operator table.
DEFAULT_TOP_OPS = 15


class MemoryReportCallback(Callback):
    """
    Profile the first training batch and write ``memory_report.json``.

    Args:
        save_dir:  Fold directory that receives ``memory_report.json``.
        top_ops:   How many operators to keep in the per-operator table
                   (ranked by self device memory).
    """

    def __init__(self, save_dir: str, top_ops: int = DEFAULT_TOP_OPS):
        self.save_dir = save_dir
        self.top_ops = int(top_ops)
        self._prof = None
        self._done = False
        self._allocated_before = 0
        self._peak_before = 0

    # ------------------------------------------------------------------
    # Profiling lifecycle
    # ------------------------------------------------------------------
    def _make_profiler(self):
        """torch.profiler over the step; CPU-only fallback if CUPTI is missing."""
        from torch.profiler import ProfilerActivity, profile

        try:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           profile_memory=True)
            prof.start()
            return prof
        except Exception as exc:
            logger.warning("MemoryReport: CUDA profiling unavailable (%s); "
                           "falling back to CPU activity", exc)
        prof = profile(activities=[ProfilerActivity.CPU], profile_memory=True)
        prof.start()
        return prof

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._done or batch_idx != 0 or not torch.cuda.is_available():
            return
        self._allocated_before = torch.cuda.memory_allocated()
        self._peak_before = torch.cuda.max_memory_allocated()
        try:
            self._prof = self._make_profiler()
        except Exception as exc:
            # Counters alone still give the headline numbers.
            logger.warning("MemoryReport: profiler disabled (%s)", exc)
            self._prof = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._done or batch_idx != 0 or not torch.cuda.is_available():
            return
        self._done = True
        # Stop BEFORE reading: key_averages() is only populated once the
        # profiler finalises its events.
        if self._prof is not None:
            try:
                self._prof.stop()
            except Exception:
                pass
        try:
            report = self._gather(batch)
            self._write_report(report)
            self._log_summary(report)
        except Exception as exc:  # a diagnostic must never break a run
            logger.warning("MemoryReport: report failed (%s)", exc)
        finally:
            self._prof = None

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _batch_shapes(batch: Any) -> Any:
        if torch.is_tensor(batch):
            return list(batch.shape)
        if isinstance(batch, (list, tuple)):
            return [list(t.shape) if torch.is_tensor(t) else str(type(t))
                    for t in batch]
        return str(type(batch))

    def _top_ops(self) -> List[Dict[str, Any]]:
        """Per-operator self device memory of the profiled step, descending."""
        if self._prof is None:
            return []
        ops = []
        for ev in self._prof.key_averages():
            self_mem = getattr(ev, "self_device_memory_usage", 0) or 0
            if self_mem > 0:
                ops.append({"op": ev.key,
                            "calls": int(ev.count),
                            "self_device_bytes": int(self_mem)})
        ops.sort(key=lambda d: d["self_device_bytes"], reverse=True)
        return ops[: self.top_ops]

    def _gather(self, batch: Any) -> Dict[str, Any]:
        props = torch.cuda.get_device_properties(0)
        return {
            "device": torch.cuda.get_device_name(0),
            "device_total_bytes": int(props.total_memory),
            "batch_idx": 0,
            "batch_shapes": self._batch_shapes(batch),
            # Counters are process-wide; "before" includes model init and the
            # sanity validation pass, so the delta isolates the first step.
            "allocated_before_bytes": int(self._allocated_before),
            "allocated_after_bytes": int(torch.cuda.memory_allocated()),
            "reserved_after_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_before_bytes": int(self._peak_before),
            "peak_allocated_after_bytes": int(torch.cuda.max_memory_allocated()),
            "top_ops_by_self_device_memory": self._top_ops(),
        }

    @rank_zero_only
    def _write_report(self, report: Dict[str, Any]) -> None:
        path = os.path.join(self.save_dir, "memory_report.json")
        os.makedirs(self.save_dir, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2)
        report["path"] = path

    @rank_zero_only
    def _log_summary(self, report: Dict[str, Any]) -> None:
        gib = 1.0 / (1024 ** 3)
        logger.info(
            "[MemoryReport] %s (%.1f GiB total) | first step: allocated "
            "%.2f -> %.2f GiB, peak %.2f GiB | full table: %s",
            report["device"], report["device_total_bytes"] * gib,
            report["allocated_before_bytes"] * gib,
            report["allocated_after_bytes"] * gib,
            report["peak_allocated_after_bytes"] * gib,
            report.get("path", os.path.join(self.save_dir, "memory_report.json")),
        )
        for op in report["top_ops_by_self_device_memory"][:5]:
            logger.info("[MemoryReport]   %8.2f MiB  x%-5d %s",
                        op["self_device_bytes"] / (1024 ** 2),
                        op["calls"], op["op"])


__all__ = ["MemoryReportCallback", "DEFAULT_TOP_OPS"]

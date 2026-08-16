"""
Tests for MemoryReportCallback (callbacks/memory_report.py).

The callback profiles the FIRST training batch and writes memory_report.json.
Everything except the CUDA profile itself is CPU-testable: the no-CUDA no-op
path, the batch-shape helper, the JSON writer, and the top-ops fallback.
"""

import json

import torch

from causaliT.training.callbacks.memory_report import MemoryReportCallback


def test_noop_without_cuda(tmp_path):
    """Without CUDA the hooks must do nothing (no file, no exception)."""
    if torch.cuda.is_available():
        import pytest
        pytest.skip("CPU-only path")
    cb = MemoryReportCallback(str(tmp_path))
    batch = [torch.zeros(4, 3, 2), torch.zeros(4, 5, 2)]
    cb.on_train_batch_start(None, None, batch, 0)
    cb.on_train_batch_end(None, None, None, batch, 0)
    assert not (tmp_path / "memory_report.json").exists()


def test_batch_shapes():
    cb = MemoryReportCallback("unused")
    assert cb._batch_shapes(torch.zeros(4, 3)) == [4, 3]
    assert cb._batch_shapes([torch.zeros(4, 3, 2), torch.zeros(4, 5, 2)]) == \
        [[4, 3, 2], [4, 5, 2]]
    assert cb._batch_shapes("weird") == str(type("weird"))


def test_top_ops_empty_without_profiler():
    assert MemoryReportCallback("unused")._top_ops() == []


def test_write_report_emits_valid_json(tmp_path):
    cb = MemoryReportCallback(str(tmp_path))
    report = {
        "device": "cpu-stub",
        "device_total_bytes": 0,
        "batch_idx": 0,
        "batch_shapes": [[8, 4, 2]],
        "allocated_before_bytes": 1,
        "allocated_after_bytes": 2,
        "reserved_after_bytes": 3,
        "peak_allocated_before_bytes": 4,
        "peak_allocated_after_bytes": 5,
        "top_ops_by_self_device_memory": [
            {"op": "aten::mm", "calls": 3, "self_device_bytes": 1024},
        ],
    }
    cb._write_report(report)
    path = tmp_path / "memory_report.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["peak_allocated_after_bytes"] == 5
    assert loaded["top_ops_by_self_device_memory"][0]["op"] == "aten::mm"
    assert loaded["batch_shapes"] == [[8, 4, 2]]

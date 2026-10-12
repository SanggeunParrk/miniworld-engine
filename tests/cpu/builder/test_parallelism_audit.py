"""The scheduling audit checks concurrency and assignments, not source spelling."""
from __future__ import annotations

import concurrent.futures

import pytest

from miniworld_engine.autotune import builder, capture
from miniworld_engine.build import audit


def _gpu_finding(monkeypatch):
    monkeypatch.setattr(capture, "_compile_jobs", lambda: 2)
    report = audit.Report()
    audit.check_parallelism(report)
    return next(row for row in report.findings if row.subject == "gpu-tune")


def test_real_scheduler_passes_without_cuda(monkeypatch):
    finding = _gpu_finding(monkeypatch)
    assert finding.level == audit.OK, finding.detail
    assert "1 and 2 slots" in finding.detail


def test_serialized_pool_is_rejected(monkeypatch):
    executor = concurrent.futures.ThreadPoolExecutor
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor",
                        lambda **kwargs: executor(max_workers=1))
    finding = _gpu_finding(monkeypatch)
    assert finding.level == audit.FAIL
    assert "BrokenBarrierError" in finding.detail


@pytest.mark.parametrize("result_count", [0, 2])
def test_missing_gpu_assignments_are_rejected(monkeypatch, result_count):
    monkeypatch.setattr(builder, "build_all", lambda *args, **kwargs:
                        [{"gpu": 0} for _ in range(result_count)])
    finding = _gpu_finding(monkeypatch)
    assert finding.level == audit.FAIL
    assert "worker assignments" in finding.detail

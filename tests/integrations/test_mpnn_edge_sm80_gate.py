"""The A100 MPNN edge path's gate and loader, on a CPU box: the switch, the engine backend, the device check, and the content-addressed staging of the kernel sources."""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.integrations import mpnn_edge_sm80 as sm80
from miniworld_engine.kernels.mpnn_edge_tail.cuda import sm80 as loader


def test_the_switch_and_the_engine_backend_turn_the_path_off(monkeypatch) -> None:
    monkeypatch.delenv(sm80.ENV, raising=False)
    assert sm80.wanted()
    monkeypatch.setenv(sm80.ENV, "0")
    assert not sm80.wanted()
    monkeypatch.setenv(sm80.ENV, "1")
    assert sm80.wanted()
    previous = settings.configure(engine_backend="triton")
    try:
        assert not sm80.wanted()
    finally:
        settings.configure(engine_backend=previous.engine_backend)
    assert sm80.wanted()


def test_a_cpu_device_is_never_an_a100() -> None:
    assert not sm80.on_a100(torch.device("cpu"))


def test_the_kernel_sources_are_staged_by_content(tmp_path, monkeypatch) -> None:
    """The extension is built from a copy named by the sources' hash: the same sources give the same directory (a snapshot of the checkout shares the build), and the copy holds every file."""
    monkeypatch.setenv("MINIWORLD_ENGINE_JIT_ROOT", str(tmp_path))
    first, tag = loader._staged()
    again, tag_again = loader._staged()
    assert first == again
    assert tag == tag_again
    assert first.parent == tmp_path
    originals = sorted(p.name for p in loader._dir.iterdir() if p.suffix in {".cu", ".cuh"})
    assert originals
    assert sorted(p.name for p in first.iterdir()) == originals
    assert "ops.cu" in originals

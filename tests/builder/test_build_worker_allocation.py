"""Build subprocesses stay on the physical GPUs allocated to their parent."""
from __future__ import annotations

import pytest

from miniworld_engine import cli
from miniworld_engine.autotune import builder


@pytest.mark.parametrize(("mask", "device", "expected"), [
    (None, 2, "2"), ("3,1", 0, "3"), ("3,1", 1, "1"),
    ("GPU-first,GPU-second", 1, "GPU-second"), ("MIG-GPU-a/1/0", 0, "MIG-GPU-a/1/0"),
])
def test_worker_inherits_allocated_device(tmp_path, monkeypatch, mask, device, expected):
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    seen = {}

    def stop_before_launch(cmd, **kwargs):
        seen.update(kwargs["env"])
        raise SystemExit

    monkeypatch.setattr(builder.subprocess, "run", stop_before_launch)
    unit = builder.OpUnit("example_triton", 128)
    with pytest.raises(SystemExit):
        builder._run_unit_subprocess(unit, device, tmp_path, tmp_path, 1)
    assert seen["CUDA_VISIBLE_DEVICES"] == expected
    assert cli._bench_visible_device(device) == expected


@pytest.mark.parametrize(("mask", "device"), [("", 0), ("-1", 0), ("3", 1), ("3", -1)])
def test_invalid_device_cannot_escape_allocation(monkeypatch, mask, device):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        builder.visible_device(device)


@pytest.mark.parametrize(("models", "selected", "valid"), [
    (["A6000", "A6000"], [0, 1], True),
    (["A6000", "A5000"], [0, 1], False),
    (["A6000", "A5000"], [1], False),
    (["A6000"], [], False),
    (["A6000"], [0, 0], False),
])
def test_single_publication_cannot_mix_gpu_models(monkeypatch, models, selected, valid):
    from types import SimpleNamespace

    monkeypatch.setattr(builder.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(builder.torch.cuda, "get_device_properties", lambda device:
                        SimpleNamespace(name=models[device], major=8, minor=6))
    if valid:
        builder.validate_build_gpus(selected)
    else:
        with pytest.raises(ValueError, match="build"):
            builder.validate_build_gpus(selected)

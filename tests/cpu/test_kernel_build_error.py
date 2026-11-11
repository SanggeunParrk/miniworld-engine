"""A kernel build / load failure raises ``_nvcc.KernelBuildError`` (chained to the cause), never a warning and another path; while
planning (the prebuilder's ``--list``) a loader's failure is not an error."""

from __future__ import annotations

import importlib

import pytest

from miniworld_engine.kernels import _nvcc


def test_build_failed_raises_chained():
    cause = OSError("nvcc: not found")
    with pytest.raises(_nvcc.KernelBuildError, match=r"my family: kernel build / load failed: OSError: nvcc: not found") as info:
        _nvcc.build_failed("my family", cause)
    assert info.value.__cause__ is cause


def test_build_failed_passes_a_kernel_build_error_through():
    inner = _nvcc.KernelBuildError("inner")
    with pytest.raises(_nvcc.KernelBuildError) as info:
        _nvcc.build_failed("outer", inner)
    assert info.value is inner


def test_build_failed_returns_while_planning():
    with _nvcc.planning() as specs:
        assert _nvcc.build_failed("x", RuntimeError("boom")) is None
        assert specs == []


def test_kernel_build_error_is_not_backend_unsupported():
    """A build failure must not be caught as an unsupported call (the measurers record BackendUnsupported as 'unsupported')."""
    from miniworld_engine.modules.primitives import BackendUnsupported

    assert not issubclass(_nvcc.KernelBuildError, BackendUnsupported)


@pytest.fixture
def failing_build(monkeypatch, tmp_path):
    """Every extension lookup misses (an empty cache, no packaged kernels) and every build fails."""
    monkeypatch.setattr(_nvcc, "_find", lambda rel, marker=None: None)
    monkeypatch.setattr(_nvcc, "_LOADED", {})
    monkeypatch.setattr(_nvcc, "engine_jit_root", lambda: tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("forced build failure (test)")

    monkeypatch.setattr(_nvcc, "_build_ext", broken)
    return broken


@pytest.mark.parametrize(
    ("module", "loader", "cache"),
    [
        ("miniworld_engine.kernels.rope.cuda.sm80", "loads", "ext"),
        ("miniworld_engine.kernels.layernorm_linear.cuda.sm80", "_loads", "_ext"),
    ],
)
def test_loader_raises_kernel_build_error(failing_build, module: str, loader: str, cache: str):
    """A real loader (the function a dispatch calls before using an extension), its build forced to fail: KernelBuildError, chained."""
    mod = importlib.import_module(module)
    getattr(mod, cache).cache_clear()
    try:
        with pytest.raises(_nvcc.KernelBuildError, match="forced build failure") as info:
            getattr(mod, loader)()
        assert isinstance(info.value.__cause__, RuntimeError)
        with _nvcc.planning():  # the prebuilder's planning: the same loader records instead of building, no error
            getattr(mod, cache).cache_clear()
            assert getattr(mod, loader)() is True
    finally:
        getattr(mod, cache).cache_clear()

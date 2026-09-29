"""Check native dependencies before a full build spends time tuning Triton kernels."""
from __future__ import annotations

import importlib


def cache_write_access() -> None:
    """Reject an unwritable installation before deriving plans or tuning kernels.

    Cache lookup and publication retain their existing package-local semantics.
    This only checks access; it creates no directories, lock files or cache data.
    """
    import os

    from miniworld_engine.autotune import cache

    root = cache._CACHE_ROOT
    existing = root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.is_dir() or not os.access(existing, os.W_OK | os.X_OK):
        raise ValueError(
            f"autotune cache is not writable: {root}. "
            "Building caches requires a writable MiniWorld installation or source checkout. "
            "Read-only installations can consume shipped caches, but cannot run build all.")


def native_dependencies(arch: str) -> None:
    """Import the selected FlashAttention backend and check SM90+ build dependencies.

    This checks imports/headers only. It does not certify that a native binary compiles
    or computes correctly; real module runs still have to do that on the target GPU.
    """
    from miniworld_engine.autotune.derive import normalise_arch

    major = int(normalise_arch(arch)[2:]) // 10
    errors = []
    if major >= 9:
        for name in ("cutlass.cute", "quack.gemm_config"):
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: PERF203 - report every broken dependency
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        from miniworld_engine.kernels._nvcc import mathdx_includes
        try:
            mathdx_includes()
        except RuntimeError as exc:
            errors.append(str(exc))
    from miniworld_engine.modules.swa_atom_attention import module as swa
    backend = swa._flash_backend()
    if backend is not None:
        name = "flash_attn.cute" if backend == "fa4" else "flash_attn.flash_attn_interface"
        try:
            native = importlib.import_module(name)
            if not callable(getattr(native, "flash_attn_varlen_func", None)):
                raise ImportError("flash_attn_varlen_func is unavailable")
        except Exception as exc:
            errors.append(f"selected {backend} backend ({name}): {type(exc).__name__}: {exc}")
    if errors:
        raise ValueError("native dependency preflight failed before tuning:\n  " + "\n  ".join(errors))

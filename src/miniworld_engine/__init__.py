"""miniworld-engine — hand-optimized GPU kernels for AF3-style ops.

Public API
----------
``miniworld_engine.modules`` (Transition, TriangleMultiplication, ConditionedTransition, DiTBlock, ...) are the model layers: each one holds its
weights and picks the kernel backend for the running GPU. ``miniworld_engine.kernels`` is the primitive surface (per-GEMM / LN / gate /
attention fusion units and their functional entry points); a model that needs only one of them (a LayerNorm, an RMSNorm) connects to its
kernel backend directly instead of building a module around it.

Both are intentionally cheap to import (no triton/cutlass/cuequivariance loaded until a kernel is first called). The kernel names are covered
by ``tests/cpu/compile/test_public_api.py``.

Importing this top-level package must stay side-effect-free: do not import ``.kernels`` / ``.modules`` here.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("miniworld-engine")
except PackageNotFoundError:  # not installed (e.g. raw source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

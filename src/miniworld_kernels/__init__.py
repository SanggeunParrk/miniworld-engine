"""miniworld-kernels — hand-optimized GPU kernels for AF3-style ops.

Public API
----------
The consumed surface is the **ops** namespace — complete, autograd-transparent
model ops (weights as arguments, backend dispatch inside)::

    from miniworld_kernels import ops
    y = ops.triangle_multiplicative_update(pair, direction="outgoing", ...)

A model layer holds its weights as ``nn.Parameter`` and calls one ``ops.*``; it
never composes primitives itself.

``miniworld_kernels.kernels`` is the **primitive** surface (per-GEMM / LN / gate /
attention fusion units). These are the implementation detail out of which the
``ops`` are built (and are also used by the internal benchmark harness); model
code should not reach into them directly.

Both ``ops`` and ``kernels`` are intentionally cheap and side-effect-free to
import (no triton/cutlass/cuequivariance loaded until an op/kernel is first
called). Their public names are covered by ``tests/test_public_api.py``.

``miniworld_kernels.modules`` (Pairformer, Transition, ...) is an internal
reference / benchmark harness; it is not part of the supported surface and pulls
the full backend + baseline stack at import.

Importing this top-level package must stay side-effect-free: do not import
``.ops`` / ``.kernels`` / ``.modules`` here.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("miniworld-kernels")
except PackageNotFoundError:  # not installed (e.g. raw source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

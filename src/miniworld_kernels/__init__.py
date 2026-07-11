"""miniworld-kernels — hand-optimized GPU kernels for AF3-style ops.

Public API
----------
The supported surface is the **kernels** namespace::

    from miniworld_kernels import kernels
    y = kernels.triton_transition_fused(...)

``miniworld_kernels.kernels`` is intentionally cheap and side-effect-free to
import (no triton/cutlass/cuequivariance loaded until a kernel is first
accessed). Its public names are stable and covered by
``tests/test_public_api.py``.

``miniworld_kernels.modules`` (Pairformer, Transition, ...) is an internal
reference / benchmark harness that composes these kernels. It is not part of the
supported surface, pulls the full backend + baseline stack at import, and may
change at any time.

Importing this top-level package must stay side-effect-free: do not import
``.kernels`` or ``.modules`` here.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("miniworld-kernels")
except PackageNotFoundError:  # not installed (e.g. raw source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

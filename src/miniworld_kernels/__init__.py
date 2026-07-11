"""miniworld-kernels — hand-optimized GPU kernels for AF3-style ops.

Public contract
---------------
The supported surface for downstream consumers (e.g. team-gm, which pins this
as a submodule) is the **kernels** namespace::

    from miniworld_kernels import kernels
    y = kernels.triton_transition_fused(...)

``miniworld_kernels.kernels`` is intentionally cheap and side-effect-free to
import (no triton/cutlass/cuequivariance loaded until a kernel is first
accessed). Its public names are frozen and covered by
``tests/test_public_api.py``.

``miniworld_kernels.modules`` (Pairformer, Transition, ...) is an *internal*
reference / benchmark harness that composes these kernels. It is NOT part of the
consumed contract, pulls the full backend + baseline stack at import, and may
change without a semver bump.

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

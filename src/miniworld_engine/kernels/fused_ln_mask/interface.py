"""Public entry point for the fused LayerNorm + pair-mask family.

One HBM pass over a ``(B, L, L, D)`` pair activation: LayerNorm over the last axis, then a
multiply by the per-row scalar of a ``(B, L, L)`` pair mask -- replacing a separate LN and mask
multiply. ``reference.py`` is the torch definition the checkers compare against.

Nothing outside the family imports this yet -- the flat ``kernels`` bridge does not name it --
but every family gets an ``interface.py`` so the layout rule holds without exceptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["fused_ln_mask"]


def fused_ln_mask(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LN(x) over the last axis times the per-row mask, in one pass. Lazy: imports on first call.

    The import is function-local so that importing this module loads no GPU backend. Every
    entry the flat ``kernels`` bridge can reach must keep ``import miniworld_engine.kernels``
    free of triton/cutlass (``tests/test_public_api.py::test_import_is_side_effect_free``), and
    the backend module builds its autotuned kernel at import. (Despite the ``cute/`` folder
    name, that backend is Triton today, not CuTeDSL -- the deferral holds either way.)
    """
    from .cute.fused_ln_mask import fused_ln_mask as _impl

    return _impl(x, weight, bias, mask, eps)

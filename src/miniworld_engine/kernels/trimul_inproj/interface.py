"""Public entry points for the trimul_inproj family (the fused triangle multiplicative update).

Triton, every arch: :func:`trimul_triton` (single direction) and
:func:`bidirectional_trimul_triton` (outgoing + incoming in one block), both weights-as-args
autograd Functions with a forward-only inference path. The hand-CUDA H100 kernels under
``cuda/`` are reached through ``kernels.integrations.trimul_h100``, which states their contract.
"""

from __future__ import annotations

from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
    bidirectional_trimul_triton,
)
from miniworld_engine.kernels.trimul_inproj.triton.unidirectional import trimul_triton

__all__ = ["bidirectional_trimul_triton", "trimul_triton"]


def triangle_multiplicative_update(x, direction, mask=None, norm_in_weight=None, norm_in_bias=None, p_in_weight=None, g_in_weight=None,
                                   norm_out_weight=None, norm_out_bias=None, p_out_weight=None, g_out_weight=None, eps: float = 1e-5):
    """The fused triangle multiplicative update with the argument signature of ``cuequivariance_torch.triangle_multiplicative_update``; returns the
    RESIDUAL form ``x + update(x)`` (see ``whole_op.triangle_multiplicative_update``, the one this delegates to, for the weights and the per-card routing)."""
    from miniworld_engine.kernels.trimul_inproj.whole_op import triangle_multiplicative_update as _update

    return _update(x, direction, mask, norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias,
                   p_out_weight, g_out_weight, eps)

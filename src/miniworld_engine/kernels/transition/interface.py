"""Public entry point for the Transition (LayerNorm + SwiGLU expand/squeeze) kernel family.

The op is ``y = (silu(LN(x)@Wa^T) * (LN(x)@Wb^T)) @ Ws^T`` over the last dimension; see
:mod:`.reference` for the exact definition. :func:`triton_transition` takes an already
normalised ``x`` and fuses only the expand pair, gate and squeeze;
:func:`triton_transition_fused` folds the input LayerNorm in as well.

The ``Transition`` module and ``ops.transition`` run the residual-fused path instead
(hand-CUDA ``cuda/fused_sm90a`` / ``cuda/fused_wide_sm90a`` on sm_90, ``cuda/fused_sm100a`` /
``cuda/fused_wide_sm100a`` on sm_100, ``cuda/fused_sm80`` on sm_80, else
``triton/residual.transition_residual``); these two are the Triton kernels behind the
raw-op benches, ``ConditionedTransition`` and the drivers.
"""

from __future__ import annotations

from miniworld_engine import settings
from miniworld_engine.kernels.transition.triton import fused as _triton_fused
from miniworld_engine.kernels.transition.triton.fused import triton_swiglu_ffn
from miniworld_engine.kernels.transition.triton.main import triton_transition

__all__ = [
    "triton_swiglu_ffn",
    "triton_transition",
    "triton_transition_fused",
]


def triton_transition_fused(x, ln_weight, ln_bias, expand_a_weight, expand_b_weight, squeeze_weight, n, eps=1e-5, save_xn=False):
    """``y = x + transition(x)`` with the LayerNorm folded in. The caller-side choice of the stages lives here: the portable Triton
    function (``triton/fused.py``, pure Triton) under ``settings.force_portable_kernels``, else the same function with the hand-CUDA
    stages plugged in where they apply (``cuda/train_route.py``)."""
    if settings.current().force_portable_kernels:
        return _triton_fused.triton_transition_fused(x, ln_weight, ln_bias, expand_a_weight, expand_b_weight, squeeze_weight, n, eps, save_xn)
    from miniworld_engine.kernels.transition.cuda import train_route

    return train_route.triton_transition_fused(x, ln_weight, ln_bias, expand_a_weight, expand_b_weight, squeeze_weight, n, eps, save_xn)

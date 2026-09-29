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

from miniworld_engine.kernels.transition.triton.fused import (
    triton_swiglu_ffn,
    triton_transition_fused,
)
from miniworld_engine.kernels.transition.triton.main import triton_transition

__all__ = [
    "triton_swiglu_ffn",
    "triton_transition",
    "triton_transition_fused",
]

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


def swiglu_ffn(x, expand_a_weight, expand_b_weight, squeeze_weight):
    """``W_down( SiLU(Wa @ x) * (Wb @ x) )``, routed per card -- see ``kernels.transition.triton.fused.triton_swiglu_ffn``. H100 / B200 launch no
    Triton: the CUDA kernels of the conditioned-transition tail (fused expand + SwiGLU GEMM, SwiGLU row kernels) where they serve the call
    (``kernels/conditioned_transition/cuda/swiglu_ffn``), the PyTorch statements otherwise. A card without a record: the PyTorch statements."""
    import torch.nn.functional as F

    from miniworld_engine.kernels._device import no_triton
    from miniworld_engine.modules.dispatch import portable_blocked

    if no_triton(x):
        # H100: the CUDA kernels of the conditioned-transition tail (fused expand + SwiGLU GEMM, SwiGLU row kernels) where they serve the call
        from miniworld_engine.kernels.conditioned_transition.cuda import swiglu_ffn as cuda_swiglu

        out = cuda_swiglu.swiglu_ffn(x, expand_a_weight, expand_b_weight, squeeze_weight)
        if out is not None:
            return out
    if no_triton(x) or (x.is_cuda and portable_blocked("swiglu_ffn", x.device, x, expand_a_weight, expand_b_weight, squeeze_weight)):
        return F.linear(F.silu(F.linear(x, expand_a_weight)) * F.linear(x, expand_b_weight), squeeze_weight)
    from miniworld_engine.kernels.transition.triton.fused import triton_swiglu_ffn

    return triton_swiglu_ffn(x, expand_a_weight, expand_b_weight, squeeze_weight)


def transition(x, *, ln_in_weight, ln_in_bias, expand_a_weight, expand_b_weight, squeeze_weight, n, eps: float = 1e-5):
    """The whole Transition layer ``x + squeeze(SwiGLU(LN(x)))`` with the weights as arguments (autograd-transparent): ``kernels.transition.whole_op.transition``."""
    from miniworld_engine.kernels.transition.whole_op import transition as _transition

    return _transition(
        x, ln_in_weight=ln_in_weight, ln_in_bias=ln_in_bias, expand_a_weight=expand_a_weight, expand_b_weight=expand_b_weight,
        squeeze_weight=squeeze_weight, n=n, eps=eps,
    )

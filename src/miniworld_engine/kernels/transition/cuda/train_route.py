"""Where the Transition TRAINING path takes a hand-CUDA stage instead of its Triton one.

``triton/fused.py`` is Triton only. Its forward and backward ``_fused_fwd`` / ``_fused_bwd`` run the portable kernels; the three stages
that have a faster hand-CUDA kernel are chosen HERE, by the caller (:func:`triton_transition_fused`, which ``transition/interface.py`` hands out
and which falls back to the pure Triton function under ``settings.force_portable_kernels``):

* forward, sm90, ``n == 4``, bf16, K in {128, 256}, Version A (no saved ``xn``): the inference b2b kernel
  (``transition_b2b_fwd``, residual folded into the squeeze epilogue) after the Triton LN statistics;
* backward, sm90, bf16, K in {256, 512}: the WGMMA fused expand + gate-backward (``transition_expand_gatebwd_wgmma``);
* backward, bf16/fp16, K <= 512, contiguous: the warp-per-row LayerNorm backward (``layernorm.cuda.layer_norm_bwd_cuda``), on every card.

Each stage falls back to its Triton launch on any build or launch failure. Behaviour is that of the former in-file branches.
"""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.transition.triton import fused as _tf

__all__ = ["triton_transition_fused"]


def _gatebwd_wgmma_enabled() -> bool:
    """The sm90 large-d (K in {256,512}) Version-A gate-backward through the hand-CUDA WGMMA kernel; ``settings.transition_gatebwd_wgmma`` A/Bs it."""
    return settings.current().transition_gatebwd_wgmma


def _cuda_b2b_train_enabled() -> bool:
    """The inference b2b CUDA kernel for the TRAINING forward too (same toggle as the inference dispatch)."""
    return settings.current().transition_cuda_b2b


def _ln_bwd_hook(dxn, x2, rstd, c1, gamma):
    """Hand-CUDA warp-per-row LN backward (~1.10x over privatized Triton at K=128). CUDA takes ``mean``; it is recovered from c1 = mean * rstd."""
    if (
        settings.current().transition_lnbwd_cuda
        and x2.dtype in (torch.float16, torch.bfloat16)
        and gamma.dtype == x2.dtype
        and x2.shape[1] <= 512
        and x2.is_contiguous()
        and dxn.is_contiguous()
    ):
        try:
            from miniworld_engine.kernels.layernorm.cuda import layer_norm_bwd_cuda

            return layer_norm_bwd_cuda(dxn, x2, gamma, c1 / rstd, rstd)
        except Exception:  # noqa: BLE001 (build unavailable / dtype edge) -> Triton
            return None
    return None


def _gatebwd_hook(x2, rstd, c1, ln_weight, ln_bias, expand_a_weight, expand_b_weight, grad_expand):
    """sm90 hand-CUDA WGMMA fused expand + gate-backward (d=256 ~1.02-1.05x, d=512 ~1.07-1.18x; d=128 stays Triton where it wins)."""
    if (
        _gatebwd_wgmma_enabled()
        and torch.cuda.get_device_capability(x2.device)[0] == 9  # Hopper (sm_90a)
        and x2.shape[-1] in (256, 512)
        and x2.dtype == torch.bfloat16
        and x2.shape[0] % 128 == 0
    ):
        try:
            from miniworld_engine.kernels.transition.cuda import transition_expand_gatebwd_wgmma

            return transition_expand_gatebwd_wgmma(x2, rstd, c1, ln_weight, ln_bias, expand_a_weight, expand_b_weight, grad_expand)
        except Exception:  # noqa: BLE001 build/launch unavailable -> Triton
            return None
    return None


@opaque(fake=_tf._fused_bwd_fake, name="transition_fused_bwd_cuda")
def _fused_bwd_cuda(
    grad_output: torch.Tensor, x2: torch.Tensor, rstd: torch.Tensor, c1: torch.Tensor, ln_weight: torch.Tensor,
    ln_bias: torch.Tensor, expand_a_weight: torch.Tensor, expand_b_weight: torch.Tensor, squeeze_weight: torch.Tensor,
    xn_saved: torch.Tensor | None, eps: float, has_xn: bool, orig_shape: list[int], shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The Triton backward with the CUDA stages above plugged in."""
    return _tf._fused_bwd_impl(
        grad_output, x2, rstd, c1, ln_weight, ln_bias, expand_a_weight, expand_b_weight, squeeze_weight,
        xn_saved, eps, has_xn, orig_shape, shape_key, ln_bwd_hook=_ln_bwd_hook, gatebwd_hook=_gatebwd_hook,
    )


@opaque(fake=_tf._fused_fwd_fake, name="transition_fused_fwd_cuda")
def _fused_fwd_cuda(
    x2: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, expand_a_weight: torch.Tensor,
    expand_b_weight: torch.Tensor, squeeze_weight: torch.Tensor, n: int, eps: float, save_xn: bool, shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-light training forward: the FAST inference hand-CUDA b2b (squeeze fused, h never in HBM) for d <= 256 where it fits smem
    (d=128 ~1.29x vs Triton). Version A saves no xn (the backward recomputes it); the LN stats are needed by the backward anyway. Anything
    else, or a build failure of the kernel, runs the Triton forward."""
    K = x2.shape[-1]
    if (
        (not save_xn)
        and n == 4
        and x2.dtype == torch.bfloat16
        and x2.is_cuda
        and x2.shape[0] % 128 == 0
        # Hopper (sm_90a) WGMMA/TMA kernel: sm_80 / sm_86 / sm_100 take the Triton b2b / split path.
        and torch.cuda.get_device_capability(x2.device)[0] == 9
        and _cuda_b2b_train_enabled()
        and K in (128, 256)
    ):
        from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

        rstd, c1 = stats_triton(x2, eps, shape_key=shape_key)
        try:
            from miniworld_engine.kernels.transition.cuda import transition_b2b_fwd

            out = transition_b2b_fwd(
                x2, rstd, c1, ln_weight.contiguous(), ln_bias.contiguous(),
                expand_a_weight.contiguous(), expand_b_weight.contiguous(), squeeze_weight.contiguous(),
            )  # the residual is folded into the squeeze epilogue
        except Exception:  # noqa: BLE001  build unavailable -> split fallback (always fits)
            expand = _tf.transition_expand_gate(
                x2, ln_weight, ln_bias, expand_a_weight, expand_b_weight, eps,
                stats=(rstd, c1), save_xn=False, shape_key=shape_key,
            )
            out = torch.matmul(expand, squeeze_weight.T) + x2
        return out, rstd, c1, x2.new_empty((0, 0))
    return _tf._fused_fwd(x2, ln_weight, ln_bias, expand_a_weight, expand_b_weight, squeeze_weight, n, eps, save_xn, shape_key)


TransitionFusedCudaStagesFunction = _tf.make_fused_function(_fused_fwd_cuda, _fused_bwd_cuda)


def triton_transition_fused(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    expand_a_weight: torch.Tensor,
    expand_b_weight: torch.Tensor,
    squeeze_weight: torch.Tensor,
    n: int,
    eps: float = 1e-5,
    save_xn: bool = False,
) -> torch.Tensor:
    """``y = x + transition(x)`` (LN folded in) with the hand-CUDA stages of the module docstring where they apply. Same contract as
    ``triton.fused.triton_transition_fused``."""
    return TransitionFusedCudaStagesFunction.apply(
        x, ln_weight, ln_bias, expand_a_weight, expand_b_weight, squeeze_weight, n, eps, save_xn,
    )

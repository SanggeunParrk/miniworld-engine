"""The ``ops.*`` entry points that were a bare Triton unit: they obey the same per-card rule as the modules.

``ops.swiglu_ffn`` and ``ops.layer_norm`` called a Triton function directly, so on a card the engine holds no record for they would
have stayed on Triton while ``modules.*`` warned and ran PyTorch. Here the call is routed first
(``modules.dispatch.portable_blocked``): CUDA / recorded Triton as before, the PyTorch statements otherwise. H100 and B200 are not touched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def swiglu_ffn(x: torch.Tensor, expand_a_weight: torch.Tensor, expand_b_weight: torch.Tensor, squeeze_weight: torch.Tensor) -> torch.Tensor:
    """``W_down( SiLU(Wa @ x) * (Wb @ x) )`` -- see ``kernels.transition.triton.fused.triton_swiglu_ffn``."""
    from miniworld_engine.modules.dispatch import portable_blocked

    if x.is_cuda and portable_blocked("swiglu_ffn", x.device, x, expand_a_weight, expand_b_weight, squeeze_weight):
        return F.linear(F.silu(F.linear(x, expand_a_weight)) * F.linear(x, expand_b_weight), squeeze_weight)
    from miniworld_engine.kernels.transition.triton.fused import triton_swiglu_ffn

    return triton_swiglu_ffn(x, expand_a_weight, expand_b_weight, squeeze_weight)


def layer_norm(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float, row_scale: torch.Tensor | None = None) -> torch.Tensor:
    """LayerNorm (autograd) with an optional per-row scale folded in -- see ``kernels.layernorm.triton.main.triton_layernorm``."""
    from miniworld_engine.modules.dispatch import portable_blocked

    if x.is_cuda and portable_blocked("layernorm", x.device, x, *(t for t in (weight, bias) if t is not None)):
        out = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        return out if row_scale is None else out * row_scale.reshape(*x.shape[:-1], 1).to(out.dtype)
    # The caller picks the backward: the hand-CUDA warp-per-row one (B200-measured, masked / opted-in bf16 N 128..512) or the Triton one.
    from miniworld_engine.kernels.layernorm.cuda.masked_bwd import layernorm_cuda_bwd_autograd, use_cuda_bwd

    if use_cuda_bwd(x, row_scale):
        return layernorm_cuda_bwd_autograd(x, weight, bias, eps, row_scale)
    from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm

    return triton_layernorm(x, weight, bias, eps, row_scale)

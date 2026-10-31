"""The ``ops.*`` entry points that were a bare Triton unit: they obey the same per-card rule as the modules.

``ops.swiglu_ffn``, ``ops.layer_norm``, ``ops.rms_norm_modulation``, ``ops.layer_norm_linear`` and ``ops.gated_residual`` called a Triton
function directly, so on a card the engine holds no record for they would have stayed on Triton while ``modules.*`` warned and ran
PyTorch. Here the call is routed first (``modules.dispatch.portable_blocked``, at the call's precision and mode): CUDA / recorded Triton as
before, the PyTorch statements otherwise. H100 and B200 are not touched (``portable_blocked`` is False there: they keep what they ran).
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
    from miniworld_engine.kernels.layernorm.cuda.masked_bwd import (
        layernorm_cuda_bwd_autograd,
        use_cuda_bwd,
    )

    if use_cuda_bwd(x, row_scale):
        return layernorm_cuda_bwd_autograd(x, weight, bias, eps, row_scale)
    from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm

    return triton_layernorm(x, weight, bias, eps, row_scale)


def rms_norm_modulation(
    q: torch.Tensor,
    c: torch.Tensor,
    w_scale: torch.Tensor,
    w_shift: torch.Tensor,
    w_gate: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rmsnorm(q) * (1 + c @ w_scale^T) + c @ w_shift^T, c @ w_gate^T)`` with ``c`` the ACTIVATED conditioning -- see
    ``kernels.rmsnorm.triton.main.triton_rmsnorm_adamod``. On a card without a record: the PyTorch reference
    (``kernels.rmsnorm.reference.rmsnorm_adamod_reference``, fp32 statistics and projections, the outputs in ``q``'s dtype)."""
    from miniworld_engine.modules.dispatch import portable_blocked

    if q.is_cuda and portable_blocked(
        "rms_norm_modulation", q.device, q, c, w_scale, w_shift, w_gate, *(t for t in (weight,) if t is not None)
    ):
        from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_adamod_reference

        return rmsnorm_adamod_reference(q, c, w_scale, w_shift, w_gate, weight, eps)
    from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm_adamod

    return triton_rmsnorm_adamod(q, c, w_scale, w_shift, w_gate, weight, eps)


def layer_norm_linear(x: torch.Tensor, ln_weight: torch.Tensor, proj_weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """``Linear(LayerNorm(x))`` over the last dim (both affine, no bias) -- see
    ``kernels.layernorm_linear.triton.pair_bias.triton_layer_norm_linear``. On a card without a record: ``F.layer_norm`` and ``F.linear``
    with the weights in the activation's dtype (the statements the modules' PyTorch legs run), the result in ``x``'s dtype."""
    from miniworld_engine.modules.dispatch import portable_blocked

    if x.is_cuda and portable_blocked("layer_norm_linear", x.device, x, ln_weight, proj_weight):
        normed = F.layer_norm(x, (x.shape[-1],), ln_weight.to(x.dtype), None, eps)
        return F.linear(normed, proj_weight.to(x.dtype))
    from miniworld_engine.kernels.layernorm_linear.triton.pair_bias import (
        triton_layer_norm_linear,
    )

    return triton_layer_norm_linear(x, ln_weight, proj_weight, eps)


def gated_residual(x: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """``x + gate * branch`` -- see ``kernels.gated_projection.triton.residual.gated_residual``. Its two Triton kernels have no registry
    row (``autotune.records.UNREGISTERED_OPS``), so no card holds a record for them: off H100 / B200 the call is the same PyTorch statement
    (the kernel rounds the product to the activation dtype before the add, as eager does), with one warning."""
    from miniworld_engine.modules.dispatch import portable_blocked

    if x.is_cuda and portable_blocked("gated_residual", x.device, x, gate, branch):
        if x.shape != gate.shape or x.shape != branch.shape:
            raise ValueError("gated_residual requires identical shapes")
        return x + gate * branch
    from miniworld_engine.kernels.gated_projection.triton.residual import (
        gated_residual as _gated_residual,
    )

    return _gated_residual(x, gate, branch)

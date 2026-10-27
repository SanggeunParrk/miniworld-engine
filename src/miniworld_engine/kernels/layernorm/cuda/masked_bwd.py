"""LayerNorm autograd whose BACKWARD is the hand-CUDA warp-per-row kernel (``layer_norm_bwd_cuda``).

This used to live inside ``triton/main.py`` (``_ln_bwd``); a Triton file must not reach for CUDA, so the choice is made by the caller
(:func:`use_cuda_bwd`, asked by ``kernels/_ops_routed.layer_norm``) and the CUDA backward lives here, in the CUDA family. The forward is the
Triton forward (``_ln_fwd``); only the three gradients differ.

Measured B200 (M=L^2, N=128, fwd+bwd): MASKED (a ``row_scale``, the AF pair mask) CUDA beats Triton 1.17x @L512, 1.28x @L1024 -- Triton pays a
real row-scale penalty (+26% bwd) the CUDA path (one FMA) avoids. DENSE is ~neutral (1.00x @L512, 1.07x @L1024, slower at L384), so a dense LN
takes this path only under ``settings.layernorm_cuda_bwd`` (opt-in). Any build or run failure of the CUDA kernel falls back to the Triton backward.
"""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.autotune.shape_key import both_key, rows_of
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.layernorm.triton.main import _ln_bwd, _ln_bwd_fake, _ln_fwd

__all__ = ["layernorm_cuda_bwd_autograd", "use_cuda_bwd"]


def use_cuda_bwd(x: torch.Tensor, row_scale: torch.Tensor | None) -> bool:
    """The routing rule the Triton ``_ln_bwd`` used to apply itself: bf16, 128 <= N <= 512, masked or opted in, portable rung not forced."""
    cfg = settings.current()
    return (
        x.is_cuda
        and not cfg.force_portable_kernels
        and x.dtype == torch.bfloat16
        and 128 <= x.shape[-1] <= 512
        and (row_scale is not None or cfg.layernorm_cuda_bwd)
    )


@opaque(fake=_ln_bwd_fake, name="layernorm_bwd_cuda")
def _ln_bwd_cuda(
    dy_2d: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    rs: torch.Tensor | None,
    has_rs: bool,
    input_shape: list[int],
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.to(dy_2d.dtype)
    if x.dtype == torch.bfloat16:
        try:
            from miniworld_engine.kernels.layernorm.cuda import layer_norm_bwd_cuda

            dx_c, dw_c, db_c = layer_norm_bwd_cuda(
                dy_2d, x.contiguous(), weight, mean, rstd, row_scale=rs if has_rs else None,
            )
            return dx_c.view(tuple(input_shape)), dw_c.float(), db_c.float()
        except Exception:  # noqa: BLE001 - the Triton backward on any CUDA-path failure
            pass
    return _ln_bwd(dy_2d, x, weight, mean, rstd, rs, has_rs, input_shape, shape_key)


class _LayerNormCudaBwdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps, row_scale=None):
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        has_rs = row_scale is not None
        rs = row_scale.reshape(-1).to(x_2d.dtype).contiguous() if has_rs else None
        y_2d, mean, rstd = _ln_fwd(
            x_2d, weight, bias, rs, eps, has_rs, both_key(rows_of(x.shape)),
            store_stats=any(ctx.needs_input_grad[:3]),
        )
        ctx.save_for_backward(x_2d, weight, mean, rstd, rs if has_rs else None)
        ctx.has_rowscale = has_rs
        ctx.input_shape = x.shape
        return y_2d.view_as(x)

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd, rs = ctx.saved_tensors
        dx, dw, db = _ln_bwd_cuda(
            dy.reshape(-1, dy.shape[-1]).contiguous(), x, weight, mean, rstd, rs,
            ctx.has_rowscale, list(ctx.input_shape), both_key(rows_of(ctx.input_shape)),
        )
        return dx, dw, db, None, None


def layernorm_cuda_bwd_autograd(x, weight, bias, eps, row_scale=None):
    """``triton_layernorm`` with the hand-CUDA backward (absent affine parameters become constants, as in the Triton entry)."""
    if weight is None:
        weight = torch.ones(x.shape[-1], device=x.device, dtype=torch.float32)
    if bias is None:
        bias = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float32)
    return _LayerNormCudaBwdFunction.apply(x, weight, bias, eps, row_scale)

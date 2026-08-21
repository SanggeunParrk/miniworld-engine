from __future__ import annotations

from miniworld_engine.kernels._compile import opaque
# The kernel that used to live here was persistent.py's, with `stride_part` spelled
# `stride_part_row` and one masked load's `other=` differing on lanes that are never
# stored. dx, part_dw and part_db all came out bitwise equal (.bench/direct.out).
# What is actually different is the LAUNCHER below: it sizes the partial buffer from M
# rather than from the SM count.
from .persistent import _ln_bwd_persistent

import torch
import triton
import triton.language as tl


from .main import layer_norm_fwd_fused
from miniworld_engine.autotune.shape_key import both_key, length_of


def _bwd_block_m(n: int) -> int:
    """Rows per PARTIAL ROW (how finely dw/db are split), not the compute tile.

    This used to be the compute BLOCK_M1 as well, which is why it had to be host-computed: the
    partial buffer is allocated before the launch, so its row count cannot depend on a config the
    autotuner picks. Splitting the two lets the compute tile be tuned while the partial buffer
    keeps exactly the shape (and the final-reduce cost) it had before.
    """
    if n <= 256:
        return 128
    return 64


def _bwd_num_warps(n: int) -> int:
    """Kept for callers; num_warps is part of the autotune sweep now."""
    if n <= 256:
        return 4
    return 8




class TritonLayerNormPartialReductionFunction(torch.autograd.Function):
    @staticmethod
    @opaque()
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y_2d = torch.empty_like(x_2d)
        m, n = x_2d.shape

        mean = torch.empty(m, dtype=torch.float32, device=x.device)
        rstd = torch.empty(m, dtype=torch.float32, device=x.device)

        grid = lambda meta: [triton.cdiv(m, meta["BLOCK_M1"])]
        layer_norm_fwd_fused[grid](
            x_2d,
            y_2d,
            weight,
            bias,
            mean,
            rstd,
            rstd,  # placeholder when HAS_ROWSCALE=False; keep the call shape identical to main.py
            x_2d.stride(0),
            x_2d.stride(1),
            m,
            n,
            eps,
            # L = x.shape[-2] BEFORE the reshape, not m. (The kernel parameter is named
            # shape_key now -- GROUP_M here was a stale name from before the rename.)
            shape_key=both_key(length_of(x.shape)),
            HAS_ROWSCALE=False,
        )

        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.input_shape = x.shape
        return y_2d.view_as(x)

    @staticmethod
    @opaque()
    def backward(ctx, dy: torch.Tensor):
        x, weight, mean, rstd = ctx.saved_tensors
        dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()
        m, n = x.shape

        # num_partials fixes how finely dw/db are split (unchanged); the compute tile BLOCK_M1 and
        # the feature tile BLOCK_N now come from the autotune sweep. Programs that get no row tile
        # write their zero-initialised accumulator, so the final sum is unaffected.
        num_partials = triton.cdiv(m, _bwd_block_m(n))

        dx_2d = torch.empty_like(dy_2d)
        partial_dw = torch.empty((num_partials, n), dtype=torch.float32, device=x.device)
        partial_db = torch.empty((num_partials, n), dtype=torch.float32, device=x.device)

        grid = lambda meta: (num_partials, triton.cdiv(n, meta["BLOCK_K"]))  # noqa: E731
        _ln_bwd_persistent[grid](
            dx_2d,
            partial_dw,
            partial_db,
            dy_2d,
            x,
            weight,
            mean,
            rstd,
            partial_dw.stride(0),
            x.stride(0),
            x.stride(1),
            m,
            N=n,
            shape_key=both_key(length_of(ctx.input_shape)),
        )

        dw = partial_dw.sum(dim=0).to(weight.dtype)
        db = partial_db.sum(dim=0).to(weight.dtype)
        return dx_2d.view(ctx.input_shape), dw, db, None


triton_layernorm_partial = TritonLayerNormPartialReductionFunction.apply

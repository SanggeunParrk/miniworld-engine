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
from miniworld_engine.autotune.shape_key import both_key, length_of, rows_of


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




def _partial_fwd_fake(x_2d, weight, bias, eps, shape_key):
    """``y`` like ``x_2d``, plus ``mean`` and ``rstd`` as (M,) fp32 against a bf16 activation."""
    m = x_2d.shape[0]
    return (
        torch.empty_like(x_2d),
        x_2d.new_empty((m,), dtype=torch.float32),   # mean
        x_2d.new_empty((m,), dtype=torch.float32),   # rstd
    )


@opaque(fake=_partial_fwd_fake, name="layernorm_partial_fwd")
def _partial_fwd(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The launch -> ``(y, mean, rstd)``; ``x_2d`` arrives flat and contiguous.

    Split out of ``TritonLayerNormPartialReductionFunction.forward`` so the reshape and
    ``save_for_backward`` stay traceable -- see ``kernels._compile``.
    """
    m, n = x_2d.shape
    y_2d = torch.empty_like(x_2d)
    mean = torch.empty(m, dtype=torch.float32, device=x_2d.device)
    rstd = torch.empty(m, dtype=torch.float32, device=x_2d.device)

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
        shape_key=shape_key,
        HAS_ROWSCALE=False,
    )
    return y_2d, mean, rstd


def _partial_bwd_fake(dy_2d, x, weight, mean, rstd, input_shape, shape_key):
    """``dx`` at the forward's PRE-flatten ``input_shape``, plus ``dweight`` and ``dbias`` as (N,)
    in ``weight``'s dtype -- the ``[num_partials, N]`` fp32 partial buffer is summed and cast back
    inside the op, so neither it nor its row count is visible here."""
    n = x.shape[-1]
    return (
        dy_2d.new_empty(tuple(input_shape)),
        weight.new_empty((n,)),
        weight.new_empty((n,)),
    )


@opaque(fake=_partial_bwd_fake, name="layernorm_partial_bwd")
def _partial_bwd(
    dy_2d: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    input_shape: list[int],
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward launch + its partial reduction -> ``(dx, dweight, dbias)``."""
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
        shape_key=shape_key,
    )

    dw = partial_dw.sum(dim=0).to(weight.dtype)
    db = partial_db.sum(dim=0).to(weight.dtype)
    return dx_2d.view(tuple(input_shape)), dw, db


class TritonLayerNormPartialReductionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y_2d, mean, rstd = _partial_fwd(
            x_2d, weight, bias, eps,
            # L = x.shape[-2] BEFORE the reshape, not m. (The kernel parameter is named
            # shape_key now -- GROUP_M here was a stale name from before the rename.)
            both_key(rows_of(x.shape)),
        )
        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.input_shape = x.shape
        return y_2d.view_as(x)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, mean, rstd = ctx.saved_tensors
        dx, dw, db = _partial_bwd(
            dy.reshape(-1, dy.shape[-1]).contiguous(), x, weight, mean, rstd,
            list(ctx.input_shape), both_key(rows_of(ctx.input_shape)),
        )
        return dx, dw, db, None   # eps takes no gradient


triton_layernorm_partial = TritonLayerNormPartialReductionFunction.apply

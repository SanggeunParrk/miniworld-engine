"""Split Transition with residual in the squeeze and input-LN gradient epilogues."""
import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key, pack, rows_of
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._tiles import tile_grid, tile_order
from miniworld_engine.kernels.transition.triton.main import _expand_swiglu, swiglu_squeeze_backward
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_ln_residual


@triton.autotune(configs=configs_for("transition_squeeze_residual_triton"), key=["shape_key"])
@triton.jit
def _squeeze_residual_kernel(
    H, W, R, Y, M, N: tl.constexpr, K: tl.constexpr,
    shm, shk, swn, swk, srm, srn,
    shape_key, BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pm, pn = tile_order(tl.program_id(0).to(tl.int64), tl.cdiv(M, BLOCK_M1),
                        tl.cdiv(N, BLOCK_N), GROUP_M)
    m = pm * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)
    for k0 in range(tl.cdiv(K, BLOCK_K)):
        kk = k0 * BLOCK_K + k
        h = tl.load(H + m[:, None] * shm + kk[None, :] * shk,
                    mask=(m[:, None] < M) & (kk[None, :] < K), other=0)
        w = tl.load(W + n[None, :] * swn + kk[:, None] * swk,
                    mask=(n[None, :] < N) & (kk[:, None] < K), other=0)
        acc = tl.dot(h, w, acc, input_precision="ieee")
    mask = (m[:, None] < M) & (n[None, :] < N)
    residual = tl.load(R + m[:, None] * srm + n[None, :] * srn, mask=mask, other=0).to(tl.float32)
    # Match mm(...).to(activation_dtype) + residual, including the old rounding boundary.
    y = acc.to(Y.dtype.element_ty).to(tl.float32) + residual
    tl.store(Y + m[:, None] * N + n[None, :], y, mask=mask)


def _squeeze_residual_fake(h, weight, residual, shape_key):
    return residual.new_empty(residual.shape)


@opaque(fake=_squeeze_residual_fake, name="transition_squeeze_residual")
def squeeze_residual(h: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor,
                     shape_key: int) -> torch.Tensor:
    m, k = h.shape
    n = weight.shape[0]
    if weight.shape[1] != k or residual.shape != (m, n):
        raise ValueError("squeeze/residual shapes must be H[M,K], W[N,K], R[M,N]")
    if h.dtype != weight.dtype or residual.dtype != h.dtype:
        raise ValueError("squeeze/residual operands must share dtype")
    out = residual.new_empty((m, n))
    _squeeze_residual_kernel[lambda c: tile_grid(m, n, c["BLOCK_M1"], c["BLOCK_N"])](
        h, weight, residual, out, m, n, k,
        *h.stride(), *weight.stride(), *residual.stride(),
        shape_key=pack(shape_key, K=k, N=n),
    )
    return out


class ResidualTransitionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xn, wa, wb, ws, residual):
        shape = xn.shape
        xn = xn.reshape(-1, shape[-1]).contiguous()
        key = both_key(rows_of(shape))
        h = _expand_swiglu(xn, wa, wb, key)
        ctx.save_for_backward(xn, wa, wb, ws)
        ctx.shape = shape
        return squeeze_residual(h, ws, residual.reshape(-1, shape[-1]), key).reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        xn, wa, wb, ws = ctx.saved_tensors
        flat = dy.reshape(-1, dy.shape[-1]).contiguous()
        dx, dwa, dwb, dws = swiglu_squeeze_backward(xn, wa, wb, ws, flat, ctx.shape, wa.shape[0])
        return dx, dwa, dwb, dws, dy


def transition_residual(x, gamma, beta, wa, wb, ws, eps):
    # Returning the identity from the SAME LN Function lets its two gradients meet
    # inside LN backward. No activation copy, separate dx add, or duplicate dgamma/dbeta.
    xn, residual = input_ln_residual(x, gamma, beta, eps)
    return ResidualTransitionFunction.apply(xn, wa, wb, ws, residual)

"""Split Transition with residual in the squeeze and input-LN gradient epilogues."""
import torch
import triton
import triton.language as tl

from miniworld_engine.kernels._tf32 import tf32_cached, tf32_dot, tf32_grid_many

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key, pack, rows_of
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._dot_precision import dot_precision
from miniworld_engine.kernels._tiles import tile_grid, tile_order
from miniworld_engine.kernels.transition.triton.main import (
    _expand_swiglu,
    _expand_swiglu_saveact,
    swiglu_squeeze_backward,
)
from miniworld_engine.kernels.transition.triton.training import swiglu_squeeze_backward_saved
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_ln_residual


@triton.autotune(configs=configs_for("transition_squeeze_residual_triton"), key=["shape_key"])
@triton.jit
def _squeeze_residual_kernel(
    H, W, R, Y, M, N: tl.constexpr, K: tl.constexpr,
    shm, shk, swn, swk, srm, srn,
    shape_key, BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    PREC: tl.constexpr = "ieee",
    PRE_H: tl.constexpr = False, PRE_W: tl.constexpr = False,
):
    n_m = tl.cdiv(M, BLOCK_M1)
    n_n = tl.cdiv(N, BLOCK_N)
    pm, pn = tile_order(tl.program_id(0).to(tl.int64), n_m, n_n, GROUP_M)
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
        acc = tf32_dot(h, w, acc, input_precision=PREC, pre_a=PRE_H, pre_b=PRE_W)
    mask = (m[:, None] < M) & (n[None, :] < N)
    residual = tl.load(R + m[:, None] * srm + n[None, :] * srn, mask=mask, other=0).to(tl.float32)
    # Match mm(...).to(activation_dtype) + residual, including the old rounding boundary.
    y = acc.to(Y.dtype.element_ty).to(tl.float32) + residual
    tl.store(Y + m[:, None] * N + n[None, :], y, mask=mask)


def _squeeze_residual_fake(h, weight, residual, shape_key, h_rounded=False):
    """Return the residual-shaped output metadata."""
    return residual.new_empty(residual.shape)


@opaque(fake=_squeeze_residual_fake, name="transition_squeeze_residual")
def squeeze_residual(h: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor,
                     shape_key: int, h_rounded: bool = False) -> torch.Tensor:
    """Project the hidden state and add the residual in one Triton launch. ``h_rounded``: fp32 ``h`` is already on the TF32 grid (the
    expand kernel stores it so); otherwise it is rounded here."""
    m, k = h.shape
    n = weight.shape[0]
    if weight.shape[1] != k or residual.shape != (m, n):
        raise ValueError("squeeze/residual shapes must be H[M,K], W[N,K], R[M,N]")
    if h.dtype != weight.dtype or residual.dtype != h.dtype:
        raise ValueError("squeeze/residual operands must share dtype")
    out = residual.new_empty((m, n))
    prec = dot_precision(h.dtype)
    pre = prec == "tf32" and h.dtype is torch.float32
    if pre:
        weight = tf32_cached((weight,), lambda: tf32_grid_many(weight))[0]
        if not h_rounded:
            h = tf32_grid_many(h)[0]
    grid = lambda c: tile_grid(m, n, c["BLOCK_M1"], c["BLOCK_N"])  # noqa: E731
    _squeeze_residual_kernel[grid](
        h, weight, residual, out, m, n, k,
        *h.stride(), *weight.stride(), *residual.stride(),
        shape_key=pack(shape_key, K=k, N=n), PREC=prec,
        PRE_H=pre, PRE_W=pre,
    )
    return out


class ResidualTransitionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xn, wa, wb, ws, residual, save_act=True, half_scale=None):
        shape = xn.shape
        xn = xn.reshape(-1, shape[-1]).contiguous()
        key = both_key(rows_of(shape))
        if save_act:
            h, a, b = _expand_swiglu_saveact(xn, wa, wb, key, half_scale)     # training: the halves, see main.TritonTransitionFunction
            ctx.save_for_backward(xn, wa, wb, ws, h, a, b, *(() if half_scale is None else (half_scale,)))
        else:
            h = _expand_swiglu(xn, wa, wb, key)
            ctx.save_for_backward()
        ctx.shape = shape
        ctx.key = key
        return squeeze_residual(h, ws, residual.reshape(-1, shape[-1]), key, True).reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        xn, wa, wb, ws, h, a, b, *scale = ctx.saved_tensors
        flat = dy.reshape(-1, dy.shape[-1]).contiguous()
        if flat.dtype != xn.dtype:
            flat = flat.to(xn.dtype)
        dx, dwa, dwb, dws = swiglu_squeeze_backward_saved(xn, wa, wb, ws, flat, ctx.key, saved=(h, a, b), half_scale=scale[0] if scale else None)
        return dx.reshape(ctx.shape), dwa, dwb, dws, dy, None, None


@torch.no_grad()
def half_scale(gamma, beta, wa, wb):
    """One fp32 on the device: ``max(1, bound / 60000)`` with ``bound`` >= every |a|, |b| of ``LN(x) @ [Wa; Wb]^T``. LayerNorm output is
    ``z * gamma + beta`` with ``|z| <= sqrt(D)``, so ``|xn_j| <= sqrt(D) |gamma|_max + |beta|_max`` and ``|a_i| <= that * ||Wa_i||_1``. The saved fp16
    halves are divided by it: they can never overflow, whatever the weights are, and the scale is 1 for any ordinary magnitude. Device
    arithmetic only (a few reductions over the weights): graph-capturable, no host sync."""
    d = wa.shape[1]
    xn_max = (d ** 0.5) * gamma.abs().max().float() + beta.abs().max().float()
    w1 = torch.maximum(wa.abs().sum(dim=1).max(), wb.abs().sum(dim=1).max()).float()
    return (xn_max * w1 * (1.0 / 60000.0)).clamp_min(1.0).reshape(1)


def transition_residual(x, gamma, beta, wa, wb, ws, eps):
    # Returning the identity from the SAME LN Function lets its two gradients meet
    # inside LN backward. No activation copy, separate dx add, or duplicate dgamma/dbeta.
    xn, residual = input_ln_residual(x, gamma, beta, eps)
    # grad mode and an operand that wants one, decided here: ``ctx.needs_input_grad`` is true for parameters under no_grad too
    save = torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))
    scale = half_scale(gamma, beta, wa, wb) if save and xn.dtype == torch.float32 else None
    return ResidualTransitionFunction.apply(xn, wa, wb, ws, residual, save, scale)

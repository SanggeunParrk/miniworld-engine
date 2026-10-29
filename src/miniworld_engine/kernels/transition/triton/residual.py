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
    SPLIT_K: tl.constexpr = 1, P=None,
):
    n_m = tl.cdiv(M, BLOCK_M1)
    n_n = tl.cdiv(N, BLOCK_N)
    pm, pn = tile_order(tl.program_id(0).to(tl.int64), n_m, n_n, GROUP_M)
    m = pm * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)
    if SPLIT_K == 1:
        for k0 in range(tl.cdiv(K, BLOCK_K)):
            kk = k0 * BLOCK_K + k
            h = tl.load(H + m[:, None] * shm + kk[None, :] * shk,
                        mask=(m[:, None] < M) & (kk[None, :] < K), other=0)
            w = tl.load(W + n[None, :] * swn + kk[:, None] * swk,
                        mask=(n[None, :] < N) & (kk[:, None] < K), other=0)
            acc = tf32_dot(h, w, acc, input_precision=PREC, pre_a=PRE_H, pre_b=PRE_W)
    else:
        n_k = tl.cdiv(K, BLOCK_K)
        per = tl.cdiv(n_k, SPLIT_K)                 # K tiles per split
        k_lo = tl.program_id(1) * per
        for k0 in range(k_lo, tl.minimum(k_lo + per, n_k)):
            kk = k0 * BLOCK_K + k
            h = tl.load(H + m[:, None] * shm + kk[None, :] * shk,
                        mask=(m[:, None] < M) & (kk[None, :] < K), other=0)
            w = tl.load(W + n[None, :] * swn + kk[:, None] * swk,
                        mask=(n[None, :] < N) & (kk[:, None] < K), other=0)
            acc = tf32_dot(h, w, acc, input_precision=PREC, pre_a=PRE_H, pre_b=PRE_W)
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT_K > 1:
        # this split's fp32 partial; _split_k_residual_kernel sums the splits in order and adds the residual
        tl.store(P + tl.program_id(1) * M * N + m[:, None] * N + n[None, :], acc, mask=mask)
    else:
        residual = tl.load(R + m[:, None] * srm + n[None, :] * srn, mask=mask, other=0).to(tl.float32)
        # Match mm(...).to(activation_dtype) + residual, including the old rounding boundary.
        y = acc.to(Y.dtype.element_ty).to(tl.float32) + residual
        tl.store(Y + m[:, None] * N + n[None, :], y, mask=mask)


@triton.jit
def _split_k_residual_kernel(P, R, Y, M, N, srm, srn, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    """``Y = (sum of the SPLIT_K fp32 partials, in split order).to(Y dtype) + R``, the squeeze epilogue of a split-K launch."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.load(P + offs, mask=mask, other=0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(P + s * M * N + offs, mask=mask, other=0)
    row, col = offs // N, offs % N
    residual = tl.load(R + row * srm + col * srn, mask=mask, other=0).to(tl.float32)
    tl.store(Y + offs, acc.to(Y.dtype.element_ty).to(tl.float32) + residual, mask=mask)


def _split_k(m: int, n: int, k: int, device) -> int:
    """K splits for the squeeze: few rows leave most SMs without a tile (128 rows x d 768: 24 tiles of 64 x 64 on 132 SMs), so the K
    loop is cut until about two tiles per SM are in flight; 1 whenever the tiles already fill the card or K is short."""
    from miniworld_engine.kernels._device import num_sms

    tiles = tile_grid(m, n, 64, 64)[0]
    s = 1
    while s < 8 and tiles * s * 2 <= 2 * num_sms(device) and k // (2 * s) >= 256:
        s *= 2
    return s


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
    split = _split_k(m, n, k, h.device)
    part = h.new_empty((split, m, n), dtype=torch.float32) if split > 1 else out
    grid = lambda c: (tile_grid(m, n, c["BLOCK_M1"], c["BLOCK_N"])[0], split)  # noqa: E731
    _squeeze_residual_kernel[grid](
        h, weight, residual, out, m, n, k,
        *h.stride(), *weight.stride(), *residual.stride(),
        shape_key=pack(shape_key, K=k, N=n), PREC=prec,
        PRE_H=pre, PRE_W=pre, SPLIT_K=split, P=part,
    )
    if split > 1:
        block = 1024
        _split_k_residual_kernel[(triton.cdiv(m * n, block),)](
            part, residual, out, m, n, *residual.stride(), SPLIT_K=split, BLOCK=block,
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


#: fp32 training at up to this many rows keeps the saved expand halves in fp32 (no fp16 scale), see transition_residual.
_FP32_HALVES_MAX_ROWS = 4096


def transition_residual(x, gamma, beta, wa, wb, ws, eps):
    # Returning the identity from the SAME LN Function lets its two gradients meet
    # inside LN backward. No activation copy, separate dx add, or duplicate dgamma/dbeta.
    xn, residual = input_ln_residual(x, gamma, beta, eps)
    # grad mode and an operand that wants one, decided here: ``ctx.needs_input_grad`` is true for parameters under no_grad too
    save = torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))
    # fp32: the saved halves are fp16 below a scale only where their traffic matters; at few rows they stay fp32 and the scale's
    # reductions over the weights (more than the halves themselves cost there) are not made
    rows = xn.numel() // xn.shape[-1]
    scale = half_scale(gamma, beta, wa, wb) if save and xn.dtype == torch.float32 and rows > _FP32_HALVES_MAX_ROWS else None
    return ResidualTransitionFunction.apply(xn, wa, wb, ws, residual, save, scale)

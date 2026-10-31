"""Row-fused portable (Triton) Transition for the B200 at the narrow pair widths (d = 64, 128).

``y = x + Ws (silu(Wa xn) * (Wb xn))``, ``xn = LN(x) gamma + beta``. On sm_100 the split portable path (LayerNorm, expand + SwiGLU,
squeeze + residual: three launches, the (M, n d) hidden activation and, in training, both expand halves through HBM) lost to
torch.compile'd PyTorch at these widths. Here a program owns a 128-row tile and the whole d row:

* ``_rows_fwd_kernel``: LayerNorm (row statistics in registers), then per hidden chunk ``a = xn Wa_c^T``, ``b = xn Wb_c^T`` (two tensor-core
  products), the SwiGLU gate, and ``acc += h_c Ws_c^T`` -- h never leaves the SM -- and ``y = bf16(acc) + x``. A training call stores
  only the row statistics (mean rstd as ``c1`` and rstd).
* ``_rows_bwd_kernel`` (training, bf16): xn again from x and the statistics, per hidden chunk ``dh = dy Ws_c`` (rounded to the activation
  dtype, the boundary of the PyTorch statements), a and b recomputed in fp32, the gate backward, ``h``, ``dA``, ``dB`` stored for the
  weight-gradient GEMMs (cuBLAS: ``dWs = dy^T h``, ``[dWa; dWb] = [dA | dB]^T xn``) and ``d_xn += dA Wa_c + dB Wb_c`` kept in fp32 in the
  tile, whose epilogue is the LayerNorm backward with the residual (``dx = dy + rstd (g - mean(g) - xhat mean(g xhat))``, g = d_xn gamma)
  and per-program dgamma / dbeta partials summed in program order (no atomics: a replay is bit-identical).

Operands move by TMA (host tensor descriptors: the row tail of the last tile is zero-filled on the way in and clipped on the way out).
The forward also takes fp32 activations for inference (TF32 products, operands rounded to nearest: weights once in the launcher, xn and
h in the kernel, the contract of ``kernels/_tf32.py``) at hidden = 4 d (``FP32_EXPANSION``).

The tile shapes are fixed per (dtype, d): measured on the B200 (CUDA-graph replay, [589 824, d] rows, n = 4) over BLOCK_M 64 / 128,
BLOCK_N 32 / 64 / 128, 4 / 8 warps, 1-3 stages; there is no autotune key for these kernels, and other cards keep the split path
(``serves`` is the whole gate).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._fastmath import sigmoid_fast
from miniworld_engine.kernels._tf32 import tf32_cached, tf32_dot, tf32_grid_many

#: (dtype, d) -> (BLOCK_M, BLOCK_N, num_warps, num_stages), measured on the B200 (fwd alone, d = 128, L768 pair: 328 us; d = 64: 128 us).
#: The fp32 (TF32) forward (inference only, see FP32_EXPANSION) is 1.7-1.9x faster than torch.compile (d128 n4 L768 833 vs 1439 us).
_FWD = {(torch.bfloat16, 64): (128, 64, 4, 2), (torch.bfloat16, 128): (128, 32, 4, 2),
        (torch.float32, 64): (64, 32, 4, 2), (torch.float32, 128): (64, 32, 4, 2)}
#: the backward (bf16 only; d = 128 L768: 1.34 ms, d = 64: 0.47 ms; BLOCK_M 64 at d = 128 ties at L768 and loses at L128, 131 vs 116 us)
_BWD = {(torch.bfloat16, 64): (128, 32, 4, 2), (torch.bfloat16, 128): (128, 64, 8, 2)}


#: fp32 (TF32) inference takes the row-fused forward at this expansion only (hidden = 4 d): there the hand-CUDA forward
#: (cuda/sm100/generic/tg_fused_f32.cu) is faster than it (d128 L384 197 vs 218 us, L768 732 vs 833); at hidden = 2 d it is not yet
#: (L128 27.0 vs 19.5, L384 130 vs 130), and the split path keeps the call
FP32_EXPANSION = 4


def _desc(t: torch.Tensor, block):
    from triton.tools.tensor_descriptor import TensorDescriptor

    return TensorDescriptor(t, list(t.shape), list(t.stride()), list(block))


def serves(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor, training: bool) -> bool:
    """sm_100 (B200), d = 64 or 128, a hidden width that is a multiple of 64, bf16 activations and weights; fp32 (TF32) for inference at
    hidden = 4 d."""
    if x.device.type != "cuda" or x.numel() == 0 or x.ndim < 2:
        return False
    if torch.cuda.get_device_capability(x.device)[0] != 10:
        return False
    d = x.shape[-1]
    nd = wa.shape[0]
    key = (x.dtype, d)
    if key not in (_BWD if training else _FWD):
        return False
    if tuple(wa.shape) != (nd, d) or tuple(ws.shape) != (d, nd) or nd % 64:
        return False
    if wa.dtype != x.dtype or ws.dtype != x.dtype:
        return False
    if x.dtype is torch.float32 and nd != FP32_EXPANSION * d:
        return False
    if x.dtype is torch.float32:   # TF32 only where torch allows it (the split path keeps IEEE fp32 otherwise)
        from miniworld_engine.kernels._dot_precision import dot_precision

        return dot_precision(x.dtype) == "tf32"
    return True


@triton.jit
def _rows_fwd_kernel(XD, RSTD, C1, GAMMA, BETA, WAD, WBD, WSD, YD, M, EPS,
                     D: tl.constexpr, ND: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     PREC: tl.constexpr, FAST: tl.constexpr, SAVE: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    r0 = pid * BLOCK_M
    rows = r0 + tl.arange(0, BLOCK_M)
    rmask = rows < M
    k = tl.arange(0, D)
    et = YD.dtype
    x = XD.load([r0, 0]).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / D)
    xc = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(xc * xc, axis=1) * (1.0 / D) + EPS)
    if SAVE:
        tl.store(RSTD + rows, rstd, mask=rmask)
        tl.store(C1 + rows, mean * rstd, mask=rmask)
    g = tl.load(GAMMA + k).to(tl.float32)
    be = tl.load(BETA + k).to(tl.float32)
    xn = (xc * rstd[:, None] * g[None, :] + be[None, :]).to(et)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    for n0 in tl.range(0, ND, BLOCK_N, num_stages=STAGES):
        a = tf32_dot(xn, WAD.load([n0, 0]).T, input_precision=PREC, pre_b=True)
        b = tf32_dot(xn, WBD.load([n0, 0]).T, input_precision=PREC, pre_b=True)
        if FAST:
            sig = sigmoid_fast(a)
        else:
            sig = tl.sigmoid(a)
        acc = tf32_dot((a * sig * b).to(et), WSD.load([0, n0]).T, acc, input_precision=PREC, pre_b=True)
    # mm(...).to(activation dtype) + residual: the rounding boundary of the PyTorch statements
    y = acc.to(et).to(tl.float32) + XD.load([r0, 0]).to(tl.float32)
    YD.store([r0, 0], y.to(et))


@triton.jit
def _rows_bwd_kernel(XD, DYD, RSTD, C1, GAMMA, BETA, WAD, WBD, WSD, XND, HD, DAD, DBD, DXD, PART, M,
                     D: tl.constexpr, ND: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     FAST: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    r0 = pid * BLOCK_M
    rows = r0 + tl.arange(0, BLOCK_M)
    rmask = rows < M
    k = tl.arange(0, D)
    et = XND.dtype
    x = XD.load([r0, 0]).to(tl.float32)
    rstd = tl.load(RSTD + rows, mask=rmask, other=0)
    c1 = tl.load(C1 + rows, mask=rmask, other=0)
    g = tl.load(GAMMA + k).to(tl.float32)
    be = tl.load(BETA + k).to(tl.float32)
    xn = ((x * rstd[:, None] - c1[:, None]) * g[None, :] + be[None, :]).to(et)
    XND.store([r0, 0], xn)
    dy = DYD.load([r0, 0])
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    for n0 in tl.range(0, ND, BLOCK_N, num_stages=STAGES):
        dh = tl.dot(dy, WSD.load([0, n0])).to(et).to(tl.float32)
        wa = WAD.load([n0, 0])
        wb = WBD.load([n0, 0])
        a = tl.dot(xn, wa.T)
        b = tl.dot(xn, wb.T)
        if FAST:
            sig = sigmoid_fast(a)
        else:
            sig = tl.sigmoid(a)
        silu = a * sig
        HD.store([r0, n0], (silu * b).to(et))
        da = (dh * b * (sig + silu * (1.0 - sig))).to(et)
        db = (dh * silu).to(et)
        DAD.store([r0, n0], da)
        DBD.store([r0, n0], db)
        acc = tl.dot(da, wa, acc)
        acc = tl.dot(db, wb, acc)
    # LayerNorm backward with the residual branch; rows past M have dy = 0, hence d_xn = 0, and add nothing to the partials
    x = XD.load([r0, 0]).to(tl.float32)
    xhat = x * rstd[:, None] - c1[:, None]
    dxh = acc * g[None, :]
    ca = tl.sum(dxh, axis=1) * (1.0 / D)
    cb = tl.sum(dxh * xhat, axis=1) * (1.0 / D)
    dx = rstd[:, None] * (dxh - ca[:, None] - xhat * cb[:, None]) + DYD.load([r0, 0]).to(tl.float32)
    DXD.store([r0, 0], dx.to(et))
    tl.store(PART + pid * (2 * D) + k, tl.sum(acc * xhat, axis=0))
    tl.store(PART + pid * (2 * D) + D + k, tl.sum(acc, axis=0))


def _fwd_fake(x, gamma, beta, wa, wb, ws, eps, save):
    m = x.shape[0]
    rows = m if save else 1
    return torch.empty_like(x), x.new_empty((rows,), dtype=torch.float32), x.new_empty((rows,), dtype=torch.float32)


@opaque(fake=_fwd_fake, name="transition_rows_fwd")
def rows_fwd(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, eps: float,
             save: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(y, rstd, c1)`` for x [M, d] (contiguous), the weights in x's dtype; ``rstd`` / ``c1`` are 1-element placeholders unless ``save``."""
    m, d = x.shape
    nd = wa.shape[0]
    bm, bn, warps, stages = _FWD[(x.dtype, d)]
    f32 = x.dtype is torch.float32
    if f32:   # TF32 operands rounded to nearest once (cached while grad mode is off); xn and h are rounded in the kernel
        wa, wb, ws = tf32_cached((wa, wb, ws), lambda: tf32_grid_many(wa, wb, ws))
    y = torch.empty_like(x)
    rows = m if save else 1
    rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
    c1 = torch.empty_like(rstd)
    _rows_fwd_kernel[(triton.cdiv(m, bm),)](
        _desc(x, (bm, d)), rstd, c1, gamma, beta, _desc(wa, (bn, d)), _desc(wb, (bn, d)), _desc(ws, (d, bn)), _desc(y, (bm, d)),
        m, float(eps), d, nd, BLOCK_M=bm, BLOCK_N=bn, PREC="tf32" if f32 else "ieee", FAST=not f32, SAVE=save, STAGES=stages,
        num_warps=warps)
    return y, rstd, c1


def _bwd_fake(x, dy, rstd, c1, gamma, beta, wa, wb, ws):
    m, d = x.shape
    nd = wa.shape[0]
    bm = _BWD.get((x.dtype, d), (128,))[0]
    return (torch.empty_like(x), x.new_empty((m, nd)), x.new_empty((m, 2 * nd)), torch.empty_like(x),
            x.new_empty((triton.cdiv(m, bm), 2 * d), dtype=torch.float32))


@opaque(fake=_bwd_fake, name="transition_rows_bwd")
def rows_bwd(x: torch.Tensor, dy: torch.Tensor, rstd: torch.Tensor, c1: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
             wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(xn, h, [dA | dB], dx, partials)``: dx carries the residual branch; partials [programs, 2 d] = [dgamma | dbeta] per program."""
    m, d = x.shape
    nd = wa.shape[0]
    bm, bn, warps, stages = _BWD[(x.dtype, d)]
    xn = torch.empty_like(x)
    h = x.new_empty((m, nd))
    dab = x.new_empty((m, 2 * nd))
    dx = torch.empty_like(x)
    nprog = triton.cdiv(m, bm)
    part = torch.empty((nprog, 2 * d), device=x.device, dtype=torch.float32)
    _rows_bwd_kernel[(nprog,)](
        _desc(x, (bm, d)), _desc(dy, (bm, d)), rstd, c1, gamma, beta, _desc(wa, (bn, d)), _desc(wb, (bn, d)), _desc(ws, (d, bn)),
        _desc(xn, (bm, d)), _desc(h, (bm, bn)), _desc(dab[:, :nd], (bm, bn)), _desc(dab[:, nd:], (bm, bn)), _desc(dx, (bm, d)), part, m,
        d, nd, BLOCK_M=bm, BLOCK_N=bn, FAST=True, STAGES=stages, num_warps=warps)
    return xn, h, dab, dx, part


class RowsTransitionFunction(torch.autograd.Function):
    """``y = x + transition(x)``; saves x, the row statistics and the (activation-dtype) parameters."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        g, b = gamma.float().contiguous(), beta.float().contiguous()
        wa, wb, ws = wa.contiguous(), wb.contiguous(), ws.contiguous()
        y, rstd, c1 = rows_fwd(x2, g, b, wa, wb, ws, float(eps), True)
        ctx.save_for_backward(x2, rstd, c1, g, b, wa, wb, ws)
        ctx.shape = shape
        ctx.dtypes = (gamma.dtype, beta.dtype)
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, rstd, c1, g, b, wa, wb, ws = ctx.saved_tensors
        nd, d = wa.shape
        dy2 = dy.reshape(-1, d).to(x2.dtype).contiguous()
        xn, h, dab, dx, part = rows_bwd(x2, dy2, rstd, c1, g, b, wa, wb, ws)
        dws = torch.matmul(dy2.t(), h)                 # (d, nd)
        dwab = torch.matmul(dab.t(), xn)               # [dWa; dWb] (2 nd, d)
        dgb = part.sum(0)
        return (dx.reshape(ctx.shape), dgb[:d].to(ctx.dtypes[0]), dgb[d:].to(ctx.dtypes[1]), dwab[:nd], dwab[nd:].clone(), dws, None)


def transition_rows(x, gamma, beta, wa, wb, ws, eps):
    """``x + transition(x)`` by the row-fused kernels; call ``serves`` first (``training`` = a gradient is wanted)."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws)):
        return RowsTransitionFunction.apply(x, gamma, beta, wa, wb, ws, eps)
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    y = rows_fwd(x2, gamma.float().contiguous(), beta.float().contiguous(), wa.contiguous(), wb.contiguous(), ws.contiguous(),
                 float(eps), False)[0]
    return y.reshape(shape)

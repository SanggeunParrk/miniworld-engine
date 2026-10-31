"""B200 portable (Triton) Transition inference at the wide widths (d = 256 / 384 / 512, bf16): the split path's two GEMMs on TMA.

The split path's expand + SwiGLU and squeeze + residual kernels (main.py / residual.py) load their tiles through pointers; on sm_100
that left the d >= 256 inference forward slower than torch.compile'd PyTorch (cuBLAS + a fused pointwise kernel). Here the same two
launches take their operands through host TMA descriptors (``tl.dot`` on tcgen05 with a 2-stage TMA pipeline):

* ``_expand_kernel``: ``h = silu(xn Wa^T) * (xn Wb^T)``, a and b as two products over one xn tile, 128 x 128 output tiles, BK 64.
* ``_squeeze_kernel``: ``y = bf16(h Ws^T) + x``, 128 x 256 (d % 256 == 0) or 128 x 128 tiles, 4 stages.

Kernels alone (CUDA graph, min of 15, us; pointer kernels / these / cuBLAS): expand d256 [589 824 rows] 1154 / 886 / 599 (mm alone, no
gate); d768 [147 456] 1642 / 1207 / 849; squeeze d256 458 / 389 / 367 (addmm); d768 599 / 541 / 503. The LayerNorm stays the split
path's. Tiles fixed per width (no autotune key: the descriptors fix the block shape at launch). d = 768 keeps the split path: there
the hand-CUDA forward (the generic tcgen05 kernels) is not faster than this one would be. Training keeps the split path (its
saved-halves gate kernel did not get under compiled PyTorch on TMA either: d256 L768 1939 -> 1641 us against 293 us of cuBLAS dh plus a
pointwise pass).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._fastmath import sigmoid_fast

#: widths served (bf16 inference on sm_100); 768 is left to the split path (see the module docstring)
WIDTHS = (256, 384, 512)
_EXP = (128, 128, 64, 4, 2)          # BLOCK_M, BLOCK_N, BLOCK_K, warps, stages
_SQ = {256: (128, 256, 64, 8, 4), 128: (128, 128, 64, 4, 3)}


def _desc(t: torch.Tensor, block):
    from triton.tools.tensor_descriptor import TensorDescriptor

    return TensorDescriptor(t, list(t.shape), list(t.stride()), list(block))


def serves(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor, training: bool) -> bool:
    if training or x.device.type != "cuda" or x.dtype is not torch.bfloat16 or x.numel() == 0 or x.ndim < 2:
        return False
    if torch.cuda.get_device_capability(x.device)[0] != 10:
        return False
    d, nd = x.shape[-1], wa.shape[0]
    return (d in WIDTHS and nd % 128 == 0 and tuple(wa.shape) == (nd, d) and tuple(ws.shape) == (d, nd)
            and wa.dtype == torch.bfloat16 and ws.dtype == torch.bfloat16)


@triton.jit
def _tile(pid, M, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr):
    n_m = tl.cdiv(M, BM)
    n_n: tl.constexpr = N // BN
    per = GROUP * n_n
    first = (pid // per) * GROUP
    gs = tl.minimum(n_m - first, GROUP)
    return first + (pid % per) % gs, (pid % per) // gs


@triton.jit
def _expand_kernel(XD, WAD, WBD, HD, M, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   STAGES: tl.constexpr):
    pm, pn = _tile(tl.program_id(0), M, N, BM, BN, 8)
    acc_a = tl.zeros((BM, BN), tl.float32)
    acc_b = tl.zeros((BM, BN), tl.float32)
    for k in tl.range(0, K, BK, num_stages=STAGES):
        x = XD.load([pm * BM, k])
        acc_a = tl.dot(x, WAD.load([pn * BN, k]).T, acc_a)
        acc_b = tl.dot(x, WBD.load([pn * BN, k]).T, acc_b)
    HD.store([pm * BM, pn * BN], (acc_a * sigmoid_fast(acc_a) * acc_b).to(HD.dtype))


@triton.jit
def _squeeze_kernel(HD, WD, RD, YD, M, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    STAGES: tl.constexpr):
    pm, pn = _tile(tl.program_id(0), M, N, BM, BN, 8)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in tl.range(0, K, BK, num_stages=STAGES):
        acc = tl.dot(HD.load([pm * BM, k]), WD.load([pn * BN, k]).T, acc)
    # mm(...).to(activation dtype) + residual: the rounding boundary of the PyTorch statements (as residual.squeeze_residual)
    y = acc.to(YD.dtype).to(tl.float32) + RD.load([pm * BM, pn * BN]).to(tl.float32)
    YD.store([pm * BM, pn * BN], y.to(YD.dtype))


def _fwd_fake(xn, x, wa, wb, ws):
    return torch.empty_like(x)


@opaque(fake=_fwd_fake, name="transition_wide_fwd_sm100")
def wide_fwd(xn: torch.Tensor, x: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """``x + Ws (silu(Wa xn) * (Wb xn))`` for xn / x [M, d] (contiguous, bf16)."""
    m, d = x.shape
    nd = wa.shape[0]
    bm, bn, bk, w, st = _EXP
    h = x.new_empty((m, nd))
    _expand_kernel[(triton.cdiv(m, bm) * (nd // bn),)](
        _desc(xn, (bm, bk)), _desc(wa, (bn, bk)), _desc(wb, (bn, bk)), _desc(h, (bm, bn)), m, d, nd,
        BM=bm, BN=bn, BK=bk, STAGES=st, num_warps=w)
    bm, bn, bk, w, st = _SQ[256 if d % 256 == 0 else 128]
    y = torch.empty_like(x)
    _squeeze_kernel[(triton.cdiv(m, bm) * (d // bn),)](
        _desc(h, (bm, bk)), _desc(ws, (bn, bk)), _desc(x, (bm, bn)), _desc(y, (bm, bn)), m, nd, d,
        BM=bm, BN=bn, BK=bk, STAGES=st, num_warps=w)
    return y


def transition_wide(x, gamma, beta, wa, wb, ws, eps):
    """Inference ``x + transition(x)``; call ``serves`` first. The LayerNorm is the split path's (``input_ln_residual``)."""
    from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_ln_residual

    shape = x.shape
    xn, _ = input_ln_residual(x, gamma, beta, eps)
    x2 = x.reshape(-1, shape[-1]).contiguous()
    return wide_fwd(xn.reshape(x2.shape).contiguous(), x2, wa.contiguous(), wb.contiguous(), ws.contiguous()).reshape(shape)

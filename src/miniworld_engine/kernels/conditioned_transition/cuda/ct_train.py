"""H100 ConditionedTransition tail, training: forward and backward (``ct_rows.cu`` + cuBLAS).

    ab = x [Wa; Wb]^T        (cuBLAS)       h = silu(a) b                 (``swiglu_fwd``)
    out = h Ws^T             (cuBLAS)       sp = cond Wsc^T + b           (cuBLAS, bias epilogue)
    y = res + sigmoid(sp) out               (``sigmul_fwd``)

and the backward as the mirror (``sigmul_bwd``, ``swiglu_bwd`` and one cuBLAS GEMM per operand gradient). ``x`` is the output of the module's AdaLN
(the kernel is the post-AdaLN tail, as the Triton family's). bf16 activations (fp32 master weights are rounded to bf16 per call and their gradients
come back in fp32) or fp32 on TF32 GEMMs (the process's ``allow_tf32``). Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from miniworld_engine.kernels._device import num_sms
from miniworld_engine.kernels import _cast_cuda

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.adaln.cuda.infer import colsum

_dir = Path(__file__).parent
_FM = os.environ.get("CT_FM", "0") == "1"   # the feature-major backward (measured: no gain at 18432 rows, see _backward_feature_major)


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="ct_rows_sm90",
        sources=[str(_dir / "ct_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(), "-O3", "-std=c++17", *gencodes("90a"), "--expt-relaxed-constexpr", f"-I{_dir.parent.parent / 'adaln' / 'cuda'}",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def supported(x: torch.Tensor, cond: torch.Tensor, nd: int) -> bool:
    """sm_90, bf16, or fp32 under TF32 (``allow_tf32``); the hidden width a multiple of 8."""
    if not (x.is_cuda and cond.is_cuda and x.dtype == cond.dtype):
        return False
    if x.dtype == torch.float32 and not torch.backends.cuda.matmul.allow_tf32:
        return False
    if x.dtype not in (torch.bfloat16, torch.float32) or nd % 8:
        return False
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(idx) == (9, 0)


def _squeeze_gate_save_fake(h, cond, ws, wsc, bsc, res):
    """(y, out, sp) [M, D]."""
    n = (h.shape[0], ws.shape[0])
    return h.new_empty(n), h.new_empty(n), h.new_empty(n)


@opaque(fake=_squeeze_gate_save_fake, name="conditioned_transition_cuda_squeeze_gate_save")
def squeeze_gate_save(h: torch.Tensor, cond: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor,
                      res: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 training forward of the tail's second half in one wgmma kernel (``ct_sm90.cu``): y = res + sigmoid(cond Wsc^T + b) (h Ws^T), and the two
    products the backward needs, out = h Ws^T and sp = cond Wsc^T + b, each rounded to bf16 as the unfused ops round them."""
    from .ct_inference import ext as ct_ext

    y = torch.empty(h.shape[0], ws.shape[0], device=h.device, dtype=h.dtype)
    out, sp = torch.empty_like(y), torch.empty_like(y)
    ct_ext().squeeze_gate(h, ws, cond, wsc, bsc, res, y, num_sms(h.device), 128, True, out, sp)
    return y, out, sp


def _cast_weights_fake(wa, wb, ws, wsc, bsc):
    """([Wa; Wb] [2 ND, D], Ws, Wsc, bsc) in bf16."""
    bf = torch.bfloat16
    return (wa.new_empty((2 * wa.shape[0], wa.shape[1]), dtype=bf), ws.new_empty(ws.shape, dtype=bf), wsc.new_empty(wsc.shape, dtype=bf),
            bsc.new_empty(bsc.shape, dtype=bf))


@opaque(fake=_cast_weights_fake, name="conditioned_transition_cuda_cast_weights")
def cast_weights(wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The five fp32 master weights of a training step as bf16 in ONE launch; Wa and Wb land in the two halves of their concatenation."""
    bf = torch.bfloat16
    nd = wa.shape[0]
    wab = torch.empty(2 * nd, wa.shape[1], device=wa.device, dtype=bf)
    wsb, wscb, bscb = (torch.empty(t.shape, device=t.device, dtype=bf) for t in (ws, wsc, bsc))
    _cast_cuda.cast_into([wa, wb, ws, wsc, bsc], [wab[:nd], wab[nd:], wsb, wscb, bscb])
    return wab, wsb, wscb, bscb


def _swiglu_fwd_fake(ab):
    """h [M, ND] of ab [M, 2 ND]."""
    return ab.new_empty((ab.shape[0], ab.shape[1] // 2))


@opaque(fake=_swiglu_fwd_fake, name="conditioned_transition_cuda_swiglu_fwd")
def swiglu_fwd(ab: torch.Tensor) -> torch.Tensor:
    """h = silu(a) b of ab = [a | b]."""
    return ext().swiglu_fwd(ab)


def _swiglu_bwd_fake(dh, ab):
    """[dA | dB] like ab."""
    return torch.empty_like(ab)


@opaque(fake=_swiglu_bwd_fake, name="conditioned_transition_cuda_swiglu_bwd")
def swiglu_bwd(dh: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """da = dh b silu'(a), db = dh silu(a), written as [dA | dB]."""
    return ext().swiglu_bwd(dh, ab)


def _sigmul_fwd_fake(out, sp, res):
    """y like out."""
    return torch.empty_like(out)


@opaque(fake=_sigmul_fwd_fake, name="conditioned_transition_cuda_sigmul_fwd")
def sigmul_fwd(out: torch.Tensor, sp: torch.Tensor, res: torch.Tensor | None) -> torch.Tensor:
    """y = res + sigmoid(sp) out (the gated update rounded to the stream dtype before the add)."""
    return ext().sigmul_fwd(out, sp, res)


def sigmul_bwd_blocks(dy: torch.Tensor) -> int:
    """Blocks of ``sigmul_bwd``'s grid (= rows of its partial-sum buffer): a function of the shape alone, so the fake knows it."""
    v = 4 if dy.dtype is torch.float32 else 8
    ndc = dy.shape[-1] // v
    threads = ndc * max(1, 128 // ndc)
    return min(-(-(dy.numel() // v) // threads), 132 * 8)


def _sigmul_bwd_fake(dy, out, sp):
    """(dout, dscale, the block partials of the column sums of dscale)."""
    return torch.empty_like(dy), torch.empty_like(dy), dy.new_empty((sigmul_bwd_blocks(dy), dy.shape[-1]), dtype=torch.float32)


@opaque(fake=_sigmul_bwd_fake, name="conditioned_transition_cuda_sigmul_bwd")
def sigmul_bwd(dy: torch.Tensor, out: torch.Tensor, sp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dout = dy g, dscale = dy out g (1 - g), and the block partials of sum(dscale)."""
    dout, dsc, part = ext().sigmul_bwd(dy, out, sp, sigmul_bwd_blocks(dy))
    return dout, dsc, part


# Measured (M = 36864, bf16): the fused wgmma squeeze + gate kernel with the two saved outputs takes 425 us against 108 + 43 + 79 for cuBLAS + addmm + sigmul_fwd
# (a 128 x 128 tile is bound by L2 -> SM traffic at ~280 TFLOPS; cuBLAS runs 192 x 192): kept for the small-M inference tail, off for training.
_FUSED_SQUEEZE_TRAIN = False


def _fused_squeeze_ok(h, cond, ws) -> bool:
    """The squeeze + gate kernel's tiling: K slabs of 64, whole 128-column output tiles."""
    return _FUSED_SQUEEZE_TRAIN and ws.shape[0] % 128 == 0 and h.shape[1] % 64 == 0 and cond.shape[1] % 64 == 0 and h.stride(1) == 1 and cond.stride(1) == 1


def _pack_fp32_fake(wa, wb, ws, wsc):
    """Output structure of ``conditioned_transition_cuda_pack_fp32``."""
    nd, d = wa.shape
    dc = wsc.shape[1]
    return wa.new_empty((2 * nd, d)), wa.new_empty((d, 2 * nd)), wa.new_empty((nd, d)), wa.new_empty((dc, d))


@opaque(fake=_pack_fp32_fake, name="conditioned_transition_cuda_pack_fp32")
def pack_fp32(wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 weights of a training step in ONE launch (``adaln.cuda.pack_f32``): [Wa; Wb] [2 ND, D], and the K-major (transposed) right operands of the three
    TF32 data-gradient GEMMs: [Wa; Wb]^T [D, 2 ND], Ws^T [ND, D], Wsc^T [DC, D] (cuBLAS "nt" kernels: 15 - 25 % faster than "nn"; the PyTorch transposing
    copies were 40 us a step at 18432 rows)."""
    from miniworld_engine.kernels.adaln.cuda import _ext as rows_ext

    nd, d = wa.shape
    dc = wsc.shape[1]
    dev, f32 = wa.device, torch.float32
    wab = torch.empty(2 * nd, d, device=dev, dtype=f32)
    wabT = torch.empty(d, 2 * nd, device=dev, dtype=f32)
    wsT = torch.empty(nd, d, device=dev, dtype=f32)
    wscT = torch.empty(dc, d, device=dev, dtype=f32)
    wa, wb, ws, wsc = (w.contiguous() for w in (wa, wb, ws, wsc))
    rows_ext().pack_f32([wa, wb, wa, wb, ws, wsc], [wab, wab, wabT, wabT, wsT, wscT], [[0, 0, 0], [0, nd, 0], [1, 0, 0], [1, 0, nd], [1, 0, 0], [1, 0, 0]])
    return wab, wabT, wsT, wscT


_XT = os.environ.get("CT_XT", "1") != "0"   # read once at import (the routing runs inside traces)


def _sigmul_bwd_x_fake(dy, out, sp):
    """(dout, dscale, the block partials of the column sums of dscale, dout^T padded to 32 rows)."""
    m = dy.numel() // dy.shape[-1]
    blocks = min(sigmul_bwd_blocks(dy), -(-m // 32))
    return torch.empty_like(dy), torch.empty_like(dy), dy.new_empty((blocks, dy.shape[-1])), dy.new_empty((dy.shape[-1], (m + 31) // 32 * 32))


@opaque(fake=_sigmul_bwd_x_fake, name="conditioned_transition_cuda_sigmul_bwd_x")
def sigmul_bwd_x(dy: torch.Tensor, out: torch.Tensor, sp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``sigmul_bwd`` (fp32) plus dout^T [D, rows padded to 32] (the K-major left operand of dWs = dout^T h)."""
    dout, dsc, part, doutT = ext().sigmul_bwd_x(dy, out, sp, sigmul_bwd_blocks(dy))
    return dout, dsc, part, doutT


def _transpose_f32_fake(t):
    """t^T: [C, R], same dtype."""
    return t.new_empty((t.shape[1], t.shape[0]))


@opaque(fake=_transpose_f32_fake, name="conditioned_transition_cuda_transpose_f32")
def transpose_f32(t: torch.Tensor) -> torch.Tensor:
    """t^T [C, R] (fp32, contiguous), rounded to the TF32 grid to nearest. The TF32 weight-gradient GEMM dW = G^T X contracts over the ROWS of both
    operands, which makes both of them MN-major for cuBLAS (G^T X: 200 TFLOP/s at 18432 x 3072 x 768); with the SMALLER operand K-major (X^T copied once, 43 us
    for 56 MB) the GEMM runs at ~290 and the copy is paid back several times."""
    from miniworld_engine.kernels.adaln.cuda import _ext as rows_ext

    t = t.contiguous()
    o = torch.empty(t.shape[1], t.shape[0], device=t.device, dtype=t.dtype)
    rows_ext().pack_f32([t], [o], [[1, 0, 0, 1]])
    return o


class _Tail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc, res):  # noqa: D102
        dt = x.dtype
        if dt is torch.bfloat16 and all(t.dtype is torch.float32 for t in (wa, wb, ws, wsc, bsc)):
            wab, wsb, wscb, bscb = cast_weights(wa, wb, ws, wsc, bsc)   # one launch for the five casts and the concatenation
            wabT = wsT = wscT = None
        elif dt is torch.float32 and all(t.dtype is torch.float32 for t in (wa, wb, ws, wsc)) and torch.backends.cuda.matmul.allow_tf32:
            wab, wabT, wsT, wscT = pack_fp32(wa, wb, ws, wsc)
            wsb, wscb, bscb = ws.contiguous(), wsc.contiguous(), bsc
        else:
            wab = torch.cat([wa, wb]).to(dt)
            wsb, wscb, bscb = ws.to(dt), wsc.to(dt), bsc.to(dt)
            wabT = wsT = wscT = None
        ab = torch.mm(x, wab.t())
        h = swiglu_fwd(ab)
        if dt is torch.bfloat16 and _fused_squeeze_ok(h, cond, wsb):   # one wgmma kernel for both GEMMs and the gate (+ residual)
            y, out, sp = squeeze_gate_save(h, cond, wsb, wscb, bsc.float().contiguous(), res)
        else:
            out = torch.mm(h, wsb.t())
            sp = torch.addmm(bscb, cond, wscb.t())
            y = sigmul_fwd(out, sp, res)
        ctx.save_for_backward(x, cond, ab, h, out, sp, wab, wsb, wscb, wa, ws, wsc, *([] if wabT is None else [wabT, wsT, wscT]))
        ctx.has_res = res is not None
        return y

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        x, cond, ab, h, out, sp, wab, wsb, wscb, wa, ws, wsc, *tw = ctx.saved_tensors
        dy = dy.contiguous()
        if x.dtype is torch.float32 and not tw:
            tw = [w.t().contiguous() for w in (wab, wsb, wscb)]
        if x.dtype is torch.float32 and _feature_major_ok(x, ws) and _FM:
            return _backward_feature_major(ctx, dy, x, cond, ab, h, out, sp, wab, wsb, wscb, tw)
        f32 = torch.float32
        o = lambda w: f32 if (w.dtype == f32 and x.dtype != f32) else None  # noqa: E731  fp32 master weights: fp32 gradients, unrounded
        xt = x.dtype is f32 and _XT and x.shape[0] % 4 == 0 and x.shape[1] >= 256   # token widths (measured at the atom width, 128: slower)
        if xt and dy.shape[-1] % 128 == 0 and dy.shape[-1] <= 1024:   # sigmul_bwd also writes dout^T (K-major left operand of dWs)
            dout, dsc, part, doutT = sigmul_bwd_x(dy, out, sp)
        else:
            dout, dsc, part = sigmul_bwd(dy, out, sp)
            doutT = transpose_f32(dout) if xt else None
        if xt:   # TF32 weight gradients: the smaller operand goes K-major (rows contiguous), the big gradient stays as it is (see transpose_f32)
            dws = torch.mm(doutT[:, :dout.shape[0]], h)   # left operand K-major, [D, ND] straight out (no transposing copy)
        else:
            dws = torch.mm(dout.t(), h, out_dtype=f32) if o(ws) else torch.mm(dout.t(), h)
        t = (lambda w: w) if x.dtype is not f32 else None   # TF32: the K-major copies packed by the forward (18432 rows: dh -21, dx -72, dcond -23 us)
        k = (lambda i, w: tw[i].t()) if x.dtype is f32 else (lambda i, w: w)
        dh = torch.mm(dout, k(1, wsb))
        dab = swiglu_bwd(dh, ab)
        if xt:
            dwab = torch.mm(dab.t(), transpose_f32(x).t())
        else:
            dwab = torch.mm(dab.t(), x, out_dtype=f32) if o(wa) else torch.mm(dab.t(), x)
        dx = torch.mm(dab, k(0, wab))
        dwsc = torch.mm(dsc.t(), cond, out_dtype=f32) if o(wsc) else torch.mm(dsc.t(), cond)
        dcond = torch.mm(dsc, k(2, wscb))
        dbsc = colsum(part).to(dsc.dtype if not o(wsc) else f32)   # the block partials of sum(dscale), summed in a fixed order
        nd = wab.shape[0] // 2
        return dx, dcond, dwab[:nd], dwab[nd:], dws, dwsc, dbsc, (dy if ctx.has_res else None)



def _feature_major_ok(x, ws) -> bool:
    """fp32 (TF32) tail with token-width gradients: the weight-gradient GEMMs run on the gradients written feature-major (tpose.cuh). Not the atom width
    (measured: no difference there)."""
    d, nd = ws.shape
    return d >= 256 and d % 128 == 0 and nd % 128 == 0 and d <= 1024


def _backward_feature_major(ctx, dy, x, cond, ab, h, out, sp, wab, wsb, wscb, tw):
    """``_Tail.backward`` for fp32 with dout / dscale / dab produced feature-major ([features, rows]): every weight gradient is a GEMM with the rows contiguous
    in its left operand (M = 18432: dWab 430 -> ~295 us, dWs 237 -> ~145, dWsc 78 -> ~61); the data-gradient GEMMs read the same buffers as column-major operands."""
    m = x.shape[0]
    doutT, dscT, part = ext().sigmul_bwd_t(dy, out, sp, sigmul_bwd_blocks(dy))
    doutT, dscT = doutT[:, :m], dscT[:, :m]
    dws = torch.mm(doutT, h)
    dh = torch.mm(doutT.t(), tw[1].t())
    dabT = ext().swiglu_bwd_t(dh, ab)[:, :m]
    dwab = torch.mm(dabT, x)
    dx = torch.mm(dabT.t(), tw[0].t())
    dwsc = torch.mm(dscT, cond)
    dcond = torch.mm(dscT.t(), tw[2].t())
    dbsc = colsum(part)
    nd = wab.shape[0] // 2
    return dx, dcond, dwab[:nd], dwab[nd:], dws, dwsc, dbsc, (dy if ctx.has_res else None)

# ---- D = 768, hidden 2 D, bf16, whole 128-row tiles: the Transition family's wgmma kernels (kernels/transition/cuda/wide), which keep a, b and h of the
# expand GEMM and fuse the SwiGLU backward into the dh GEMM. The tail's squeeze (h Ws^T) is their squeeze kernel with a zero residual (its rounding is the
# plain GEMM's: bf16(h Ws^T + 0)); the gate and the residual follow in ``sigmul_fwd``.
# Measured on the ladder shapes (A = 48, bf16): their expand_saved + squeeze 164 + 87 us against cuBLAS + swiglu_fwd + cuBLAS 124 + 61 + 55, their gate_saved
# 178 us against cuBLAS dh + swiglu_bwd 63 + 98: a net loss of ~10 us, so the row kernels around cuBLAS stay the route; the path is kept for a faster kernel.
_USE_WIDE = False
_ZEROS: dict = {}


def _zeros_like_rows(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, tuple(x.shape))
    z = _ZEROS.get(key)
    if z is None:
        z = _ZEROS[key] = torch.zeros_like(x)
    return z


def _wide_expand_saved_fake(xn, zeros, w1p, ws):
    """(out, h, a, b) shapes."""
    m, d = xn.shape
    h = ws.shape[1]
    return xn.new_empty((m, d)), xn.new_empty((m, h)), xn.new_empty((m, h)), xn.new_empty((m, h))


@opaque(fake=_wide_expand_saved_fake, name="conditioned_transition_cuda_wide_expand_saved")
def _wide_expand_saved(xn: torch.Tensor, zeros: torch.Tensor, w1p: torch.Tensor, ws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h Ws^T, h, a, b) with h = silu(a) b, [a | b] = xn [Wa; Wb]^T (the Transition family's expand_saved + squeeze, zero residual)."""
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as fw

    out, h, a, b = fw._ext_for(xn, True, 2).fwd_split(xn, zeros, w1p, ws)
    return out, h, a, b


def _wide_gate_saved_fake(dy, wst, a, b):
    """[dA | dB] [M, 2 H]."""
    return dy.new_empty((a.shape[0], 2 * a.shape[1]))


@opaque(fake=_wide_gate_saved_fake, name="conditioned_transition_cuda_wide_gate_saved")
def _wide_gate_saved(dy: torch.Tensor, wst: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[dA | dB] = SwiGLU backward of dh = dy Ws from the saved a, b (one kernel; dh never reaches HBM)."""
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as fw

    return fw._ext_for(dy, True, 2).gate_saved(dy, wst, a, b)


def wide_supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    if x.shape[-1] != 768 or wa.shape[0] != 1536 or x.dtype is not torch.bfloat16:
        return False
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as fw

    return fw.available(x, wa.to(torch.bfloat16) if wa.dtype != torch.bfloat16 else wa, ws)


class _TailWide(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc, res):  # noqa: D102
        from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as fw
        from miniworld_engine.kernels.transition.cuda.fused_sm90a import _cuda_prepare

        dt = x.dtype
        wa_, wb_, ws_, wst = _cuda_prepare(wa, wb, ws, True)
        w1p = fw._pack(wa_, wb_, 128)
        out, h, a, b = _wide_expand_saved(x, _zeros_like_rows(x), w1p, ws_)
        wscb, bscb = wsc.to(dt), bsc.to(dt)
        sp = torch.addmm(bscb, cond, wscb.t())
        y = sigmul_fwd(out, sp, res)
        ctx.save_for_backward(x, cond, a, b, h, out, sp, wa_, wb_, wst, wscb, wa, ws, wsc)
        ctx.has_res = res is not None
        return y

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        x, cond, a, b, h, out, sp, wa_, wb_, wst, wscb, wa, ws, wsc = ctx.saved_tensors
        dy = dy.contiguous()
        dout, dsc, part = sigmul_bwd(dy, out, sp)
        f32 = torch.float32
        o = lambda w: w.dtype == f32  # noqa: E731  fp32 master weights: fp32 gradients, unrounded
        dws = torch.mm(dout.t(), h, out_dtype=f32) if o(ws) else torch.mm(dout.t(), h)
        dab = _wide_gate_saved(dout, wst, a, b)
        dwab = torch.mm(dab.t(), x, out_dtype=f32) if o(wa) else torch.mm(dab.t(), x)
        dx = torch.mm(dab, torch.cat([wa_, wb_]))
        dwsc = torch.mm(dsc.t(), cond, out_dtype=f32) if o(wsc) else torch.mm(dsc.t(), cond)
        dcond = torch.mm(dsc, wscb)
        dbsc = colsum(part).to(dsc.dtype if not o(wsc) else f32)
        nd = a.shape[1]
        return dx, dcond, dwab[:nd], dwab[nd:], dws, dwsc, dbsc, (dy if ctx.has_res else None)


def ct_tail_train(x, cond, wa, wb, ws, wsc, bsc, residual=None):
    """``residual + sigmoid(cond Wsc^T + b) * (silu(x Wa^T) * (x Wb^T)) Ws^T`` with a backward; ``x`` (M, D), ``cond`` (M, DC) of one dtype, the weights in
    any dtype (rounded to the activations' dtype per call; gradients in the weights' dtype). bf16 D = 768 / hidden 1536 on whole 128-row tiles runs the
    Transition family's wgmma expand / dh kernels, everything else the row kernels around cuBLAS."""
    x, cond = x.contiguous(), cond.contiguous()
    if _USE_WIDE and x.shape[0] % 128 == 0 and wide_supported(x, wa, ws):
        return _TailWide.apply(x, cond, wa, wb, ws, wsc, bsc, residual)
    return _Tail.apply(x, cond, wa, wb, ws, wsc, bsc, residual)


__all__ = ["ct_tail_train", "ext", "supported"]

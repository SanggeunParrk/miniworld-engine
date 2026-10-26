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
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.adaln.cuda.infer import colsum

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="ct_rows_sm90",
        sources=[str(_dir / "ct_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(), "-O3", "-std=c++17", *gencodes("90a"), "--expt-relaxed-constexpr",
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
    ct_ext().squeeze_gate(h, ws, cond, wsc, bsc, res, y, torch.cuda.get_device_properties(h.device).multi_processor_count, 128, True, out, sp)
    return y, out, sp


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


def _fused_squeeze_ok(h, cond, ws) -> bool:
    """The squeeze + gate kernel's tiling: K slabs of 64, whole 128-column output tiles."""
    return ws.shape[0] % 128 == 0 and h.shape[1] % 64 == 0 and cond.shape[1] % 64 == 0 and h.stride(1) == 1 and cond.stride(1) == 1


class _Tail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc, res):  # noqa: D102
        dt = x.dtype
        wab = torch.cat([wa, wb]).to(dt)
        wsb, wscb, bscb = ws.to(dt), wsc.to(dt), bsc.to(dt)
        ab = torch.mm(x, wab.t())
        h = swiglu_fwd(ab)
        if dt is torch.bfloat16 and _fused_squeeze_ok(h, cond, wsb):   # one wgmma kernel for both GEMMs and the gate (+ residual)
            y, out, sp = squeeze_gate_save(h, cond, wsb, wscb, bsc.float().contiguous(), res)
        else:
            out = torch.mm(h, wsb.t())
            sp = torch.addmm(bscb, cond, wscb.t())
            y = sigmul_fwd(out, sp, res)
        ctx.save_for_backward(x, cond, ab, h, out, sp, wab, wsb, wscb, wa, ws, wsc)
        ctx.has_res = res is not None
        return y

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        x, cond, ab, h, out, sp, wab, wsb, wscb, wa, ws, wsc = ctx.saved_tensors
        dy = dy.contiguous()
        dout, dsc, part = sigmul_bwd(dy, out, sp)
        f32 = torch.float32
        o = lambda w: f32 if (w.dtype == f32 and x.dtype != f32) else None  # noqa: E731  fp32 master weights: fp32 gradients, unrounded
        dws = torch.mm(dout.t(), h, out_dtype=f32) if o(ws) else torch.mm(dout.t(), h)
        t = (lambda w: w.t().contiguous().t()) if x.dtype is f32 else (lambda w: w)   # TF32: the operand layouts cuBLAS runs fastest (18432 rows: dh -21, dx -72, dcond -23 us)
        dh = torch.mm(dout, t(wsb))
        dab = swiglu_bwd(dh, ab)
        dwab = torch.mm(dab.t(), x, out_dtype=f32) if o(wa) else torch.mm(dab.t(), x)
        dx = torch.mm(dab, t(wab))
        dwsc = torch.mm(dsc.t(), cond, out_dtype=f32) if o(wsc) else torch.mm(dsc.t(), cond)
        dcond = torch.mm(dsc, t(wscb))
        dbsc = colsum(part).to(dsc.dtype if not o(wsc) else f32)   # the block partials of sum(dscale), summed in a fixed order
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

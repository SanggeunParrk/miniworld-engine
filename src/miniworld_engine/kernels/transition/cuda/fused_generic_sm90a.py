"""Hand-CUDA sm_90a Transition for every shape the tuned ``fused_sm90a`` / ``fused_wide_sm90a`` builds do not take.

``y = x + Ws (silu(Wa LN(x)) * (Wb LN(x)))`` with

* any hidden width ``HID`` that is a multiple of 128 (n = 2, 4, ... : the expansion factor is read off the weights),
* any channel width ``D`` that is a multiple of 64 up to 1024 (pair 64 / 128 / 256, single / token 384 / 768, MSA ...),
* any row count (the tail of the last 128-row tile is zero-filled by TMA on the way in, clipped / guarded on the way out),
* bf16 activations on bf16 tensor cores, or fp32 activations on TF32 tensor cores (``torch.set_float32_matmul_precision`` not
  ``"highest"``, or ``torch.backends.cuda.matmul.allow_tf32``: the same switch cuBLAS follows; accumulation is fp32 either way).

Forward (training saves ``xn`` and the LayerNorm statistics, inference saves nothing): ``ln_fwd`` -> ``swiglu`` (wgmma GEMM, SwiGLU in
the epilogue, h [M, HID]) -> ``squeeze_res`` (wgmma GEMM, residual in the epilogue).  Backward: ``gate`` (dy Ws, xn [Wa;Wb]^T and the
SwiGLU backward in one pass -> h, [dA|dB]) -> ``dxn_gemm`` (fp32) -> ``ln_bwd`` (+ residual, per-CTA dgamma / dbeta partials);
the three weight gradients are cuBLAS GEMMs with fp32 output.  All kernels in ``generic_sm90a/``.  Sources of every kernel: one
persistent producer-warpgroup (TMA) + two-consumer-warpgroup (wgmma) GEMM skeleton, see ``generic_sm90a/tr90.cuh``.

``supported()`` is the whole gate.  Every launch has a fake, so the module traces under ``torch.compile``.
"""

import functools
import warnings
from pathlib import Path

import torch

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_dir = Path(__file__).parent / "generic_sm90a"
_REVISION = "v28"
MAX_D = 1024
# `ext.prep` job masks: bit 0 Ws^T, 1-2 the two halves of [Wa;Wb]^T, 3-5 copies of Wa / Wb / Ws (TF32-rounded for fp32 activations)
_PREP_BWD, _PREP_BWD_ROUNDED, _PREP_ROUNDED_COPIES = 0b000111, 0b011111, 0b111000


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name=f"transition_generic_sm90a_{_REVISION}",
        sources=[str(_dir / "bind.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"),
                           f"-I{_dir.parent / 'anthropic_v5'}", f"-I{_dir}", "--expt-relaxed-constexpr",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                           "-U__CUDA_NO_BFLOAT162_OPERATORS__"],
        extra_cflags=["-std=c++17"], verbose=False,
    )


@functools.lru_cache(maxsize=8)
def _is_hopper(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (9, 0)


def _tf32_allowed() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32) or torch.get_float32_matmul_precision() != "highest"


def _settings():
    from miniworld_engine import settings

    return settings.current()


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """sm_90, bf16 (or fp32 with TF32 allowed), D a multiple of 64 up to 1024, hidden a multiple of 128, weights in the activation dtype.
    Any row count."""
    if not _settings().transition_generic_sm90a:
        return False
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float32):
        return False
    if wa.dtype != x.dtype or ws.dtype != x.dtype:
        return False
    if x.dtype is torch.float32 and not _tf32_allowed():
        return False
    if not _is_hopper(x.device.index if x.device.index is not None else torch.cuda.current_device()):
        return False
    d = x.shape[-1]
    if d % 64 != 0 or d > MAX_D or x.numel() == 0:
        return False
    hid = wa.shape[0]
    return hid % 128 == 0 and tuple(wa.shape) == (hid, d) and tuple(ws.shape) == (d, hid)


# Rows up to which the generic kernels (tiles over the output columns too) beat the row-tile-per-CTA fused / wide builds, which leave SMs
# idle below ~132 tiles: (inference, training) per D, measured at n = 4 on H100 against those builds, anthropic and the compiled PyTorch reference.
_PREFER_GENERIC_ROWS = {64: (3072, 3072), 128: (6144, 3072), 256: (6144, 12288), 384: (16384, 16384), 512: (16384, 12288)}


def prefers_generic(x: torch.Tensor, training: bool, n: int = 4) -> bool:
    """Whether the generic kernels are the faster pick for this call even though a fused / wide build would also take it.
    D = 64 inference: the wide kernel (one launch, 64-row tiles up to 20 480 rows) beats the three generic ones at every row count."""
    if x.dtype is not torch.bfloat16 or not _settings().transition_generic_small:
        return False
    limits = _PREFER_GENERIC_ROWS.get(x.shape[-1])
    if limits is None:
        return False
    rows = x.numel() // x.shape[-1]
    if x.shape[-1] == 64 and not training:       # the 64-row-tile wide forward (one launch) is the fastest D = 64 inference path
        return False
    if x.shape[-1] == 128 and not training:      # D128 inference: the fused build (x first, fragment LayerNorm) vs three generic launches
        return rows <= (512 if n == 2 else 4096)     # measured (ping-pong fused build): n = 2 fused 7.56 / generic 7.55 us at 768 rows and fused
                                                     # ahead of generic by 0.3 us per call in the single-call graph; n = 4 generic 10.9 / fused 11.8
                                                     # at 3 072, fused 11.8 / generic 12.2 at 4 608
    return rows <= limits[1 if training else 0]


_BUILD_FAILED = False


def _is_fake(*tensors) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor
    return any(isinstance(t, FakeTensor) for t in tensors)


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful build (cached).  A build failure is an environment problem: it warns once and keeps the
    existing path."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(x, wa, ws):
        return False
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"generic sm90a Transition unavailable, keeping the existing path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _saves_ab(x: torch.Tensor, save: bool) -> bool:
    """Training bf16 forwards keep a | b for the backward (no recompute GEMM there); TF32 keeps the recompute (a, b in fp32 would double the bytes)."""
    return save and x.dtype is torch.bfloat16 and _settings().transition_generic_save_ab


def _cublas_plan(rows: int, d: int, hid: int) -> tuple[bool, bool, bool]:
    """(F3 as cuBLAS addmm, gate as cuBLAS dh GEMM + SwiGLU-backward kernel, dxn as cuBLAS over [Wa;Wb]) for a bf16 call: below the row
    counts where the hand-written tiles fill the SMs, cuBLAS's tile shapes win (measured; ``settings.transition_generic_cublas`` = any of
    ``f3`` / ``gate`` / ``dxn`` comma-separated, ``none`` or ``all`` forces)."""
    force = _settings().transition_generic_cublas
    if force is not None:
        parts = {p.strip() for p in force.split(",")}
        if "all" in parts:
            parts = {"f3", "gate", "dxn"}
        return "f3" in parts, "gate" in parts, "dxn" in parts
    return False, False, rows <= 12288                        # dxn only: same rounding as the hand tiles, 3-12 % faster below 12k rows


def _ln_fused(x: torch.Tensor, wa: torch.Tensor) -> bool:
    """bf16, D in {128, 256, 384, 512}: the expand GEMM normalizes its own x tile.  OFF by default (``settings.transition_generic_lnf``
    turns it on): measured 5-25 % SLOWER than ln_fwd + expand at 768 .. 12 288 rows -- the per-CTA normalization sits on the critical
    path, while the separate kernel runs on every SM at once and overlaps its successor through PDL."""
    return (x.dtype is torch.bfloat16 and x.shape[-1] in (128, 256, 384, 512) and wa.shape[0] % 128 == 0
            and _settings().transition_generic_lnf)


_SIDE_STREAMS: dict = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    key = (device.type, device.index if device.index is not None else torch.cuda.current_device())
    st = _SIDE_STREAMS.get(key)
    if st is None:
        st = _SIDE_STREAMS[key] = torch.cuda.Stream(device=device)
    return st


# Rows up to which the weight gradients overlap the dxn -> ln_bwd chain profitably (measured, n = 4, training step):
# 1.25 - 1.28x at D64 / 128 for 768 - 3072 rows, 1.03 - 1.13x at D256 / 384, nothing at 12 288 rows and 1-3 % LOSS at 49 152+ (the SMs are full)
_DW_ASIDE_ROWS = {64: 12288, 128: 3072, 256: 3072, 384: 3072, 512: 768}


def _dw_aside_enabled(rows: int | None = None, d: int | None = None) -> bool:
    if not _settings().transition_generic_dw_aside:
        return False
    if rows is None or d is None:
        return True
    limit = _DW_ASIDE_ROWS.get(d)
    return limit is not None and rows <= limit


class _weight_grads_aside:
    """Run the enclosed weight-gradient GEMMs on a second stream (fork after the gate, join at the end of the backward).  Inside a CUDA
    graph this is a parallel branch.  The tensors the branch reads are ``record_stream``-ed so the allocator does not reuse them under it.
    ``MINIWORLD_TRANSITION_GENERIC_DW_ASIDE=0`` runs them in line."""

    def __init__(self, *reads: torch.Tensor, rows: int | None = None, d: int | None = None):
        self.reads = reads
        self.on = _dw_aside_enabled(rows, d)
        self.keep: list = []

    def __enter__(self):
        if not self.on:
            return self.keep
        dev = self.reads[0].device
        side = _side_stream(dev)
        side.wait_stream(torch.cuda.current_stream())
        for t in self.reads:
            t.record_stream(side)
        self._ctx = torch.cuda.stream(side)
        self._ctx.__enter__()
        return self.keep

    def __exit__(self, *exc):
        if self.on:
            self._ctx.__exit__(*exc)
        return False


def _fwd_fake(x, gamma, beta, wa, wb, ws, eps, save):
    rows = x.shape[0] if save else 1
    f32 = dict(dtype=torch.float32, device=x.device)
    ab = torch.empty((x.shape[0], 2 * wa.shape[0]), dtype=x.dtype, device=x.device) if _saves_ab(x, save) else x.new_empty((1, 1))
    hs = torch.empty((x.shape[0], wa.shape[0]), dtype=x.dtype, device=x.device) if _saves_ab(x, save) else x.new_empty((1, 1))
    return (torch.empty_like(x), torch.empty_like(x), torch.empty((rows,), **f32), torch.empty((rows,), **f32), ab, hs)


@opaque(fake=_fwd_fake, name="transition_generic_fwd_sm90a")
def _fwd_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor,
                eps: float, save: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, D]; returns (out, xn, rstd, c1, ab, h), out = transition(x) + x; rstd / c1 are 1-element placeholders unless ``save``; ``ab``
    [M, 2 HID] = [a | b] and ``h`` [M, HID] (from the fp32 accumulators) when the bf16 training forward keeps them for the backward,
    else 1-element placeholders."""
    if _is_fake(x, wa):
        return _fwd_fake(x, gamma, beta, wa, wb, ws, eps, save)
    ext = _ext()
    if x.dtype is torch.float32:                                  # the MMA would truncate raw fp32 weights: hand it TF32-rounded copies
        _, _, wa, wb, ws = ext.prep(wa, wb, ws, _PREP_ROUNDED_COPIES, True)
    if _ln_fused(x, wa):
        # LayerNorm formed in shared memory by the expand GEMM itself: one launch fewer, no xn round trip
        h, ab, xn, rstd, c1 = ext.swiglu_ln(x, gamma, beta, float(eps), wa, wb, bool(save) and _saves_ab(x, save))
        if save and not _saves_ab(x, save):                       # (saved-a/b switched off: stats still needed)
            xn, rstd, c1 = ext.ln_fwd(x, gamma, beta, float(eps), True)
    else:
        xn, rstd, c1 = ext.ln_fwd(x, gamma, beta, float(eps), bool(save))
        if _saves_ab(x, save):
            h, ab = ext.swiglu_save(xn, wa, wb)
        else:
            h, ab = ext.swiglu(xn, wa, wb), x.new_empty((1, 1))
    if x.dtype is torch.bfloat16 and _cublas_plan(x.shape[0], x.shape[1], wa.shape[0])[0]:
        out = torch.addmm(x, h, ws.t())                        # y = h Ws^T + x, residual as beta * C
    else:
        out = ext.squeeze_res(h, ws, x)
    return out, xn, rstd, c1, ab, (h if ab.numel() > 1 else x.new_empty((1, 1)))


def _bwd_fake(dy, x, xn, rstd, c1, ab, h, wcat, gamma, wa, wb, ws):
    return (torch.empty_like(x), torch.empty_like(gamma), torch.empty_like(gamma),
            torch.empty((2 * wa.shape[0], wa.shape[1]), dtype=wa.dtype, device=wa.device), torch.empty(ws.shape, dtype=ws.dtype, device=ws.device))


@opaque(fake=_bwd_fake, name="transition_generic_bwd_sm90a")
def _bwd_launch(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, rstd: torch.Tensor, c1: torch.Tensor, ab: torch.Tensor, h: torch.Tensor, wcat: torch.Tensor, gamma: torch.Tensor,
                wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dx, dgamma, dbeta, dWab, dWs): ``dx`` already carries the residual branch, dWab = [dWa; dWb] [2 HID, D], the weight
    gradients come out of cuBLAS (fp32 accumulation, one rounding to the weight dtype) in the weight dtype."""
    if _is_fake(dy, x):
        return _bwd_fake(dy, x, xn, rstd, c1, ab, h, wcat, gamma, wa, wb, ws)
    ext = _ext()
    tf32 = x.dtype is torch.float32
    if ab.numel() > 1:                  # bf16 training: a | b saved by the forward; the weights are read as stored (MN-major wgmma operand)
        _, gate_cublas, _ = _cublas_plan(x.shape[0], x.shape[1], wa.shape[0])
        if gate_cublas:
            dab = ext.swiglu_bwd(torch.mm(dy, ws), ab)         # dh by cuBLAS, SwiGLU backward in one elementwise pass
        else:
            dab = ext.gate_ab(dy, ws, ab)
        with _weight_grads_aside(dy, h, dab, xn, rows=x.shape[0], d=x.shape[1]) as grads:     # dW GEMMs on a second stream: off the dxn -> ln_bwd critical path
            dws = torch.mm(dy.t(), h)                          # [D, HID]
            dwab = torch.mm(dab.t(), xn)                       # [2 HID, D]: dWa then dWb (dab holds dA | dB)
            grads.append((dwab, dws))
        del h
        dxn = torch.mm(dab, wcat) if wcat.numel() > 1 else ext.dxn_gemm_t(dab, wa, wb)
    else:
        wst, wabt, wa_r, wb_r, _ = ext.prep(wa, wb, ws, _PREP_BWD_ROUNDED if tf32 else _PREP_BWD, tf32)   # Ws^T, [Wa;Wb]^T (+ rounded Wa, Wb)
        h, dab = ext.gate(xn, dy, wa_r if tf32 else wa, wb_r if tf32 else wb, wst)
        dws = torch.mm(dy.t(), h)
        dwab = torch.mm(dab.t(), xn)
        del h
        dxn = ext.dxn_gemm(dab, wabt)
    del dab
    dx, dgam, dbeta = ext.ln_bwd(dxn, x, dy, rstd, c1, gamma)
    if ab.numel() > 1 and _dw_aside_enabled(x.shape[0], x.shape[1]):
        torch.cuda.current_stream().wait_stream(_side_stream(x.device))     # join: the weight gradients are ready
    return dx, dgam, dbeta, dwab, dws


class _GenericTransitionSM90A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        gf, bf = gamma.float().contiguous(), beta.float().contiguous()
        wa, wb, ws = wa.contiguous(), wb.contiguous(), ws.contiguous()
        out, xn, rstd, c1, ab, h = _fwd_launch(flat, gf, bf, wa, wb, ws, float(eps), True)
        wcat = (torch.cat([wa, wb], 0) if ab.numel() > 1 and _cublas_plan(flat.shape[0], flat.shape[1], wa.shape[0])[2]
                else flat.new_empty((1, 1)))
        ctx.save_for_backward(flat, xn, rstd, c1, ab, h, wcat, gf, wa, wb, ws)
        ctx.shape = shape
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, rstd, c1, ab, h, wcat, gf, wa, wb, ws = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        dx, dgam, dbeta, dwab, dws = _bwd_launch(dy.reshape(-1, dy.shape[-1]).contiguous(), flat, xn, rstd, c1, ab, h, wcat, gf, wa, wb, ws)
        hid = wa.shape[0]
        return (dx.reshape(ctx.shape), dgam.to(gdt), dbeta.to(bdt), dwab[:hid].to(adt), dwab[hid:].to(bwdt), dws.to(sdt), None)


def transition_generic_sm90a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry (same signature as ``fused_sm90a.transition_fused_sm90a``).  Call ``supported()`` first.  Inference (grad
    off, or nothing requiring grad) runs the forward without the saves."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))):
        shape = x.shape
        out, *_ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), gamma.float().contiguous(), beta.float().contiguous(),
                                   wa.contiguous(), wb.contiguous(), ws.contiguous(), float(eps), False)
        return out.reshape(shape)
    return _GenericTransitionSM90A.apply(x, gamma, beta, wa, wb, ws, eps)

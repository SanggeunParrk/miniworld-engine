"""Wide-width sm_80 Transition: every (D, n) of the registry that ``fused_sm80`` (D = 128 / n = 4) does not take, and the bare SwiGLU FFN.

The A100 counterpart of ``fused_wide_sm90a`` / ``fused_wide_sm100a`` -- their data flow (LayerNorm, one dual GEMM + SwiGLU, the squeeze with the
residual folded in; in the backward the gate with the SwiGLU backward, the weight-gradient GEMMs, d_xn and the LayerNorm backward), without TMA /
wgmma / tcgen05.  What is hand CUDA and what cuBLAS was decided by measurement (docs/gpus/a100/transition/transition.md):

    forward   ln_fwd (row kernel: xn, mean / rstd in fp32)  ->  dual_swiglu (a | b = xn [Wa; Wb]^T, h = rn(silu(a) b))  ->  gemm_res (h Ws^T + x)
    backward  dh = dy Ws (cuBLAS)  ->  gate_bwd (a, b recomputed, dA | dB | h)  ->  dWs = dy^T h, dWa = dA^T xn, dWb = dB^T xn, d_xn = [dA | dB] [Wa; Wb]
              (cuBLAS, fp32 d_xn)  ->  ln_bwd (row kernel: dx = LN backward + dy, dgamma / dbeta partials per CTA, reduced in a fixed order)

``supported()`` is the whole gate: sm_80, bf16, width D in ``WIDTHS`` with hidden H a multiple of 64, any row count.  The reductions run in a
fixed order (no atomics), so a replay is bit-identical.  ``MINIWORLD_TRANSITION_WIDE_SM80=0`` (or ``MINIWORLD_TRANSITION_FUSED_SM80=0``, the one switch
of every A100 Transition path) turns it off.
"""

import functools
import hashlib
import os
import warnings
from pathlib import Path

import torch

from ... import _capture
from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from . import fused_bwd_sm80, fused_fwd_sm80
from .fused_sm80 import _is_ampere, _is_fake

_dir = Path(__file__).parent / "sm80"

#: Widths with a row-kernel instantiation (and a tile config); the hidden width is any multiple of 64.
WIDTHS = (64, 128, 256, 384, 512, 768)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    # MINIWORLD_TRANSITION_WIDE_SM80_FLAGS: extra nvcc flags for experiments (their own build; "-Xptxas -v" shows the registers / spills)
    extra = os.environ.get("MINIWORLD_TRANSITION_WIDE_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"transition_wide_sm80{tag}",
        sources=[str(_dir / "transition_wide_sm80.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=bool(extra),
    )


_BUILD_FAILED = False


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """The extension builds / loads (cached); a failure warns once and keeps the existing path.  Constant for dynamo: the JIT build is never traced."""
    global _BUILD_FAILED
    if _BUILD_FAILED:
        return False
    try:
        _ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"wide sm80 Transition unavailable, keeping the existing path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def switched_off() -> bool:
    """``MINIWORLD_TRANSITION_WIDE_SM80=0`` or ``MINIWORLD_TRANSITION_FUSED_SM80=0`` (the one switch of every A100 Transition path)."""
    return os.environ.get("MINIWORLD_TRANSITION_WIDE_SM80", "1") == "0" or os.environ.get("MINIWORLD_TRANSITION_FUSED_SM80", "1") == "0"


_WEIGHT_DTYPES = (torch.bfloat16, torch.float32)


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """The kernels' own requirements: sm_80, bf16, D in WIDTHS, H % 64 == 0, any row count; the weights bf16 or fp32 (an fp32 master: the entries cast them to bf16 outside autograd); ``MINIWORLD_TRANSITION_WIDE_SM80=0`` /
    ``MINIWORLD_TRANSITION_FUSED_SM80=0`` turn it off."""
    if switched_off():
        return False
    if not x.is_cuda or x.dtype is not torch.bfloat16 or wa.dtype not in _WEIGHT_DTYPES or ws.dtype not in _WEIGHT_DTYPES:
        return False
    if not _is_ampere(x.device.index if x.device.index is not None else torch.cuda.current_device()):
        return False
    d, h = x.shape[-1], wa.shape[0]
    if d not in WIDTHS or wa.shape != (h, d) or ws.shape != (d, h) or h % 64:
        return False
    return x.numel() > 0


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful (cached) build."""
    return supported(x, wa, ws) and _loads()


#: Fewest rows for which ``fused_sm80`` (the D = 128 / n = 4 forward and backward kernels, 256-row tiles) beats this module's path: a tile costs ~55 us
#: whatever the row count, so below it most SMs idle (msa_token rows are 1024 .. 6144: the kernel is 1.2-2x slower than Triton there).  Measured, docs page.
FUSED_MIN_ROWS = 8192


def route(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> str | None:
    """The A100 path that serves a call: ``"fused"`` (``fused_sm80``), ``"wide"`` (this module) or ``None`` (the Triton path)."""
    from . import fused_sm80

    if fused_sm80.available(x, wa, ws) and x.numel() // x.shape[-1] >= FUSED_MIN_ROWS:
        return "fused"
    return "wide" if available(x, wa, ws) else None


# ----------------------------------------------------------------------------------------------------------------------- tile choice
def _tiles(m: int, d: int, h: int) -> tuple[int, int]:
    """(dual-GEMM tile config, residual-GEMM tile config) of the extension for an M-row call (measured: docs page)."""
    dual = 0 if m > 4096 else 2
    if d <= 128:
        res = 3             # 128 x 64 tile: the best at D <= 128
    elif d == 256 and m < 32768:
        res = 2             # 64 x 128: 256 CTAs of the 128-row tile are 1.2 waves of the 216 resident slots, 512 are 2.4
    else:
        res = 0
    return dual, res


def _mm_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM (cuBLAS with an fp32 output); torch builds without ``out_dtype`` round through bf16."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a, b).float()


# ------------------------------------------------------------------------------------------------------------------------ launches
_affine: dict = {}


def _f32(t: torch.Tensor, cache: bool) -> torch.Tensor:
    """The contiguous fp32 view of a LayerNorm affine parameter: the parameter itself when it is fp32; a cast launch otherwise (the module's parameters are bf16 once
    it is ``.to(bfloat16)``), reused across inference calls by parameter version (scoped by ``_capture``)."""
    if t.dtype is torch.float32 and t.is_contiguous():
        return t
    if not cache:
        return t.float().contiguous()
    key = (t.data_ptr(), t._version)
    # keyed on the parameter and scoped by ``_capture``; the parameter stays alive with its cast: a recycled data_ptr cannot alias it
    return _capture.lookup(_affine, key, lambda: (t.float().contiguous(), t), limit=16)[0]


#: Squeeze projections with the residual (the Transition) run the hand GEMM with the residual in its epilogue from HAND_SQUEEZE_MIN_ROWS rows on: with the lean pipelined
#: mainloop it is 6-20 % faster than cuBLAS + a residual add pass at every width (one launch, no second pass over the output); with fewer rows its tiles leave most SMs
#: idle and cuBLAS (which splits K) wins.  The bare FFN (no residual) is always cuBLAS.  Measured, docs page.
HAND_SQUEEZE_MIN_ROWS = 8192


def _hand_squeeze(m: int, d: int, ln: bool) -> bool:
    """True: the hand ``gemm_res``; False: cuBLAS ``mm`` (+ ``add_res`` with the residual)."""
    return ln and m >= HAND_SQUEEZE_MIN_ROWS


def _squeeze(ext, hid: torch.Tensor, ws: torch.Tensor, x: torch.Tensor, ln: bool, res_cfg: int) -> torch.Tensor:
    """h Ws^T (+ x): the hand GEMM with the residual folded in where ``_hand_squeeze``, else cuBLAS and ``add_res``."""
    if _hand_squeeze(hid.shape[0], ws.shape[0], ln):
        return ext.gemm_res(hid, ws, x, res_cfg)
    out = torch.mm(hid, ws.t())
    if ln:
        ext.add_res(out, x)
    return out


def _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save, ln):
    """(out like x, xn [M, D] when ``save`` and ``ln`` else empty, stats [M, 2] f32 when ``save`` and ``ln`` else empty)."""
    keep = save and ln
    return (torch.empty_like(x), x.new_empty(x.shape) if keep else x.new_empty((0,)),
            torch.empty((x.shape[0] if keep else 0, 2), dtype=torch.float32, device=x.device))


@opaque(fake=_fwd_launch_fake, name="transition_wide_fwd_sm80")
def _fwd_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor,
                eps: float, save: bool, ln: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, D] -> out = x + Ws (silu(Wa xn) * (Wb xn)), xn = LN(x) (``ln``; the FFN has neither the LayerNorm nor the residual; gamma / beta are the module's affine
    parameters in any float dtype).  Training also returns xn and (mean, rstd)."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save, ln)
    m, d = x.shape
    if ln:
        gamma, beta = _f32(gamma, not save), _f32(beta, not save)
    # the weights are bf16 or fp32 (a master); the fused forward casts (and packs, cached by the parameters' version) itself, the other path casts here
    if fused_fwd_sm80.supports(d, wa.shape[0], m) and fused_fwd_sm80.loads():
        # D = 64 / 128: the whole forward in one kernel (a warp owns 32 rows for the whole hidden dimension)
        out, xn, stats = fused_fwd_sm80.forward(x, gamma, beta, wa, wb, ws, eps, save, cache=not save, ln=ln)
        return out, xn if save and ln else x.new_empty((0,)), stats
    ext = _ext()
    wa, wb, ws = (w.to(x.dtype).contiguous() for w in (wa, wb, ws))
    dual, res = _tiles(m, d, wa.shape[0])
    if ln:
        xn, stats = ext.ln_fwd(x, gamma, beta, eps, save)
    else:
        xn, stats = x, torch.empty((0, 2), dtype=torch.float32, device=x.device)
    out = _squeeze(ext, ext.dual_swiglu(xn, wa, wb, dual), ws, x, ln, res)
    return out, (xn if save and ln else x.new_empty((0,))), stats


def _bwd_launch_fake(dy, x, xn, stats, gamma, wa, wb, ws, ln, gdt, wdt):
    """(dx like x, dgamma, dbeta [D] in ``gdt`` (empty without ``ln``), dW = [dWa; dWb] [2H, D], dWs [D, H] in ``wdt``)."""
    d = x.shape[1]
    return (torch.empty_like(x), torch.empty((d if ln else 0,), dtype=gdt, device=x.device), torch.empty((d if ln else 0,), dtype=gdt, device=x.device),
            torch.empty((2 * wa.shape[0], d), dtype=wdt, device=x.device), torch.empty((d, wa.shape[0]), dtype=wdt, device=x.device))


@opaque(fake=_bwd_launch_fake, name="transition_wide_bwd_sm80")
def _bwd_launch(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, stats: torch.Tensor, gamma: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
                ws: torch.Tensor, ln: bool, gdt: torch.dtype, wdt: torch.dtype,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dgamma, dbeta, [dWa; dWb], dWs); dx carries the residual branch (``ln``); dgamma / dbeta come out in ``gdt``, the weight gradients in ``wdt`` (f32: the accumulators unrounded, for an fp32 master).  a, b are recomputed from xn (x for the FFN)."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, stats, gamma, wa, wb, ws, ln, gdt, wdt)
    m, d = x.shape
    h = wa.shape[0]
    if ln:
        gamma = _f32(gamma, False)
    if fused_bwd_sm80.supports(d, h, m) and fused_bwd_sm80.loads():
        # D = 64 / 128, whole 256-row tiles: PW (a, b, dh, SwiGLU backward, dW partials, dA | dB blocks) + X (d_xn, LayerNorm backward) + the partial sums
        return fused_bwd_sm80.backward(dy, x, xn, stats, gamma, wa, wb, ws, ln, gdt, wdt)
    ext = _ext()
    a_in = xn if ln else x
    dual, _ = _tiles(m, d, h)
    dh = torch.mm(dy, ws)                                       # [M, H]
    dab, hid = ext.gate_bwd(a_in, wa, wb, dh, dual)             # dA | dB [M, 2H], h [M, H]
    mm_w = _mm_f32 if wdt is torch.float32 else torch.mm       # fp32 master: the f32 accumulators unrounded
    dws = mm_w(dy.t(), hid)                                     # [D, H]
    dwab = mm_w(dab.t(), a_in)                                  # [2H, D] = [dWa; dWb]
    wab = torch.cat((wa, wb), 0)                                # [2H, D]
    if ln:
        dxn = _mm_f32(dab, wab)                                 # [M, D] f32
        dx, dgam, dbeta = ext.ln_bwd(dxn, x, stats, gamma, dy, gdt)
    else:
        dx = torch.mm(dab, wab)
        dgam, dbeta = (torch.empty((0,), dtype=gdt, device=x.device) for _ in range(2))   # distinct placeholders: custom-op outputs must not alias
    return dx, dgam, dbeta, dwab, dws


def _bwd_dx_fake(dy, x, xn, stats, gamma, wa, wb, ws, ln):
    """Shape of the input gradient: a fresh tensor like ``x``."""
    return torch.empty_like(x)


@opaque(fake=_bwd_dx_fake, name="transition_wide_bwd_dx_sm80")
def _bwd_dx(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, stats: torch.Tensor,
            gamma: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor,
            ln: bool) -> torch.Tensor:
    """Input gradient for frozen parameters; omit the two weight-gradient GEMMs."""
    ext = _ext()
    dual, _ = _tiles(x.shape[0], x.shape[1], wa.shape[0])
    dab, _ = ext.gate_bwd(xn if ln else x, wa, wb, torch.mm(dy, ws), dual)
    wab = torch.cat((wa, wb), 0)
    if not ln:
        return torch.mm(dab, wab)
    dx, _, _ = ext.ln_bwd(_mm_f32(dab, wab), x, stats, _f32(gamma, False), dy, gamma.dtype)
    return dx


class _WideTransitionSM80(torch.autograd.Function):
    """``y = transition(x) + x`` (``ln``) or the bare FFN ``squeeze(silu(a) b)``; the forward saves xn and (mean, rstd) (the FFN saves x only)."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps, ln):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        if not ln:
            gamma = beta = flat.new_empty((0,), dtype=torch.float32)
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        wa, wb, ws = (w.to(x.dtype).contiguous() for w in (wa, wb, ws))      # the kernels' dtype; cast here, outside autograd
        out, xn, stats = _fwd_launch(flat, gamma, beta, wa, wb, ws, float(eps), True, ln)
        ctx.save_for_backward(flat, xn, stats, gamma, wa, wb, ws)
        ctx.shape, ctx.ln = shape, ln
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, stats, gamma, wa, wb, ws = ctx.saved_tensors
        if not any(ctx.needs_input_grad[1:6]):
            dx = _bwd_dx(dy.reshape(-1, dy.shape[-1]).contiguous(), flat, xn, stats, gamma, wa, wb, ws, ctx.ln)
            return dx.reshape(ctx.shape), None, None, None, None, None, None, None
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        wdt = adt if adt in (torch.float32, torch.bfloat16) and adt == bwdt == sdt else torch.float32
        dx, dgam, dbeta, dwab, dws = _bwd_launch(dy.reshape(-1, dy.shape[-1]).contiguous(), flat, xn, stats, gamma, wa, wb, ws, ctx.ln, gdt, wdt)
        h = wa.shape[0]
        dbeta = dbeta if bdt == gdt else dbeta.to(bdt)
        return (dx.reshape(ctx.shape), dgam if ctx.ln else None, dbeta if ctx.ln else None,
                dwab[:h].to(adt), dwab[h:].to(bwdt), dws.to(sdt), None, None)


def _needs_grad(*tensors) -> bool:
    return torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)


def transition_wide_sm80(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as the Triton ``transition_residual``.  Call ``available(x, wa, ws)`` first.  No grad needed: the forward
    runs without the xn / statistics saves."""
    if not _needs_grad(x, gamma, beta, wa, wb, ws):
        shape = x.shape
        out, _, _ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), gamma, beta, wa, wb, ws, float(eps), False, True)
        return out.reshape(shape)
    return _WideTransitionSM80.apply(x, gamma, beta, wa, wb, ws, eps, True)


def swiglu_ffn_sm80(x, wa, wb, ws):
    """The bare SwiGLU FFN ``squeeze(silu(x Wa^T) * (x Wb^T))`` (no LayerNorm, no residual).  Call ``available(x, wa, ws)`` first."""
    if not _needs_grad(x, wa, wb, ws):
        shape = x.shape
        empty = x.new_empty((0,), dtype=torch.float32)
        out, _, _ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), empty, empty, wa, wb, ws, 0.0, False, False)
        return out.reshape(shape)
    return _WideTransitionSM80.apply(x, None, None, wa, wb, ws, 0.0, False)

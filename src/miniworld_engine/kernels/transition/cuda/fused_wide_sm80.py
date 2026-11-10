"""Wide-width sm_80 Transition: every (D, n) of the registry that ``fused_sm80`` (D = 128 / n = 4) does not take, and the bare SwiGLU FFN.

The A100 counterpart of ``fused_wide_sm90a`` / ``fused_wide_sm100a`` -- their data flow (LayerNorm, one dual GEMM + SwiGLU, the squeeze with the
residual folded in; in the backward the gate with the SwiGLU backward, the weight-gradient GEMMs, d_xn and the LayerNorm backward), without TMA /
wgmma / tcgen05.  What is hand CUDA and what cuBLAS was decided by measurement (docs/gpus/a100/transition/transition.md):

    forward   ln_fwd (row kernel: xn, mean / rstd in fp32)  ->  dual_swiglu (a | b = xn [Wa; Wb]^T, h = rn(silu(a) b))  ->  gemm_res (h Ws^T + x)
    backward  dh = dy Ws (cuBLAS)  ->  gate_bwd (a, b recomputed, dA | dB | h)  ->  dWs = dy^T h, dWa = dA^T xn, dWb = dB^T xn, d_xn = [dA | dB] [Wa; Wb]
              (cuBLAS, fp32 d_xn)  ->  ln_bwd (row kernel: dx = LN backward + dy, dgamma / dbeta partials per CTA, reduced in a fixed order)

``supported()`` is the whole gate: sm_80, bf16, width D in ``WIDTHS`` with hidden H a multiple of 64, any row count.  The reductions run in a
fixed order (no atomics), so a replay is bit-identical.
"""

import functools
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
    return load_extension(
        name="transition_wide_sm80",
        sources=[str(_dir / "transition_wide_sm80.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_80,code=sm_80",
            f"-I{_dir}",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
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
        warnings.warn(
            f"wide sm80 Transition unavailable, keeping the existing path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


_WEIGHT_DTYPES = (torch.bfloat16, torch.float32)


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """The kernels' own requirements: sm_80, bf16, D in WIDTHS, H % 64 == 0, any row count; the weights bf16 or fp32 (an fp32 master: the entries cast them to bf16 outside autograd)."""
    if (
        not x.is_cuda
        or x.dtype is not torch.bfloat16
        or wa.dtype not in _WEIGHT_DTYPES
        or ws.dtype not in _WEIGHT_DTYPES
    ):
        return False
    if not _is_ampere(
        x.device.index if x.device.index is not None else torch.cuda.current_device()
    ):
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
        res = 3  # 128 x 64 tile: the best at D <= 128
    elif d == 256 and m < 32768:
        res = 2  # 64 x 128: 256 CTAs of the 128-row tile are 1.2 waves of the 216 resident slots, 512 are 2.4
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
    if not torch.is_grad_enabled():   # grad off: a capture reads the eager cast (no cast kernel per replay)
        return _capture.param_forms("transition_sm80_affine", [t], lambda: t.float().contiguous(), capture_reads_eager=True)
    key =(t.data_ptr(), t._version)
    # keyed on the parameter and scoped by ``_capture``; the parameter stays alive with its cast: a recycled data_ptr cannot alias it
    return _capture.lookup(_affine, key, lambda: (t.float().contiguous(), t), limit=16)[
        0
    ]


#: Squeeze projections with the residual (the Transition) run the hand GEMM with the residual in its epilogue from HAND_SQUEEZE_MIN_ROWS rows on: with the lean pipelined
#: mainloop it is 6-20 % faster than cuBLAS + a residual add pass at every width (one launch, no second pass over the output); with fewer rows its tiles leave most SMs
#: idle and cuBLAS (which splits K) wins.  The bare FFN (no residual) is always cuBLAS.  Measured, docs page.
HAND_SQUEEZE_MIN_ROWS = 8192


def _hand_squeeze(m: int, d: int, ln: bool) -> bool:
    """True: the hand ``gemm_res``; False: cuBLAS ``mm`` (+ ``add_res`` with the residual)."""
    return ln and m >= HAND_SQUEEZE_MIN_ROWS


def _squeeze(
    ext, hid: torch.Tensor, ws: torch.Tensor, x: torch.Tensor, ln: bool, res_cfg: int
) -> torch.Tensor:
    """h Ws^T (+ x): the hand GEMM with the residual folded in where ``_hand_squeeze``, else cuBLAS and ``add_res``."""
    if _hand_squeeze(hid.shape[0], ws.shape[0], ln):
        return ext.gemm_res(hid, ws, x, res_cfg)
    out = torch.mm(hid, ws.t())
    if ln:
        ext.add_res(out, x)
    return out


_kernel_forms: dict = {}


def _kernel_weights(wa, wb, ws, dtype, cache: bool):
    """The three projection weights as contiguous ``dtype`` tensors: themselves when they already are; otherwise (an fp32 master) the casts, which an inference
    call reuses by parameter version (scoped by ``_capture``: a graph replay under ``static_weights`` runs no cast kernel, a training graph recasts the current weights)."""
    ws_ = (wa, wb, ws)
    if all(w.dtype == dtype and w.is_contiguous() for w in ws_):
        return ws_
    if not cache:
        return tuple(w.to(dtype).contiguous() for w in ws_)
    if not torch.is_grad_enabled():
        # grad off: a CUDA-graph capture reads the casts its eager warm-up made (``capture_reads_eager``): no cast kernel per replay
        return _capture.param_forms(("transition_sm80_weights", dtype), list(ws_), lambda: tuple(w.to(dtype).contiguous() for w in ws_),
                                    capture_reads_eager=True)
    key =tuple((w.data_ptr(), w._version, w.dtype) for w in ws_) + (dtype,)
    # the parameters stay alive with their casts: a recycled data_ptr cannot alias the entry
    return _capture.lookup(
        _kernel_forms,
        key,
        lambda: (*(w.to(dtype).contiguous() for w in ws_), ws_),
        limit=8,
    )[:3]


def _save_ab(m: int, d: int, h: int, save: bool) -> bool:
    """True: the training forward also writes h and a | b, and the backward is the one-kernel ``gate_dh`` (B200's saved-activation scheme at D >= 256)."""
    return (
        save
        and d >= SAVE_AB_MIN_D
        and m >= SAVE_AB_MIN_ROWS
        and not (fused_bwd_sm80.supports(d, h, m) and fused_bwd_sm80.loads())
    )


#: widths / rows from which the saved a | b scheme is used (measured: docs page)
SAVE_AB_MIN_D = 128
SAVE_AB_MIN_ROWS = 1024


def _dh_cfg(m: int, d: int, h: int) -> int:
    """Tile config of ``gate_dh``: config 0 (128 x 64, two CTAs per SM). In isolation the 128 x 128 tile (config 1) is 6-10 % faster at D384 / D512 (147 k rows), but inside the training step it is
    0.5-2.5 % slower (interleaved A/B, D256 / D384 / D512), so it is not selected."""
    return 0


def _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save, ln):
    """(out like x, xn [M, D] and stats [M, 2] f32 when ``save`` and ``ln`` else empty, h [M, H] and a | b [M, 2H] when the saved-activation scheme applies else empty)."""
    keep = save and ln
    sab = _save_ab(x.shape[0], x.shape[1], wa.shape[0], save)
    h = wa.shape[0]
    return (
        torch.empty_like(x),
        x.new_empty(x.shape) if keep else x.new_empty((0,)),
        torch.empty(
            (x.shape[0] if keep else 0, 2), dtype=torch.float32, device=x.device
        ),
        x.new_empty((x.shape[0], h)) if sab else x.new_empty((0,)),
        x.new_empty((x.shape[0], 2 * h)) if sab else x.new_empty((0,)),
    )


@opaque(fake=_fwd_launch_fake, name="transition_wide_fwd_sm80")
def _fwd_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    eps: float,
    save: bool,
    ln: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, D] -> out = x + Ws (silu(Wa xn) * (Wb xn)), xn = LN(x) (``ln``; the FFN has neither the LayerNorm nor the residual; gamma / beta are the module's affine
    parameters in any float dtype).  Training also returns xn and (mean, rstd), and where ``_save_ab`` applies h and a | b."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save, ln)
    m, d = x.shape
    if ln:
        gamma, beta = _f32(gamma, not save), _f32(beta, not save)
    # the weights are bf16 or fp32 (a master); the fused forward casts (and packs, cached by the parameters' version) itself, the other path casts here
    if fused_fwd_sm80.supports(d, wa.shape[0], m) and fused_fwd_sm80.loads():
        # D = 64 / 128: the whole forward in one kernel (a warp owns 32 rows for the whole hidden dimension)
        out, xn, stats = fused_fwd_sm80.forward(
            x, gamma, beta, wa, wb, ws, eps, save, cache=not save, ln=ln
        )
        return (
            out,
            xn if save and ln else x.new_empty((0,)),
            stats,
            x.new_empty((0,)),
            x.new_empty((0,)),
        )
    ext = _ext()
    wa, wb, ws = _kernel_weights(wa, wb, ws, x.dtype, cache=not save)
    dual, res = _tiles(m, d, wa.shape[0])
    if ln:
        xn, stats = ext.ln_fwd(x, gamma, beta, eps, save)
    else:
        xn, stats = x, torch.empty((0, 2), dtype=torch.float32, device=x.device)
    if _save_ab(m, d, wa.shape[0], save):
        hid, ab = ext.dual_swiglu_save(xn, wa, wb, dual)
    else:
        hid, ab = ext.dual_swiglu(xn, wa, wb, dual), x.new_empty((0,))
    out = _squeeze(ext, hid, ws, x, ln, res)
    return (
        out,
        (xn if save and ln else x.new_empty((0,))),
        stats,
        hid if ab.numel() else x.new_empty((0,)),
        ab,
    )


def _bwd_launch_fake(dy, x, xn, stats, gamma, wa, wb, ws, hid, ab, ln, gdt, wdt):
    """(dx like x, dgamma, dbeta [D] in ``gdt`` (empty without ``ln``), dW = [dWa; dWb] [2H, D], dWs [D, H] in ``wdt``)."""
    d = x.shape[1]
    return (
        torch.empty_like(x),
        torch.empty((d if ln else 0,), dtype=gdt, device=x.device),
        torch.empty((d if ln else 0,), dtype=gdt, device=x.device),
        torch.empty((2 * wa.shape[0], d), dtype=wdt, device=x.device),
        torch.empty((d, wa.shape[0]), dtype=wdt, device=x.device),
    )


@opaque(fake=_bwd_launch_fake, name="transition_wide_bwd_sm80")
def _bwd_launch(
    dy: torch.Tensor,
    x: torch.Tensor,
    xn: torch.Tensor,
    stats: torch.Tensor,
    gamma: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    hid: torch.Tensor,
    ab: torch.Tensor,
    ln: bool,
    gdt: torch.dtype,
    wdt: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dgamma, dbeta, [dWa; dWb], dWs); dx carries the residual branch (``ln``); dgamma / dbeta come out in ``gdt``, the weight gradients in ``wdt`` (f32: the accumulators unrounded, for an fp32 master).  a, b are the forward's saved ``ab`` (with h = ``hid``) where the saved-activation scheme applied, else recomputed from xn (x for the FFN)."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(
            dy, x, xn, stats, gamma, wa, wb, ws, hid, ab, ln, gdt, wdt
        )
    m, d = x.shape
    h = wa.shape[0]
    if ln:
        gamma = _f32(gamma, False)
    if fused_bwd_sm80.supports(d, h, m) and fused_bwd_sm80.loads():
        # D = 64 / 128, whole 256-row tiles: PW (a, b, dh, SwiGLU backward, dW partials, dA | dB blocks) + X (d_xn, LayerNorm backward) + the partial sums
        return fused_bwd_sm80.backward(
            dy, x, xn, stats, gamma, wa, wb, ws, ln, gdt, wdt
        )
    ext = _ext()
    a_in = xn if ln else x
    dual, _ = _tiles(m, d, h)
    if ab.numel():
        # one kernel: dh = dy Ws and the SwiGLU backward from the saved a | b (the forward saved h as well)
        dab = ext.gate_dh(dy, ws.t().contiguous(), ab, _dh_cfg(m, d, h))
    else:
        dh = torch.mm(dy, ws)  # [M, H]
        dab, hid = ext.gate_bwd(a_in, wa, wb, dh, dual)  # dA | dB [M, 2H], h [M, H]
    # B200's wide backward also keeps weight accumulators in FP32. Direct BF16
    # GEMM outputs fail the existing expand-weight gradient band at protocol D256.
    full_precision_grad = d == 256
    mm_w = _mm_f32 if wdt is torch.float32 or full_precision_grad else torch.mm
    dws = mm_w(dy.t(), hid)  # [D, H]
    dwab = mm_w(dab.t(), a_in)  # [2H, D] = [dWa; dWb]
    dwab, dws = dwab.to(wdt), dws.to(wdt)   # (the grad_cast kernel of the extension is the opt-in alternative; not used)
    wab = _wab(wa, wb)  # [2H, D]
    if ln:
        dxn = _mm_f32(dab, wab)  # [M, D] f32
        dx, dgam, dbeta = ext.ln_bwd(dxn, x, stats, gamma, dy, gdt)
    else:
        dx = torch.mm(dab, wab)
        dgam, dbeta = (
            torch.empty((0,), dtype=gdt, device=x.device) for _ in range(2)
        )  # distinct placeholders: custom-op outputs must not alias
    return dx, dgam, dbeta, dwab, dws


def _bwd_dx_fake(dy, x, xn, stats, gamma, wa, wb, ws, ab, ln):
    """Shape of the input gradient: a fresh tensor like ``x``."""
    return torch.empty_like(x)


@opaque(fake=_bwd_dx_fake, name="transition_wide_bwd_dx_sm80")
def _bwd_dx(
    dy: torch.Tensor,
    x: torch.Tensor,
    xn: torch.Tensor,
    stats: torch.Tensor,
    gamma: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    ab: torch.Tensor,
    ln: bool,
) -> torch.Tensor:
    """Input gradient for frozen parameters; omit the two weight-gradient GEMMs."""
    ext = _ext()
    m, d, h = x.shape[0], x.shape[1], wa.shape[0]
    if ab.numel():
        dab = ext.gate_dh(dy, ws.t().contiguous(), ab, _dh_cfg(m, d, h))
    else:
        dual, _ = _tiles(m, d, h)
        dab, _ = ext.gate_bwd(xn if ln else x, wa, wb, torch.mm(dy, ws), dual)
    wab = _wab(wa, wb)
    if not ln:
        return torch.mm(dab, wab)
    dx, _, _ = ext.ln_bwd(
        _mm_f32(dab, wab), x, stats, _f32(gamma, False), dy, gamma.dtype
    )
    return dx


def _master_cast_fake(weights):
    """One fresh contiguous bf16 tensor per weight, same shapes."""
    return [
        torch.empty_like(w, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
        for w in weights
    ]


@opaque(fake=_master_cast_fake, name="transition_master_cast_sm80")
def _master_cast(weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """Batched cast of fp32 master weights to bf16 with the existing CUDA caster."""
    from miniworld_engine.kernels.transition.cuda import (
        parameter_cast_sm80 as parameter_cast,
    )

    return list(parameter_cast.result(weights))


def _master_cast_wab_fake(wa, wb, ws):
    """([wa; wb] [2H, D], ws), fresh contiguous bf16 tensors."""
    return (
        wa.new_empty((2 * wa.shape[0], wa.shape[1]), dtype=torch.bfloat16),
        torch.empty_like(
            ws, dtype=torch.bfloat16, memory_format=torch.contiguous_format
        ),
    )


@opaque(fake=_master_cast_wab_fake, name="transition_master_cast_wab_sm80")
def _master_cast_wab(
    wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One launch: the fp32 masters as bf16, wa and wb stacked in one buffer (the backward's d_xn operand needs no cat copy)."""
    from miniworld_engine.kernels.transition.cuda import (
        parameter_cast_sm80 as parameter_cast,
    )

    return parameter_cast.result_wab(wa.contiguous(), wb.contiguous(), ws.contiguous())


def _wab(wa: torch.Tensor, wb: torch.Tensor) -> torch.Tensor:
    """[wa; wb] [2H, D]: the stacked buffer the masters were cast into (wa, wb are its halves), else a cat copy."""
    if (
        wa.is_contiguous()
        and wb.is_contiguous()
        and wa.untyped_storage().data_ptr() == wb.untyped_storage().data_ptr()
        and wb.storage_offset() == wa.storage_offset() + wa.numel()
    ):
        return torch.as_strided(
            wa,
            (wa.shape[0] + wb.shape[0], wa.shape[1]),
            (wa.shape[1], 1),
            wa.storage_offset(),
        )
    return torch.cat((wa, wb), 0)


def _training_weights(wa, wb, ws, dtype):
    # fp32 masters (AMP): one launch casts the three projection weights, wa / wb into a stacked buffer, and the autograd Function keeps the casts out of autograd
    # (unrounded fp32 master gradients).
    if dtype is torch.bfloat16 and all(w.dtype is torch.float32 for w in (wa, wb, ws)):
        wab, ws16 = _master_cast_wab(wa, wb, ws)
        return wab[: wa.shape[0]], wab[wa.shape[0] :], ws16
    return tuple(w.to(dtype).contiguous() for w in (wa, wb, ws))


class _WideTransitionSM80(torch.autograd.Function):
    """``y = transition(x) + x`` (``ln``) or the bare FFN ``squeeze(silu(a) b)``; the forward saves xn and (mean, rstd) (the FFN saves x only)."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps, ln):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        if not ln:
            gamma = beta = flat.new_empty((0,), dtype=torch.float32)
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        wa, wb, ws = _training_weights(wa, wb, ws, x.dtype)
        out, xn, stats, hid, ab = _fwd_launch(
            flat, gamma, beta, wa, wb, ws, float(eps), True, ln
        )
        ctx.save_for_backward(flat, xn, stats, gamma, wa, wb, ws, hid, ab)
        ctx.shape, ctx.ln = shape, ln
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, stats, gamma, wa, wb, ws, hid, ab = ctx.saved_tensors
        if not any(ctx.needs_input_grad[1:6]):
            dx = _bwd_dx(
                dy.reshape(-1, dy.shape[-1]).contiguous(),
                flat,
                xn,
                stats,
                gamma,
                wa,
                wb,
                ws,
                ab,
                ctx.ln,
            )
            return dx.reshape(ctx.shape), None, None, None, None, None, None, None
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        wdt = (
            adt
            if adt in (torch.float32, torch.bfloat16) and adt == bwdt == sdt
            else torch.float32
        )
        dx, dgam, dbeta, dwab, dws = _bwd_launch(
            dy.reshape(-1, dy.shape[-1]).contiguous(),
            flat,
            xn,
            stats,
            gamma,
            wa,
            wb,
            ws,
            hid,
            ab,
            ctx.ln,
            gdt,
            wdt,
        )
        h = wa.shape[0]
        dbeta = dbeta if bdt == gdt else dbeta.to(bdt)
        return (
            dx.reshape(ctx.shape),
            dgam if ctx.ln else None,
            dbeta if ctx.ln else None,
            dwab[:h].to(adt),
            dwab[h:].to(bwdt),
            dws.to(sdt),
            None,
            None,
        )


def _needs_grad(*tensors) -> bool:
    return torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in tensors
    )


def transition_wide_sm80(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as the Triton ``transition_residual``.  Call ``available(x, wa, ws)`` first.  No grad needed: the forward
    runs without the xn / statistics saves."""
    if not _needs_grad(x, gamma, beta, wa, wb, ws):
        shape = x.shape
        out, _, _, _, _ = _fwd_launch(
            x.reshape(-1, shape[-1]).contiguous(),
            gamma,
            beta,
            wa,
            wb,
            ws,
            float(eps),
            False,
            True,
        )
        return out.reshape(shape)
    return _WideTransitionSM80.apply(x, gamma, beta, wa, wb, ws, eps, True)


def swiglu_ffn_sm80(x, wa, wb, ws):
    """The bare SwiGLU FFN ``squeeze(silu(x Wa^T) * (x Wb^T))`` (no LayerNorm, no residual).  Call ``available(x, wa, ws)`` first."""
    if not _needs_grad(x, wa, wb, ws):
        shape = x.shape
        empty = x.new_empty((0,), dtype=torch.float32)
        out, _, _, _, _ = _fwd_launch(
            x.reshape(-1, shape[-1]).contiguous(),
            empty,
            empty,
            wa,
            wb,
            ws,
            0.0,
            False,
            False,
        )
        return out.reshape(shape)
    return _WideTransitionSM80.apply(x, None, None, wa, wb, ws, 0.0, False)

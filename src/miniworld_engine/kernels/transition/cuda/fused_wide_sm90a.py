"""Hand-CUDA sm_90a Transition for the channel widths other than 128: D = 64, 256, 384, 512 (n = 4, bf16).

The companion of ``fused_sm90a`` (D = 128), developed in ``archive/experiments-20260928:experiments/transition_fused`` (record:
``records/widths.md``). What runs depends on the width, because what fits the register file does:

    D     forward                              backward
    64    one fused kernel (2 CTAs / SM)       one fused two-role kernel + partial reduction
    256   one fused kernel                     gate kernel -> fp32 dW GEMMs -> d_xn + LayerNorm-bwd kernel
    384   ln_swiglu_gemm + squeeze_gemm        gate kernel -> fp32 dW GEMMs -> d_xn GEMM -> LayerNorm bwd
    512   ln_swiglu_gemm + squeeze_gemm        same as 384

The gate kernel computes dh = dy Ws, [a | b] = xn [Wa; Wb]^T and the SwiGLU backward in one pass and writes h and [dA | dB];
the weight gradients are cuBLAS GEMMs with an fp32 output (more accurate than the bf16-output GEMMs of the Triton path, at
no measured cost). At D >= 256 a single fused backward does not fit sm_90 (the dW accumulators alone exceed the register
file), so the backward is bound by the HBM traffic of h and [dA | dB].

Measured on an H100 SXM, modules.Transition backward = (fwd + bwd) - fwd, both captured in CUDA graphs, L = 384 / 768:
D64 2.28x / 2.31x, D256 1.28x / 1.23x, D384 1.26x / 1.23x, D512 1.22x / 1.20x against the engine's Triton residual path.

The wide widths use a tanh.approx sigmoid (forward and backward alike); D = 64 keeps the rcp/ex2 one of the D = 128 kernels.
``supported()`` is the whole gate; everything it rejects keeps the existing path.
"""

import functools
import os
import warnings
from pathlib import Path

import torch

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
from .fused_sm90a import _cuda_prepare, _native_ext, native_aux_enabled

_dir = Path(__file__).parent / "wide"

#: Channel widths with a build. Each is its own extension, so a model using one width never compiles the others.
WIDTHS = (64, 256, 384, 512)
#: Row tile of every persistent grid here. ``M`` must be a whole number of these.
ROWS = 128

_SOURCES = {
    64: ("d64_fwd.cu", "d64_bwd.cu"),
    256: ("d256_fwd.cu", "gate.cu", "dxln.cu"),
    384: ("lnsg.cu", "squeeze.cu", "gate.cu", "gate_noh.cu"),
    512: ("lnsg.cu", "squeeze.cu", "gate.cu", "gate_noh.cu"),
}


def save_h_enabled(d: int) -> bool:
    """Keep the forward's SwiGLU intermediate h [M, 4D] for the D >= 384
    backward instead of recomputing-and-storing it in the gate kernel. The D >= 384 forward writes h to HBM anyway (between
    its two kernels), so this costs no forward time -- only activation memory, M x 4D bf16 per layer (453 MB at D = 384,
    604 MB at D = 512 for a 384 x 384 pair). Native D384 enables this by default;
    ``MINIWORLD_TRANSITION_WIDE_SAVE_H`` explicitly selects either variant."""
    default = "1" if d == 384 and native_aux_enabled(d) else "0"
    return d >= 384 and os.environ.get("MINIWORLD_TRANSITION_WIDE_SAVE_H", default) == "1"


def save_ab_enabled(d: int) -> bool:
    """B200 tensor lifetimes: retain h/a/b, then run only dh GEMM in the gate backward."""
    return (d >= 384 and native_aux_enabled(d)
            and os.environ.get("MINIWORLD_TRANSITION_H100_SAVE_AB", "0") == "1")


def split_fwd_enabled(d: int) -> bool:
    return (d == 512 and save_ab_enabled(d)
            and os.environ.get("MINIWORLD_TRANSITION_H100_SPLIT_FWD", "0") == "1")


def _sm_count(device: torch.device | int | None = None) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def fused_d384_enabled(d: int) -> bool:
    return (d == 384 and native_aux_enabled(d)
            and os.environ.get("MINIWORLD_TRANSITION_H100_FUSED_D384", "0") == "1")


@functools.lru_cache(maxsize=8)
def _ext(width: int, ctas: int, d64_repl: int = 16, save_ab: bool = False, split_fwd: bool = False,
         fused_d384: bool = False, multicast: bool = False, affine_parallel: bool = False,
         gate_multicast: bool = False, gate_k64: int = 0, gate_full_stage: bool = False,
         dxln_full_row: bool = False, gate_saved_tma: bool = False):
    """Build the extension for one channel width and SM count (both in the module name, so a second device with a
    different multiprocessor count gets its own build rather than a silently wrong grid)."""
    ensure_cuda_home()
    # This change lives in an included header. Give the corrected D64 kernel
    # its own cache identity instead of accepting a prebuilt pre-barrier binary.
    revision = "_input_barrier_v1" if width == 64 else ""
    if width == 64 and d64_repl != 16:
        revision += f"_repl{d64_repl}"
    if save_ab:
        revision += "_saved_ab_v1"
    if split_fwd:
        revision += "_split_fwd_v1"
    if fused_d384:
        revision += "_fused_d384_v3_converged"
    if multicast:
        revision += "_multicast_v1"
    if affine_parallel:
        revision += "_affine_parallel_v1"
    if gate_multicast:
        revision += "_gate_multicast_v1"
    if gate_k64:
        revision += f"_gate_k64_v1_{gate_k64}"
    if gate_full_stage:
        revision += "_gate_full_stage_v1"
    if dxln_full_row:
        revision += "_dxln_full_row_xsm_v1"
    if gate_saved_tma:
        revision += "_saved_ab_tma_ring_v1"
    return load_extension(
        name=f"transition_wide_sm90a_d{width}_c{ctas}{revision}",
        sources=[str(_dir / "bind.cu"), *(str(_dir / s) for s in _SOURCES[width])],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"),
                           f"-I{_dir.parent / 'anthropic_v5'}", f"-I{_dir}", f"-I{_dir / 'kernels'}",
                           f"-DWIDE_D={width}", f"-DWIDE_SMS={ctas}",
                           f"-DMW_D64_DW_REPL={d64_repl}",
                           *(["-DMW_D64_AFFINE_PARALLEL=1"] if affine_parallel else []),
                           *(["-DMW_GATE_MULTICAST=1"] if gate_multicast else []),
                           *([f"-DMW_GATE_K64={gate_k64}"] if gate_k64 else []),
                           *(["-DMW_GATE_FULL_STAGE=1"] if gate_full_stage else []),
                           *(["-DMW_DXLN_FULL_ROW=1"] if dxln_full_row else []),
                           *(["-DMW_GATE_SAVED_TMA=1"] if gate_saved_tma else []),
                           *([ "-DMW_SAVE_AB=1" ] if save_ab else []),
                           *([ "-DMW_SPLIT_FWD=1" ] if split_fwd else []),
                           *([ "-DMW_FUSED_D384=1" ] if fused_d384 else []),
                           *([ "-DMW_FUSED_D384_MC=1" ] if multicast else []),
                           "--expt-relaxed-constexpr",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                           "-U__CUDA_NO_BFLOAT162_OPERATORS__"],
        extra_cflags=["-std=c++17"], verbose=False,
    )


def _ext_for(x: torch.Tensor, save_ab: bool = False):
    repl = int(os.environ.get("MINIWORLD_TRANSITION_D64_REPL", "16")) if native_aux_enabled(x.shape[-1]) else 16
    gate_k64 = (int(os.environ.get("MINIWORLD_TRANSITION_H100_GATE_K64", "0"))
                if x.shape[-1] in (256, 512) and not save_ab else 0)
    if gate_k64 not in (0, 1, 2):
        raise ValueError("MINIWORLD_TRANSITION_H100_GATE_K64 must be 0, 1 or 2")
    gate_full_stage = (x.shape[-1] in (256, 512) and not save_ab
                       and os.environ.get("MINIWORLD_TRANSITION_H100_GATE_FULL_STAGE", "0") == "1")
    if gate_full_stage:
        gate_k64 = 0
    return _ext(x.shape[-1], _sm_count(x.device), repl, save_ab, save_ab and split_fwd_enabled(x.shape[-1]),
                fused_d384_enabled(x.shape[-1]),
                fused_d384_enabled(x.shape[-1])
                and os.environ.get("MINIWORLD_TRANSITION_H100_FUSED_D384_MC", "0") == "1",
                x.shape[-1] == 64 and native_aux_enabled(64)
                and os.environ.get("MINIWORLD_TRANSITION_H100_AFFINE_PARALLEL", "1") == "1",
                x.shape[-1] in (256, 512) and not save_ab and not gate_k64 and not gate_full_stage
                and os.environ.get("MINIWORLD_TRANSITION_H100_GATE_MULTICAST", "0") == "1",
                gate_k64, gate_full_stage,
                x.shape[-1] == 256 and native_aux_enabled(256)
                and os.environ.get("MINIWORLD_TRANSITION_H100_DXLN_FULL_ROW", "0") == "1",
                x.shape[-1] == 512 and save_ab
                and os.environ.get("MINIWORLD_TRANSITION_H100_SAVED_AB_TMA", "0") == "1")


@functools.lru_cache(maxsize=8)
def _is_hopper(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (9, 0)


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """Whether these kernels can run this call: sm_90, bf16, D in ``WIDTHS`` with hidden 4 D, whole 128-row tiles.
    The requirements are the kernels' own (literal tile shapes, persistent grids without a ragged tail), not a policy."""
    if os.environ.get("MINIWORLD_TRANSITION_WIDE_SM90A", "1") == "0":
        return False
    if not x.is_cuda or x.dtype is not torch.bfloat16:
        return False
    if not _is_hopper(x.device.index if x.device.index is not None else torch.cuda.current_device()):
        return False
    d = x.shape[-1]
    if d not in WIDTHS or wa.shape != (4 * d, d) or ws.shape != (d, 4 * d):
        return False
    return (x.numel() // d) % ROWS == 0


_BUILD_FAILED: set[int] = set()


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful build of this width, both cached. A build failure is an environment problem, not
    a dispatch bug: it warns once per width and keeps the existing path."""
    if not supported(x, wa, ws) or x.shape[-1] in _BUILD_FAILED:
        return False
    # FakeTensor dispatch records shapes only; it must not invoke nvcc or
    # wait for another process's extension lock. Launch wrappers have fakes.
    # Dynamo's TensorVariable is not itself a FakeTensor. Defer its build to
    # the opaque runtime launch instead of tracing filesystem/compiler calls.
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _ext_for(x)
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED.add(x.shape[-1])
        warnings.warn(f"wide sm90a Transition (D={x.shape[-1]}) unavailable, keeping the existing path: {exc!r}",
                      RuntimeWarning, stacklevel=2)
        return False
    return True


def _is_fake(*tensors) -> bool:
    """True under FakeTensorMode, where a native extension must not be entered (see ``fused_sm90a._is_fake``)."""
    from torch._subclasses.fake_tensor import FakeTensor
    return any(isinstance(t, FakeTensor) for t in tensors)


def _pack(wa: torch.Tensor, wb: torch.Tensor, block: int) -> torch.Tensor:
    """[Wa; Wb] interleaved in ``block``-row blocks [Wa_0 | Wb_0 | Wa_1 | ...] -- one GEMM tile then holds a AND b of the
    same hidden units."""
    if native_aux_enabled(wa.shape[1]):
        return _native_ext().pair_layout(wa, wb, block, False)
    h, d = wa.shape
    return torch.stack([wa.reshape(h // block, block, d), wb.reshape(h // block, block, d)], 1).reshape(2 * h, d).contiguous()


def _mm_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM (cuBLAS with an fp32 output); torch builds without ``out_dtype`` round through bf16."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a, b).float()


def _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save, wst=None):
    """Output structure only: (out, xn, rstd, c1, h); the saves are 1-element placeholders when ``save`` is false, and h is
    [M, 4D] at D >= 384 (the two-kernel forward materializes it) and a placeholder otherwise."""
    rows = x.shape[0] if save else 1
    d = x.shape[1]
    keep_h = d >= 384 and (save or not fused_d384_enabled(d))
    h = x.new_empty((x.shape[0], 4 * d)) if keep_h else x.new_empty((1, 1))
    return (torch.empty_like(x), torch.empty_like(x) if save else x.new_empty((1, d)),
            torch.empty((rows,), dtype=torch.float32, device=x.device), torch.empty((rows,), dtype=torch.float32, device=x.device), h)


@opaque(fake=_fwd_launch_fake, name="transition_wide_fwd_sm90a")
def _fwd_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
                ws: torch.Tensor, eps: float, save: bool, wst: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, D] bf16, gamma / beta fp32. Returns (out, xn, rstd, c1, h), out = transition(x) + x; h see the fake."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save, wst)
    ext = _ext_for(x)
    if fused_d384_enabled(x.shape[-1]) and not save:
        wst = _cuda_prepare(wa, wb, ws)[3]
        return tuple(ext.fwd_fused_d384(x, gamma, beta, wa, wb, wst, float(eps), False)[:5])
    if x.shape[-1] in (64, 256):
        if wst is None:
            wst = _cuda_prepare(wa, wb, ws)[3] if native_aux_enabled(x.shape[-1]) else ws.t().contiguous()
        return (*ext.fwd(x, gamma, beta, wa, wb, wst, float(eps), bool(save)), x.new_empty((1, 1)))
    return tuple(ext.fwd(x, gamma, beta, _pack(wa, wb, 64), ws, float(eps), bool(save)))


def _saved_fwd_fake(x, gamma, beta, wa, wb, ws, eps):
    return (*_fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, True),
            x.new_empty((x.shape[0], wa.shape[0])), x.new_empty((x.shape[0], wa.shape[0])))


@opaque(fake=_saved_fwd_fake, name="transition_wide_saved_fwd_sm90a")
def _saved_fwd_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                     wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, eps: float,
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor, torch.Tensor]:
    if _is_fake(x, wa):
        return _saved_fwd_fake(x, gamma, beta, wa, wb, ws, eps)
    if fused_d384_enabled(x.shape[-1]):
        wst = _cuda_prepare(wa, wb, ws)[3]
        return tuple(_ext_for(x, True).fwd_fused_d384(x, gamma, beta, wa, wb, wst, float(eps), True))
    if split_fwd_enabled(x.shape[-1]):
        xn, rstd, c1 = _native_ext().ln_fwd(x, gamma, beta, float(eps))
        out, h, a, b = _ext_for(x, True).fwd_split(xn, x, _pack(wa, wb, 128), ws)
        return out, xn, rstd, c1, h, a, b
    return tuple(_ext_for(x, True).fwd(x, gamma, beta, _pack(wa, wb, 64), ws, float(eps), True))


def _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hsaved, fp32_dw=False, asaved=None, bsaved=None):
    """Output structure only: (dx, dgamma, dbeta, dWa, dWb, dWs); the weight gradients are fp32 at D >= 256."""
    wdt = (torch.float32 if fp32_dw else wa.dtype) if native_aux_enabled(x.shape[-1]) else (
        torch.float32 if x.shape[-1] >= 256 or fp32_dw else wa.dtype)
    return (torch.empty_like(x), torch.empty_like(gamma), torch.empty_like(gamma),
            torch.empty(wa.shape, dtype=wdt, device=wa.device), torch.empty(wb.shape, dtype=wdt, device=wb.device),
            torch.empty(ws.shape, dtype=wdt, device=ws.device))


@opaque(fake=_bwd_launch_fake, name="transition_wide_bwd_sm90a")
def _bwd_launch(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, rstd: torch.Tensor, c1: torch.Tensor,
                gamma: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, hsaved: torch.Tensor, fp32_dw: bool = False,
                asaved: torch.Tensor | None = None, bsaved: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dx, dgamma, dbeta, dWa, dWb, dWs); ``dx`` already carries the residual branch. ``hsaved`` is the forward's
    h [M, 4D] when it was kept (save_h) and a placeholder otherwise."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hsaved, fp32_dw)
    ext = _ext_for(x, asaved is not None)
    d = x.shape[-1]
    if d == 64:
        return tuple(ext.bwd_master(dy, x, xn, rstd, c1, gamma, wa, wb, ws, fp32_dw))
    h = 4 * d
    have_h = hsaved.shape[0] == x.shape[0]
    wst = _cuda_prepare(wa, wb, ws)[3] if native_aux_enabled(d) else ws.t().contiguous()
    if asaved is not None:
        assert have_h and bsaved is not None
        hid, dab = hsaved, ext.gate_saved(dy, wst, asaved, bsaved)
    else:
        hid, dab = ext.gate(xn, dy, _pack(wa, wb, 128), wst, not have_h)
    if have_h:
        hid = hsaved
    dws = _mm_f32(dy.t(), hid)                            # [D, H]
    dwab = _mm_f32(dab.t(), xn)                           # [2H, D]: dWa then dWb (dab holds dA | dB, unpacked)
    del hid
    split_dx = (d == 256 and native_aux_enabled(d)
                and os.environ.get("MINIWORLD_TRANSITION_H100_SPLIT_DX", "0") == "1")
    if native_aux_enabled(d):
        w_ab = _native_ext().pair_layout(wa, wb, 0, d == 256 and not split_dx)
    else:
        w_ab = torch.cat((wa, wb), 0)
    if d == 256 and not split_dx:
        if native_aux_enabled(d):
            dx, pg, pb = ext.dxln_partials(dab, w_ab, x, dy, gamma, rstd, c1)
            dgam, dbeta = _native_ext().reduce_affine(pg, pb)
        else:
            dx, dgam, dbeta = ext.dxln(dab, w_ab.t().contiguous(), x, dy, gamma, rstd, c1)
    elif native_aux_enabled(d):
        aux = _native_ext()
        compact = (d == 512
                   and os.environ.get("MINIWORLD_TRANSITION_H100_COMPACT_LN", "0") == "1")
        compact384 = (d == 384 and x.shape[0] == 768 * 768
                      and os.environ.get("MINIWORLD_TRANSITION_H100_COMPACT_LN_D384", "0") == "1")
        ln = aux.ln_bwd_compact if compact or compact384 else aux.ln_bwd
        # The compact kernel uses 89 registers without spills, allowing five
        # resident 128-thread CTAs instead of four for the 112-register plan.
        dx, dgam, dbeta = ln(dab @ w_ab, x, gamma, rstd, c1, dy, 6 if compact384 else 5 if compact else 4)
    else:
        from ..triton.fused import _transition_ln_bwd

        dx, dgam, dbeta = _transition_ln_bwd(dab @ w_ab, x, rstd, c1, gamma)
        dx = dx.add_(dy)
    if native_aux_enabled(d):
        # The opaque op forbids even disjoint output views of one storage. Copy/cast
        # all three leaves together, producing the final parameter-gradient dtype.
        dwa, dwb, dws = _native_ext().cast([dwab[:h], dwab[h:], dws], [fp32_dw] * 3, True)
        return dx, dgam, dbeta, dwa, dwb, dws
    return dx, dgam, dbeta, dwab[:h], dwab[h:].clone(), dws



def _cast_weights(weights, dtype):
    if native_aux_enabled(weights[0].shape[1]):
        return _cuda_prepare(*weights, False)[:3]
    if all(w.dtype == dtype for w in weights):
        return tuple(w.contiguous() for w in weights)
    outs = [torch.empty(w.shape, device=w.device, dtype=dtype) for w in weights]
    torch._foreach_copy_(outs, list(weights))
    return tuple(outs)

class _WideTransitionSM90A(torch.autograd.Function):
    """``y = transition(x) + x`` at D in ``WIDTHS``; the forward saves xn and the LayerNorm statistics."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        raw = (wa, wb, ws)
        param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        if native_aux_enabled(shape[-1]):
            wa, wb, ws, wst = _cuda_prepare(*raw, shape[-1] in (64, 256))
        elif shape[-1] == 64 and all(w.dtype == torch.float32 and w.is_contiguous() for w in raw):
            from .fused_sm90a import _master_pack
            wa, wb, ws, wst = _master_pack(*raw)
        else:
            wa, wb, ws = _cast_weights(raw, x.dtype)
            wst = None
        flat = x.reshape(-1, shape[-1]).contiguous()
        gf, bf = gamma.float().contiguous(), beta.float().contiguous()
        wa, wb, ws = wa.contiguous(), wb.contiguous(), ws.contiguous()
        keep_ab = save_ab_enabled(flat.shape[-1])
        if keep_ab:
            out, xn, rstd, c1, h, a, b = _saved_fwd_launch(flat, gf, bf, wa, wb, ws, float(eps))
        else:
            out, xn, rstd, c1, h = _fwd_launch(flat, gf, bf, wa, wb, ws, float(eps), True, wst)
            a, b = None, None
        if not (save_h_enabled(flat.shape[-1]) or keep_ab):
            h = h.new_empty((1, 1))                       # not kept: the gate kernel recomputes and stores it
        ctx.save_for_backward(flat, xn, rstd, c1, gf, wa, wb, ws, h, a, b, *raw)
        ctx.shape = shape
        ctx.param_dtypes = param_dtypes
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, rstd, c1, gf, wa, wb, ws, h, a, b, *_raw = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        dx, dgam, dbeta, dwa, dwb, dws = _bwd_launch(
            dy.reshape(-1, dy.shape[-1]).contiguous(), flat, xn, rstd, c1, gf, wa, wb, ws, h, adt == torch.float32, a, b)
        if native_aux_enabled(flat.shape[-1]) and any(
                dw.dtype != dt for dw, dt in zip((dwa, dwb, dws), (adt, bwdt, sdt))):
            dwa, dwb, dws = _native_ext().cast([dwa, dwb, dws],
                                             [dt == torch.float32 for dt in (adt, bwdt, sdt)], False)
        return (dx.reshape(ctx.shape), dgam.to(gdt), dbeta.to(bdt), dwa.to(adt), dwb.to(bwdt), dws.to(sdt), None)


def transition_wide_sm90a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as ``fused_sm90a.transition_fused_sm90a``. Call ``supported()`` first: this
    raises rather than falling back. Inference (grad mode off, or nothing requiring grad) runs the forward without the
    saves, decided here and not inside the autograd Function, for the reason ``fused_sm90a`` documents."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))):
        shape = x.shape
        if (shape[-1] == 64 and native_aux_enabled(64)
                and os.environ.get("MINIWORLD_TRANSITION_H100_PREPARE_FUSED", "1") == "1"):
            # The forward consumes Ws transposed. Prepare its live master cast
            # and transpose together, as the training entry already does.
            wa, wb, ws, wst = _cuda_prepare(wa, wb, ws)
        else:
            wa, wb, ws = _cast_weights((wa, wb, ws), x.dtype)
            wst = None
        out, _, _, _, _ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), gamma.float().contiguous(),
                                   beta.float().contiguous(), wa.contiguous(), wb.contiguous(), ws.contiguous(),
                                   float(eps), False, wst)
        return out.reshape(shape)
    out = _WideTransitionSM90A.apply(x, gamma, beta, wa, wb, ws, eps)
    if native_aux_enabled(x.shape[-1]):
        from miniworld_engine.integrations.token_dit_train import _native_caller_output
        out = _native_caller_output(out, x, gamma, beta, wa, wb, ws, force=True)
    return out

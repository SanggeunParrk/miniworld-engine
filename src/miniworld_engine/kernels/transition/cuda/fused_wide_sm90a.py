"""Hand-CUDA sm_90a Transition for the channel widths other than 128: D = 64, 256, 384, 512 (n = 4, bf16).

The companion of ``fused_sm90a`` (D = 128), developed in ``experiments/transition_fused`` (record:
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
    """Opt-in (``MINIWORLD_TRANSITION_WIDE_SAVE_H=1``), D >= 384 only: keep the forward's SwiGLU intermediate h [M, 4D] for the
    backward instead of recomputing-and-storing it in the gate kernel. The D >= 384 forward writes h to HBM anyway (between
    its two kernels), so this costs no forward time -- only activation memory, M x 4D bf16 per layer (453 MB at D = 384,
    604 MB at D = 512 for a 384 x 384 pair). Measured: gate kernel -12 % / -12 %, the backward ~-5 %."""
    return d >= 384 and os.environ.get("MINIWORLD_TRANSITION_WIDE_SAVE_H", "0") == "1"


def _sm_count(device: torch.device | int | None = None) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.lru_cache(maxsize=8)
def _ext(width: int, ctas: int):
    """Build the extension for one channel width and SM count (both in the module name, so a second device with a
    different multiprocessor count gets its own build rather than a silently wrong grid)."""
    ensure_cuda_home()
    return load_extension(
        name=f"transition_wide_sm90a_d{width}_c{ctas}",
        sources=[str(_dir / "bind.cu"), *(str(_dir / s) for s in _SOURCES[width])],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"),
                           f"-I{_dir.parent / 'anthropic_v5'}", f"-I{_dir}", f"-I{_dir / 'kernels'}",
                           f"-DWIDE_D={width}", f"-DWIDE_SMS={ctas}",
                           "--expt-relaxed-constexpr",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                           "-U__CUDA_NO_BFLOAT162_OPERATORS__"],
        extra_cflags=["-std=c++17"], verbose=False,
    )


def _ext_for(x: torch.Tensor):
    return _ext(x.shape[-1], _sm_count(x.device))


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
    h, d = wa.shape
    return torch.stack([wa.reshape(h // block, block, d), wb.reshape(h // block, block, d)], 1).reshape(2 * h, d).contiguous()


def _mm_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM (cuBLAS with an fp32 output); torch builds without ``out_dtype`` round through bf16."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a, b).float()


def _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save):
    """Output structure only: (out, xn, rstd, c1, h); the saves are 1-element placeholders when ``save`` is false, and h is
    [M, 4D] at D >= 384 (the two-kernel forward materializes it) and a placeholder otherwise."""
    rows = x.shape[0] if save else 1
    f32 = dict(dtype=torch.float32, device=x.device)
    d = x.shape[1]
    h = x.new_empty((x.shape[0], 4 * d)) if d >= 384 else x.new_empty((1, 1))
    return (torch.empty_like(x), torch.empty_like(x) if save else x.new_empty((1, d)),
            torch.empty((rows,), **f32), torch.empty((rows,), **f32), h)


@opaque(fake=_fwd_launch_fake, name="transition_wide_fwd_sm90a")
def _fwd_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
                ws: torch.Tensor, eps: float, save: bool,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, D] bf16, gamma / beta fp32. Returns (out, xn, rstd, c1, h), out = transition(x) + x; h see the fake."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save)
    ext = _ext_for(x)
    if x.shape[-1] in (64, 256):
        return (*ext.fwd(x, gamma, beta, wa, wb, ws.t().contiguous(), float(eps), bool(save)), x.new_empty((1, 1)))
    return tuple(ext.fwd(x, gamma, beta, _pack(wa, wb, 64), ws, float(eps), bool(save)))


def _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hsaved):
    """Output structure only: (dx, dgamma, dbeta, dWa, dWb, dWs); the weight gradients are fp32 at D >= 256."""
    wdt = torch.float32 if x.shape[-1] >= 256 else wa.dtype
    return (torch.empty_like(x), torch.empty_like(gamma), torch.empty_like(gamma),
            torch.empty(wa.shape, dtype=wdt, device=wa.device), torch.empty(wb.shape, dtype=wdt, device=wb.device),
            torch.empty(ws.shape, dtype=wdt, device=ws.device))


@opaque(fake=_bwd_launch_fake, name="transition_wide_bwd_sm90a")
def _bwd_launch(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, rstd: torch.Tensor, c1: torch.Tensor,
                gamma: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, hsaved: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dx, dgamma, dbeta, dWa, dWb, dWs); ``dx`` already carries the residual branch. ``hsaved`` is the forward's
    h [M, 4D] when it was kept (save_h) and a placeholder otherwise."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hsaved)
    ext = _ext_for(x)
    d = x.shape[-1]
    if d == 64:
        return tuple(ext.bwd(dy, x, xn, rstd, c1, gamma, wa, wb, ws))
    h = 4 * d
    have_h = hsaved.shape[0] == x.shape[0]
    hid, dab = ext.gate(xn, dy, _pack(wa, wb, 128), ws.t().contiguous(), not have_h)
    if have_h:
        hid = hsaved
    dws = _mm_f32(dy.t(), hid)                            # [D, H]
    dwab = _mm_f32(dab.t(), xn)                           # [2H, D]: dWa then dWb (dab holds dA | dB, unpacked)
    del hid
    w_ab = torch.cat((wa, wb), 0)
    if d == 256:
        dx, dgam, dbeta = ext.dxln(dab, w_ab.t().contiguous(), x, dy, gamma, rstd, c1)
    else:
        from ..triton.fused import _transition_ln_bwd

        dx, dgam, dbeta = _transition_ln_bwd(dab @ w_ab, x, rstd, c1, gamma)
        dx = dx.add_(dy)
    return dx, dgam, dbeta, dwab[:h], dwab[h:].clone(), dws


class _WideTransitionSM90A(torch.autograd.Function):
    """``y = transition(x) + x`` at D in ``WIDTHS``; the forward saves xn and the LayerNorm statistics."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        gf, bf = gamma.float().contiguous(), beta.float().contiguous()
        wa, wb, ws = wa.contiguous(), wb.contiguous(), ws.contiguous()
        out, xn, rstd, c1, h = _fwd_launch(flat, gf, bf, wa, wb, ws, float(eps), True)
        if not save_h_enabled(flat.shape[-1]):
            h = h.new_empty((1, 1))                       # not kept: the gate kernel recomputes and stores it
        ctx.save_for_backward(flat, xn, rstd, c1, gf, wa, wb, ws, h)
        ctx.shape = shape
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, rstd, c1, gf, wa, wb, ws, h = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        dx, dgam, dbeta, dwa, dwb, dws = _bwd_launch(
            dy.reshape(-1, dy.shape[-1]).contiguous(), flat, xn, rstd, c1, gf, wa, wb, ws, h)
        return (dx.reshape(ctx.shape), dgam.to(gdt), dbeta.to(bdt), dwa.to(adt), dwb.to(bwdt), dws.to(sdt), None)


def transition_wide_sm90a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as ``fused_sm90a.transition_fused_sm90a``. Call ``supported()`` first: this
    raises rather than falling back. Inference (grad mode off, or nothing requiring grad) runs the forward without the
    saves, decided here and not inside the autograd Function, for the reason ``fused_sm90a`` documents."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))):
        shape = x.shape
        out, _, _, _, _ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), gamma.float().contiguous(),
                                   beta.float().contiguous(), wa.contiguous(), wb.contiguous(), ws.contiguous(),
                                   float(eps), False)
        return out.reshape(shape)
    return _WideTransitionSM90A.apply(x, gamma, beta, wa, wb, ws, eps)

"""Hand-CUDA sm_90a Transition for the channel widths other than 128: D = 64, 256, 384, 512 (n = 4 and n = 2, bf16).

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
from pathlib import Path

import torch

from miniworld_engine.kernels._device import num_sms
from miniworld_engine.kernels import _cast_cuda

from ..._compile import opaque
from ._weights import transposed_contiguous
from ..._nvcc import build_failed, ensure_cuda_home, gencodes, host_flags, load_extension

from .fused_sm90a import _cuda_prepare, _native_ext

_dir = Path(__file__).parent / "wide"

#: Channel widths with a build. Each is its own extension, so a model using one width never compiles the others.
WIDTHS = (64, 256, 384, 512, 768)
#: Hidden multiples (the n of ``Transition(d, n)``, hidden = n D) with a build, per width.
NS = {64: (4, 2), 256: (4, 2), 384: (4, 2), 512: (4, 2), 768: (2,)}
#: Row tile of every persistent grid here. ``M`` must be a whole number of these.
ROWS = 128

_SOURCES = {
    64: ("d64_fwd.cu", "d64_fwd_small_f2.cu", "d64_fwd_small_f3.cu", "d64_bwd.cu"),
    256: ("d256_fwd.cu", "gate.cu", "dxln.cu"),
    384: ("lnsg.cu", "squeeze.cu", "gate.cu", "gate_noh.cu"),
    512: ("lnsg.cu", "squeeze.cu", "gate.cu", "gate_noh.cu"),
    768: (
        "lnsg.cu",
        "squeeze.cu",
        "gate.cu",
        "gate_noh.cu",
    ),  # lnsg.cu holds only the split-forward expand at D768
}


#: D384 keeps the forward's h by default; tests flip this to exercise the recompute cell.
_SAVE_H_D384 = True



_D512_SAVED_SIDES = (768, 384)


def _d512_saved_rows(rows: int) -> bool:
    """Rows of the D512 training cells that retain a / b (see ``save_ab_enabled``): a square pair of these sides."""
    return any(rows == s * s for s in _D512_SAVED_SIDES)


def save_h_enabled(d: int) -> bool:
    """Keep the forward's SwiGLU intermediate h [M, 4D] for the D >= 384
    backward instead of recomputing-and-storing it in the gate kernel. The D >= 384 forward writes h to HBM anyway (between
    its two kernels), so this costs no forward time -- only activation memory, M x 4D bf16 per layer (453 MB at D = 384,
    604 MB at D = 512 for a 384 x 384 pair). D384 keeps it (``_SAVE_H_D384``); the other widths recompute it (the
    old switch that kept it for every D >= 384 is gone)."""
    if d < 384:
        return False
    return (d == 384 and _SAVE_H_D384)


def save_ab_enabled(d: int, rows: int = 0) -> bool:
    """B200 tensor lifetimes: retain h/a/b, then run only dh GEMM in the gate backward. D768 (n = 2) has no other path:
    its forward is LayerNorm kernel + split expand (the resident LayerNorm tile of ``lnsg`` does not fit at 768)."""
    if d == 768:
        return True
    return d == 512 and _d512_saved_rows(rows)


def split_fwd_enabled(d: int, rows: int = 0) -> bool:
    if d == 768:
        return True
    return d == 512 and save_ab_enabled(d, rows) and _d512_saved_rows(rows)




@functools.lru_cache(maxsize=8)
def _ext(
    width: int,
    ctas: int,
    d64_repl: int = 16,
    save_ab: bool = False,
    split_fwd: bool = False,
    affine_parallel: bool = False,
    gate_multicast: bool = False,
    gate_k64: int = 0,
    gate_full_stage: bool = False,
    dxln_full_row: bool = False,
    gate_saved_tma: bool = False,
    gate_saved_no_h_stage: bool = False,
    n: int = 4,
):
    """Build the extension for one channel width and SM count (both in the module name, so a second device with a
    different multiprocessor count gets its own build rather than a silently wrong grid)."""
    ensure_cuda_home()
    # This change lives in an included header. Give the corrected D64 kernel
    # its own cache identity instead of accepting a prebuilt pre-barrier binary.
    revision = "_input_barrier_v1_small3_frag2_pf" if width == 64 else ""
    if n != 4:
        revision += f"_n{n}"
    if width == 64 and d64_repl != 16:
        revision += f"_repl{d64_repl}"
    if save_ab:
        revision += "_saved_ab_v1"
    if split_fwd:
        revision += "_split_fwd_v1"
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
    if gate_saved_no_h_stage:
        revision += "_no_unused_h_stage_v1"
    if width == 512 and (gate_saved_tma or split_fwd):
        revision += "_k64_v1"   # 64-wide K slabs of the saved gate (2 stages) and of the split expand GEMM (3 stages)
    return load_extension(
        name=f"transition_wide_sm90a_d{width}_c{ctas}{revision}",
        sources=[str(_dir / "bind.cu"), *(str(_dir / s) for s in _SOURCES[width])],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            *gencodes("90a"),
            f"-I{_dir.parent / 'anthropic_v5'}",
            f"-I{_dir}",
            f"-I{_dir / 'kernels'}",
            f"-DWIDE_D={width}",
            f"-DWIDE_N={n}",
            f"-DWIDE_SMS={ctas}",
            f"-DMW_D64_DW_REPL={d64_repl}",
            *(["-DMW_D64_AFFINE_PARALLEL=1"] if affine_parallel else []),
            *(["-DMW_GATE_MULTICAST=1"] if gate_multicast else []),
            *([f"-DMW_GATE_K64={gate_k64}"] if gate_k64 else []),
            *(["-DMW_GATE_FULL_STAGE=1"] if gate_full_stage else []),
            *(["-DMW_DXLN_FULL_ROW=1"] if dxln_full_row else []),
            *(["-DMW_GATE_SAVED_TMA=1"] if gate_saved_tma else []),
            *(["-DMW_GATE_SAVED_NO_H_STAGE=1"] if gate_saved_no_h_stage else []),
            *(["-DMW_SAVE_AB=1"] if save_ab else []),
            *(["-DMW_SPLIT_FWD=1"] if split_fwd else []),
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def _ext_for(x: torch.Tensor, save_ab: bool = False, n: int = 4):
    args, kwargs = _ext_args(x.shape[-1], x.shape[0], save_ab, n, num_sms(x.device))
    return _ext(*args, **kwargs)


def _ext_args(d: int, rows: int, save_ab: bool, n: int, sms: int) -> tuple[tuple, dict]:
    """The ``_ext`` build that ``_ext_for`` picks for width ``d``, ``rows`` rows and ``sms`` SMs (no device asked)."""
    if d == 768:  # n = 2 only: saved a / b and the split forward are the whole path
        return (768, sms), {"save_ab": True, "split_fwd": True, "n": n}
    if n != 4:  # the n = 4 experiments below are tuned for hidden = 4 D; other n runs the plain build
        # D64 backward: weight CTAs = hidden slices (n) x replicas; measured best 26 replicas at n = 2 (16 at n = 4)
        repl = 26 if d == 64 else 16
        return (d, sms, repl), {"n": n}
    # The selected L768/D512 training path retains a/b; inference passes save_ab=False.
    saved_tma = d == 512 and _d512_saved_rows(rows) and save_ab
    return (d, sms, 16, save_ab, save_ab and split_fwd_enabled(d, rows)), {
        "affine_parallel": d == 64, "gate_saved_tma": saved_tma, "gate_saved_no_h_stage": saved_tma}


@functools.lru_cache(maxsize=8)
def _is_hopper(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (9, 0)


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """Whether these kernels can run this call: sm_90, bf16, D in ``WIDTHS`` with hidden 4 D, whole 128-row tiles.
    The requirements are the kernels' own (literal tile shapes, persistent grids without a ragged tail), not a policy."""
    if not x.is_cuda or x.dtype is not torch.bfloat16:
        return False
    if not _is_hopper(
        x.device.index if x.device.index is not None else torch.cuda.current_device()
    ):
        return False
    d = x.shape[-1]
    if d not in WIDTHS or wa.shape[1] != d or wa.shape[0] // d not in NS[d]:
        return False
    if wa.shape != (wa.shape[0], d) or ws.shape != (d, wa.shape[0]) or wa.shape[0] % d:
        return False
    return (x.numel() // d) % ROWS == 0


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful build of this width, both cached. A build / load failure
    raises ``_nvcc.KernelBuildError`` (never a fallback)."""
    key = (x.shape[-1], wa.shape[0] // x.shape[-1])
    if not supported(x, wa, ws):
        return False
    # FakeTensor dispatch records shapes only; it must not invoke nvcc or
    # wait for another process's extension lock. Launch wrappers have fakes.
    # Dynamo's TensorVariable is not itself a FakeTensor. Defer its build to
    # the opaque runtime launch instead of tracing filesystem/compiler calls.
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _ext_for(x, n=key[1])
    except Exception as exc:  # a build / load failure is an error, never a fallback
        build_failed(f"wide sm90a Transition (D={x.shape[-1]}, n={key[1]})", exc)
        return False  # reached only while planning (build_failed returns there)
    return True


def _is_fake(*tensors) -> bool:
    """True under FakeTensorMode, where a native extension must not be entered (see ``fused_sm90a._is_fake``)."""
    from torch._subclasses.fake_tensor import FakeTensor

    return any(isinstance(t, FakeTensor) for t in tensors)


def _pack(wa: torch.Tensor, wb: torch.Tensor, block: int) -> torch.Tensor:
    """[Wa; Wb] interleaved in ``block``-row blocks [Wa_0 | Wb_0 | Wa_1 | ...] -- one GEMM tile then holds a AND b of the
    same hidden units."""
    return _native_ext().pair_layout(wa, wb, block, False)


def _mm_dx(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def _mm_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM (cuBLAS with an fp32 output); torch builds without ``out_dtype`` round through bf16."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a, b).float()


def _d64_variant(rows: int, sms: int = 132) -> int:
    """D = 64 inference build for ``rows`` (a multiple of 64): 13 / 12 = the 64-row tile with LayerNorm in the wgmma A-fragment layout (xn
    stays in registers) at 3 / 2 CTAs per SM -- the fastest builds at every measured row count (n = 2 and 4); 0 = the 128-row tile.  3 / SM
    unless the 2 / SM build's waves are fuller by more than 2 % (a wave: 3 x 64 or 2 x 64 rows per SM)."""
    if rows % 64 != 0 or rows < 64:
        return 0
    if rows <= 6144:
        return 13
    cap3, cap2 = 3 * sms * 64, 2 * sms * 64

    def fill(cap: int) -> float:
        waves = -(-rows // cap)
        return rows / (cap * waves)

    return 12 if fill(cap2) > fill(cap3) + 0.02 else 13


def _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save, wst=None):
    """Output structure only: (out, xn, rstd, c1, h); the saves are 1-element placeholders when ``save`` is false, and h is
    [M, 4D] at D >= 384 (the two-kernel forward materializes it) and a placeholder otherwise."""
    rows = x.shape[0] if save else 1
    d = x.shape[1]
    h = x.new_empty((x.shape[0], wa.shape[0])) if d >= 384 else x.new_empty((1, 1))
    return (
        torch.empty_like(x),
        torch.empty_like(x) if save else x.new_empty((1, d)),
        torch.empty((rows,), dtype=torch.float32, device=x.device),
        torch.empty((rows,), dtype=torch.float32, device=x.device),
        h,
    )


@opaque(fake=_fwd_launch_fake, name="transition_wide_fwd_sm90a")
def _fwd_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    eps: float,
    save: bool,
    wst: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, D] bf16, gamma / beta fp32. Returns (out, xn, rstd, c1, h), out = transition(x) + x; h see the fake."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save, wst)
    n = wa.shape[0] // x.shape[-1]
    if x.shape[-1] == 768:
        out, xn, rstd, c1, h, _a, _b = _saved_fwd_launch(
            x, gamma, beta, wa, wb, ws, eps
        )
        if save:
            return out, xn, rstd, c1, h
        return out, x.new_empty((1, 768)), rstd.new_empty((1,)), c1.new_empty((1,)), h
    ext = _ext_for(x, n=n)
    variant = _d64_variant(x.shape[0], num_sms(x.device)) if x.shape[-1] == 64 and not save else 0
    if variant:
        # 64-row tiles: one consumer warpgroup per CTA (2 or 3 CTAs per SM), more tiles in flight than the 128-row build
        out = ext.fwd_small(x, gamma, beta, wa, wb, wst if wst is not None else transposed_contiguous(ws), float(eps), variant)
        f32 = dict(dtype=torch.float32, device=x.device)
        return (out, x.new_empty((1, 64)), torch.empty((1,), **f32), torch.empty((1,), **f32), x.new_empty((1, 1)))
    if x.shape[-1] in (64, 256):
        if wst is None:
            wst = _cuda_prepare(wa, wb, ws)[3]
        return (
            *ext.fwd(x, gamma, beta, wa, wb, wst, float(eps), bool(save)),
            x.new_empty((1, 1)),
        )
    return tuple(ext.fwd(x, gamma, beta, _pack(wa, wb, 64), ws, float(eps), bool(save)))


def _saved_fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps):
    """Output structure only: the forward's five outputs, then the saved activations a and b, each [M, wa.shape[0]]."""
    return (
        *_fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, True),
        x.new_empty((x.shape[0], wa.shape[0])),
        x.new_empty((x.shape[0], wa.shape[0])),
    )


@opaque(fake=_saved_fwd_launch_fake, name="transition_wide_saved_fwd_sm90a")
def _saved_fwd_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """The forward that also saves what the backward needs: (out, xn, rstd, c1, h, a, b)."""
    if _is_fake(x, wa):
        return _saved_fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps)
    if split_fwd_enabled(x.shape[-1], x.shape[0]):
        xn, rstd, c1 = _native_ext().ln_fwd(x, gamma, beta, float(eps))
        out, h, a, b = _ext_for(x, True, wa.shape[0] // x.shape[-1]).fwd_split(
            xn, x, _pack(wa, wb, 128), ws
        )
        return out, xn, rstd, c1, h, a, b
    return tuple(
        _ext_for(x, True).fwd(x, gamma, beta, _pack(wa, wb, 64), ws, float(eps), True)
    )


def _bwd_launch_fake(
    dy,
    x,
    xn,
    rstd,
    c1,
    gamma,
    wa,
    wb,
    ws,
    hsaved,
    fp32_dw=False,
    asaved=None,
    bsaved=None,
):
    """Output structure only: (dx, dgamma, dbeta, dWa, dWb, dWs); the weight gradients are fp32 at D >= 256."""
    wdt = torch.float32 if fp32_dw else wa.dtype
    return (
        torch.empty_like(x),
        torch.empty_like(gamma),
        torch.empty_like(gamma),
        torch.empty(wa.shape, dtype=wdt, device=wa.device),
        torch.empty(wb.shape, dtype=wdt, device=wb.device),
        torch.empty(ws.shape, dtype=wdt, device=ws.device),
    )


@opaque(fake=_bwd_launch_fake, name="transition_wide_bwd_sm90a")
def _bwd_launch(
    dy: torch.Tensor,
    x: torch.Tensor,
    xn: torch.Tensor,
    rstd: torch.Tensor,
    c1: torch.Tensor,
    gamma: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    hsaved: torch.Tensor,
    fp32_dw: bool = False,
    asaved: torch.Tensor | None = None,
    bsaved: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Returns (dx, dgamma, dbeta, dWa, dWb, dWs); ``dx`` already carries the residual branch. ``hsaved`` is the forward's
    h [M, 4D] when it was kept (save_h) and a placeholder otherwise."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hsaved, fp32_dw)
    ext = _ext_for(x, asaved is not None, n=wa.shape[0] // x.shape[-1])
    d = x.shape[-1]
    if d == 64:
        return tuple(ext.bwd_master(dy, x, xn, rstd, c1, gamma, wa, wb, ws, fp32_dw))
    h = wa.shape[0]
    have_h = hsaved.shape[0] == x.shape[0]
    wst = _cuda_prepare(wa, wb, ws)[3]
    if asaved is not None:
        assert have_h and bsaved is not None
        hid, dab = hsaved, ext.gate_saved(dy, wst, asaved, bsaved)
    else:
        hid, dab = ext.gate(xn, dy, _pack(wa, wb, 128), wst, not have_h)
    if have_h:
        hid = hsaved
    dws = _mm_f32(dy.t(), hid)  # [D, H]
    dwab = _mm_f32(dab.t(), xn)  # [2H, D]: dWa then dWb (dab holds dA | dB, unpacked)
    del hid
    w_ab = _native_ext().pair_layout(wa, wb, 0, d == 256)
    if d == 256:
        dx, pg, pb = ext.dxln_partials(dab, w_ab, x, dy, gamma, rstd, c1)
        dgam, dbeta = _native_ext().reduce_affine(pg, pb)
    else:
        aux = _native_ext()
        # The compact kernel uses 89 registers without spills, allowing five
        # resident 128-thread CTAs instead of four for the 112-register plan.
        compact = d == 512 and asaved is not None and _d512_saved_rows(x.shape[0])
        ln = aux.ln_bwd_compact if compact else aux.ln_bwd
        dx, dgam, dbeta = ln(
            _mm_dx(dab, w_ab),
            x,
            gamma,
            rstd,
            c1,
            dy,
            5 if compact else 3 if d == 768 else 4,
        )  # D768: 3 resident CTAs per SM (170 registers)
    # The opaque op forbids even disjoint output views of one storage. Copy/cast
    # all three leaves together, producing the final parameter-gradient dtype.
    dwa, dwb, dws = _cast_cuda.cast(
        [dwab[:h], dwab[h:], dws], [torch.float32 if fp32_dw else torch.bfloat16] * 3, force=True
    )
    return dx, dgam, dbeta, dwa, dwb, dws


def _cast_weights(weights, dtype):
    return _cuda_prepare(weights[0], weights[1], weights[2], False)[:3]


def _inference_forms(wa, wb, ws, transpose):
    """The inference forward's bf16 weights (and ``Ws^T``) from fp32 masters: cast once per parameter version and reused (``param_forms``,
    a capture reads the entry its warm-up made, as the fp32 path's TF32-rounded weights) instead of a cast launch per call; bf16 weights
    pass through ``_cuda_prepare`` as before."""
    from miniworld_engine.kernels import _capture

    if all(w.dtype == torch.bfloat16 for w in (wa, wb, ws)):
        return _cuda_prepare(wa, wb, ws, transpose)
    return _capture.param_forms(("transition_wide_bf16_forms", transpose), (wa, wb, ws), lambda: _cuda_prepare(wa, wb, ws, transpose),
                                capture_reads_eager=True)


class _WideTransitionSM90A(torch.autograd.Function):
    """``y = transition(x) + x`` at D in ``WIDTHS``; the forward saves xn and the LayerNorm statistics."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        raw = (wa, wb, ws)
        param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        wa, wb, ws, wst = _cuda_prepare(*raw, shape[-1] in (64, 256))
        flat = x.reshape(-1, shape[-1]).contiguous()
        gf, bf = gamma.float().contiguous(), beta.float().contiguous()
        wa, wb, ws = wa.contiguous(), wb.contiguous(), ws.contiguous()
        keep_ab = (
            wa.shape[0] == 4 * shape[-1] or shape[-1] == 768
        ) and save_ab_enabled(flat.shape[-1], flat.shape[0])
        if keep_ab:
            out, xn, rstd, c1, h, a, b = _saved_fwd_launch(
                flat, gf, bf, wa, wb, ws, float(eps)
            )
        else:
            out, xn, rstd, c1, h = _fwd_launch(
                flat, gf, bf, wa, wb, ws, float(eps), True, wst
            )
            a, b = None, None
        if not (save_h_enabled(flat.shape[-1]) or keep_ab):
            h = h.new_empty(
                (1, 1)
            )  # not kept: the gate kernel recomputes and stores it
        ctx.save_for_backward(flat, xn, rstd, c1, gf, wa, wb, ws, h, a, b, *raw)
        ctx.shape = shape
        ctx.param_dtypes = param_dtypes
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, rstd, c1, gf, wa, wb, ws, h, a, b, *_raw = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        dx, dgam, dbeta, dwa, dwb, dws = _bwd_launch(
            dy.reshape(-1, dy.shape[-1]).contiguous(),
            flat,
            xn,
            rstd,
            c1,
            gf,
            wa,
            wb,
            ws,
            h,
            adt == torch.float32,
            a,
            b,
        )
        if any(dw.dtype != dt for dw, dt in zip((dwa, dwb, dws), (adt, bwdt, sdt))):
            dwa, dwb, dws = _cast_cuda.cast(
                [dwa, dwb, dws], [torch.float32 if dt == torch.float32 else torch.bfloat16 for dt in (adt, bwdt, sdt)]
            )
        return (
            dx.reshape(ctx.shape),
            dgam.to(gdt),
            dbeta.to(bdt),
            dwa.to(adt),
            dwb.to(bwdt),
            dws.to(sdt),
            None,
        )


def transition_wide_sm90a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as ``fused_sm90a.transition_fused_sm90a``. Call ``supported()`` first: this
    raises rather than falling back. Inference (grad mode off, or nothing requiring grad) runs the forward without the
    saves, decided here and not inside the autograd Function, for the reason ``fused_sm90a`` documents."""
    if not (
        torch.is_grad_enabled()
        and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))
    ):
        shape = x.shape
        if shape[-1] == 64:
            # The forward consumes Ws transposed. bf16 weights reuse a cached transpose (an inference loop calls with the same
            # weights); fp32 masters are cast and transposed together, as the training entry does.
            if all(w.dtype == torch.bfloat16 for w in (wa, wb, ws)):
                wst = transposed_contiguous(ws)
            else:
                wa, wb, ws, wst = _inference_forms(wa, wb, ws, True)
        else:
            wa, wb, ws = _inference_forms(wa, wb, ws, False)[:3]
            wst = None
        out, _, _, _, _ = _fwd_launch(
            x.reshape(-1, shape[-1]).contiguous(),
            gamma.float().contiguous(),
            beta.float().contiguous(),
            wa.contiguous(),
            wb.contiguous(),
            ws.contiguous(),
            float(eps),
            False,
            wst,
        )
        return out.reshape(shape)
    out = _WideTransitionSM90A.apply(x, gamma, beta, wa, wb, ws, eps)
    from miniworld_engine.kernels.integrations.token_dit_train import _native_caller_output

    return _native_caller_output(out, x, gamma, beta, wa, wb, ws, force=True)


# ---------------------------------------------------------------------------------------------------------- fp32 (f32io family)
#: fp32 activations, the bf16 builds' structure (``wide/f32io.cu`` binds every kernel; one extension per (D, n)):
#:   D 64 / 128  one fused forward (``kernels/transition_fwd_f32io.cu``: x tile by TMA, LayerNorm in the A fragments)
#:   D 256       one fused forward (``kernels/transition_fwd_f32xs.cu``: xn in shared memory, x / y in the fragment layout)
#:   D 512       LayerNorm inside the expand GEMM (``kernels/transition_lnsg_f32.cu``, h in fp16) + the generic ``squeeze_res_h16``
#: The products run on fp16 tensor cores with fp32 accumulation: fp16's 10-bit mantissa is TF32's, so each operand is rounded as the TF32
#: path rounds it; device scales (``forms``) keep every fp16 operand inside fp16's range. Training (D 128 / 256 / 512, n 2: the requirement
#: rows) runs the saving build of the same forward (xn, rstd, c1) and a native-TF32 recompute backward: at D 128 (from
#: ``_F32_FUSED_BWD_MIN_ROWS``) the bf16 D 128 backward's two roles as two kernels (``kernels/transition_bwd_f32dx.cu``: dh, a, b, SwiGLU
#: backward, d_xn, LayerNorm backward and residual in one pass; ``kernels/transition_bwd_f32dw.cu``: dWa / dWb / dWs per hidden slice on chip),
#: else the generic build's (``gate``: dh, a, b and the SwiGLU backward in one pass -> cuBLAS TF32 dW -> ``dxn_gemm`` -> ``ln_bwd``), the
#: bf16 D 384 / 512 backward structure. A
#: TF32 d_xn GEMM with the LayerNorm backward in its epilogue (the bf16 D 256 ``dxln``) was built and measured slower than the two launches
#: (H100, 768^2 rows: D 128 1.12 against 0.99 ms, D 256 2.67 against 1.91 ms: the row epilogue does not overlap the GEMM); dropped.
F32_SHAPES = {64: (2, 4), 128: (2, 4), 256: (2,), 512: (2,)}
#: (D, n) with the training forward (the float32 train requirement rows)
F32_TRAIN_SHAPES = {128: (2,), 256: (2,), 512: (2,)}
_F32_REVISION = "v3"


@functools.lru_cache(maxsize=8)
def _f32_ext(width: int, n: int):
    ensure_cuda_home()
    return load_extension(
        name=f"transition_f32io_sm90a_d{width}_n{n}_{_F32_REVISION}",
        sources=[str(_dir / "f32io.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            *gencodes("90a"),
            f"-I{_dir.parent / 'anthropic_v5'}",
            f"-I{_dir}",
            f"-DF32_D={width}",
            f"-DF32_N={n}",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def f32_supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor, training: bool) -> bool:
    """fp32 activations on sm_90 with TF32 allowed (the precision it reproduces), D / n in ``F32_SHAPES`` (``F32_TRAIN_SHAPES`` when training),
    weights fp32 or bf16, any row count (the forwards zero-fill / guard the tail tile)."""
    from .fused_generic_sm90a import _tf32_allowed

    if not x.is_cuda or x.dtype is not torch.float32 or x.numel() == 0 or not _tf32_allowed():
        return False
    if not _is_hopper(x.device.index if x.device.index is not None else torch.cuda.current_device()):
        return False
    d = x.shape[-1]
    shapes = F32_TRAIN_SHAPES if training else F32_SHAPES
    if d not in shapes or wa.dim() != 2 or wa.shape[1] != d or wa.shape[0] % d or wa.shape[0] // d not in shapes[d]:
        return False
    if tuple(ws.shape) != (d, wa.shape[0]):
        return False
    return all(w.dtype in (torch.float32, torch.bfloat16) for w in (wa, ws))


def f32_available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor, training: bool) -> bool:
    """``f32_supported()`` plus a successful build (cached); a failed build / load raises ``_nvcc.KernelBuildError`` (never a fallback)."""
    if not f32_supported(x, wa, ws, training):
        return False
    key = (x.shape[-1], wa.shape[0] // x.shape[-1])
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _f32_ext(*key)
        if training or key[0] == 512:
            from . import fused_generic_sm90a

            fused_generic_sm90a._ext()   # the TF32 backward / the D 512 squeeze
    except Exception as exc:  # a build / load failure is an error, never a fallback
        build_failed(f"fp32 sm90a Transition (D={key[0]}, n={key[1]})", exc)
        return False  # reached only while planning (build_failed returns there)
    return True


#: fp16 range kept by the scales (below 65504 with room for the rounding); the ``forms`` kernel carries the same constant
_F16_BOUND = 30000.0


def _f32_forms_fake(gamma, beta, wa, wb, ws):
    h, d = wa.shape
    f16 = dict(dtype=torch.float16, device=wa.device)
    return (torch.empty((h, d), **f16), torch.empty((h, d), **f16), torch.empty((h, d), **f16), torch.empty((4,), dtype=torch.float32, device=wa.device))


@opaque(fake=_f32_forms_fake, name="transition_f32io_forms_sm90a")
def _f32_forms(gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
               ws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(Wa16, Wb16, (Ws / s_ws)^T fp16, sc fp32 [4]) -- see the kernels' header for ``sc``. Bounds: |xn| <= sqrt(D) max|gamma| + max|beta|,
    |a|, |b| <= |xn|max * the largest row L1 norm of Wa, Wb, |h| <= |a|max |b|max. Each scale is 1 unless its bound passes ``_F16_BOUND``.
    Two launches of device arithmetic (no host sync): made once per parameter version at inference, once per step in training."""
    if _is_fake(wa):
        return _f32_forms_fake(gamma, beta, wa, wb, ws)
    d, h = wa.shape[1], wa.shape[0]
    f = [t.float().contiguous() for t in (gamma, beta, wa, wb, ws)]
    return tuple(_f32_ext(d, h // d).forms(*f))


def _f32_variant(d: int, n: int, rows: int) -> int:
    """The build of ``wide/f32io.cu`` for (D, n, rows) (see its switch), measured on H100 at n 2 / 4, 16 384 - 589 824 rows:
    D 64: 1 (64-row tiles, 3 CTAs / SM), D 128: 1 (128-row tiles over two consumer warpgroups, double-buffered x), D 256: 1 (three weight
    slots)."""
    return 1


def _f32_launch_fake(x, gamma, beta, wa16, wb16, wst16, sc, eps, variant, save=False):
    rows, d = (x.shape[0], x.shape[1]) if save else (1, 1)
    f32 = dict(dtype=torch.float32, device=x.device)
    return torch.empty_like(x), torch.empty((rows, d), **f32), torch.empty((rows,), **f32), torch.empty((rows,), **f32)


@opaque(fake=_f32_launch_fake, name="transition_f32io_fwd_sm90a")
def _f32_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa16: torch.Tensor, wb16: torch.Tensor, wst16: torch.Tensor,
                sc: torch.Tensor, eps: float, variant: int, save: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x fp32 [M, D] -> (x + transition(x) fp32, xn, rstd, c1); the last three are 1-element placeholders unless ``save``."""
    if _is_fake(x, wa16):
        return _f32_launch_fake(x, gamma, beta, wa16, wb16, wst16, sc, eps, variant, save)
    d = x.shape[1]
    ext = _f32_ext(d, wa16.shape[0] // d)
    if d == 512:
        from . import fused_generic_sm90a

        h16, xn, rstd, c1 = ext.expand(x, gamma, beta, wa16, wb16, sc, float(eps), bool(save), num_sms(x.device))
        out = fused_generic_sm90a._ext().squeeze_res_h16(h16, _f32_ws16(wst16), x, sc[3:4])
        return out, xn, rstd, c1
    if save:
        return tuple(ext.fwd_train(x, gamma, beta, wa16, wb16, wst16, sc, float(eps), int(variant), num_sms(x.device)))
    out = ext.fwd(x, gamma, beta, wa16, wb16, wst16, sc, float(eps), int(variant), num_sms(x.device))
    f32 = dict(dtype=torch.float32, device=x.device)
    return out, torch.empty((1, 1), **f32), torch.empty((1,), **f32), torch.empty((1,), **f32)


def _f32_ws16(wst16: torch.Tensor) -> torch.Tensor:
    """Ws16 [D, H] (the generic squeeze's B operand, K-major over the hidden units) from the forms' Ws16^T, once per forms tensor."""
    from miniworld_engine.kernels import _capture

    return _capture.param_forms("transition_f32io_ws16", (wst16,), lambda: wst16.t().contiguous(), capture_reads_eager=True)


def _f32_inputs(x, gamma, beta, wa, wb, ws, cache: bool):
    """(flat x, fp32 gamma, fp32 beta, wa16, wb16, wst16, sc); ``cache`` (inference): the forms once per parameter version."""
    from miniworld_engine.kernels import _capture

    flat = x.reshape(-1, x.shape[-1]).contiguous()
    gf, bf = gamma.float().contiguous(), beta.float().contiguous()
    if cache:
        forms = _capture.param_forms("transition_f32io_forms", (gamma, beta, wa, wb, ws), lambda: _f32_forms(gamma, beta, wa, wb, ws),
                                     capture_reads_eager=True)
    else:
        forms = _f32_forms(gamma, beta, wa, wb, ws)
    return (flat, gf, bf, *forms)


@functools.lru_cache(maxsize=8)
def _k8_perm(k: int, device: str) -> torch.Tensor:
    """Column order (0, 2, 4, 6, 1, 3, 5, 7) inside every 8 of ``k``: the K order in which ``bwd_dx`` reads its accumulator / dy fragments as
    TF32 A fragments (see ``wide/kernels/transition_bwd_f32dx.cu``)."""
    base = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    return (torch.arange(0, k, 8).view(-1, 1) + base).reshape(-1).to(device)


#: D 128 training rows from which the two-kernel TF32 backward (`_f32_bwd`) runs; below, the generic recompute backward (H100 graph, 128^2
#: rows: 0.106 against 0.101 ms -- one 128-row tile per CTA leaves the per-tile latency exposed; 384^2: 0.74 against 0.85, 768^2: 2.82 against
#: 2.94 ms).
_F32_FUSED_BWD_MIN_ROWS = 65536


#: D 128 weight-gradient kernel: replicas of each 32-unit hidden slice (grid = H / 32 * replicas, at most one CTA per SM)
def _dw_replicas(hid: int, sms: int) -> int:
    return max(1, sms // (hid // 32))


def _f32_bwd_fake(dy, xn, x, rstd, c1, gamma, wst, wa, wb, wabt):
    return (torch.empty_like(x), torch.empty_like(gamma), torch.empty_like(gamma), torch.empty_like(wa), torch.empty_like(wb),
            wa.new_empty((wa.shape[1], wa.shape[0])))


@opaque(fake=_f32_bwd_fake, name="transition_f32_bwd_sm90a")
def _f32_bwd(dy: torch.Tensor, xn: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor, c1: torch.Tensor, gamma: torch.Tensor,
             wst: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
             wabt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """D 128 backward, the bf16 D 128 backward's two roles as two kernels (TF32): the fused dgrad (dh, a, b, SwiGLU backward, d_xn, LayerNorm
    backward, residual) and the weight gradients (dh, a, b recomputed per hidden slice, dWa / dWb / dWs accumulated on chip); h and [dA | dB]
    never reach memory. Returns (dx, dgamma, dbeta, dWa, dWb, dWs) from the forward's saves and the step's TF32-rounded weight copies."""
    if _is_fake(dy, x):
        return _f32_bwd_fake(dy, xn, x, rstd, c1, gamma, wst, wa, wb, wabt)
    d, hid = x.shape[1], wa.shape[0]
    dev = str(x.device)
    ext = _f32_ext(d, hid // d)
    sms = num_sms(x.device)
    wst_p = wst.index_select(1, _k8_perm(d, dev))
    wabt_p = wabt.index_select(1, _k8_perm(2 * hid, dev))
    dx, _h, _dab, dgam, dbeta = ext.bwd_dx(dy, xn, x, rstd, c1, gamma, wst_p, wa, wb, wabt_p, sms, False)
    dwa, dwb, dws = ext.bwd_dw(dy, xn, wst, wa, wb, _dw_replicas(hid, sms))
    return dx, dgam, dbeta, dwa, dwb, dws


class _F32TransitionSM90A(torch.autograd.Function):
    """fp32 training: the saving fused forward (fp16 products at TF32's rounding) and a native-TF32 recompute backward (D 128: ``_f32_bwd``;
    else the generic build's)."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        from . import fused_generic_sm90a as G

        shape = x.shape
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        masters = (wa, wb, ws)
        flat, gf, bf, wa16, wb16, wst16, sc = _f32_inputs(x, gamma, beta, wa, wb, ws, cache=False)
        out, xn, rstd, c1 = _f32_launch(flat, gf, bf, wa16, wb16, wst16, sc, float(eps), _f32_variant(shape[-1], wa.shape[0] // shape[-1],
                                                                                                         flat.shape[0]), True)
        # the backward's TF32-rounded weight copies (Ws^T, [Wa;Wb]^T, Wa, Wb, Ws) in one launch
        wst, wabt, wa_r, wb_r, ws_r = G._prep_step(*G._kernel_weights(flat, wa, wb, ws))
        ctx.save_for_backward(flat, xn, rstd, c1, gf, wa_r, wb_r, ws_r, wst, wabt, *masters)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        from . import fused_generic_sm90a as G

        flat, xn, rstd, c1, gf, wa_r, wb_r, ws_r, wst, wabt, *_m = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        hid = wa_r.shape[0]
        if flat.shape[1] == 128 and flat.shape[0] >= _F32_FUSED_BWD_MIN_ROWS:
            dx, dgam, dbeta, dwa, dwb, dws = _f32_bwd(dy.reshape(-1, dy.shape[-1]).contiguous(), xn, flat, rstd, c1, gf, wst, wa_r, wb_r, wabt)
            return dx.reshape(ctx.shape), dgam.to(gdt), dbeta.to(bdt), dwa.to(adt), dwb.to(bwdt), dws.to(sdt), None
        ph = flat.new_empty((1, 1))
        dx, dgam, dbeta, dwab, dws = G._bwd_launch(dy.reshape(-1, dy.shape[-1]).contiguous(), flat, xn, rstd, c1, ph, ph, ph, gf, wa_r, wb_r,
                                                   ws_r, False, wst, wabt)
        hid = wa_r.shape[0]
        return dx.reshape(ctx.shape), dgam.to(gdt), dbeta.to(bdt), dwab[:hid].to(adt), dwab[hid:].to(bwdt), dws.to(sdt), None


def transition_f32_sm90a(x, gamma, beta, wa, wb, ws, eps, variant: int | None = None):
    """fp32 entry (call ``f32_available`` first): ``x + transition(x)``. Inference: one fused launch (two at D 512), the fp16 weight forms and
    the scales made once per parameter version. Training (grad on and an operand requiring it): ``_F32TransitionSM90A``."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws)):
        return _F32TransitionSM90A.apply(x, gamma, beta, wa, wb, ws, eps)
    shape = x.shape
    d = shape[-1]
    flat, gf, bf, wa16, wb16, wst16, sc = _f32_inputs(x, gamma, beta, wa, wb, ws, cache=True)
    if variant is None:
        variant = _f32_variant(d, wa.shape[0] // d, flat.shape[0])
    return _f32_launch(flat, gf, bf, wa16, wb16, wst16, sc, float(eps), int(variant), False)[0].reshape(shape)

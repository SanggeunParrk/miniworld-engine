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
import warnings
from pathlib import Path

import torch

from miniworld_engine.kernels._device import num_sms
from miniworld_engine.kernels import _cast_cuda

from ..._compile import opaque
from ._weights import transposed_contiguous
from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

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


_BUILD_FAILED: set[tuple[int, int]] = set()


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful build of this width, both cached. A build failure is an environment problem, not
    a dispatch bug: it warns once per width and keeps the existing path."""
    key = (x.shape[-1], wa.shape[0] // x.shape[-1])
    if not supported(x, wa, ws) or key in _BUILD_FAILED:
        return False
    # FakeTensor dispatch records shapes only; it must not invoke nvcc or
    # wait for another process's extension lock. Launch wrappers have fakes.
    # Dynamo's TensorVariable is not itself a FakeTensor. Defer its build to
    # the opaque runtime launch instead of tracing filesystem/compiler calls.
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _ext_for(x, n=key[1])
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED.add(key)
        warnings.warn(
            f"wide sm90a Transition (D={x.shape[-1]}, n={key[1]}) unavailable, keeping the existing path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
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


# ---------------------------------------------------------------------------------------------------------- fp32 inference (f32io)
#: Channel widths / hidden multiples of the fp32 inference forward (``wide/kernels/transition_fwd_f32io.cu``): one fused kernel, x and y
#: fp32, the LayerNorm in fp32, the products on fp16 tensor cores with fp32 accumulation -- fp16's 10-bit mantissa is TF32's, so each
#: operand is rounded as the TF32 path rounds it; device scales (``_f32_forms``) keep every fp16 operand inside fp16's range.
F32_SHAPES = {64: (2, 4), 128: (2, 4)}
_F32_REVISION = "v2"


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
    """fp32 inference on sm_90 with TF32 allowed (the precision it reproduces), D / n in ``F32_SHAPES``, weights fp32 or bf16, any row
    count (the tail tile is zero-filled on the way in, clipped on the way out)."""
    from .fused_generic_sm90a import _tf32_allowed

    if training or not x.is_cuda or x.dtype is not torch.float32 or x.numel() == 0 or not _tf32_allowed():
        return False
    if not _is_hopper(x.device.index if x.device.index is not None else torch.cuda.current_device()):
        return False
    d = x.shape[-1]
    if d not in F32_SHAPES or wa.dim() != 2 or wa.shape[1] != d or wa.shape[0] % d or wa.shape[0] // d not in F32_SHAPES[d]:
        return False
    if tuple(ws.shape) != (d, wa.shape[0]):
        return False
    return all(w.dtype in (torch.float32, torch.bfloat16) for w in (wa, ws))


_F32_BUILD_FAILED: set[tuple[int, int]] = set()


def f32_available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor, training: bool) -> bool:
    """``f32_supported()`` plus a successful build (cached); a build failure warns once and keeps the existing (generic TF32) path."""
    if not f32_supported(x, wa, ws, training):
        return False
    key = (x.shape[-1], wa.shape[0] // x.shape[-1])
    if key in _F32_BUILD_FAILED:
        return False
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _f32_ext(*key)
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _F32_BUILD_FAILED.add(key)
        warnings.warn(f"fp32 sm90a Transition (D={key[0]}, n={key[1]}) unavailable, keeping the existing path: {exc!r}", RuntimeWarning,
                      stacklevel=2)
        return False
    return True


#: fp16 range kept by the scales (below 65504 with room for the rounding)
_F16_BOUND = 30000.0


def _f32_forms(gamma, beta, wa, wb, ws):
    """(Wa16, Wb16, (Ws / s_ws)^T fp16, sc fp32 [4]) -- see the kernel's header for ``sc``. Bounds: |xn| <= sqrt(D) max|gamma| + max|beta|,
    |a|, |b| <= |xn|max * the largest row L1 norm of Wa, Wb, |h| <= |a|max |b|max. Each scale is 1 unless its bound passes ``_F16_BOUND``.
    Device arithmetic only (no host sync); made once per parameter version (``param_forms``)."""
    with torch.no_grad():
        g, b = gamma.float(), beta.float()
        wa, wb, ws = wa.float(), wb.float(), ws.float()
        d = wa.shape[1]
        xn_max = (d ** 0.5) * g.abs().max() + b.abs().max()
        s_x = (xn_max / _F16_BOUND).clamp_min(1.0)
        s_w = (torch.maximum(wa.abs().max(), wb.abs().max()) / _F16_BOUND).clamp_min(1.0)
        a_max = xn_max * torch.maximum(wa.abs().sum(dim=1).max(), wb.abs().sum(dim=1).max())
        s_h = (a_max * a_max / _F16_BOUND).clamp_min(1.0)
        s_ws = (ws.abs().max() / _F16_BOUND).clamp_min(1.0)
        sc = torch.stack([1.0 / s_x, s_x * s_w, 1.0 / s_h, s_h * s_ws]).contiguous()
        return (wa / s_w).half().contiguous(), (wb / s_w).half().contiguous(), (ws / s_ws).t().contiguous().half(), sc


def _f32_variant(d: int, n: int, rows: int) -> int:
    """The build of ``wide/f32io.cu`` for (D, n, rows) (see its switch), measured on H100 at n 2 / 4, 16 384 - 589 824 rows:
    D 64: 1 (64-row tiles, 3 CTAs / SM), D 128: 1 (128-row tiles over two consumer warpgroups, double-buffered x)."""
    return 1


def _f32_launch_fake(x, gamma, beta, wa16, wb16, wst16, sc, eps, variant):
    return torch.empty_like(x)


@opaque(fake=_f32_launch_fake, name="transition_f32io_fwd_sm90a")
def _f32_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa16: torch.Tensor, wb16: torch.Tensor, wst16: torch.Tensor,
                sc: torch.Tensor, eps: float, variant: int) -> torch.Tensor:
    """x fp32 [M, D] -> x + transition(x) (fp32)."""
    if _is_fake(x, wa16):
        return _f32_launch_fake(x, gamma, beta, wa16, wb16, wst16, sc, eps, variant)
    return _f32_ext(x.shape[1], wa16.shape[0] // x.shape[1]).fwd(x, gamma, beta, wa16, wb16, wst16, sc, float(eps), int(variant),
                                                                 num_sms(x.device))


def transition_f32_sm90a(x, gamma, beta, wa, wb, ws, eps, variant: int | None = None):
    """fp32 inference entry (call ``f32_supported`` first): ``x + transition(x)`` from one launch, the fp16 weight forms and the scales made
    once per parameter version."""
    from miniworld_engine.kernels import _capture

    shape = x.shape
    d = shape[-1]
    flat = x.reshape(-1, d).contiguous()
    gf, bf = gamma.float().contiguous(), beta.float().contiguous()
    wa16, wb16, wst16, sc = _capture.param_forms("transition_f32io_forms", (gamma, beta, wa, wb, ws),
                                                 lambda: _f32_forms(gamma, beta, wa, wb, ws), capture_reads_eager=True)
    if variant is None:
        variant = _f32_variant(d, wa.shape[0] // d, flat.shape[0])
    return _f32_launch(flat, gf, bf, wa16, wb16, wst16, sc, float(eps), int(variant)).reshape(shape)

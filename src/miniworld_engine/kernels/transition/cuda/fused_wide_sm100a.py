"""Hand-CUDA sm_100a Transition for the channel widths other than 128: D = 64, 256, 384, 512 (n = 4, bf16; B200).

The companion of ``fused_sm100a`` (D = 128), developed in ``experiments/transition_fused_sm100`` (branch
``perf/transition-sm100-b200``, rounds w1-w6 and s1; the kernels under ``sm100/widths/`` are generated from that capsule by its
``export_engine.py``). What runs depends on the width, because what fits tensor memory (512 columns) and shared memory does:

    D     forward                                          backward
    64    one fused kernel (weights resident)              one fused two-role kernel + partial reduction
    256   one fused kernel                                 gate (recomputes a, b from xn) -> fp32 dW GEMMs -> d_xn GEMM -> LN bwd
    384   LN -> expand + SwiGLU (saves h, a, b) -> squeeze  gate (dh GEMM + SwiGLU bwd from the saved a, b) -> fp32 dW GEMMs ->
    512   same as 384                                        d_xn GEMM -> LN bwd

All kernels are tcgen05 / TMA / 2-CTA-cluster kernels except the LayerNorm ones; the weight gradients are cuBLAS GEMMs with an
fp32 output. At D >= 384 the forward keeps a and b (bf16) for the backward instead of recomputing them in fp32: ~2 x M x 4D bf16
more activation memory per layer, and the D >= 384 step is at this algorithm's energy floor on the power-capped card (w5).
The expand + SwiGLU at D >= 384 deals (tile pair, 128-unit chunk) items over all SMs up to 1152 tiles (L = 384 for an L x L pair):
the whole forward 5-15 % faster at L256 / L384, equal from L512 (s1 §3).

Measured on a B200 (1000 W cap), CUDA-graph replay, µs, against torch.compile of the bf16 module:

    inference   L128: D64 8.5 (20.9), D256 29.1 (58.9), D384 57.4 (91.3), D512 84.8 (125.0)
                L768: D64 108.5 (389.3), D256 802.3 (1845.2), D384 2038.0 (3256.4), D512 3314.9 (4874.6)
    training    L384: D64 123.5 (329.1), D256 1017.9 (1290.3), D384 1855.4 (2273.1), D512 3049.9 (3538.3)
                L768: D64 438.6 (1089.0), D256 4015.7 (5057.5), D384 7616.4 (9390.0), D512 12125.3 (14389.3)

``supported()`` is the whole gate; everything it rejects keeps the existing path. The reductions run in a fixed order (no
atomics), so a replay is bit-identical.
"""

import functools
import os
import warnings

import torch

from ..._compile import opaque
from .fused_sm100a import _ext, _is_b200, build_cubins

#: Channel widths with a build here (128 is ``fused_sm100a``). Each builds its own cubins on first use.
WIDTHS = (64, 256, 384, 512)
#: Row tile of every persistent grid here. ``M`` must be a whole number of these.
ROWS = 128
#: D = 64 backward: hidden-slice replicas of the weight role (4 slices x R CTAs; the rest run the input role). 20 measured
#: fastest at L384 and L768 on a 148-SM B200 (w1).
D64_REPL = 20
#: D >= 384: the expand + SwiGLU deals (tile pair, chunk) items over all SMs up to this many tiles, whole tile pairs above.
ITEM_TILES = 1152


def _specs(d: int) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    if d == 64:
        return (("fwd", "widths/tfwd_d64.cu", ()), ("bwd", "widths/tbwd_d64.cu", ()))
    dim = (f"-DDIM={d}",)
    lnb = ("lnbwd", "widths/tlnbwd_w.cu", (*dim, f"-DLPR={32 if d == 512 else 16}"))
    dxn = ("dxn", "widths/tgemm_nd.cu", (*dim, f"-DKDIM={8 * d}", "-DEPI_PLAIN"))
    if d == 256:
        return (("fwd", "widths/tfwd_d256.cu", ()), ("gate", "widths/tgate_w.cu", dim), dxn, lnb)
    return (("ln", "widths/tln_w.cu", (*dim, f"-DLPR={32 if d == 512 else 16}")),
            ("swiglu", "widths/tswiglu_w.cu", dim), ("swiglu_is", "widths/tswiglu_w.cu", (*dim, "-DITEM_SCHED")),
            ("swiglu_ab", "widths/tswiglu_w.cu", (*dim, "-DSAVE_AB")),
            ("swiglu_abis", "widths/tswiglu_w.cu", (*dim, "-DSAVE_AB", "-DITEM_SCHED")),
            ("squeeze", "widths/tgemm_nd.cu", dim), ("gate", "widths/tgate_ab.cu", (*dim, "-DNO_H")), dxn, lnb)


@functools.lru_cache(maxsize=8)
def _load(d: int):
    """Build this width's cubins and register them with the extension (loaded on the current device: a cubin the driver rejects
    fails here, inside ``available()``). Returns the extension."""
    ext = _ext()
    for name, path in build_cubins(f"d{d}", _specs(d)).items():
        ext.load_cubin(f"d{d}_{name}", path)
    return ext


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """Whether these kernels can run this call: sm_100 (B200), bf16, D in ``WIDTHS`` with hidden 4 D, whole 128-row tiles, and at
    D = 64 enough SMs for the backward's two roles. The requirements are the kernels' own, not a policy."""
    if os.environ.get("MINIWORLD_TRANSITION_FUSED_SM100A", "1") == "0":
        return False
    if not x.is_cuda or x.dtype is not torch.bfloat16:
        return False
    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if not _is_b200(index):
        return False
    d = x.shape[-1]
    if d not in WIDTHS or wa.shape != (4 * d, d) or ws.shape != (d, 4 * d):
        return False
    if wa.dtype is not torch.bfloat16 or ws.dtype is not torch.bfloat16:
        return False
    if d == 64 and _sm_count(index) - 4 * D64_REPL < 2:
        return False
    rows = x.numel() // d
    return rows > 0 and rows % ROWS == 0


_BUILD_FAILED: set[int] = set()


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful build of this width, both cached; a build failure warns once per width and keeps the
    existing path."""
    if not supported(x, wa, ws) or x.shape[-1] in _BUILD_FAILED:
        return False
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _load(x.shape[-1])
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED.add(x.shape[-1])
        warnings.warn(f"wide sm100a Transition (D={x.shape[-1]}) unavailable, keeping the existing path: {exc!r}",
                      RuntimeWarning, stacklevel=2)
        return False
    return True


def _is_fake(*tensors) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor
    return any(isinstance(t, FakeTensor) for t in tensors)


@functools.lru_cache(maxsize=8)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _mm_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM (cuBLAS with an fp32 output); torch builds without ``out_dtype`` round through bf16."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a, b).float()


class _Launch:
    """The kernels of one width on one device, with the grids of an M-row call."""

    def __init__(self, x: torch.Tensor):
        self.d, self.m = x.shape[1], x.shape[0]
        self.h = 4 * self.d
        self.tiles = self.m // ROWS
        self.nsm = _sm_count(x.device.index)
        g = min(self.nsm, self.tiles)
        self.grid = max(2, g - g % 2)                          # persistent 2-CTA clusters, a pair walks the leader's tile count
        self.ext = _load(self.d)

    def tmap(self, t, inner, outer, box_inner=64, box_outer=64):
        return self.ext.tmap(t, inner, outer, box_inner, box_outer)

    def run(self, cubin, kernel, grid, args, block=512, cluster=2):
        self.ext.launch_kernel(f"d{self.d}_{cubin}", kernel, grid, block, cluster, list(args))

    def rows(self, t):                                         # [M, D] activation map
        return self.tmap(t, self.d, self.m)


def _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save):
    """Output structure only: (out, xn, rstd, c1, h, a, b). xn / rstd / c1 are 1-element placeholders unless ``save``; h / a / b
    are [M, 4D] when ``save`` at D >= 384 (kept for the backward) and placeholders otherwise."""
    m, d = x.shape
    rows = m if save else 1
    f32 = dict(dtype=torch.float32, device=x.device)
    keep = save and d >= 384
    hab = [x.new_empty((m, 4 * d)) if keep else x.new_empty((1, 1)) for _ in range(3)]
    return (torch.empty_like(x), torch.empty_like(x) if save else x.new_empty((1, d)),
            torch.empty((rows,), **f32), torch.empty((rows,), **f32), *hab)


@opaque(fake=_fwd_launch_fake, name="transition_wide_fwd_sm100a")
def _fwd_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
                ws: torch.Tensor, eps: float, save: bool,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, D] bf16, gamma / beta fp32. Returns (out, xn, rstd, c1, h, a, b), out = transition(x) + x; see the fake."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save)
    k = _Launch(x)
    d, h, m, tiles = k.d, k.h, k.m, k.tiles
    f32 = dict(dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    rstd = torch.empty((m if save else 1,), **f32)
    c1 = torch.empty_like(rstd)
    def ph():                                                  # distinct placeholders: custom-op outputs must not alias each other
        return x.new_empty((1, 1))

    if d in (64, 256):
        xn = torch.empty_like(x) if save else x.new_empty((1, d))
        mout = k.rows(out)
        k.run("fwd", f"transition_fwd_d{d}_sm100", k.grid,
              (k.rows(x), k.tmap(wa, d, h), k.tmap(wb, d, h), k.tmap(ws, h, d, 64, d // 2), mout,
               k.rows(xn) if save else mout, gamma, beta, rstd, c1, tiles, float(eps), int(save)))
        return out, xn, rstd, c1, ph(), ph(), ph()
    xn = torch.empty_like(x)                                   # D >= 384: the expand's A operand, always materialised
    hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
    k.run("ln", "transition_ln_w", min(m // 16, k.nsm * 4), (x, gamma, beta, xn, rstd, c1, m, float(eps), int(save)), 256, 1)
    item = tiles <= ITEM_TILES
    mh = k.tmap(hid, h, m)
    maps = (k.rows(xn), k.tmap(wa, d, h, 64, 128), k.tmap(wb, d, h, 64, 128), mh)
    if save:
        a, b = torch.empty_like(hid), torch.empty_like(hid)
        k.run("swiglu_abis" if item else "swiglu_ab", "transition_swiglu_w_sm100", k.nsm if item else k.grid,
              (*maps, k.tmap(a, h, m), k.tmap(b, h, m), 1, tiles))
    else:
        k.run("swiglu_is" if item else "swiglu", "transition_swiglu_w_sm100", k.nsm if item else k.grid, (*maps, tiles))
    k.run("squeeze", "transition_gemm_nd_sm100", k.grid, (mh, k.tmap(ws, h, d), k.rows(x), k.rows(out), tiles))
    if not save:
        return out, xn.new_empty((1, d)), rstd, c1, ph(), ph(), ph()
    return out, xn, rstd, c1, hid, a, b


def _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hid, a, b):
    """Output structure only: (dx, dgamma, dbeta, dWa, dWb, dWs); the weight gradients are fp32 at D >= 256."""
    wdt = torch.float32 if x.shape[-1] >= 256 else wa.dtype
    return (torch.empty_like(x), torch.empty_like(gamma), torch.empty_like(gamma),
            torch.empty(wa.shape, dtype=wdt, device=wa.device), torch.empty(wb.shape, dtype=wdt, device=wb.device),
            torch.empty(ws.shape, dtype=wdt, device=ws.device))


@opaque(fake=_bwd_launch_fake, name="transition_wide_bwd_sm100a")
def _bwd_launch(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, rstd: torch.Tensor, c1: torch.Tensor,
                gamma: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, hid: torch.Tensor,
                a: torch.Tensor, b: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dx, dgamma, dbeta, dWa, dWb, dWs); ``dx`` already carries the residual branch. ``hid`` / ``a`` / ``b`` are the
    forward's h / a / b at D >= 384 and placeholders otherwise."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hid, a, b)
    k = _Launch(x)
    d, h, m, tiles = k.d, k.h, k.m, k.tiles
    f32 = dict(dtype=torch.float32, device=x.device)
    dx = torch.empty_like(x)
    if d == 64:
        ndw = (h // 64) * D64_REPL
        ndx = k.nsm - ndw
        partab = torch.empty((ndw, 128, d), **f32)
        parts = torch.empty((ndw, d, 64), **f32)
        dgbw = torch.empty((ndx * 4, 2 * d), **f32)
        dwa, dwb, dws = torch.empty_like(wa), torch.empty_like(wb), torch.empty_like(ws)
        dgam, dbeta = torch.empty((d,), **f32), torch.empty((d,), **f32)
        k.run("bwd", "transition_bwd_d64_sm100", k.nsm,
              (k.rows(dy), k.rows(xn), k.rows(x), k.tmap(ws, h, d), k.tmap(wa, d, h), k.tmap(wb, d, h), k.rows(dx),
               rstd, c1, gamma, partab, parts, dgbw, tiles, ndw))
        nred = 3 * h * d + 2 * d
        k.run("bwd", "transition_bwd_d64_reduce", (nred + 255) // 256,
              (partab, parts, dgbw, dwa, dwb, dws, dgam, dbeta, ndw, ndx * 4), 256, 1)
        return dx, dgam, dbeta, dwa, dwb, dws
    wst = ws.t().contiguous()                                  # [H, D]
    wab_t = torch.cat((wa, wb), 0).t().contiguous()            # [D, 2H]
    dab = torch.empty((m, 2 * h), dtype=x.dtype, device=x.device)
    mdab = k.tmap(dab, 2 * h, m)
    if d == 256:
        hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
        k.run("gate", "transition_gate_w_sm100", k.grid,
              (k.rows(dy), k.rows(xn), k.tmap(wst, d, h, 64, 32), k.tmap(wa, d, h), k.tmap(wb, d, h), k.tmap(hid, h, m), mdab,
               tiles, 1))
    else:
        mh = k.tmap(hid, h, m)
        k.run("gate", "transition_gate_ab_sm100", k.grid,
              (k.rows(dy), k.tmap(wst, d, h), k.tmap(a, h, m), k.tmap(b, h, m), mh, mdab, tiles, 0))
    dws = _mm_f32(dy.t(), hid)                                # [D, H]
    dwab = _mm_f32(dab.t(), xn)                               # [2H, D]: dWa then dWb
    dxn = torch.empty_like(x)
    mdy = k.rows(dy)
    k.run("dxn", "transition_gemm_nd_sm100", k.grid, (mdab, k.tmap(wab_t, 2 * h, d), mdy, k.rows(dxn), tiles))
    nb = min(m // 16, k.nsm * 4)
    lpart = torch.empty((nb, 2 * d), **f32)
    dgb = torch.empty((2 * d,), **f32)
    k.run("lnbwd", "transition_lnbwd_w", nb, (dxn, x, dy, rstd, c1, gamma, dx, lpart, m), 256, 1)
    k.run("lnbwd", "transition_lnbwd_w_reduce", (2 * d + 255) // 256, (lpart, dgb, nb), 256, 1)
    return dx, dgb[:d], dgb[d:].clone(), dwab[:h], dwab[h:].clone(), dws   # outputs of a custom op must not alias each other


class _WideTransitionSM100A(torch.autograd.Function):
    """``y = transition(x) + x`` at D in ``WIDTHS``; the forward saves xn and the LayerNorm statistics (and h, a, b at D >= 384)."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        gf, bf = gamma.float().contiguous(), beta.float().contiguous()
        wa, wb, ws = wa.contiguous(), wb.contiguous(), ws.contiguous()
        out, xn, rstd, c1, hid, a, b = _fwd_launch(flat, gf, bf, wa, wb, ws, float(eps), True)
        ctx.save_for_backward(flat, xn, rstd, c1, gf, wa, wb, ws, hid, a, b)
        ctx.shape = shape
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, rstd, c1, gf, wa, wb, ws, hid, a, b = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        dx, dgam, dbeta, dwa, dwb, dws = _bwd_launch(
            dy.reshape(-1, dy.shape[-1]).contiguous(), flat, xn, rstd, c1, gf, wa, wb, ws, hid, a, b)
        return (dx.reshape(ctx.shape), dgam.to(gdt), dbeta.to(bdt), dwa.to(adt), dwb.to(bwdt), dws.to(sdt), None)


def transition_wide_sm100a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as ``fused_sm100a.transition_fused_sm100a``. Call ``available()`` first: this raises
    rather than falling back. Inference (grad mode off, or nothing requiring grad) runs the forward without the saves."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))):
        shape = x.shape
        out, *_ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), gamma.float().contiguous(), beta.float().contiguous(),
                              wa.contiguous(), wb.contiguous(), ws.contiguous(), float(eps), False)
        return out.reshape(shape)
    return _WideTransitionSM100A.apply(x, gamma, beta, wa, wb, ws, eps)

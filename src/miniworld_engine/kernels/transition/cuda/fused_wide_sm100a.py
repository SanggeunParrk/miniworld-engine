"""Hand-CUDA sm_100a Transition for every (width, expansion) but D = 128 / n = 4: n = 4 at D = 64, 256, 384, 512 and n = 2 at
D = 64, 128, 256, 384, 512, 768 (bf16; B200).

The companion of ``fused_sm100a`` (D = 128, n = 4), developed in ``experiments/transition_fused_sm100`` (branch
``perf/transition-sm100-b200``, rounds w1-w6, s1 and n2; the kernels under ``sm100/`` are generated from that capsule by its
``export_engine.py``, each built with ``-DHID=<n D>``). What runs depends on the width, because what fits tensor memory (512 columns)
and shared memory does:

    D     forward                                          backward
    64    one fused kernel (weights resident)              one fused two-role kernel + partial reduction
    128   (n = 2) the D = 128 fused kernel                 (n = 2) the D = 128 fused two-role kernel + partial reduction
    256   one fused kernel                                 gate (recomputes a, b from xn) -> fp32 dW GEMMs -> d_xn GEMM -> LN bwd
    384   one fused kernel (saves h, a, b)                 one fused kernel (gate + d_xn + LN bwd) -> fp32 dW GEMMs
    512   LN -> expand + SwiGLU (saves h, a, b) -> squeeze  gate (dh GEMM + SwiGLU bwd from the saved a, b) -> fp32 dW GEMMs ->
                                                           d_xn GEMM -> LN bwd
    768   (n = 2) as 512, the squeeze / d_xn GEMMs run as two 384-column halves (the accumulator would need 768 TMEM columns)

All kernels are tcgen05 / TMA / 2-CTA-cluster kernels except the LayerNorm ones; the weight gradients are cuBLAS GEMMs with an
fp32 output. At D >= 384 the forward keeps a and b (bf16) for the backward instead of recomputing them in fp32: ~2 x M x nD bf16
more activation memory per layer, and the D >= 384 step is at this algorithm's energy floor on the power-capped card (w5).
The expand + SwiGLU at D >= 384 deals (tile pair, 128-unit chunk) items over all SMs up to 1152 tiles (L = 384 for an L x L pair):
the whole forward 5-15 % faster at L256 / L384, equal from L512 (s1 §3).

Measured on a B200 (1000 W cap), CUDA-graph replay, µs, against torch.compile of the bf16 module (n = 4; the D384 values
are the three-kernel chain before round n2, current numbers in docs/gpus/b200/transition/transition.md):

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

#: (D, n) with a build here; (128, 4) is ``fused_sm100a``. Each builds its own cubins on first use.
SHAPES = ((64, 4), (256, 4), (384, 4), (512, 4), (64, 2), (128, 2), (256, 2), (384, 2), (512, 2), (768, 2))
#: Channel widths with a build for some n (kept for callers that ask by width only).
WIDTHS = tuple(sorted({d for d, _ in SHAPES}))
#: Row tile of every persistent grid here. ``M`` must be a whole number of these.
ROWS = 128
#: Two-role backward (D = 64, and D = 128 at n = 2): replicas of each 64-unit hidden slice in the weight role; the other CTAs run
#: the input role. (64, 4) = 20 measured fastest at L384 and L768 on a 148-SM B200 (w1); the n = 2 values are from round n2.
#: MINIWORLD_TRANSITION_WIDE_REPL overrides them for A/B runs.
REPL = {(64, 4): 20, (64, 2): 30, (128, 2): 14}
#: D >= 384: the expand + SwiGLU deals (tile pair, chunk) items over all SMs up to this many tiles, whole tile pairs above.
ITEM_TILES = 1152
#: Column width of one squeeze / d_xn GEMM launch at D = 768 (two halves; one launch's accumulator must fit 512 TMEM columns).
HALF = 384
#: D >= 384 at small M (at most this many 128-row tiles, e.g. a single-stream [B, L, D] activation): whole tile pairs would keep
#: only tiles / 2 cluster pairs busy, so the backward gate deals its (tile pair, 256-unit block) items over all SMs. Round n2.
#: MINIWORLD_TRANSITION_WIDE_SMALL_TILES overrides it for A/B runs.
SMALL_TILES = 16


def _small(tiles: int) -> bool:
    return tiles <= int(os.environ.get("MINIWORLD_TRANSITION_WIDE_SMALL_TILES", SMALL_TILES))


#: D = 384 forward at small M: one kernel per tile cluster, LayerNorm + expand + SwiGLU + squeeze + residual with the hidden
#: dimension split over the cluster's CL CTAs and the partial accumulators summed through an fp32 L2 scratch
#: (sm100/widths/tsmall_w.cu, round n2). CL is the largest of SMALL_CLS whose clusters are all resident at once (the driver's
#: cuOccupancyMaxActiveClusters: 12-CTA clusters do not pack a 148-SM B200 twelve to a wave), up to SMALL_FWD_TILES tiles (the
#: measured break-even with the per-tile forward: 16 tiles; at 24 the per-tile forward is 1.6x faster).
#: MINIWORLD_TRANSITION_WIDE_SMALL_FWD_TILES overrides it (0: off).
SMALL_FWD_TILES = 16
SMALL_CLS = (12, 6, 4, 2)


@functools.lru_cache(maxsize=64)
def _resident_clusters(d: int, n: int, cl: int, index: int) -> int:
    return int(_load(d, n).max_active_clusters(f"d{d}n{n}_small{cl}", "transition_small_w_sm100", 512, cl))


def _small_cl(d: int, n: int, tiles: int, index: int) -> int:
    """Cluster size of the small-M forward for this call, 0 when the per-tile path runs."""
    if d != 384 or tiles > int(os.environ.get("MINIWORLD_TRANSITION_WIDE_SMALL_FWD_TILES", SMALL_FWD_TILES)):
        return 0
    return next((c for c in SMALL_CLS if tiles <= _resident_clusters(d, n, c, index)), 0)


def _dn(x: torch.Tensor, wa: torch.Tensor) -> tuple[int, int]:
    d = x.shape[-1]
    return d, wa.shape[0] // d


def _repl(d: int, n: int) -> int:
    return int(os.environ.get("MINIWORLD_TRANSITION_WIDE_REPL", REPL[(d, n)]))


def _ndw(d: int, n: int) -> int:
    """Weight-role CTAs of the two-role backward: one per (64-unit hidden slice, replica)."""
    return (n * d // 64) * _repl(d, n)


def _specs(d: int, n: int) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    h = n * d
    hid = (f"-DHID={h}",)
    if d == 64:
        return (("fwd", "widths/tfwd_d64.cu", hid), ("bwd", "widths/tbwd_d64.cu", hid))
    if d == 128:
        return (("fwd", "tr_fwd_sm100.cu", hid), ("bwd", "tr_bwd_sm100.cu", hid))
    lpr = f"-DLPR={32 if d in (512, 768) else 16}"
    nd = HALF if d == 768 else d
    lnb = ("lnbwd", "widths/tlnbwd_w.cu", (f"-DDIM={d}", lpr))
    dxn = ("dxn", "widths/tgemm_nd.cu", (f"-DDIM={nd}", f"-DKDIM={2 * h}", "-DEPI_PLAIN"))
    if d == 256:
        return (("fwd", "widths/tfwd_d256.cu", hid), ("gate", "widths/tgate_w.cu", ("-DDIM=256", *hid)), dxn, lnb)
    dim = (f"-DDIM={d}", *hid)
    fwd = ((("fwd", "widths/tfwd_d384.cu", hid), ("bwd", "widths/tbwd_d384.cu", hid)) if d == 384 else
           (("ln", "widths/tln_w.cu", (f"-DDIM={d}", lpr)),
            ("swiglu", "widths/tswiglu_w.cu", dim), ("swiglu_is", "widths/tswiglu_w.cu", (*dim, "-DITEM_SCHED")),
            ("swiglu_ab", "widths/tswiglu_w.cu", (*dim, "-DSAVE_AB")),
            ("swiglu_abis", "widths/tswiglu_w.cu", (*dim, "-DSAVE_AB", "-DITEM_SCHED")),
            ("squeeze", "widths/tgemm_nd.cu", (f"-DDIM={nd}", f"-DKDIM={h}"))))
    return (*fwd,
            ("gate", "widths/tgate_ab.cu", (*dim, "-DNO_H")), ("gate_is", "widths/tgate_ab.cu", (*dim, "-DNO_H", "-DITEM_SCHED")),
            dxn, lnb,
            *((spec for c in SMALL_CLS for spec in (
                (f"small{c}", "widths/tsmall_w.cu", (*dim, f"-DCL={c}")),
                (f"small{c}_ab", "widths/tsmall_w.cu", (*dim, f"-DCL={c}", "-DSAVE_AB")))) if d == 384 else ()))


@functools.lru_cache(maxsize=16)
def _load(d: int, n: int):
    """Build this (width, n)'s cubins and register them with the extension (loaded on the current device: a cubin the driver
    rejects fails here, inside ``available()``). Returns the extension."""
    ext = _ext()
    for name, path in build_cubins(f"d{d}n{n}", _specs(d, n)).items():
        ext.load_cubin(f"d{d}n{n}_{name}", path)
    return ext


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """Whether these kernels can run this call: sm_100 (B200), bf16, (D, n) in ``SHAPES`` with hidden n D, whole 128-row tiles,
    and for the two-role backward enough SMs for both roles. The requirements are the kernels' own, not a policy."""
    if os.environ.get("MINIWORLD_TRANSITION_FUSED_SM100A", "1") == "0":
        return False
    if not x.is_cuda or x.dtype is not torch.bfloat16:
        return False
    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if not _is_b200(index):
        return False
    d, n = _dn(x, wa)
    if (d, n) not in SHAPES or wa.shape != (n * d, d) or ws.shape != (d, n * d):
        return False
    if wa.dtype is not torch.bfloat16 or ws.dtype is not torch.bfloat16:
        return False
    if (d, n) in REPL:
        ndx = _sm_count(index) - _ndw(d, n)
        if ndx < 2 or ndx % 2:
            return False
    rows = x.numel() // d
    return rows > 0 and rows % ROWS == 0


_BUILD_FAILED: set[tuple[int, int]] = set()


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful build of this (width, n), both cached; a build failure warns once and keeps the existing
    path."""
    if not supported(x, wa, ws) or _dn(x, wa) in _BUILD_FAILED:
        return False
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _load(*_dn(x, wa))
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED.add(_dn(x, wa))
        warnings.warn(f"wide sm100a Transition (D={x.shape[-1]}, n={wa.shape[0] // x.shape[-1]}) unavailable, keeping the existing "
                      f"path: {exc!r}", RuntimeWarning, stacklevel=2)
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
    """The kernels of one (width, n) on one device, with the grids of an M-row call."""

    def __init__(self, x: torch.Tensor, h: int):
        self.d, self.m, self.h = x.shape[1], x.shape[0], h
        self.n = h // self.d
        self.tiles = self.m // ROWS
        self.nsm = _sm_count(x.device.index)
        # persistent 2-CTA clusters, a pair walks the leader's tile count; rounded UP to the pair (an odd tile count below the SM
        # count gets a pair with one dummy CTA rather than a second round for one pair)
        self.grid = max(2, min(self.nsm - self.nsm % 2, self.tiles + self.tiles % 2))
        self.ext = _load(self.d, self.n)

    def tmap(self, t, inner, outer, box_inner=64, box_outer=64, row_stride=-1):
        return self.ext.tmap(t, inner, outer, box_inner, box_outer, row_stride)

    def run(self, cubin, kernel, grid, args, block=512, cluster=2):
        self.ext.launch_kernel(f"d{self.d}n{self.n}_{cubin}", kernel, grid, block, cluster, list(args))

    def rows(self, t):                                         # [M, D] activation map
        return self.tmap(t, self.d, self.m)

    def halves(self):
        """(first column, width) of each squeeze / d_xn GEMM launch."""
        return [(c, HALF) for c in range(0, self.d, HALF)] if self.d == 768 else [(0, self.d)]

    def cols(self, t, c0, nd):                                 # [M, nd] column slice of an [M, D] activation
        return self.tmap(t[:, c0:c0 + nd], nd, self.m, row_stride=self.d)


_FUSED_FWD = {64: "transition_fwd_d64_sm100", 128: "transition_fwd2_sm100", 256: "transition_fwd_d256_sm100"}


def _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save):
    """Output structure only: (out, xn, rstd, c1, h, a, b). xn / rstd / c1 are 1-element placeholders unless ``save``; h / a / b
    are [M, nD] when ``save`` at D >= 384 (kept for the backward) and placeholders otherwise."""
    m, d = x.shape
    rows = m if save else 1
    f32 = dict(dtype=torch.float32, device=x.device)
    keep = save and d >= 384
    hab = [x.new_empty((m, wa.shape[0])) if keep else x.new_empty((1, 1)) for _ in range(3)]
    return (torch.empty_like(x), torch.empty_like(x) if save else x.new_empty((1, d)),
            torch.empty((rows,), **f32), torch.empty((rows,), **f32), *hab)


@opaque(fake=_fwd_launch_fake, name="transition_wide_fwd_sm100a")
def _fwd_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
                ws: torch.Tensor, eps: float, save: bool,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, D] bf16, gamma / beta fp32. Returns (out, xn, rstd, c1, h, a, b), out = transition(x) + x; see the fake."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save)
    k = _Launch(x, wa.shape[0])
    d, h, m, tiles = k.d, k.h, k.m, k.tiles
    f32 = dict(dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    rstd = torch.empty((m if save else 1,), **f32)
    c1 = torch.empty_like(rstd)

    def ph():                                                  # distinct placeholders: custom-op outputs must not alias each other
        return x.new_empty((1, 1))

    if d in _FUSED_FWD:
        xn = torch.empty_like(x) if save else x.new_empty((1, d))
        mout = k.rows(out)
        k.run("fwd", _FUSED_FWD[d], k.grid,
              (k.rows(x), k.tmap(wa, d, h), k.tmap(wb, d, h), k.tmap(ws, h, d, 64, d // 2), mout,
               k.rows(xn) if save else mout, gamma, beta, rstd, c1, tiles, float(eps), int(save)))
        return out, xn, rstd, c1, ph(), ph(), ph()
    cl = _small_cl(d, k.n, tiles, x.device.index)
    if cl:
        xn = torch.empty_like(x) if save else x.new_empty((1, d))
        hid, a, b = ((torch.empty((m, h), dtype=x.dtype, device=x.device) if save else ph()) for _ in range(3))
        part = torch.empty((tiles * cl * 128, d), dtype=torch.float32, device=x.device)
        mx = k.tmap(x, d, m, 64, 128)
        k.run(f"small{cl}_ab" if save else f"small{cl}", "transition_small_w_sm100", tiles * cl,
              (mx, k.tmap(wa, d, h), k.tmap(wb, d, h), k.tmap(ws, h, d), k.tmap(xn, d, m, 64, 128) if save else mx,
               k.tmap(part, d, tiles * cl * 128, 32, 128), x, out, gamma, beta, rstd, c1, hid, a, b, float(eps), int(save)),
              block=512, cluster=cl)
        return out, xn, rstd, c1, hid, a, b
    if d == 384:                                               # one fused kernel (round n2, tfwd_d384.cu)
        xn = torch.empty_like(x) if save else x.new_empty((1, d))
        hid, a, b = ((torch.empty((m, h), dtype=x.dtype, device=x.device) if save else ph()) for _ in range(3))
        mout = k.rows(out)
        k.run("fwd", "transition_fwd_d384_sm100", k.grid,
              (k.rows(x), k.tmap(wa, d, h, 64, 32), k.tmap(wb, d, h, 64, 32), k.tmap(ws, h, d), k.rows(xn) if save else mout, mout,
               gamma, beta, rstd, c1, hid, a, b, tiles, float(eps), int(save)))
        return out, xn, rstd, c1, hid, a, b
    xn = torch.empty_like(x)                                   # D >= 512: the expand's A operand, always materialised
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
    for c0, nd in k.halves():
        k.run("squeeze", "transition_gemm_nd_sm100", k.grid,
              (mh, k.tmap(ws[c0:c0 + nd], h, nd), k.cols(x, c0, nd), k.cols(out, c0, nd), tiles))
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
    k = _Launch(x, wa.shape[0])
    d, h, m, tiles = k.d, k.h, k.m, k.tiles
    f32 = dict(dtype=torch.float32, device=x.device)
    dx = torch.empty_like(x)
    if d in (64, 128):                                         # the fused two-role backward
        ndw = _ndw(d, k.n)
        ndx = k.nsm - ndw
        partab = torch.empty((ndw, 128, d), **f32)
        parts = torch.empty((ndw, d, 64), **f32)
        dgbw = torch.empty((ndx * 4, 2 * d), **f32)
        dwa, dwb, dws = torch.empty_like(wa), torch.empty_like(wb), torch.empty_like(ws)
        dgam, dbeta = torch.empty((d,), **f32), torch.empty((d,), **f32)
        maps = (k.rows(dy), k.rows(xn), k.rows(x), k.tmap(ws, h, d), k.tmap(wa, d, h), k.tmap(wb, d, h), k.rows(dx))
        if d == 64:
            k.run("bwd", "transition_bwd_d64_sm100", k.nsm, (*maps, rstd, c1, gamma, partab, parts, dgbw, tiles, ndw))
            reduce = "transition_bwd_d64_reduce"
        else:
            k.run("bwd", "transition_bwd_sm100", k.nsm, (*maps, rstd, c1, gamma, x, partab, parts, dgbw, tiles, ndw))
            reduce = "transition_bwd_reduce"
        nred = 3 * h * d + 2 * d
        k.run("bwd", reduce, (nred + 255) // 256, (partab, parts, dgbw, dwa, dwb, dws, dgam, dbeta, ndw, ndx * 4), 256, 1)
        return dx, dgam, dbeta, dwa, dwb, dws
    small = d >= 384 and _small(tiles)
    if d == 384 and not small:                                 # gate + d_xn + LayerNorm backward in one kernel (round n2, tbwd_d384.cu)
        wst = ws.t().contiguous()                              # [H, D]
        dab = torch.empty((m, 2 * h), dtype=x.dtype, device=x.device)
        part = torch.empty((k.grid, 2 * d), **f32)
        dgb = torch.empty((2 * d,), **f32)
        k.run("bwd", "transition_bwd_d384_sm100", k.grid,
              (k.rows(dy), k.rows(x), k.tmap(wst, d, h, 64, 32), k.tmap(wa, d, h), k.tmap(wb, d, h), k.tmap(a, h, m), k.tmap(b, h, m),
               k.rows(dx), rstd, c1, gamma, dab, part, tiles))
        k.run("bwd", "transition_bwd_d384_reduce", 2 * d // 64, (part, dgb, k.grid), 256, 1)
        dws = _mm_f32(dy.t(), hid)                            # [D, H]
        dwab = _mm_f32(dab.t(), xn)                           # [2H, D]: dWa then dWb
        return dx, dgb[:d], dgb[d:].clone(), dwab[:h], dwab[h:].clone(), dws
    wst = ws.t().contiguous()                                  # [H, D]
    wab = torch.cat((wa, wb), 0)                               # [2H, D]
    dab = torch.empty((m, 2 * h), dtype=x.dtype, device=x.device)
    mdab = k.tmap(dab, 2 * h, m)
    if d == 256:
        hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
        k.run("gate", "transition_gate_w_sm100", k.grid,
              (k.rows(dy), k.rows(xn), k.tmap(wst, d, h, 64, 32), k.tmap(wa, d, h), k.tmap(wb, d, h), k.tmap(hid, h, m), mdab,
               tiles, 1))
    else:
        mh = k.tmap(hid, h, m)
        k.run("gate_is" if small else "gate", "transition_gate_ab_sm100", k.nsm if small else k.grid,
              (k.rows(dy), k.tmap(wst, d, h), k.tmap(a, h, m), k.tmap(b, h, m), mh, mdab, tiles, 0))
    dws = _mm_f32(dy.t(), hid)                                # [D, H]
    dwab = _mm_f32(dab.t(), xn)                               # [2H, D]: dWa then dWb
    dxn = torch.empty_like(x)
    wab_t = wab.t().contiguous()                               # [D, 2H]
    for c0, nd in k.halves():
        mo = k.cols(dxn, c0, nd)                               # (the x slot of the plain-epilogue GEMM is never read)
        k.run("dxn", "transition_gemm_nd_sm100", k.grid, (mdab, k.tmap(wab_t[c0:c0 + nd], 2 * h, nd), mo, mo, tiles))
    nb = min(m // 16, k.nsm * 4)
    lpart = torch.empty((nb, 2 * d), **f32)
    dgb = torch.empty((2 * d,), **f32)
    k.run("lnbwd", "transition_lnbwd_w", nb, (dxn, x, dy, rstd, c1, gamma, dx, lpart, m), 256, 1)
    k.run("lnbwd", "transition_lnbwd_w_reduce", (2 * d + 255) // 256, (lpart, dgb, nb), 256, 1)
    return dx, dgb[:d], dgb[d:].clone(), dwab[:h], dwab[h:].clone(), dws   # outputs of a custom op must not alias each other


class _WideTransitionSM100A(torch.autograd.Function):
    """``y = transition(x) + x`` for (D, n) in ``SHAPES``; the forward saves xn and the LayerNorm statistics (and h, a, b at
    D >= 384)."""

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

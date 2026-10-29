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
    384   at most 16 row tiles (a single-stream [B, L, 384] activation; round sx, ``SMALL_FWD_TILES``): LayerNorm -> expand + SwiGLU ->
          squeeze as three programmatic dependent launches sized for a few tiles on all SMs; backward gate -> d_xn + LayerNorm
          backward (one cluster per tile) -> weight gradients + dgamma / dbeta, three more

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

import torch

from miniworld_engine.kernels._device import num_sms

from ..._compile import opaque
from .fused_sm100a import _ext, _is_b200, build_cubins
from miniworld_engine import settings

#: (D, n) with a build here; (128, 4) is ``fused_sm100a``. Each builds its own cubins on first use.
SHAPES = (
    (64, 4),
    (256, 4),
    (384, 4),
    (512, 4),
    (64, 2),
    (128, 2),
    (256, 2),
    (384, 2),
    (512, 2),
    (768, 2),
)
#: Channel widths with a build for some n (kept for callers that ask by width only).
WIDTHS = tuple(sorted({d for d, _ in SHAPES}))
#: Row tile of every persistent grid here. ``M`` must be a whole number of these.
ROWS = 128
#: Two-role backward (D = 64, and D = 128 at n = 2): replicas of each 64-unit hidden slice in the weight role; the other CTAs run
#: the input role. (64, 4) = 20 measured fastest at L384 and L768 on a 148-SM B200 (w1); the n = 2 values are from round n2.
#: settings.transition_wide_repl overrides them for A/B runs.
REPL = {(64, 4): 20, (64, 2): 30, (128, 2): 14}
#: D >= 384: the expand + SwiGLU deals (tile pair, chunk) items over all SMs up to this many tiles, whole tile pairs above.
ITEM_TILES = 1152
#: Column width of one squeeze / d_xn GEMM launch at D = 768 (two halves; one launch's accumulator must fit 512 TMEM columns).
HALF = 384
#: D >= 384 at small M (at most this many 128-row tiles, e.g. a single-stream [B, L, D] activation): whole tile pairs would keep
#: only tiles / 2 cluster pairs busy, so the backward gate deals its (tile pair, 256-unit block) items over all SMs. Round n2.
#: settings.transition_wide_small_tiles overrides it for A/B runs.
SMALL_TILES = 16


def _small(tiles: int) -> bool:
    return tiles <= (settings.current().transition_wide_small_tiles or SMALL_TILES)


#: D = 384 at small M: up to SMALL_FWD_TILES tiles both directions take the round-sx kernels below (settings.
#: transition_wide_small_fwd_tiles overrides the bound for A/B runs).
SMALL_FWD_TILES = 16


#: D = 384 at small M (at most SMALL_FWD_TILES tiles; a single-stream [B, L, 384] activation): chains of programmatic dependent
#: launches sized for a few row tiles on all 148 SMs (sm100/widths/tsx_*.cu, round sx). Forward: LayerNorm (one warp per row) ->
#: expand + SwiGLU over (tile, CH-unit hidden chunk) CTAs -> squeeze + residual over (tile, column block, hidden split) clusters.
#: Backward: gate over (tile, 64-unit chunk) CTAs -> d_xn + LayerNorm backward over one cluster per tile (6 column blocks x a K
#: split) -> dWs, dWa | dWb (fp32, or bf16 for bf16 parameters) and dgamma / dbeta over (output tile) CTAs. Every operand is read in
#: its own layout (MN-major where needed): no transposed or concatenated weight copies per call; with fp32 masters the training
#: forward's casts ride in the LayerNorm launch.
SX_CH = (32, 64)
#: squeeze builds: (output columns per cluster, hidden split = cluster size)
SX_SQ = ((32, 1), (64, 1), (32, 2), (64, 2))


def _sx(d: int, tiles: int) -> bool:
    return d == 384 and tiles <= (settings.current().transition_wide_small_fwd_tiles or SMALL_FWD_TILES)


def _sx_ch(h: int, tiles: int, nsm: int) -> int:
    """Hidden units per expand CTA: the narrowest chunk whose CTAs fit one wave."""
    return next((c for c in SX_CH if tiles * (h // c) <= nsm), SX_CH[-1])


#: (columns, split) of the squeeze, None: automatic (kept for A/B runs)
SX_SQ_FORCE = None


def _sx_sq(d: int, tiles: int, h: int, nsm: int) -> tuple[int, int]:
    """Squeeze decomposition (output columns per cluster, hidden split = cluster size). The per-SM TMA intake of the h tile bounds this
    kernel, so the h rows are split over a CTA pair whose fp32 partials meet in distributed shared memory while the pairs fit one wave
    (2-CTA clusters pack 74 to a wave on a 148-SM B200), and 32-column blocks keep every SM busy; round sx, measured at
    [1, 768 / 384, 384] (n = 2: 4.1 vs 4.5 us unsplit; n = 4: 5.8 vs 7.4)."""
    if SX_SQ_FORCE is not None:
        return SX_SQ_FORCE
    if tiles * (d // 32) * 2 <= nsm:
        return 32, 2
    if tiles * (d // 64) * 2 <= nsm:
        return (64, 2) if h >= 4 * d else (32, 1)
    return 64, 1


@functools.lru_cache(maxsize=16)
def _sx_dxn_resident(d: int, n: int, index: int) -> int:
    """Resident 12-CTA clusters of the K-split d_xn kernel (7 on a 148-SM B200: clusters must fit a GPC)."""
    return int(_load(d, n).max_active_clusters(f"d{d}n{n}_sx_dxn2", "transition_sx_dxn", 384, 2 * (d // 64)))


def _sx_ks(d: int, n: int, tiles: int, index: int) -> int:
    """K split of the d_xn kernel: the two halves (dA Wa, dB Wb) on a CTA pair while every tile's cluster is resident at once, which
    halves the per-SM TMA intake that bounds it; one CTA per column block otherwise."""
    return 2 if tiles <= _sx_dxn_resident(d, n, index) else 1


def _dn(x: torch.Tensor, wa: torch.Tensor) -> tuple[int, int]:
    d = x.shape[-1]
    return d, wa.shape[0] // d


def _repl(d: int, n: int) -> int:
    return settings.current().transition_wide_repl or REPL[(d, n)]


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
    dxn = (
        "dxn",
        "widths/tgemm_nd.cu",
        (f"-DDIM={nd}", f"-DKDIM={2 * h}", "-DEPI_PLAIN"),
    )
    if d == 256:
        return (
            ("fwd", "widths/tfwd_d256.cu", hid),
            ("gate", "widths/tgate_w.cu", ("-DDIM=256", *hid)),
            dxn,
            lnb,
        )
    dim = (f"-DDIM={d}", *hid)
    fwd = (
        (("fwd", "widths/tfwd_d384.cu", hid), ("bwd", "widths/tbwd_d384.cu", hid))
        if d == 384
        else (
            ("ln", "widths/tln_w.cu", (f"-DDIM={d}", lpr)),
            ("swiglu", "widths/tswiglu_w.cu", dim),
            ("swiglu_is", "widths/tswiglu_w.cu", (*dim, "-DITEM_SCHED")),
            ("swiglu_ab", "widths/tswiglu_w.cu", (*dim, "-DSAVE_AB")),
            ("swiglu_abis", "widths/tswiglu_w.cu", (*dim, "-DSAVE_AB", "-DITEM_SCHED")),
            ("squeeze", "widths/tgemm_nd.cu", (f"-DDIM={nd}", f"-DKDIM={h}")),
        )
    )
    sx = (
        (
            ("sx_ln", "widths/tsx_ln.cu", (f"-DDIM={d}",)),
            *((f"sx_expand{c}", "widths/tsx_expand.cu", (*dim, f"-DCH={c}")) for c in SX_CH),
            *(
                (f"sx_sq{nb}_{k}", "widths/tsx_squeeze.cu", (*dim, f"-DNBW={nb}", f"-DSK={k}"))
                for nb, k in SX_SQ
            ),
            ("sx_gate", "widths/tsx_gate.cu", dim),
            *((f"sx_dxn{k}", "widths/tsx_dxn.cu", (*dim, f"-DKS={k}")) for k in (1, 2)),
            *((f"sx_dw{nt}", "widths/tsx_dw.cu", (*dim, f"-DNT={nt}")) for nt in (64, 128)),
        )
        if d == 384
        else ()
    )
    return (
        *fwd,
        *sx,
        ("gate", "widths/tgate_ab.cu", (*dim, "-DNO_H")),
        ("gate_is", "widths/tgate_ab.cu", (*dim, "-DNO_H", "-DITEM_SCHED")),
        dxn,
        lnb,
    )


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
    if not settings.current().transition_fused_sm100a:
        return False
    if not x.is_cuda or x.dtype is not torch.bfloat16:
        return False
    index = (
        x.device.index if x.device.index is not None else torch.cuda.current_device()
    )
    if not _is_b200(index):
        return False
    d, n = _dn(x, wa)
    if (d, n) not in SHAPES or wa.shape != (n * d, d) or ws.shape != (d, n * d):
        return False
    if wa.dtype not in (torch.bfloat16, torch.float32) or ws.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        return False  # bf16, or an fp32 master the entry casts to bf16 outside autograd
    if (d, n) in REPL:
        ndx = num_sms(index) - _ndw(d, n)
        if ndx < 2 or ndx % 2:
            return False
    rows = x.numel() // d
    return rows > 0 and rows % ROWS == 0


_BUILD_FAILED: set[tuple[int, int]] = set()


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful build of this (width, n), both cached; a build failure raises."""
    if not supported(x, wa, ws) or _dn(x, wa) in _BUILD_FAILED:
        return False
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _load(*_dn(x, wa))
    except Exception as exc:  # B200 has no engine Triton / silent PyTorch fallback: a build or load failure is an error
        raise RuntimeError(f"wide sm100a Transition kernels failed to build or load on B200: {exc!r}") from exc
    return True


def _is_fake(*tensors) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor

    return any(isinstance(t, FakeTensor) for t in tensors)




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
        self.nsm = num_sms(x.device.index)
        # persistent 2-CTA clusters, a pair walks the leader's tile count; rounded UP to the pair (an odd tile count below the SM
        # count gets a pair with one dummy CTA rather than a second round for one pair)
        self.grid = max(2, min(self.nsm - self.nsm % 2, self.tiles + self.tiles % 2))
        self.ext = _load(self.d, self.n)

    def tmap(self, t, inner, outer, box_inner=64, box_outer=64, row_stride=-1):
        return self.ext.tmap(t, inner, outer, box_inner, box_outer, row_stride)

    def run(self, cubin, kernel, grid, args, block=512, cluster=2, pdl=False):
        self.ext.launch_kernel(
            f"d{self.d}n{self.n}_{cubin}", kernel, grid, block, cluster, list(args), pdl
        )

    def rows(self, t):  # [M, D] activation map
        return self.tmap(t, self.d, self.m)

    def halves(self):
        """(first column, width) of each squeeze / d_xn GEMM launch."""
        return (
            [(c, HALF) for c in range(0, self.d, HALF)]
            if self.d == 768
            else [(0, self.d)]
        )

    def cols(self, t, c0, nd):  # [M, nd] column slice of an [M, D] activation
        return self.tmap(t[:, c0 : c0 + nd], nd, self.m, row_stride=self.d)


_FUSED_FWD = {
    64: "transition_fwd_d64_sm100",
    128: "transition_fwd2_sm100",
    256: "transition_fwd_d256_sm100",
}


def _sx_forward(k, x, gamma, beta, wa, wb, ws, eps, save, out, rstd, c1, cast=None):
    """The round-sx forward (tsx_ln -> tsx_expand -> tsx_squeeze, programmatic dependent launches) into ``out`` (and ``rstd`` / ``c1``
    with save). ``cast``: the fp32 masters whose bf16 copies ``wa`` / ``wb`` / ``ws`` the LayerNorm launch writes first. Returns
    (out, xn, rstd, c1, h, a, b) as ``_fwd_launch``."""
    d, h, m, tiles = k.d, k.h, k.m, k.tiles

    def ph():
        return x.new_empty((1, 1))

    xn = torch.empty_like(x)  # the expand's A operand, kept for the backward with save
    hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
    a, b = ((torch.empty((m, h), dtype=x.dtype, device=x.device) if save else ph()) for _ in range(2))
    nrb = (m + 7) // 8
    k.run(
        "sx_ln",
        "transition_sx_ln",
        nrb + ((3 * h * d // 8 + 511) // 512 if cast is not None else 0),  # cast blocks: 2 x 8 elements per thread
        (x, gamma, beta, xn, rstd, c1, m, float(eps), int(save), *(cast or (x, x, x)), wa, wb, ws, h * d if cast else 0),
        block=256,
        cluster=1,
        pdl=True,
    )
    ch = _sx_ch(h, tiles, k.nsm)
    k.run(
        f"sx_expand{ch}",
        "transition_sx_expand",
        tiles * (h // ch),
        (k.tmap(xn, d, m, 64, 128), k.tmap(wa, d, h, 64, ch), k.tmap(wb, d, h, 64, ch), hid, a, b, int(save)),
        block=384,
        cluster=1,
        pdl=True,
    )
    nb, sk = _sx_sq(d, tiles, h, k.nsm)
    k.run(
        f"sx_sq{nb}_{sk}",
        "transition_sx_squeeze",
        tiles * (d // nb) * sk,
        (k.tmap(hid, h, m, 64, 128), k.tmap(ws, h, d, 64, nb), x, out),
        block=256,
        cluster=sk,
        pdl=True,
    )
    if not save:
        return out, x.new_empty((1, d)), rstd, c1, ph(), ph(), ph()
    return out, xn, rstd, c1, hid, a, b


def _fwd_sx_cast_fake(x, gamma, beta, wa, wb, ws, eps):
    m, d = x.shape
    f32 = dict(dtype=torch.float32, device=x.device)
    bf = dict(dtype=torch.bfloat16, device=x.device)
    return (
        torch.empty_like(x),
        torch.empty_like(x),
        torch.empty((m,), **f32),
        torch.empty((m,), **f32),
        *(x.new_empty((m, wa.shape[0])) for _ in range(3)),
        torch.empty(wa.shape, **bf),
        torch.empty(wb.shape, **bf),
        torch.empty(ws.shape, **bf),
    )


@opaque(fake=_fwd_sx_cast_fake, name="transition_wide_fwd_sx_cast_sm100a")
def _fwd_sx_cast_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    eps: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """The training forward of the round-sx path from fp32 master matrices: the LayerNorm launch also writes their bf16 copies (no
    cast launch of its own). Returns (out, xn, rstd, c1, h, a, b, wa16, wb16, ws16)."""
    if _is_fake(x, wa):
        return _fwd_sx_cast_fake(x, gamma, beta, wa, wb, ws, eps)
    k = _Launch(x, wa.shape[0])
    f32 = dict(dtype=torch.float32, device=x.device)
    w16 = [torch.empty(w.shape, dtype=torch.bfloat16, device=w.device) for w in (wa, wb, ws)]
    out = torch.empty_like(x)
    rstd, c1 = torch.empty((k.m,), **f32), torch.empty((k.m,), **f32)
    res = _sx_forward(k, x, gamma, beta, *w16, eps, True, out, rstd, c1, cast=(wa, wb, ws))
    return (*res, *w16)


def _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save):
    """Output structure only: (out, xn, rstd, c1, h, a, b). xn / rstd / c1 are 1-element placeholders unless ``save``; h / a / b
    are [M, nD] when ``save`` at D >= 384 (kept for the backward) and placeholders otherwise."""
    m, d = x.shape
    rows = m if save else 1
    f32 = dict(dtype=torch.float32, device=x.device)
    keep = save and d >= 384
    hab = [
        x.new_empty((m, wa.shape[0])) if keep else x.new_empty((1, 1)) for _ in range(3)
    ]
    return (
        torch.empty_like(x),
        torch.empty_like(x) if save else x.new_empty((1, d)),
        torch.empty((rows,), **f32),
        torch.empty((rows,), **f32),
        *hab,
    )


@opaque(fake=_fwd_launch_fake, name="transition_wide_fwd_sm100a")
def _fwd_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    eps: float,
    save: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """x [M, D] bf16, gamma / beta fp32. Returns (out, xn, rstd, c1, h, a, b), out = transition(x) + x; see the fake."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save)
    k = _Launch(x, wa.shape[0])
    d, h, m, tiles = k.d, k.h, k.m, k.tiles
    f32 = dict(dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    rstd = torch.empty((m if save else 1,), **f32)
    c1 = torch.empty_like(rstd)

    def ph():  # distinct placeholders: custom-op outputs must not alias each other
        return x.new_empty((1, 1))

    if d in _FUSED_FWD:
        xn = torch.empty_like(x) if save else x.new_empty((1, d))
        mout = k.rows(out)
        k.run(
            "fwd",
            _FUSED_FWD[d],
            k.grid,
            (
                k.rows(x),
                k.tmap(wa, d, h),
                k.tmap(wb, d, h),
                k.tmap(ws, h, d, 64, d // 2),
                mout,
                k.rows(xn) if save else mout,
                gamma,
                beta,
                rstd,
                c1,
                tiles,
                float(eps),
                int(save),
            ),
        )
        return out, xn, rstd, c1, ph(), ph(), ph()
    if _sx(d, tiles):  # three programmatic-dependent kernels sized for a few tiles (round sx: tsx_ln, tsx_expand, tsx_squeeze)
        return _sx_forward(k, x, gamma, beta, wa, wb, ws, eps, save, out, rstd, c1)
    if d == 384:  # one fused kernel (round n2, tfwd_d384.cu)
        xn = torch.empty_like(x) if save else x.new_empty((1, d))
        hid, a, b = (
            (torch.empty((m, h), dtype=x.dtype, device=x.device) if save else ph())
            for _ in range(3)
        )
        mout = k.rows(out)
        k.run(
            "fwd",
            "transition_fwd_d384_sm100",
            k.grid,
            (
                k.rows(x),
                k.tmap(wa, d, h, 64, 32),
                k.tmap(wb, d, h, 64, 32),
                k.tmap(ws, h, d),
                k.rows(xn) if save else mout,
                mout,
                gamma,
                beta,
                rstd,
                c1,
                hid,
                a,
                b,
                tiles,
                float(eps),
                int(save),
            ),
        )
        return out, xn, rstd, c1, hid, a, b
    xn = torch.empty_like(x)  # D >= 512: the expand's A operand, always materialised
    hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
    k.run(
        "ln",
        "transition_ln_w",
        min(m // 16, k.nsm * 4),
        (x, gamma, beta, xn, rstd, c1, m, float(eps), int(save)),
        256,
        1,
    )
    item = tiles <= ITEM_TILES
    mh = k.tmap(hid, h, m)
    maps = (k.rows(xn), k.tmap(wa, d, h, 64, 128), k.tmap(wb, d, h, 64, 128), mh)
    if save:
        a, b = torch.empty_like(hid), torch.empty_like(hid)
        k.run(
            "swiglu_abis" if item else "swiglu_ab",
            "transition_swiglu_w_sm100",
            k.nsm if item else k.grid,
            (*maps, k.tmap(a, h, m), k.tmap(b, h, m), 1, tiles),
        )
    else:
        k.run(
            "swiglu_is" if item else "swiglu",
            "transition_swiglu_w_sm100",
            k.nsm if item else k.grid,
            (*maps, tiles),
        )
    for c0, nd in k.halves():
        k.run(
            "squeeze",
            "transition_gemm_nd_sm100",
            k.grid,
            (
                mh,
                k.tmap(ws[c0 : c0 + nd], h, nd),
                k.cols(x, c0, nd),
                k.cols(out, c0, nd),
                tiles,
            ),
        )
    if not save:
        return out, xn.new_empty((1, d)), rstd, c1, ph(), ph(), ph()
    return out, xn, rstd, c1, hid, a, b


def _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hid, a, b):
    """Output structure only: (dx, dgamma, dbeta, dWa, dWb, dWs); the weight gradients are fp32."""
    wdt = torch.float32
    return (
        torch.empty_like(x),
        torch.empty_like(gamma),
        torch.empty_like(gamma),
        torch.empty(wa.shape, dtype=wdt, device=wa.device),
        torch.empty(wb.shape, dtype=wdt, device=wb.device),
        torch.empty(ws.shape, dtype=wdt, device=ws.device),
    )


@opaque(fake=_bwd_launch_fake, name="transition_wide_bwd_sm100a")
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
    hid: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Returns (dx, dgamma, dbeta, dWa, dWb, dWs); ``dx`` already carries the residual branch. ``hid`` / ``a`` / ``b`` are the
    forward's h / a / b at D >= 384 and placeholders otherwise."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hid, a, b)
    k = _Launch(x, wa.shape[0])
    d, h, m, tiles = k.d, k.h, k.m, k.tiles
    f32 = dict(dtype=torch.float32, device=x.device)
    dx = torch.empty_like(x)
    if d in (64, 128):  # the fused two-role backward
        ndw = _ndw(d, k.n)
        ndx = k.nsm - ndw
        partab = torch.empty((ndw, 128, d), **f32)
        parts = torch.empty((ndw, d, 64), **f32)
        dgbw = torch.empty((ndx * 4, 2 * d), **f32)
        dwa, dwb, dws = (
            torch.empty(wa.shape, **f32),
            torch.empty(wb.shape, **f32),
            torch.empty(ws.shape, **f32),
        )  # unrounded
        dgam, dbeta = torch.empty((d,), **f32), torch.empty((d,), **f32)
        maps = (
            k.rows(dy),
            k.rows(xn),
            k.rows(x),
            k.tmap(ws, h, d),
            k.tmap(wa, d, h),
            k.tmap(wb, d, h),
            k.rows(dx),
        )
        if d == 64:
            k.run(
                "bwd",
                "transition_bwd_d64_sm100",
                k.nsm,
                (*maps, rstd, c1, gamma, partab, parts, dgbw, tiles, ndw),
            )
            reduce = "transition_bwd_d64_reduce"
        else:
            k.run(
                "bwd",
                "transition_bwd_sm100",
                k.nsm,
                (*maps, rstd, c1, gamma, x, partab, parts, dgbw, tiles, ndw),
            )
            reduce = "transition_bwd_reduce"
        nred = 3 * h * d + 2 * d
        k.run(
            "bwd",
            reduce,
            (nred + 255) // 256,
            (partab, parts, dgbw, dwa, dwb, dws, dgam, dbeta, ndw, ndx * 4),
            256,
            1,
        )
        return dx, dgam, dbeta, dwa, dwb, dws
    small = d >= 384 and _small(tiles)
    if (
        d == 384 and not small
    ):  # gate + d_xn + LayerNorm backward in one kernel (round n2, tbwd_d384.cu)
        wst = ws.t().contiguous()  # [H, D]
        dab = torch.empty((m, 2 * h), dtype=x.dtype, device=x.device)
        part = torch.empty((k.grid, 2 * d), **f32)
        dgb = torch.empty((2 * d,), **f32)
        k.run(
            "bwd",
            "transition_bwd_d384_sm100",
            k.grid,
            (
                k.rows(dy),
                k.rows(x),
                k.tmap(wst, d, h, 64, 32),
                k.tmap(wa, d, h),
                k.tmap(wb, d, h),
                k.tmap(a, h, m),
                k.tmap(b, h, m),
                k.rows(dx),
                rstd,
                c1,
                gamma,
                dab,
                part,
                tiles,
            ),
        )
        k.run(
            "bwd",
            "transition_bwd_d384_reduce",
            2 * d // 64,
            (part, dgb, k.grid),
            256,
            1,
        )
        dws = _mm_f32(dy.t(), hid)  # [D, H]
        dwab = _mm_f32(dab.t(), xn)  # [2H, D]: dWa then dWb
        return dx, dgb[:d], dgb[d:].clone(), dwab[:h], dwab[h:].clone(), dws
    wst = ws.t().contiguous()  # [H, D]
    wab = torch.cat((wa, wb), 0)  # [2H, D]
    dab = torch.empty((m, 2 * h), dtype=x.dtype, device=x.device)
    mdab = k.tmap(dab, 2 * h, m)
    if d == 256:
        hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
        k.run(
            "gate",
            "transition_gate_w_sm100",
            k.grid,
            (
                k.rows(dy),
                k.rows(xn),
                k.tmap(wst, d, h, 64, 32),
                k.tmap(wa, d, h),
                k.tmap(wb, d, h),
                k.tmap(hid, h, m),
                mdab,
                tiles,
                1,
            ),
        )
    else:
        mh = k.tmap(hid, h, m)
        k.run(
            "gate_is" if small else "gate",
            "transition_gate_ab_sm100",
            k.nsm if small else k.grid,
            (
                k.rows(dy),
                k.tmap(wst, d, h),
                k.tmap(a, h, m),
                k.tmap(b, h, m),
                mh,
                mdab,
                tiles,
                0,
            ),
        )
    dws = _mm_f32(dy.t(), hid)  # [D, H]
    dwab = _mm_f32(dab.t(), xn)  # [2H, D]: dWa then dWb
    dxn = torch.empty_like(x)
    wab_t = wab.t().contiguous()  # [D, 2H]
    for c0, nd in k.halves():
        mo = k.cols(
            dxn, c0, nd
        )  # (the x slot of the plain-epilogue GEMM is never read)
        k.run(
            "dxn",
            "transition_gemm_nd_sm100",
            k.grid,
            (mdab, k.tmap(wab_t[c0 : c0 + nd], 2 * h, nd), mo, mo, tiles),
        )
    nb = min(m // 16, k.nsm * 4)
    lpart = torch.empty((nb, 2 * d), **f32)
    dgb = torch.empty((2 * d,), **f32)
    k.run(
        "lnbwd",
        "transition_lnbwd_w",
        nb,
        (dxn, x, dy, rstd, c1, gamma, dx, lpart, m),
        256,
        1,
    )
    k.run(
        "lnbwd",
        "transition_lnbwd_w_reduce",
        (2 * d + 255) // 256,
        (lpart, dgb, nb),
        256,
        1,
    )
    return (
        dx,
        dgb[:d],
        dgb[d:].clone(),
        dwab[:h],
        dwab[h:].clone(),
        dws,
    )  # outputs of a custom op must not alias each other


def _bwd_sx_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hid, a, b, wgrad_bf16):
    """Output structure only: (dx, dgamma | dbeta [2 D] fp32, dWa | dWb [2 H, D], dWs [D, H]); the weight gradients fp32, or bf16 with
    ``wgrad_bf16`` (bf16 parameters)."""
    f32 = dict(dtype=torch.float32, device=x.device)
    wd = dict(dtype=torch.bfloat16 if wgrad_bf16 else torch.float32, device=x.device)
    return (
        torch.empty_like(x),
        torch.empty((2 * x.shape[1],), **f32),
        torch.empty((2 * wa.shape[0], wa.shape[1]), **wd),
        torch.empty(ws.shape, **wd),
    )


@opaque(fake=_bwd_sx_fake, name="transition_wide_bwd_sx_sm100a")
def _bwd_sx_launch(
    dy: torch.Tensor,
    x: torch.Tensor,
    xn: torch.Tensor,
    rstd: torch.Tensor,
    c1: torch.Tensor,
    gamma: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    hid: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    wgrad_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The small-M backward at D = 384 (round sx), three programmatic dependent launches: gate (tsx_gate.cu) -> d_xn + LayerNorm
    backward + dgamma / dbeta partials (tsx_dxn.cu) -> dWs, dWa | dWb (fp32) and dgamma | dbeta (tsx_dw.cu). Returns (dx, dgamma | dbeta,
    dWa | dWb, dWs); the caller splits the packed ones (outputs of a custom op must not alias each other)."""
    if _is_fake(dy, x):
        return _bwd_sx_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hid, a, b, wgrad_bf16)
    k = _Launch(x, wa.shape[0])
    d, h, m, tiles = k.d, k.h, k.m, k.tiles
    f32 = dict(dtype=torch.float32, device=x.device)
    dx = torch.empty_like(x)
    dab = torch.empty((m, 2 * h), dtype=x.dtype, device=x.device)
    k.run(
        "sx_gate",
        "transition_sx_gate",
        tiles * (h // 64),
        (k.tmap(dy, d, m, 64, 128), k.tmap(ws, h, d), a, b, dab),
        block=384,
        cluster=1,
        pdl=True,
    )
    part = torch.empty((tiles, 2 * d), **f32)
    ks = _sx_ks(d, k.n, tiles, x.device.index)
    k.run(
        f"sx_dxn{ks}",
        "transition_sx_dxn",
        tiles * (d // 64) * ks,
        (
            k.tmap(dab, 2 * h, m, 64, 128),
            k.tmap(wa, d, h),
            k.tmap(wb, d, h),
            x,
            dy,
            rstd,
            c1,
            gamma,
            dx,
            part,
        ),
        block=384,
        cluster=(d // 64) * ks,
        pdl=True,
    )
    dgb = torch.empty((2 * d,), **f32)
    wd = dict(dtype=torch.bfloat16 if wgrad_bf16 else torch.float32, device=x.device)
    dwab = torch.empty((2 * h, d), **wd)  # dWa then dWb
    dws = torch.empty((d, h), **wd)
    nt = 64 if (d // 128) * (h // 64) + (2 * h // 128) * (d // 64) <= k.nsm else 128
    k.run(
        f"sx_dw{nt}",
        "transition_sx_dw",
        (d // 128) * (h // nt) + (2 * h // 128) * (d // nt),
        (
            k.tmap(dy, d, m, 64, 64),
            k.tmap(hid, h, m, 64, 64),
            k.tmap(dab, 2 * h, m, 64, 64),
            k.tmap(xn, d, m, 64, 64),
            dws,
            dwab,
            part,
            dgb,
            tiles,
            int(wgrad_bf16),
        ),
        block=256,
        cluster=1,
        pdl=True,
    )
    return dx, dgb, dwab, dws


def _bf16_forms(wa, wb, ws, dtype):
    """The kernels' bf16 copies of the matrices: one launch for all three (``_cast_cuda``, bit-identical to ``.to``) where it can run,
    else ``.to``; a bf16 contiguous parameter is used as it is."""
    if (
        dtype is torch.bfloat16
        and not torch.compiler.is_compiling()
        and not _is_fake(wa, wb, ws)
        and all(w.is_contiguous() for w in (wa, wb, ws))
    ):
        from miniworld_engine.kernels import _cast_cuda

        return tuple(_cast_cuda.cast((wa, wb, ws), [dtype] * 3))
    return tuple(w.to(dtype).contiguous() for w in (wa, wb, ws))


class _WideTransitionSM100A(torch.autograd.Function):
    """``y = transition(x) + x`` for (D, n) in ``SHAPES``; the forward saves xn and the LayerNorm statistics (and h, a, b at
    D >= 384). The parameters keep their dtype (bf16, or an fp32 master): the kernels get bf16 casts made here, outside autograd,
    and the parameters get the fp32 gradients in their own dtype (fp32 ones unrounded)."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        flat = x.reshape(-1, shape[-1]).contiguous()
        gf, bf = gamma.float().contiguous(), beta.float().contiguous()
        if (
            _sx(flat.shape[1], flat.shape[0] // ROWS)
            and x.dtype is torch.bfloat16
            and all(w.dtype is torch.float32 and w.is_contiguous() for w in (wa, wb, ws))
        ):  # fp32 masters at small M: the LayerNorm launch writes the bf16 copies
            out, xn, rstd, c1, hid, a, b, wa, wb, ws = _fwd_sx_cast_launch(flat, gf, bf, wa, wb, ws, float(eps))
        else:
            wa, wb, ws = _bf16_forms(wa, wb, ws, x.dtype)
            out, xn, rstd, c1, hid, a, b = _fwd_launch(
                flat, gf, bf, wa, wb, ws, float(eps), True
            )
        ctx.save_for_backward(flat, xn, rstd, c1, gf, wa, wb, ws, hid, a, b)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, rstd, c1, gf, wa, wb, ws, hid, a, b = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        d, h = flat.shape[1], wa.shape[0]
        if _sx(d, flat.shape[0] // ROWS):
            dx, dgb, dwab, dws = _bwd_sx_launch(
                dy.reshape(-1, d).contiguous(), flat, xn, rstd, c1, gf, wa, wb, ws, hid, a, b,
                adt is torch.bfloat16 and bwdt is torch.bfloat16 and sdt is torch.bfloat16,
            )
            return (
                dx.reshape(ctx.shape),
                dgb[:d].to(gdt),
                dgb[d:].to(bdt),
                dwab[:h].to(adt),
                dwab[h:].to(bwdt),
                dws.to(sdt),
                None,
            )
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
            hid,
            a,
            b,
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


def transition_wide_sm100a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as ``fused_sm100a.transition_fused_sm100a``. Call ``available()`` first: this raises
    rather than falling back. Inference (grad mode off, or nothing requiring grad) runs the forward without the saves."""
    if not (
        torch.is_grad_enabled()
        and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))
    ):
        shape = x.shape
        from miniworld_engine.kernels import _capture

        g, b, ww = _capture.param_forms(
            f"transition_wide_sm100a_{x.dtype}",
            (gamma, beta, wa, wb, ws),
            lambda: (
                gamma.float().contiguous(),
                beta.float().contiguous(),
                _bf16_forms(wa, wb, ws, x.dtype),
            ),
        )
        out, *_ = _fwd_launch(
            x.reshape(-1, shape[-1]).contiguous(), g, b, *ww, float(eps), False
        )
        return out.reshape(shape)
    return _WideTransitionSM100A.apply(x, gamma, beta, wa, wb, ws, eps)

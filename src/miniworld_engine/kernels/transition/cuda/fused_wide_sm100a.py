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
from ..._nvcc import build_failed
from .fused_sm100a import _ext, _is_b200, build_cubins


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
#: Module constants (A/B runs edit them).
REPL = {(64, 4): 19, (64, 2): 30, (128, 2): 14}
#: (S90, 2026-11-18) {(D, n): ((first row-tile count, replicas), ...)}: more replicas at many tiles, where the role balance shifts (the counts above were
#: tuned at 128 tiles, [16 384, D]). Step (graph, L2 flushed, us) at [65 536 / 147 456 / 262 144 / 409 600 / 589 824, 128] n 2, 14 -> new:
#: 98.1 / 186.4 / 309.2 / 475.0 / 673.8 -> 96.1 (15) / 182.3 (15) / 292.9 (16) / 436.2 (16) / 616.2 (16); (64, 4) 20 -> 19 at [16 384 .. 589 824, 64]:
#: 32.7 = 32.7, 65.5 / 114.7 / 194.5 / 395.3 -> 63.5 / 112.6 / 190.5 / 385.1. REPL_TILES = {}: the counts of REPL only (A/B).
REPL_TILES: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {(128, 2): ((129, 15), (2048, 16)), (64, 2): ((129, 34),)}
#: (S90) build flags of the D 64 two-role backward (tbwd_d64.cu) by n: XSEP the DX input stage carries x (no reload into the dead xn slot behind the
#: producer's in-order issue), EPI2 the gate on one warpgroup and two LayerNorm-backward epilogues on alternate tiles, DXN2 the d_xn accumulator
#: double-buffered, TB3 three tile buffers (n 2 only: the shared memory of n 4 holds two). Measured in docs/b200/transition.md (S90 part 5).
D64_BWD_FLAGS: dict[int, tuple[str, ...]] = {2: ("-DEPI2", "-DDXN2", "-DXSEP", "-DTB3"), 4: ("-DXSEP",)}
#: (round 3, training part 4) the two-role backwards (D 64, D 128 n 2) and the D 128 n 2 reduce launched as programmatic dependents: the
#: backward's prologue (barriers, tensor memory, cluster sync) runs as the forward's CTAs leave, griddepcontrol.wait before its first
#: read. A module constant (A/B runs edit it); D 128 n 4 (`transition_sm100.cu`) always.
BWD_PDL = True
#: (training part 5) A/B builds of the two-role backwards (D 64 `widths/tbwd_d64.cu`, D 128 n 2 `tr_bwd_sm100.cu`): None runs "bwd"; a flag
#: tuple (e.g. ("-DGOV=0",)) builds and runs cubin "bwdab" with those flags. A module constant (A/B runs edit it).
BWD_AB_FLAGS: tuple[str, ...] | None = None
#: D >= 384: the expand + SwiGLU deals (tile pair, chunk) items over all SMs up to this many tiles, whole tile pairs above.
ITEM_TILES = 1152
#: Column width of one squeeze / d_xn GEMM launch at D = 768 (two halves; one launch's accumulator must fit 512 TMEM columns).
HALF = 384
#: D >= 384 at small M (at most this many 128-row tiles, e.g. a single-stream [B, L, D] activation): whole tile pairs would keep
#: only tiles / 2 cluster pairs busy, so the backward gate deals its (tile pair, 256-unit block) items over all SMs. Round n2.
#: A module constant (A/B runs edit it).
SMALL_TILES = 16


def _small(tiles: int) -> bool:
    return tiles <= SMALL_TILES


#: D = 384 at small M: up to SMALL_FWD_TILES tiles both directions take the round-sx kernels below (a module constant).
SMALL_FWD_TILES = 16


#: D = 384 at small M (at most SMALL_FWD_TILES tiles; a single-stream [B, L, 384] activation): chains of programmatic dependent
#: launches sized for a few row tiles on all 148 SMs (sm100/widths/tsx_*.cu, round sx). Forward: LayerNorm (one warp per row) ->
#: expand + SwiGLU over (tile, CH-unit hidden chunk) CTAs -> squeeze + residual over (tile, column block, hidden split) clusters.
#: Backward: gate + d_xn + LayerNorm backward on one cluster per tile (12 CTAs: the gate's 64-unit chunks, then 6 column blocks x
#: the dA Wa / dB Wb halves; round sx3) -- above the resident cluster count the gate over (tile, 64-unit chunk) CTAs and d_xn as a
#: second launch -- and dWs, dWa | dWb (fp32, or bf16 for bf16 parameters) and dgamma / dbeta over (output tile) CTAs. Every operand is read in
#: its own layout (MN-major where needed): no transposed or concatenated weight copies per call; with fp32 masters the training
#: forward's casts ride in the LayerNorm launch.
#: (S90 part 9) extra build flags of the small-M chain kernels by name (gdx): A/B runs edit it
SX_X: dict[str, tuple[str, ...]] = {}
SX_ST: dict[str, torch.Tensor] = {}  # (N1 probe) stamp buffers by kernel name when SX_X carries -DTSX_STAMP
SX_CH = (32, 64)
#: squeeze builds: (output columns per cluster, hidden split = cluster size)
SX_SQ = ((32, 1), (64, 1), (32, 2), (64, 2))
#: weight-gradient output tile widths (columns per CTA), narrowest first
SX_DW_NT = (64, 128, 192)
#: weight-gradient tile width, None: automatic (kept for A/B runs)
SX_DW_NT_FORCE = None


def _sx(d: int, tiles: int) -> bool:
    return d == 384 and tiles <= SMALL_FWD_TILES


def _sx_ch(h: int, tiles: int, nsm: int, fold: bool = False) -> int:
    """Hidden units per expand CTA: the narrowest chunk whose CTAs fit one wave (the folded expand also has the 128-unit chunk)."""
    cs = SX_FOLD_CH if fold else SX_CH
    return next((c for c in cs if tiles * (h // c) <= nsm), cs[-1])


SX_FOLD_CH = (32, 64, 128)
FOLD_PAIR_SQ = True


#: (columns, split) of the squeeze, None: automatic (kept for A/B runs)
SX_SQ_FORCE = None


#: (N1) inference: the 64-column pair squeeze also at 7 - 12 tiles of n 2 (the old rule fell back to (32, 1) there: a hole, measured at 3 / 6 tiles only). False: the old rule (A/B)
SX_SQ_PAIR_INFER = True


def _sx_sq(d: int, tiles: int, h: int, nsm: int, save: bool = True) -> tuple[int, int]:
    """Squeeze decomposition (output columns per cluster, hidden split = cluster size). The per-SM TMA intake of the h tile bounds this
    kernel, so the h rows are split over a CTA pair whose fp32 partials meet in distributed shared memory while the pairs fit one wave
    (2-CTA clusters pack 74 to a wave on a 148-SM B200), and 32-column blocks keep every SM busy; round sx, measured at
    [1, 768 / 384, 384] (n = 2: 4.1 vs 4.5 us unsplit; n = 4: 5.8 vs 7.4)."""
    if SX_SQ_FORCE is not None:
        return SX_SQ_FORCE
    if tiles * (d // 32) * 2 <= nsm:
        return 32, 2
    if tiles * (d // 64) * 2 <= nsm:
        return (64, 2) if h >= 4 * d or (SX_SQ_PAIR_INFER and not save) else (32, 1)
    return 64, 1


@functools.lru_cache(maxsize=16)
def _sx_dxn_resident(d: int, n: int, index: int) -> int:
    """Resident 12-CTA clusters of the K-split d_xn kernel (7 on a 148-SM B200: clusters must fit a GPC)."""
    return int(_load(d, n).max_active_clusters(f"d{d}n{n}_sx_dxn2", "transition_sx_dxn", 384, 2 * (d // 64)))


def _sx_dw_nt(d: int, h: int, busy: int, nsm: int) -> int:
    """Output tile width of the weight-gradient kernel: the narrowest whose CTAs fit the SMs the d_xn clusters (``busy`` CTAs, one SM
    each) leave free -- the products run while d_xn finishes, and a CTA left waiting for a d_xn SM starts its whole K loop after it --
    else the widest. Round sx2, [1, 768, 384] training: n = 2 (72 d_xn CTAs) 64 -> 128, n = 4 128 -> 192."""
    if SX_DW_NT_FORCE is not None:
        return SX_DW_NT_FORCE
    fits = [nt for nt in SX_DW_NT if d % nt == 0 and h % nt == 0]
    for nt in fits:
        if (d // 128) * (h // nt) + (2 * h // 128) * (d // nt) + busy <= nsm:
            return nt
    return fits[-1]


@functools.lru_cache(maxsize=16)
def _sx_gdx_resident(d: int, n: int, index: int) -> int:
    """Resident clusters of the fused gate + d_xn kernel (one 2 D / 64-CTA cluster per tile; 7 on a 148-SM B200)."""
    return int(_load(d, n).max_active_clusters(f"d{d}n{n}_sx_gdx", "transition_sx_gdx", 384, 2 * (d // 64)))


#: the small-M backward's gate and d_xn in one launch (tsx_gdx.cu) while every tile's cluster is resident; None: automatic, False: the
#: two launches (kept for A/B runs)
SX_GDX_FORCE = None


def _sx_gdx(d: int, n: int, tiles: int, index: int) -> bool:
    if SX_GDX_FORCE is not None:
        return SX_GDX_FORCE
    return tiles <= _sx_gdx_resident(d, n, index)


def _sx_ks(d: int, n: int, tiles: int, index: int) -> int:
    """K split of the d_xn kernel: the two halves (dA Wa, dB Wb) on a CTA pair while every tile's cluster is resident at once, which
    halves the per-SM TMA intake that bounds it; one CTA per column block otherwise."""
    return 2 if tiles <= _sx_dxn_resident(d, n, index) else 1


def _dn(x: torch.Tensor, wa: torch.Tensor) -> tuple[int, int]:
    d = x.shape[-1]
    return d, wa.shape[0] // d


def _repl(d: int, n: int, tiles: int = 0) -> int:
    r = REPL[(d, n)]
    for t0, v in REPL_TILES.get((d, n), ()):
        if tiles >= t0:
            r = v
    return r


def _ndw(d: int, n: int, tiles: int = 0) -> int:
    """Weight-role CTAs of the two-role backward: one per (64-unit hidden slice, replica)."""
    return (n * d // 64) * _repl(d, n, tiles)


def _specs(d: int, n: int) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    h = n * d
    hid = (f"-DHID={h}",)
    if d == 64:
        return (
            ("fwd", "widths/tfwd_d64.cu", (*hid, *FWD_TRAIN_FLAGS.get((d, n), ()))),
            *((("fwdi", "widths/tfwd_d64.cu", (*hid, *FWD_INFER_FLAGS[(d, n)])),) if (d, n) in FWD_INFER_FLAGS else ()),
            ("bwd", "widths/tbwd_d64.cu", (*hid, *(("-DDXN32",) if n == 2 else ()), *D64_BWD_FLAGS[n])),
            *((("bwdab", "widths/tbwd_d64.cu", (*hid, *(("-DDXN32",) if n == 2 else ()), *BWD_AB_FLAGS)),) if BWD_AB_FLAGS else ()),
        )
    if d == 128:
        return (
            ("fwd", "tr_fwd_sm100.cu", hid),
            *((("fwdi", "tr_fwd_sm100.cu", (*hid, *FWD_INFER_FLAGS[(d, n)])),) if (d, n) in FWD_INFER_FLAGS else ()),
            ("bwd", "tr_bwd_sm100.cu", hid),
            *((("bwdab", "tr_bwd_sm100.cu", (*hid, *BWD_AB_FLAGS)),) if BWD_AB_FLAGS else ()),
        )
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
            *((("fwdi", "widths/tfwd_d256.cu", (*hid, *FWD_INFER_FLAGS[(d, n)])),) if (d, n) in FWD_INFER_FLAGS else ()),
            *((("fwdi1", "widths/tfwd_d256.cu", (*hid, *(f for f in FWD_INFER_FLAGS[(d, n)] if f != "-DX2=1"))),) if (d, n) in FWD_X2 else ()),
            ("gate", "widths/tgate_w.cu", ("-DDIM=256", *hid)),
            dxn,
            lnb,
        )
    dim = (f"-DDIM={d}", *hid)
    fwd = (
        (
            ("fwd", "widths/tfwd_d384.cu", hid),
            ("fwdi", "widths/tfwd_d384.cu", (*hid, "-DFOLD=1")),
            ("stat", "widths/tstat_w.cu", ("-DDIM=384", "-DLPR=16")),
            ("bwd", "widths/tbwd_d384.cu", hid),
        )
        if d == 384
        else (
            ("ln", "widths/tln_w.cu", (f"-DDIM={d}", lpr)),
            ("swiglu", "widths/tswiglu_w.cu", dim),
            ("swiglu_is", "widths/tswiglu_w.cu", (*dim, "-DITEM_SCHED")),
            ("swiglu_ab", "widths/tswiglu_w.cu", (*dim, "-DSAVE_AB")),
            ("swiglu_abis", "widths/tswiglu_w.cu", (*dim, "-DSAVE_AB", "-DITEM_SCHED")),
            ("squeeze", "widths/tgemm_nd.cu", (f"-DDIM={nd}", f"-DKDIM={h}")),
            *((("squeezep", "widths/tgemm_nd.cu", (f"-DDIM={nd}", f"-DKDIM={h}", "-DPARTS=1", "-DXEF=1", *SQ_FLAGS)),) if d == 512 else ()),
            *(
                (
                    ("stat", "widths/tstat_w.cu", (f"-DDIM={d}", f"-DLPR={STAT_LPR}")),
                    ("swiglu_fold", "widths/tswiglu_w.cu", (*dim, "-DFOLD")),
                    ("swiglu_fold_is", "widths/tswiglu_w.cu", (*dim, "-DITEM_SCHED", "-DFOLD")),
                )
                if d == 512
                else ()
            ),
        )
    )
    sx = (
        (
            ("sx_ln", "widths/tsx_ln.cu", (f"-DDIM={d}", *SX_X.get("ln", ()))),
            *((f"sx_expand{c}", "widths/tsx_expand.cu", (*dim, f"-DCH={c}", *SX_X.get("expand", ()))) for c in SX_CH),
            *((f"sx_expand{c}f", "widths/tsx_expand.cu", (*dim, f"-DCH={c}", "-DFOLD=1", *SX_X.get("expandf", ()))) for c in SX_FOLD_CH),
            *((f"sx_sq{nb}_{k}", "widths/tsx_squeeze.cu", (*dim, f"-DNBW={nb}", f"-DSK={k}", *SX_X.get("sq", ()))) for nb, k in SX_SQ),
            ("sx_gate", "widths/tsx_gate.cu", dim),
            ("sx_gdx", "widths/tsx_gdx.cu", (*dim, *SX_X.get("gdx", ()))),
            *((f"sx_dxn{k}", "widths/tsx_dxn.cu", (*dim, f"-DKS={k}")) for k in (1, 2)),
            *((f"sx_dw{nt}", "widths/tsx_dw.cu", (*dim, f"-DNT={nt}")) for nt in SX_DW_NT if d % nt == 0 and h % nt == 0),
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


def _load(d: int, n: int):
    """Build this (width, n)'s cubins and register them with the extension (loaded on the current device: a cubin the driver
    rejects fails here, inside ``available()``). Returns the extension."""
    return _load_cached(d, n, BWD_AB_FLAGS)


@functools.lru_cache(maxsize=16)
def _load_cached(d: int, n: int, ab: tuple[str, ...] | None):
    ext = _ext()
    for name, path in build_cubins(f"d{d}n{n}", _specs(d, n)).items():
        ext.load_cubin(f"d{d}n{n}_{name}", path)
    return ext


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """Whether these kernels can run this call: sm_100 (B200), bf16, (D, n) in ``SHAPES`` with hidden n D, whole 128-row tiles,
    and for the two-role backward enough SMs for both roles. The requirements are the kernels' own, not a policy."""
    if not x.is_cuda or x.dtype is not torch.bfloat16:
        return False
    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
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
        build_failed("wide sm100a Transition", exc)
        return False
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


def _dw(dy: torch.Tensor, hid: torch.Tensor, dab: torch.Tensor, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(dWs = dy^T h [D, H], [dWa; dWb] = [dA | dB]^T xn [2H, D]) fp32: the generic ``tg_dw`` launch (round 3 part 5, a deterministic
    split over the rows) where ``fused_generic_sm100a._dw_plan`` takes the shape, else the two cuBLAS GEMMs."""
    from .fused_generic_sm100a import weight_grads

    r = weight_grads(dy, hid, dab, xn)
    return r if r is not None else (_mm_f32(dy.t(), hid), _mm_f32(dab.t(), xn))


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
        self.ext.launch_kernel(f"d{self.d}n{self.n}_{cubin}", kernel, grid, block, cluster, list(args), pdl)

    def rows(self, t):  # [M, D] activation map
        return self.tmap(t, self.d, self.m)

    def halves(self):
        """(first column, width) of each squeeze / d_xn GEMM launch."""
        return [(c, HALF) for c in range(0, self.d, HALF)] if self.d == 768 else [(0, self.d)]

    def cols(self, t, c0, nd):  # [M, nd] column slice of an [M, D] activation
        return self.tmap(t[:, c0 : c0 + nd], nd, self.m, row_stride=self.d)


_FUSED_FWD = {
    64: "transition_fwd_d64_sm100",
    128: "transition_fwd2_sm100",
    256: "transition_fwd_d256_sm100",
}

#: Inference builds (save = 0) of the fused forwards, bf16 output the same as the training build or equal within the LayerNorm's
#: summation order (round 3, part 9, docs/b200/transition.md): D 64 n 2 with the epilogue on its own warpgroup (`ROLES=1`, the same bits),
#: D 256 with the LayerNorm and the epilogue on two warpgroups and the x / weight boxes issued by one lane each. The training forward
#: (save = 1) keeps the ``fwd`` cubin the backward was built against. A module constant (A/B runs edit it).
#: (S90 part 6) build flags of the D 64 training forward (cubin "fwd", save = 1): n 2 with ROLES 1 (one SwiGLU warpgroup, LayerNorm, epilogue each on its own
#: warps: the LayerNorm + epilogue pair on the same four warps set the 2.1 us tile period; now 1.7 = two SwiGLU chunks on one warpgroup): [589 824 / 147 456, 64] n 2
#: step 239.7 / 77.8 -> 227.3 / 75.8 us; n 4 is slower with it (370.7 -> 378.9). A module constant (A/B runs edit it)
FWD_TRAIN_FLAGS: dict[tuple[int, int], tuple[str, ...]] = {(64, 2): ("-DROLES=1",)}

#: Inference builds (save = 0), see above.
#: (S90) the D 256 inference builds with two xn buffers and no x buffer (tfwd_d256.cu -DX2=1): the kernel's rstd / c1 arguments carry x / y. X2_RESID False: the 2.5.14 build.
X2_RESID = True
FWD_X2 = {(256, 2), (256, 4)}
#: (S90) the LayerNorm folded into the D 512 inference expand (tstat_w.cu + tswiglu_w.cu -DFOLD) from this many rows
FOLD_LN = True
#: (S90) the D 512 inference squeeze with the K loop per N part and a double-buffered 256-column accumulator (tgemm_nd.cu -DPARTS)
SQ_PARTS = True   # n 2 only: row 33 master -2 to -2.5 % (L 256 190.5 -> 186.2, 512 665.4 -> 649.1, 640 1048.0 -> 1027.4); row 34 (n 4) +1 - 3 %
SQ_FLAGS: tuple[str, ...] = ()
#: (S90) the LayerNorm folded into the fused D 384 inference kernel (tfwd_d384.cu -DFOLD, K16 correction MMA)
FOLD_D384 = True
#: from this many rows (in-process A/B, graph medians fold / unfolded, [M, 384]: n 2 16 384 rows 1.133, 65 536 0.967, 262 144 0.948, 589 824 0.947, 1 048 576 0.920, 2 359 296 0.927; n 4 16 384 1.141, 65 536 1.000,
#: 262 144 0.908, 589 824 0.938, 1 048 576 0.952, 2 359 296 0.986): below it the prep launch (4.5 us) costs more than the shorter tile saves
FOLD_D384_MIN_ROWS = 65536
#: the statistics pass: lanes a row (16: two rows a warp, 4 x 16-B loads a lane) and blocks an SM
FOLD_N4 = False  # n 4 (row 34) kept on the LayerNorm launch: with the fold its expand takes 1437 us against 1328 (L 640) although the statistics pass saves 73 us (76 against 149)
STAT_LPR = 16
STAT_BPS = 8
FOLD_MIN_ROWS = 1 << 16
#: n 2 only, up to 524 288 rows: measure.speed row 33 master (us, LN + expand + squeeze -> fold) L 256 191.7 -> 186.1, 384 393.0 -> 380.7, 512 706.3 -> 673.7, 640 1105.8 -> 1067.9, 768 1602.3 -> 1603.5 (level);
#: row 34 (n 4) +/- 3 % (L 256 317.2 -> 325.4, 512 1275.7 -> 1257.2, 640 2007.0 -> 2075.5, 768 3230.0 -> 3328.8): kept on the LayerNorm launch
FOLD_MAX_ROWS = 524288
#: (S90 part 7) the training forward at D 256 on the same X2 build (saves xn, rstd, c1 as before); False: the 2.5.14 training build
X2_TRAIN = True
FWD_INFER_FLAGS: dict[tuple[int, int], tuple[str, ...]] = {
    (64, 2): ("-DROLES=1",),
    (128, 2): ("-DLNH=1", "-DNXS_=3", "-DNW_=2"),
    (256, 2): ("-DXPAR=1", "-DWPAR=1", "-DLN2=1", "-DPF2=1", "-DX2=1"),
    (256, 4): ("-DXPAR=1", "-DWPAR=1", "-DLN2=1", "-DPF2=1", "-DX2=1"),
}


def _sx_forward(k, x, gamma, beta, wa, wb, ws, eps, save, out, rstd, c1, cast=None, fold=None):
    """The round-sx forward (tsx_ln -> tsx_expand -> tsx_squeeze, programmatic dependent launches) into ``out`` (and ``rstd`` / ``c1``
    with save). ``cast``: the fp32 masters whose bf16 copies ``wa`` / ``wb`` / ``ws`` the LayerNorm launch writes first. Returns
    (out, xn, rstd, c1, h, a, b) as ``_fwd_launch``. ``fold`` (N1, inference): (Wp, aug) of ``_sx_fold_forms``: no LayerNorm launch, the
    expand takes the raw x (``tsx_expand.cu -DFOLD``), two launches."""
    d, h, m, tiles = k.d, k.h, k.m, k.tiles

    def ph():
        return x.new_empty((1, 1))

    xn = x if fold is not None else torch.empty_like(x)  # the expand's A operand, kept for the backward with save
    hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
    a, b = ((torch.empty((m, h), dtype=x.dtype, device=x.device) if save else ph()) for _ in range(2))
    nrb = (m + 7) // 8
    if fold is None:
        k.run(
            "sx_ln",
            "transition_sx_ln",
            nrb + ((3 * h * d // 8 + 511) // 512 if cast is not None else 0),  # cast blocks: 2 x 8 elements per thread
            (x, gamma, beta, xn, rstd, c1, m, float(eps), int(save), *(cast or (x, x, x)), wa, wb, ws, h * d if cast else 0, *([SX_ST["ln"]] if SX_ST else [])),
            block=256,
            cluster=1,
            pdl=True,
        )
    extra = ()
    if fold is not None:
        wp, aug = fold
        wa, wb, extra = wp[:h], wp[h:], (aug, float(eps))
    ch = _sx_ch(h, tiles, k.nsm, fold is not None)
    k.run(
        f"sx_expand{ch}" + ("f" if fold is not None else ""),
        "transition_sx_expand",
        tiles * (h // ch),
        (k.tmap(xn, d, m, 64, 128), k.tmap(wa, d, h, 64, ch), k.tmap(wb, d, h, 64, ch), hid, a, b, int(save), int(cast is None), h, 0, *extra, *([SX_ST["expand"]] if SX_ST else [])),
        block=384,
        cluster=1,
        pdl=True,
    )
    pnb = 0
    if fold is not None and FOLD_PAIR_SQ:  # (N1) the CTA-pair squeeze of fused_small's chain where it has one (n 4 8 - 12 tiles)
        from . import fused_small_sm100a as _fs

        pnb = _fs.BF16_PAIR_SQ.get((d, h // d, tiles), 0)
    if pnb:
        ext = _fs._load(d, h // d, False)
        kq = _fs._kbx(h // 64)
        ext.launch_kernel(
            f"{_fs._tag(d, h // d, False)}_psq{pnb}",
            "transition_sx_squeeze2",
            tiles * (d // pnb) * 2,
            256,
            2,
            [ext.tmap(hid, h, m, 64, 64, -1, 0, kq), ext.tmap(ws, h, d, 64, pnb // 2), x, out],
            True,
        )
        return out, x.new_empty((1, d)), rstd, c1, ph(), ph(), ph()
    nb, sk = _sx_sq(d, tiles, h, k.nsm, save)
    k.run(
        f"sx_sq{nb}_{sk}",
        "transition_sx_squeeze",
        tiles * (d // nb) * sk,
        (k.tmap(hid, h, m, 64, 128), k.tmap(ws, h, d, 64, nb), x, out, *([SX_ST["sq"]] if SX_ST else [])),
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
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
    hab = [x.new_empty((m, wa.shape[0])) if keep else x.new_empty((1, 1)) for _ in range(3)]
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
        # (S90) D 256 inference: the two-xn-buffer build (X2) from the second tile of a CTA; one tile a CTA keeps the single-buffer build (the tile's own latency, no ring to overlap)
        x2_cell = X2_RESID and (not save or X2_TRAIN) and (d, k.n) in FWD_X2
        x2_off = x2_cell and tiles <= k.grid
        x2 = x2_cell and not x2_off
        xn = torch.empty_like(x) if save else x.new_empty((1, d))
        mout = k.rows(out)
        k.run(
            "fwd" if (save and (not x2_cell or x2_off)) or (d, k.n) not in FWD_INFER_FLAGS else ("fwdi1" if x2_off else "fwdi"),
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
                *((x, out) if d == 256 else ()),
            ),
            pdl=save and d == 128 and BWD_PDL,  # (training part 4) tr_fwd_sm100.cu waits (griddepcontrol.wait) at its entry
        )
        return out, xn, rstd, c1, ph(), ph(), ph()
    if _sx(d, tiles):  # three programmatic-dependent kernels sized for a few tiles (round sx: tsx_ln, tsx_expand, tsx_squeeze)
        return _sx_forward(k, x, gamma, beta, wa, wb, ws, eps, save, out, rstd, c1)
    if d == 384:  # one fused kernel (round n2, tfwd_d384.cu)
        xn = torch.empty_like(x) if save else x.new_empty((1, d))
        hid, a, b = ((torch.empty((m, h), dtype=x.dtype, device=x.device) if save else ph()) for _ in range(3))
        mout = k.rows(out)
        fold = FOLD_D384 and not save and m >= FOLD_D384_MIN_ROWS and wa.dtype == x.dtype == torch.bfloat16
        if fold:  # (S90) the LayerNorm folded into the expand with a K16 correction MMA: raw x as A, W gamma as B (one prep launch also makes the correction images)
            wp = torch.empty((2 * h, d), dtype=x.dtype, device=x.device)
            u, v = (torch.empty(2 * h, **f32) for _ in range(2))
            aug = torch.zeros(h // 64 * 2 * 64 * 8, dtype=torch.bfloat16, device=x.device)
            k.run("stat", "transition_ln_stats_w", k.nsm, (x, gamma, beta, wa, wb, wp, u, v, rstd, c1, 0, h, float(eps), 0, aug, 1), 256, 1)
            wa_, wb_ = wp[:h], wp[h:]
        else:
            wa_, wb_ = wa, wb
        k.run(
            "fwdi" if fold else "fwd",
            "transition_fwd_d384_sm100",
            k.grid,
            (
                k.rows(x),
                k.tmap(wa_, d, h, 64, 32),
                k.tmap(wb_, d, h, 64, 32),
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
                *((aug,) if fold else ()),
            ),
        )
        return out, xn, rstd, c1, hid, a, b
    hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
    item = tiles <= ITEM_TILES
    mh = k.tmap(hid, h, m)
    fold = FOLD_LN and not save and d == 512 and (h == 2 * d or FOLD_N4) and FOLD_MIN_ROWS <= m <= FOLD_MAX_ROWS and wa.dtype == x.dtype == torch.bfloat16
    if fold:
        # (S90) the LayerNorm folded into the expand: raw x as the A operand, W gamma as B, statistics (one read of x, no xn) and the folded weights in ONE launch
        xn = x.new_empty((1, d))
        wp = torch.empty((2 * h, d), dtype=x.dtype, device=x.device)
        u, v = (torch.empty(2 * h, **f32) for _ in range(2))
        rstd, c1 = torch.empty(m, **f32), torch.empty(m, **f32)
        nrb = min(m // 16, k.nsm * STAT_BPS)
        k.run("stat", "transition_ln_stats_w", nrb + 32, (x, gamma, beta, wa, wb, wp, u, v, rstd, c1, m, h, float(eps), nrb, wp, 0), 256, 1)
        k.run(
            "swiglu_fold_is" if item else "swiglu_fold",
            "transition_swiglu_w_sm100",
            k.nsm if item else k.grid,
            (k.rows(x), k.tmap(wp[:h], d, h, 64, 128), k.tmap(wp[h:], d, h, 64, 128), mh, tiles, rstd, c1, u, v),
        )
    else:
        xn = torch.empty_like(x)  # D >= 512: the expand's A operand, always materialised
        k.run(
            "ln",
            "transition_ln_w",
            min(m // 16, k.nsm * 4),
            (x, gamma, beta, xn, rstd, c1, m, float(eps), int(save)),
            256,
            1,
        )
    maps = None if fold else (k.rows(xn), k.tmap(wa, d, h, 64, 128), k.tmap(wb, d, h, 64, 128), mh)
    if fold:
        pass
    elif save:
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
            "squeezep" if (SQ_PARTS and not save and d == 512 and nd == d and h == 2 * d) else "squeeze",
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
        return out, xn.new_empty((1, d)), (x.new_empty((1,), dtype=torch.float32) if fold else rstd), (x.new_empty((1,), dtype=torch.float32) if fold else c1), ph(), ph(), ph()
    return out, xn, rstd, c1, hid, a, b


def _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hid, a, b, wst, wabt):
    """Output structure only: (dx, dgamma, dbeta, dWa, dWb, dWs); the weight gradients are fp32. At D >= 256 (``_packed``) dgamma is
    dgamma | dbeta [2 D] and dWa is [dWa; dWb] [2 H, D], dbeta / dWb 1-element placeholders (the caller splits: no clone launches)."""
    wdt = torch.float32
    if _packed(x.shape[1]):
        e = lambda *s: torch.empty(s, dtype=wdt, device=x.device)  # noqa: E731
        return torch.empty_like(x), e(2 * x.shape[1]), e(1), e(2 * wa.shape[0], wa.shape[1]), e(1, 1), e(*ws.shape)
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
    wst: torch.Tensor,
    wabt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dx, dgamma, dbeta, dWa, dWb, dWs); ``dx`` already carries the residual branch. ``hid`` / ``a`` / ``b`` are the
    forward's h / a / b at D >= 384 and placeholders otherwise; ``wst`` / ``wabt`` Ws^T [H, D] and [Wa; Wb]^T [D, 2H] made by the forward's
    ``tg_prep`` (``_t_forms``), or 1 x 1 placeholders (made here)."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hid, a, b, wst, wabt)
    k = _Launch(x, wa.shape[0])
    d, h, m, tiles = k.d, k.h, k.m, k.tiles
    f32 = dict(dtype=torch.float32, device=x.device)
    dx = torch.empty_like(x)
    if d in (64, 128):  # the fused two-role backward
        ndw = _ndw(d, k.n, tiles)
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
                "bwdab" if BWD_AB_FLAGS else "bwd",
                "transition_bwd_d64_sm100",
                k.nsm,
                (*maps, rstd, c1, gamma, partab, parts, dgbw, tiles, ndw),
                pdl=BWD_PDL,
            )
            # a programmatic dependent of the backward (it waits for the whole grid): 16-byte weight-gradient blocks, then dgamma /
            # dbeta 16 columns per block over 16 row groups (fixed order; tbwd_d64.cu)
            k.run(
                "bwd",
                "transition_bwd_d64_reduce",
                3 * h * d // 1024 + 2 * d // 16,
                (partab, parts, dgbw, dwa, dwb, dws, dgam, dbeta, ndw, ndx * 4),
                256,
                1,
                pdl=True,
            )
            return dx, dgam, dbeta, dwa, dwb, dws
        else:
            k.run(
                "bwdab" if BWD_AB_FLAGS else "bwd",
                "transition_bwd_sm100",
                k.nsm,
                (*maps, rstd, c1, gamma, x, partab, parts, dgbw, tiles, ndw),
                pdl=BWD_PDL,
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
            pdl=BWD_PDL,
        )
        return dx, dgam, dbeta, dwa, dwb, dws
    small = d >= 384 and _small(tiles)
    have_t = wst.dim() == 2 and wst.shape[0] == h and wst.shape[1] == d
    if not have_t:
        wst = ws.t().contiguous()  # [H, D]
    if d == 384 and not small:  # gate + d_xn + LayerNorm backward in one kernel (round n2, tbwd_d384.cu)
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
        dws, dwab = _dw(dy, hid, dab, xn)  # [D, H], [2H, D]: dWa then dWb
        return dx, dgb, dgb.new_empty((1,)), dwab, dwab.new_empty((1, 1)), dws  # packed (_packed)
    wab_t = wabt if have_t else torch.cat((wa, wb), 0).t().contiguous()  # [D, 2H]
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
    dxn = torch.empty_like(x)
    for c0, nd in k.halves():
        mo = k.cols(dxn, c0, nd)  # (the x slot of the plain-epilogue GEMM is never read)
        k.run(
            "dxn",
            "transition_gemm_nd_sm100",
            k.grid,
            (mdab, k.tmap(wab_t[c0 : c0 + nd], 2 * h, nd), mo, mo, tiles),
        )
    nb = min(m // 16, k.nsm * LNBWD_PER_SM)
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
    # the weight gradients last, with the dgamma | dbeta sum folded into their launch (round 3 part 8: the 2-block
    # transition_lnbwd_w_reduce took 15.5 us of a 112.6 us step at [16384, 256] n 2); cuBLAS + that reduce where tg_dw does not run
    from .fused_generic_sm100a import weight_grads

    r = weight_grads(dy, hid, dab, xn, lpart, dgb) if LN_SUM_IN_DW else None
    if r is None:
        dws, dwab = _dw(dy, hid, dab, xn)  # [D, H], [2H, D]: dWa then dWb
        k.run("lnbwd", "transition_lnbwd_w_reduce", (2 * d + 255) // 256, (lpart, dgb, nb), 256, 1)
    else:
        dws, dwab = r
    return dx, dgb, dgb.new_empty((1,)), dwab, dwab.new_empty((1, 1)), dws  # packed (_packed); outputs must not alias each other


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
    """The small-M backward at D = 384 (round sx), programmatic dependent launches: while every tile's 12-CTA cluster is resident
    (``_sx_gdx``, round sx3) gate + d_xn + LayerNorm backward + dgamma / dbeta partials in one launch (tsx_gdx.cu), else gate
    (tsx_gate.cu) -> d_xn (tsx_dxn.cu); then dWs, dWa | dWb (fp32, or bf16 for bf16 parameters) and dgamma | dbeta (tsx_dw.cu, beside
    the fused launch: its dWa | dWb read a tile's [dA | dB] once the tile's flag is set). Returns (dx, dgamma | dbeta, dWa | dWb, dWs);
    the caller splits the packed ones (outputs of a custom op must not alias each other)."""
    if _is_fake(dy, x):
        return _bwd_sx_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws, hid, a, b, wgrad_bf16)
    k = _Launch(x, wa.shape[0])
    d, h, m, tiles = k.d, k.h, k.m, k.tiles
    f32 = dict(dtype=torch.float32, device=x.device)
    dx = torch.empty_like(x)
    dab = torch.empty((m, 2 * h), dtype=x.dtype, device=x.device)
    part = torch.empty((tiles, 2 * d), **f32)
    fused = _sx_gdx(d, k.n, tiles, x.device.index)
    flags = torch.empty((tiles,), dtype=torch.int32, device=x.device)  # cleared and set by the fused kernel
    if fused:  # gate + d_xn + LayerNorm backward on one cluster per tile (round sx3)
        ks = 2
        k.run(
            "sx_gdx",
            "transition_sx_gdx",
            tiles * (d // 64) * ks,
            (
                k.tmap(dy, d, m, 64, 128),
                k.tmap(ws, h, d),
                a,
                b,
                dab,
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
                flags,
            ),
            block=384,
            cluster=(d // 64) * ks,
            pdl=True,
        )
    else:
        k.run(
            "sx_gate",
            "transition_sx_gate",
            tiles * (h // 64),
            (k.tmap(dy, d, m, 64, 128), k.tmap(ws, h, d), a, b, dab),
            block=384,
            cluster=1,
            pdl=True,
        )
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
    nt = _sx_dw_nt(d, h, tiles * (d // 64) * ks, k.nsm)
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
            flags,
            int(fused),
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


#: (round 3 part 8, 2026-11-17) training from fp32 masters at D >= 256 off the small-M chains: the bf16 forms AND the backward's Ws^T /
#: [Wa; Wb]^T in ONE launch (the generic ``tg_prep``, the same round-to-nearest bf16 bits as ``.to``) in the forward, in place of the
#: cast launch + two transposing copies + a cat per step in the backward. False: the old forms (A/B).
T_FORMS = True
#: (round 3 part 8) the dgamma | dbeta column sum inside the weight-gradient launch (tg_dw / tg_wsum) at D >= 256; False: A/B
LN_SUM_IN_DW = True
#: blocks per SM of transition_lnbwd_w (each writes one dgamma | dbeta partial row that the column sum reads); round 3 part 8: 4 -> 2
#: (graph, us, 4 / 2 / 1: [16384, 256] n 2 104.5-106.5 / 100.4 / 98.3-100.4, n 4 157.7 / 149.2-149.5 / 149.5, [16384, 512] n 2 218.4-219.1 /
#: 210.9 / 213.0, [65536, 256] n 2 321.5 / 305.2-306.2 / 311.3)
LNBWD_PER_SM = 2


def _packed(d: int) -> bool:
    """(round 3 part 8) D >= 256: the backward op returns dgamma | dbeta and [dWa; dWb] packed, split by the caller (no clone launches)."""
    return d >= 256


def _t_forms(wa, wb, ws, x: torch.Tensor, tiles: int):
    """(wa, wb, ws bf16, Ws^T, [Wa; Wb]^T) by one ``tg_prep`` where the backward reads the transposes, else None."""
    d, h = wa.shape[1], wa.shape[0]
    if not (
        T_FORMS
        and x.dtype is torch.bfloat16
        and d >= 256
        and d % 32 == 0
        and h % 32 == 0
        and not _sx(d, tiles)
        and not torch.compiler.is_compiling()
        and not _is_fake(wa, wb, ws)
        and all(w.dtype is torch.float32 and w.is_contiguous() for w in (wa, wb, ws))
    ):
        return None
    from .fused_generic_sm100a import _prep

    return _prep(wa, wb, ws, False, True)


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
        t = None
        if (
            _sx(flat.shape[1], flat.shape[0] // ROWS)
            and x.dtype is torch.bfloat16
            and all(w.dtype is torch.float32 and w.is_contiguous() for w in (wa, wb, ws))
        ):  # fp32 masters at small M: the LayerNorm launch writes the bf16 copies
            out, xn, rstd, c1, hid, a, b, wa, wb, ws = _fwd_sx_cast_launch(flat, gf, bf, wa, wb, ws, float(eps))
        else:
            t = _t_forms(wa, wb, ws, flat, flat.shape[0] // ROWS)
            if t is not None:
                wa, wb, ws, wst, wabt = t
            else:
                wa, wb, ws = _bf16_forms(wa, wb, ws, x.dtype)
            out, xn, rstd, c1, hid, a, b = _fwd_launch(flat, gf, bf, wa, wb, ws, float(eps), True)
        if t is None:
            wst = wabt = flat.new_empty((1, 1))
        ctx.save_for_backward(flat, xn, rstd, c1, gf, wa, wb, ws, hid, a, b, wst, wabt)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, rstd, c1, gf, wa, wb, ws, hid, a, b, wst, wabt = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        d, h = flat.shape[1], wa.shape[0]
        if _sx(d, flat.shape[0] // ROWS):
            dx, dgb, dwab, dws = _bwd_sx_launch(
                dy.reshape(-1, d).contiguous(),
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
            wst,
            wabt,
        )
        if _packed(d):  # views of the packed gradients (no clone launch)
            dgam, dbeta, dwa, dwb = dgam[:d], dgam[d:], dwa[:h], dwa[h:]
        return (
            dx.reshape(ctx.shape),
            dgam.to(gdt),
            dbeta.to(bdt),
            dwa.to(adt),
            dwb.to(bwdt),
            dws.to(sdt),
            None,
        )


#: (N1) bf16 D 384 small-M inference (the ``_sx`` cells): the LayerNorm folded into the expand (``tsx_expand.cu -DFOLD``: raw x as A, Wp = bf16(W gamma)
#: as B, one K16 correction MMA for -mean u + q v, statistics from the x tile in shared memory), no LayerNorm launch: two launches. False: A/B
SX_FOLD = True
#: fewest 128-row tiles of the folded chain by n (measured, device span fold / unfolded: n 2 2 tiles +0.08 us, 3 tiles level, 4 - 12 tiles -0.3 .. -0.8; n 4 2 / 3 tiles +0.16 / +0.16, 4 tiles -0.07, 5 / 6 tiles -0.5 / -0.3)
SX_FOLD_MIN_TILES = {2: 4, 4: 5}  # n 2 3 tiles level (8.72 / 8.72), n 4 3 / 4 tiles +0.16 / -0.07 us


def _sx_fold_forms(wa, wb, gamma, beta):
    """(Wp, aug) of the folded expand, from the bf16 forms of Wa / Wb: Wp = bf16(W gamma) for [Wa; Wb] ([2 H, D]); aug [2 H, 8] bf16, row n =
    [u_hi u_lo u_hi v_hi v_lo v_hi 0 0] with u_n = sum_k Wp_nk (of the rounded values) and v_n = sum_k beta_k W_nk (``tstat_w.cu``'s prep as torch
    ops). A weight form: made once per weights (``param_forms``), not per call."""
    w = torch.cat((wa, wb)).float()
    wp = (w * gamma.float()).to(torch.bfloat16)
    u = wp.float().sum(1)
    v = (w * beta.float()).sum(1)
    uh, vh = u.to(torch.bfloat16), v.to(torch.bfloat16)
    ul, vl = (u - uh.float()).to(torch.bfloat16), (v - vh.float()).to(torch.bfloat16)
    z = torch.zeros_like(uh)
    return wp.contiguous(), torch.stack((uh, ul, uh, vh, vl, vh, z, z), 1).contiguous()


def _fwd_sxf_fake(x, wp, aug, ws, eps):
    return torch.empty_like(x)


@opaque(fake=_fwd_sxf_fake, name="transition_wide_fwd_sxf_sm100a")
def _fwd_sxf_launch(x: torch.Tensor, wp: torch.Tensor, aug: torch.Tensor, ws: torch.Tensor, eps: float) -> torch.Tensor:
    """x [M, D] bf16, wp [2 H, D] bf16 (Wp), aug [2 H, 8] bf16, ws [D, H] bf16: out = x + h Ws^T, h from the LayerNorm-folded expand."""
    if _is_fake(x, wp):
        return _fwd_sxf_fake(x, wp, aug, ws, eps)
    k = _Launch(x, ws.shape[1])
    out = torch.empty_like(x)
    _sx_forward(k, x, None, None, None, None, ws, eps, False, out, None, None, fold=(wp, aug))
    return out


#: (N1) the fused D 384 inference kernel's LayerNorm fold with its weight prep (Wp = bf16(W gamma), u / v correction images; ``tstat_w.cu`` ran it as a 4.5 us launch in
#: front of every call) made once per weights (``param_forms``) from this many rows; False: the per-call prep launch (A/B)
FOLD_D384_CACHED = True
#: by n (graph, us, L2 flushed, unfolded -> cached prep, [M, 384]: n 2 16 384 rows 41.0 -> 38.9, 32 768 69.6 -> 67.6, 65 536 124.9 -> 116.7, 147 456 227.6 -> 217.1; n 4 16 384 59.4 -> 59.4 (device span +1.1 us), 32 768 106.5 -> 102.4, 65 536 192.5 -> 185, 98 304 276.5 -> 272.4)
FOLD_D384_CACHED_MIN_ROWS = {2: 16384, 4: 32768}


def _f384_aug(aug_rows, h):
    """[2 H, 8] correction rows (``_sx_fold_forms``) -> the kernel's chunk images: 32 units of Wa then the same 32 of Wb, 16 B a row (the prep's layout)."""
    a, b = aug_rows[:h].reshape(h // 32, 32, 8), aug_rows[h:].reshape(h // 32, 32, 8)
    return torch.cat((a, b), 1).contiguous().reshape(-1)


def _fwd_f384_fake(x, gamma, beta, wp, aug, ws, eps):
    return torch.empty_like(x)


@opaque(fake=_fwd_f384_fake, name="transition_wide_fwd_f384_sm100a")
def _fwd_f384_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wp: torch.Tensor, aug: torch.Tensor, ws: torch.Tensor, eps: float) -> torch.Tensor:
    """The fused D 384 inference kernel with the LayerNorm folded (``tfwd_d384.cu`` cubin ``fwdi``) from cached weight forms: x [M, 384] bf16, wp [2 H, 384] (Wp),
    aug the chunk images, ws [384, H] bf16."""
    if _is_fake(x, wp):
        return _fwd_f384_fake(x, gamma, beta, wp, aug, ws, eps)
    h = ws.shape[1]
    k = _Launch(x, h)
    d, tiles = k.d, k.tiles
    f32 = dict(dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    rstd = torch.empty((1,), **f32)
    c1 = torch.empty_like(rstd)
    ph = x.new_empty((1, 1))
    mout = k.rows(out)
    k.run(
        "fwdi",
        "transition_fwd_d384_sm100",
        k.grid,
        (k.rows(x), k.tmap(wp[:h], d, h, 64, 32), k.tmap(wp[h:], d, h, 64, 32), k.tmap(ws, h, d), mout, mout, gamma, beta, rstd, c1, ph, ph, ph, tiles, float(eps), 0, aug),
    )
    return out


#: (N1) the D 512 n 2 inference fold with its weight prep (Wp, u, v: the 32 prep blocks of the statistics launch) made once per weights: from this many rows (0: off)
#: (device span, us, L2 flushed, LayerNorm launch -> cached-prep fold; the per-call prep fold was 55.7 -> 63.3 at 16 384 rows) [M, 512] n 2: 16 384 rows 55.7 -> 53.6, 32 768 97.4 -> 94.7, 65 536 182.6 -> 174.9, 147 456 382.1 -> 363.4, 262 144 686.6 -> 655.1 (the prep-per-call fold 654.9)
FOLD_D512_CACHED_MIN_ROWS = 16384
#: n 4 as well (device span, us, LayerNorm launch -> cached-prep fold, [M, 512] n 4: 16 384 rows 88.09 -> 87.23, 32 768 163.0 -> 161.3, 65 536 314.2 -> 307.0, 147 456 680.1 -> 667.5, 262 144 1261 -> 1246, 409 600 1996 -> 1959; the per-call prep fold was +20 us at 16 384 rows: FOLD_N4)
FOLD_D512_N4_CACHED = True


def _f512_forms(wa, wb, gamma, beta):
    """(Wp, u, v) of the D 512 fold (``tstat_w.cu``'s prep as torch ops)."""
    w = torch.cat((wa, wb)).float()
    wp = (w * gamma.float()).to(torch.bfloat16)
    return wp.contiguous(), wp.float().sum(1), (w * beta.float()).sum(1)


def _fwd_f512_fake(x, gamma, beta, wp, u, v, ws, eps):
    return torch.empty_like(x)


@opaque(fake=_fwd_f512_fake, name="transition_wide_fwd_f512_sm100a")
def _fwd_f512_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wp: torch.Tensor, u: torch.Tensor, v: torch.Tensor, ws: torch.Tensor, eps: float) -> torch.Tensor:
    """The D 512 n 2 inference chain with the LayerNorm folded into the expand from cached weight forms: the statistics launch (rows only), the folded expand + SwiGLU, the squeeze."""
    if _is_fake(x, wp):
        return _fwd_f512_fake(x, gamma, beta, wp, u, v, ws, eps)
    h = ws.shape[1]
    k = _Launch(x, h)
    d, m, tiles = k.d, k.m, k.tiles
    f32 = dict(dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
    rstd, c1 = torch.empty(m, **f32), torch.empty(m, **f32)
    item = tiles <= ITEM_TILES
    mh = k.tmap(hid, h, m)
    nrb = min(m // 16, k.nsm * STAT_BPS)
    k.run("stat", "transition_ln_stats_w", nrb, (x, gamma, beta, wp[:h], wp[:h], wp, u, v, rstd, c1, m, h, float(eps), nrb, wp, 0), 256, 1)
    k.run(
        "swiglu_fold_is" if item else "swiglu_fold",
        "transition_swiglu_w_sm100",
        k.nsm if item else k.grid,
        (k.rows(x), k.tmap(wp[:h], d, h, 64, 128), k.tmap(wp[h:], d, h, 64, 128), mh, tiles, rstd, c1, u, v),
    )
    k.run(
        "squeezep" if (SQ_PARTS and h == 2 * d) else "squeeze",
        "transition_gemm_nd_sm100",
        k.grid,
        (mh, k.tmap(ws, h, d), k.cols(x, 0, d), k.cols(out, 0, d), tiles),
    )
    return out


def _f512_ok(x, wa, rows):
    return (
        FOLD_LN
        and x.dtype is torch.bfloat16
        and x.shape[-1] == 512
        and (wa.shape[0] == 1024 or (wa.shape[0] == 2048 and FOLD_D512_N4_CACHED))
        and rows % ROWS == 0
        and FOLD_D512_CACHED_MIN_ROWS > 0
        and FOLD_D512_CACHED_MIN_ROWS <= rows <= FOLD_MAX_ROWS
    )


def _f384_ok(x, wa, rows):
    return (
        FOLD_D384_CACHED
        and FOLD_D384
        and x.dtype is torch.bfloat16
        and x.shape[-1] == 384
        and wa.shape[0] in (768, 1536)
        and rows % ROWS == 0
        and rows // ROWS > SMALL_FWD_TILES
        and rows >= FOLD_D384_CACHED_MIN_ROWS.get(wa.shape[0] // 384, 1 << 40)
    )


def _sx_fold_ok(x, wa, tiles):
    return (
        SX_FOLD
        and x.dtype is torch.bfloat16
        and x.shape[-1] == 384
        and wa.shape[0] in (768, 1536)
        and _sx(384, tiles)
        and tiles >= SX_FOLD_MIN_TILES.get(wa.shape[0] // 384, 99)
        and tiles * (wa.shape[0] // SX_FOLD_CH[-1]) <= num_sms(x.device.index)  # one wave of 128-unit CTAs at most (more tiles keep the unfolded chain)
    )


def fold_first(x, wa, ws):
    """Whether the folded small-M chain serves this inference call (``route_sm100a`` asks before the ``fused_small`` chains)."""
    d = x.shape[-1]
    rows = x.numel() // d
    return rows % ROWS == 0 and rows > 0 and _sx_fold_ok(x, wa, rows // ROWS) and available(x, wa, ws)


def transition_wide_sm100a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as ``fused_sm100a.transition_fused_sm100a``. Call ``available()`` first: this raises
    rather than falling back. Inference (grad mode off, or nothing requiring grad) runs the forward without the saves."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))):
        shape = x.shape
        from miniworld_engine.kernels import _capture

        rows = x.numel() // shape[-1]
        fold = _sx_fold_ok(x, wa, rows // ROWS) and rows % ROWS == 0
        f384 = _f384_ok(x, wa, rows)
        f512 = _f512_ok(x, wa, rows)

        def forms():
            w16 = _bf16_forms(wa, wb, ws, x.dtype)
            fo = _sx_fold_forms(w16[0], w16[1], gamma, beta) if (fold or f384) else None
            if f384:
                fo = (fo[0], _f384_aug(fo[1], wa.shape[0]))
            if f512:
                fo = _f512_forms(w16[0], w16[1], gamma, beta)
            return (gamma.float().contiguous(), beta.float().contiguous(), w16, *((fo,) if fo is not None else ()))

        g, b, ww, *ff = _capture.param_forms(
            f"transition_wide_sm100a_{x.dtype}" + ("_fold" if fold else "_f384" if f384 else "_f512" if f512 else ""), (gamma, beta, wa, wb, ws), forms, capture_reads_eager=True
        )
        if fold:
            out = _fwd_sxf_launch(x.reshape(-1, shape[-1]).contiguous(), ff[0][0], ff[0][1], ww[2], float(eps))
            return out.reshape(shape)
        if f512:
            out = _fwd_f512_launch(x.reshape(-1, shape[-1]).contiguous(), g, b, ff[0][0], ff[0][1], ff[0][2], ww[2], float(eps))
            return out.reshape(shape)
        if f384:
            out = _fwd_f384_launch(x.reshape(-1, shape[-1]).contiguous(), g, b, ff[0][0], ff[0][1], ww[2], float(eps))
            return out.reshape(shape)
        out, *_ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), g, b, *ww, float(eps), False)
        return out.reshape(shape)
    return _WideTransitionSM100A.apply(x, gamma, beta, wa, wb, ws, eps)

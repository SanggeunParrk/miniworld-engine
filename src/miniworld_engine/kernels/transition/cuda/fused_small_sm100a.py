"""B200 (sm_100a) Transition at SMALL M for the cells the D 384 bf16 chains of ``fused_wide_sm100a`` do not take (round 3, 2026-11-17).

A single-stream activation ``[B, L, D]`` is a few 128-row tiles: a GEMM grid over (row tile, output tile) leaves most SMs idle and each
CTA is bound by its operand intake (every CTA of a tile takes the tile's whole A operand). The small-M chains (``sm100/widths/tsx_*.cu``)
are programmatic dependent launches sized for that regime; this module runs them for

* bf16 activations at D 768 (n 2) -- the chain's K-blocks go through a ring there (the whole K does not fit shared memory), and
* fp32 activations on the TF32 tensor cores at D 384 / 768 (n 2) (``-DF32=1`` builds; only where torch allows TF32),

with the expand's K split over a cluster (``tsx_expand.cu -DSK``: the fp32 partials of ``[a | b]`` meet in distributed shared memory
before the SwiGLU), as the squeeze's already is, so that a tile's operands are spread over many SMs.

    forward    tsx_ln (xn; fp32: on the TF32 grid) -> tsx_expand (h = silu(xn Wa^T) (xn Wb^T)) -> tsx_squeeze (y = h Ws^T + x)

Whole 128-row tiles only (``takes``); everything else keeps the generic kernels. Every reduction has one order: a replay is bit-identical.
"""

from __future__ import annotations

import functools

import torch

from miniworld_engine.kernels._device import num_sms

from ..._compile import opaque
from ..._nvcc import build_failed
from .fused_sm100a import _ext, _is_b200, build_cubins

ROWS = 128
#: (D, n, fp32) cells -> the largest row-tile count the chains take in inference (measured against the generic kernels:
#: docs/b200/transition.md, round 3 part 2)
CELLS: dict[tuple[int, int, bool], int] = {
    (768, 2, False): 6,
    (384, 2, True): 16,
    (768, 2, True): 6,
}
#: expand builds (hidden units per cluster, K split = cluster size) and squeeze builds (output columns per cluster, hidden split)
EXPAND = ((32, 1), (64, 1), (128, 1), (32, 2), (64, 2), (64, 4))
SQUEEZE = ((32, 1), (64, 1), (32, 2), (64, 2), (64, 4))
#: fp32 (accuracy rule: a YELLOW case may not exceed the Triton / H100 tokens). The expand stays unsplit: its partials' order moved the
#: max-norm error (fp32 [1, 128, 768] out 1.0371 x tol with (32, 2), 1.0336 = the Triton token unsplit; grad:ln_in.weight at [1, 128, 384]
#: 1.0298 against 1.0147), and D 384 takes no 4-way squeeze ([1, 384 / 512, 384]: 1.0033 / 1.0067 against 1.0031 with the 2-way one)
EXPAND_F32 = ((32, 1), (64, 1), (128, 1))
SQUEEZE_F32 = {384: ((32, 1), (64, 1), (32, 2), (64, 2)), 768: (*SQUEEZE, (32, 4), (64, 8))}
#: fp32 INFERENCE on the fp16 tensor cores (round 3 part 4, ``-DH16=1``): xn, Wa, Wb, h, Ws as fp16 under ``fused_generic_sm100a._h16_forms`` scales
#: (fp16 = TF32's 10-bit mantissa, rounded to nearest; the H100 f32io forward's scheme and ``tg_fused_h16``'s), at half the TF32 chain's
#: operand bytes. The training forward keeps the TF32 builds (its backward takes TF32 xn / a / b). H16 = False: the TF32 chain (A/B).
H16 = True
EXPAND_H16 = {384: ((32, 1), (64, 1), (128, 1), (32, 2), (64, 2)), 768: EXPAND}
SQUEEZE_H16 = {384: SQUEEZE, 768: (*SQUEEZE, (32, 4), (64, 8))}
#: the fp32 inference cells of the fp16 chain and their largest tile counts (measured against ``fused_generic_sm100a``'s fp16 forward).
#: Round 3 part 7: D 768 up to 12 tiles (row 37 float32 L 1536: TF32 45.1 -> 26.6 us; above it the three-launch generic fp16
#: forward is faster: L 2048 55.3 -> 38.9, the chain 43.0). Part 4 had kept D 768 above 8 tiles on TF32 for YELLOW 1.0054 / 1.0067 x
#: tol, which is inside the engine standard (the accuracy rule since 2026-11-17 evening: GREEN or YELLOW pass)
CELLS_H16: dict[tuple[int, int, bool], int] = {(384, 2, True): 16, (768, 2, True): 12}
#: the fewest tiles a cell takes the fp16 chain from. Round 3 part 7: none (fp32 [1, 128, 384]: TF32 18.0 -> 14.3 us, YELLOW 1.0014 x tol,
#: inside the standard; part 4 had kept that one tile on the TF32 chain for it)
H16_MIN_TILES: dict[int, int] = {}
#: A/B overrides of the picks: None (automatic) or a build of the lists above
EXPAND_FORCE = None
SQUEEZE_FORCE = None
#: (round 3 part 17) the fp16 chain's squeeze on a CTA pair (``tsx_squeeze2.cu -DH16``) at {(D, n, tiles): output columns per pair};
#: H16_PAIR_FORCE: A/B override (0 off, else the columns per pair), None automatic
#: Measured (fine device time, L2 flushed, us, K-split chain -> pair): D 384 1 / 2 / 3 tiles 8.15 / 8.13 / 8.27 -> 7.84 / 7.94 / 8.09 (32),
#: 8 / 12 tiles 10.65 / 11.62 -> 10.35 / 11.34 (64), 16 tiles 14.33 -> 13.78 (128), 4-6 tiles level (kept); D 768 1-3 tiles 10.02 / 11.9-12.5 /
#: 13.2-13.9 -> 9.87 / 11.81 / 12.6-13.4 (32), 4-6 tiles 14.9-15.2 / 15.0 / 15.4 -> 13.9-14.2 / 14.3-14.9 / 14.7-15.4 (64), 8 / 12 tiles
#: 19.7 / 20.9 -> 17.7 / 19.1-19.5 (128). Outputs within 6e-6 of the K-split chain (bit-identical where it was unsplit), same error vs fp64
H16_PAIR_SQ: dict[tuple[int, int, int], int] = {
    **{(384, 2, t): 32 for t in (1, 2, 3)},
    (384, 2, 8): 64,
    (384, 2, 12): 64,
    (384, 2, 16): 128,
    **{(768, 2, t): 32 for t in (1, 2, 3)},
    **{(768, 2, t): 64 for t in (4, 5, 6)},
    **{(768, 2, t): 128 for t in (7, 8, 9, 10, 11, 12)},
}
H16_PAIR_NB = (32, 64, 128)
#: the fp16 chain's expand on a CTA pair (``tsx_expand2.cu -DH16``) at {(D, n, tiles): hidden units per CTA}; H16_PAIR_EX_FORCE as above
#: Measured (fine us, K-split / unsplit chain expand -> pair, with the pair squeezes above): D 384 1 tile 7.86 -> 7.44 (16), 3 / 6 / 8 / 12
#: tiles 8.11 / 9.35 / 10.31 / 11.32 -> 7.85 / 9.25 / 9.75 / 11.18 (32), 16 tiles 13.75 -> 14.57 (not taken); D 768 1 tile 9.90 -> 9.55 (16),
#: 3 tiles 12.47 -> 11.70 (32), 6 tiles 14.74 -> 14.55 (64), 8 tiles 17.73 -> 20.7 (not taken)
H16_PAIR_EX: dict[tuple[int, int, int], int] = {
    (384, 2, 1): 16,
    **{(384, 2, t): 32 for t in range(2, 13)},
    (768, 2, 1): 16,
    **{(768, 2, t): 32 for t in (2, 3)},
    **{(768, 2, t): 64 for t in (4, 5, 6)},
}
H16_PAIR_EX_FORCE: int | None = None
H16_PAIR_UH = (16, 32, 64)
H16_PAIR_FORCE: int | None = None
#: (round 3 part 18) bf16 INFERENCE cells beyond ``CELLS`` / ``PAIR_CELLS`` that take the bf16 chain (K-split expand, then the squeeze
#: on a CTA pair, ``tsx_squeeze2.cu`` bf16, NB columns per pair): {(D, n): (fewest, most tiles)} (``takes`` only: the training forward
#: keeps ``CELLS``) and {(D, n, tiles): NB}; BF16_PAIR_SQ_FORCE: A/B override (0: the K-split squeeze). Measured (fab17.py, graph median
#: of 400 L2-flushed replays / fine device time, us, route -> chain + pair squeeze): D 768 n 2 8 / 12 tiles (generic) 22.5 / 24.5 (fine
#: 18.7 / 19.7) -> 20.5 / 22.5 (16.7 / 18.0) at NB 128 (64: 24.6 / 26.6; K-split squeeze 24.4 / 24.6), 16 tiles 32.8 -> 34.8 (not taken);
#: D 384 n 2 16 tiles (generic) 20.5 (15.2) -> 16.5-18.4 (12.6) at NB 128; D 384 n 4 8 / 12 tiles (generic) 20.5 / 20.5 (15.8 / 16.3) ->
#: 18.4 / 18.4 (12.8 / 13.7) at NB 64, 16 / 31 tiles 24.5 / 30.7 -> 24.6 / 36.9 (not taken). The pair expand (tsx_expand2.cu) in place
#: of the K-split one measured slower at every one of these (UH 16 / 32 / 64: +1-17 us; part 18 notes).
INFER_CELLS: dict[tuple[int, int], tuple[int, int]] = {(768, 2): (7, 16), (384, 2): (16, 16), (384, 4): (8, 31)}
EXPAND_BF16 = {384: EXPAND_H16[384], 768: EXPAND}
BF16_PAIR_SQ: dict[tuple[int, int, int], int] = {
    **{(768, 2, t): 128 for t in range(7, 13)},
    (384, 2, 16): 128,
    **{(384, 4, t): 64 for t in range(8, 13)},
}
BF16_PAIR_NB = (32, 64, 128)
BF16_PAIR_SQ_FORCE: int | None = None
#: (round 3 part 19) the PERSISTENT pair expand (``tsx_expand2p.cu``) of the bf16 inference chain: {(D, n, tiles): (UH, groups, squeeze)},
#: grid 2 x H / (2 UH) x groups CTAs (one wave), each pair's weight panel resident, looping over its 256-row blocks (M = 256 pair products);
#: squeeze 0: the chain's squeeze pick, -1: cuBLAS addmm onto the x copy the LayerNorm launch writes. PEX_FORCE: A/B override ((0, 0, 0):
#: the K-split expand). Measured (docs/b200/transition.md part 19): the cuBLAS squeeze beat the M = 128 pair squeeze (NB 64 / 96 / 128) and
#: an M = 256 pair squeeze (rejected/transition_p19_tsx_squeeze2w.cu) at these cells; UH 64 (N = 256) leaves two A stages beside its
#: panel at D 768 and its SwiGLU epilogue is twice as long (slower everywhere)
PEX_UH = (32,)
PEX: dict[tuple[int, int, int], tuple[int, int, int]] = {
    **{(768, 2, t): (32, 3, -1) for t in range(13, 17)},
    **{(384, 4, t): (32, 3, -1) for t in range(13, 32)},
}
PEX_FORCE: tuple[int, int, int] | None = None


def _pex_kbx(d: int, uh: int) -> int:
    """K-blocks per A ring stage of ``tsx_expand2p.cu``: the largest of 4 / 3 / 2 / 1 dividing D / 64 that leaves three ring stages
    beside the resident weights (two if none does; 0: none fits)."""
    nkb = d // 64
    avail = 232448 - 1024 - 512 - nkb * 2 * uh * 128
    ok = [k for k in (4, 3, 2, 1) if nkb % k == 0]
    return next((k for k in ok if avail // (k * 16384) >= 3), next((k for k in ok if avail // (k * 16384) >= 2), 0))  # 0: no build


def _tf32_allowed() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32) or torch.get_float32_matmul_precision() != "highest"


def _specs(d: int, n: int, f32: bool, h16: bool = False) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """(name, source, flags) of every cubin of one cell (``_load`` builds them; the sm100 prebuild plan lists them); ``h16``: the fp16-operand
    inference builds of an fp32 cell."""
    if h16:
        dh = (f"-DDIM={d}", f"-DHID={n * d}", "-DH16=1")
        return (
            ("ln", "widths/tsx_ln.cu", (f"-DDIM={d}", "-DH16=1")),
            *((f"expand{c}_{k}", "widths/tsx_expand.cu", (*dh, f"-DCH={c}", f"-DSK={k}")) for c, k in EXPAND_H16[d]),
            *((f"sq{nb}_{k}", "widths/tsx_squeeze.cu", (*dh, f"-DNBW={nb}", f"-DSK={k}")) for nb, k in SQUEEZE_H16[d]),
            *((f"psq{nb}", "widths/tsx_squeeze2.cu", (*dh, f"-DNBW={nb}", f"-DKBX={_kbx(n * d // 64)}")) for nb in H16_PAIR_NB),
            *((f"pex{u}", "widths/tsx_expand2.cu", (*dh, f"-DUH={u}", f"-DKBX={_kbx(d // 64)}")) for u in H16_PAIR_UH),
        )
    dim = (f"-DDIM={d}", f"-DHID={n * d}", f"-DF32={int(f32)}")
    dh = (f"-DDIM={d}", f"-DHID={n * d}")
    pair = (
        ()
        if f32
        else (
            *((f"psq{nb}", "widths/tsx_squeeze2.cu", (*dh, f"-DNBW={nb}", f"-DKBX={_kbx(n * d // 64)}")) for nb in BF16_PAIR_NB),
            *((f"pex{u}", "widths/tsx_expand2p.cu", (*dh, f"-DUH={u}", f"-DKBX={_pex_kbx(d, u)}")) for u in PEX_UH if _pex_kbx(d, u)),
        )
    )
    return (
        ("ln", "widths/tsx_ln.cu", (f"-DDIM={d}", f"-DF32={int(f32)}")),
        *((f"expand{c}_{k}", "widths/tsx_expand.cu", (*dim, f"-DCH={c}", f"-DSK={k}", *EXPAND_X)) for c, k in (EXPAND_F32 if f32 else EXPAND_BF16[d])),
        *((f"sq{nb}_{k}", "widths/tsx_squeeze.cu", (*dim, f"-DNBW={nb}", f"-DSK={k}")) for nb, k in (SQUEEZE_F32[d] if f32 else SQUEEZE)),
        *pair,
    )


def _tag(d: int, n: int, f32: bool, h16: bool = False) -> str:
    return f"small_d{d}n{n}{'h16' if h16 else 'f32' if f32 else 'bf16'}"


@functools.lru_cache(maxsize=8)
def _load(d: int, n: int, f32: bool, h16: bool = False):
    ext = _ext()
    tag = _tag(d, n, f32, h16)
    for name, path in build_cubins(tag, _specs(d, n, f32, h16)).items():
        ext.load_cubin(f"{tag}_{name}", path)
    return ext


def _is_fake(*tensors) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor

    return any(isinstance(t, FakeTensor) for t in tensors)


def _h16(d: int, rows: int, f32: bool) -> bool:
    """Whether an fp32 inference call of these rows takes the fp16 chain (``H16``, ``H16_MIN_TILES``)."""
    return f32 and H16 and rows // ROWS >= H16_MIN_TILES.get(d, 1)


def takes(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor, grad: bool) -> bool:
    """Whether the small-M chains serve this call (shape / dtype / card only: no build): a cell of ``CELLS``, whole 128-row tiles up to
    the cell's tile count, inference."""
    if grad or x.device.type != "cuda" or x.dtype not in (torch.bfloat16, torch.float32) or x.numel() == 0:
        return False
    d = x.shape[-1]
    hid = wa.shape[0]
    f32 = x.dtype is torch.float32
    rows = x.numel() // d
    if hid % d or tuple(wa.shape) != (hid, d) or tuple(ws.shape) != (d, hid):
        return False
    cell = (d, hid // d, f32)
    cells = CELLS_H16 if _h16(d, rows, f32) else CELLS
    if rows % ROWS and (f32 or (PEX_FORCE or PEX.get((d, hid // d, -(-rows // ROWS)), (0,)))[0] == 0):
        return False  # (part 19) ragged rows: the persistent expand only (TMA zero-fills, stores masked)
    tiles = -(-rows // ROWS)
    lo, hi = INFER_CELLS.get(cell[:2], (1, 0)) if not f32 else (1, 0)
    if not (cell in cells and tiles <= cells[cell]) and not lo <= tiles <= hi:
        return False
    if any(w.dtype not in (torch.bfloat16, torch.float32) for w in (wa, ws)) or (f32 and not _tf32_allowed()):
        return False
    return _is_b200(x.device.index if x.device.index is not None else torch.cuda.current_device())


_BUILD_FAILED = False


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor, grad: bool) -> bool:
    """``takes()`` plus a successful (cached) build."""
    if _BUILD_FAILED or not takes(x, wa, ws, grad):
        return False
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        d = x.shape[-1]
        f32 = x.dtype is torch.float32
        _load(d, wa.shape[0] // d, f32, _h16(d, x.numel() // d, f32))
    except Exception as exc:
        build_failed("small-M sm100a Transition", exc)
        return False
    return True


#: bytes of per-CTA operand intake one more CTA in a cluster is worth (``_pick``)
_SPLIT_COST = {2: 120000, 1: 40000}  # by accumulators per CTA: the expand (2), the squeeze (1)


def _pick(builds, tiles: int, units: int, a_rows: int, k: int, esz: int, nsm: int, nacc: int) -> tuple[int, int]:
    """(width, K split) of a small-M GEMM launch. Each CTA takes the tile's A rows and ``nacc`` x width B rows over its K slice (a
    cluster of ``split`` CTAs per (tile, width) block) and runs once: among the builds whose clusters fit ONE wave, the least per-CTA
    operand bytes plus ``_SPLIT_COST`` per extra CTA of the cluster (the exchange of the partials); when none fits, the fewest CTAs.
    ``units``: the N of the product (hidden units of the expand, output columns of the squeeze). Measured (fp32 [1, 128, 768] n 2, us,
    L2 flushed): expand (32, 1) 22.5, (32, 2) 20.5, (64, 4) 22.5; squeeze (32, 1) 28.7, (64, 2) 24.5, (64, 4) / (32, 4) / (64, 8) 22.5;
    fp32 [1, 768, 384]: expand (32, 1) / (64, 1) 22.5, (64, 2) 24.6; squeeze (32, 1) 22.5, (32, 2) / (64, 2) 20.5.
    A second wave costs more than any split saves ([1, 768, 768]: expand (64, 1) 30.7, (64, 2) 36.9, (64, 4) 55.3)."""
    best = None
    for w, s in builds:
        if units % w:
            continue
        clusters = tiles * (units // w)
        # resident clusters: pairs pack every SM; wider clusters must fit a GPC, measured ~3 / 4 of the SMs (a second wave of
        # 4-CTA clusters at fp32 [1, 768, 384]: squeeze (64, 4) 24.6 us against (32, 2) 20.5)
        fits = clusters <= (nsm // s if s <= 2 else (3 * nsm // 4) // s)
        cost = (a_rows + nacc * w) * (k // s) * esz + _SPLIT_COST[nacc] * (s - 1) if fits else clusters * s
        if best is None or (not fits, cost) < best[0]:
            best = ((not fits, cost), w, s)
    assert best is not None
    return best[1], best[2]


def chain_takes(m: int, d: int, hid: int, f32: bool) -> bool:
    """Whether the forward chain runs an [m, d] activation (shape only; the route and the generic training forward both ask)."""
    return hid % d == 0 and (d, hid // d, f32) in CELLS and m % ROWS == 0 and 0 < m // ROWS <= CELLS[(d, hid // d, f32)]


#: (round 3 part 14) the CTA-pair chain of the bf16 D 768 n 2 INFERENCE cell (row 37 single stream): tsx_ln -> tsx_expand2.cu
#: -> #: tsx_squeeze2.cu (cta_group::2, M = 128 pair products: each SM takes 64 rows of the tile and half the pair's weights, the whole K
#: accumulates in tensor memory, no K split and no partial exchange) with the 64-row operand K-blocks KBX at a time in one 3-D TMA box
#: (the per-SM TMA box rate, not the bytes, paced the single-CTA chain: docs/b200/transition.md, part 14). {(D, n): largest tile count}.
#: D 384 (rows 35 / 36) measured equal to its chain (14.3 / 16.4 us) and is not routed.
PAIR_CELLS: dict[tuple[int, int], int] = {(768, 2): 6}
PAIR = True
PAIR_UH = (16, 32, 64)  # hidden units per expand CTA (a pair: 2 UH)
PAIR_NB = (32, 64)  # output columns per squeeze pair
#: measured picks {(D, n, tiles): (UH, NB)} (graph medians, L2 flushed, part 14 sweep: [1, 128, 768] (16, 32) 14.3 us against (16, 16)
#: 16.4; 2 / 3 tiles (32, 32) 16.4; 4-6 tiles (64, 64) 18.4-20.5)
PAIR_PICK: dict[tuple[int, int, int], tuple[int, int]] = {
    (768, 2, 1): (16, 32),
    (768, 2, 2): (32, 32),
    (768, 2, 3): (32, 32),
    (768, 2, 4): (64, 64),
    (768, 2, 5): (64, 64),
    (768, 2, 6): (64, 64),
}
#: A/B override: ((UH, KBX), (NB, KBX)) or None
PAIR_FORCE = None


def _kbx(nkb: int, want: int = 4) -> int:
    return max(k for k in range(1, want + 1) if nkb % k == 0)


def pair_takes(m: int, d: int, h: int) -> bool:
    """Whether the pair chain runs a bf16 inference call of [m, d] (whole 128-row tiles of a ``PAIR_CELLS`` cell)."""
    cell = (d, h // d) if h % d == 0 else None
    return PAIR and cell in PAIR_CELLS and m % ROWS == 0 and 0 < m // ROWS <= PAIR_CELLS[cell]


def _pair_pick(d: int, h: int, tiles: int, nsm: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """((UH, KBX) of the expand, (NB, KBX) of the squeeze): ``PAIR_PICK``, else the narrowest CTA work whose grid fits one wave."""
    uh, nb = PAIR_PICK.get((d, h // d, tiles), (0, 0))
    uh = uh or next((u for u in PAIR_UH if tiles * (h // u) <= nsm), PAIR_UH[-1])
    nb = nb or next((b for b in PAIR_NB if tiles * (d // b) * 2 <= nsm), PAIR_NB[-1])
    return (uh, _kbx(d // 64)), (nb, _kbx(h // 64))


def _pair_specs(d: int, n: int) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    dh = (f"-DDIM={d}", f"-DHID={n * d}")
    ke, kq = _kbx(d // 64), _kbx(n * d // 64)
    return (
        ("ln", "widths/tsx_ln.cu", (f"-DDIM={d}", "-DF32=0")),
        *((f"ex{u}_{ke}", "widths/tsx_expand2.cu", (*dh, f"-DUH={u}", f"-DKBX={ke}")) for u in PAIR_UH),
        *((f"sq{b}_{kq}", "widths/tsx_squeeze2.cu", (*dh, f"-DNBW={b}", f"-DKBX={kq}")) for b in PAIR_NB),
        *((f"exf{u}", "widths/tsx_expand2.cu", (*dh, f"-DUH={u}", f"-DKBX={PAIR_FOLD_KBX[u]}", "-DFOLD=1", *PAIR_FOLD_X)) for u in PAIR_UH if PAIR_FOLD),
    )


#: (N5) the LayerNorm folded into the pair expand (``tsx_expand2.cu -DFOLD=1``): no ``tsx_ln`` launch, two launches. False: the three-launch pair chain (A/B)
PAIR_FOLD = True
PAIR_FOLD_X: tuple[str, ...] = ()  # extra nvcc flags of the folded expand (probes)
EXPAND_X: tuple[str, ...] = ()  # (N9) extra nvcc flags of the bf16 and fp32 expand (SIGNEW, SIGPAT)
PAIR_STAMP = None  # probe: the %globaltimer stamp buffer of a -DTSX_STAMP=1 build
#: the tile counts of a pair cell that take the fold (set by measurement: ``noln.py`` / ``pfold.py``)
PAIR_FOLD_TILES = {(768, 2): (4, 5, 6)}


def pair_fold_takes(m: int, d: int, h: int) -> bool:
    """Whether the folded pair chain runs a bf16 inference call of [m, d] (a ``pair_takes`` call whose tile count is in ``PAIR_FOLD_TILES``)."""
    return PAIR_FOLD and pair_takes(m, d, h) and m // ROWS in PAIR_FOLD_TILES.get((d, h // d), ())


def _pair_fold_forms(wa, wb, gamma, beta):
    """(Wp, aug) of the folded pair expand, from the bf16 forms of Wa / Wb: Wp = bf16(W gamma) for [Wa; Wb] ([2 H, D]); aug [H, 4] fp32 =
    [u_a, v_a, u_b, v_b] with u_n = sum_k Wp_nk (the rounded values) and v_n = sum_k beta_k W_nk. A weight form: made once per weights."""
    h = wa.shape[0]
    w = torch.cat((wa, wb)).float()
    wp = (w * gamma.float()).to(torch.bfloat16).contiguous()
    u = wp.float().sum(1)
    v = (w * beta.float()).sum(1)
    d = wp.shape[1]
    # the B operand per expand build: chunk c of UH units = [Wa_c; Wb_c] (2 UH rows) contiguous, so a CTA's weight stage is ONE 3-D box
    il = tuple(torch.cat((wp[:h].reshape(h // uh, uh, d), wp[h:].reshape(h // uh, uh, d)), 1).reshape(2 * h, d).contiguous() for uh in PAIR_UH)
    return (*il, torch.stack((u[:h], v[:h], u[h:], v[h:]), 1).contiguous())


#: K-blocks per stage of the folded expand by UH (the x tile and the interleaved Wp each in one 3-D box a stage; the ring holds the stages that fit)
PAIR_FOLD_KBX = {16: 4, 32: 4, 64: 2}


def pair_fold_chain(x, wi16, wi32, wi64, aug, ws, eps: float, out=None):
    """x bf16 [M, D] -> out = transition(x) + x in two programmatic dependent launches: the pair expand with the LayerNorm folded in
    (raw x as the A operand, Wp = bf16(W gamma) interleaved by chunk, the rank-1 terms from ``aug`` in the epilogue), the pair squeeze + residual."""
    m, d = x.shape
    h = ws.shape[1]
    tiles = m // ROWS
    tag = f"pair_d{d}n{h // d}"
    ext = _pair_load(d, h // d)
    (uh, _), (nb, kq) = PAIR_FORCE or _pair_pick(d, h, tiles, num_sms(x.device))
    kx = PAIR_FOLD_KBX[uh]
    wi = {16: wi16, 32: wi32, 64: wi64}[uh]
    t = ext.tmap
    hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x) if out is None else out
    ext.launch_kernel(
        f"{tag}_exf{uh}",
        "transition_sx_expand2",
        tiles * (h // uh),
        256 if uh <= 32 else 384,
        2,
        [t(x, d, m, 64, 64, -1, 0, kx), t(wi, d, 2 * h, 64, 2 * uh, -1, 0, kx), t(wi, d, 2 * h, 64, 2 * uh, -1, 0, kx), hid, aug, float(eps), x, *([PAIR_STAMP] if PAIR_STAMP is not None else [])],
        True,
    )
    sq_args = [t(hid, h, m, 64, 64, -1, 0, kq), t(ws, h, d, 64, nb // 2), x, out]
    ext.launch_kernel(f"{tag}_sq{nb}_{kq}", "transition_sx_squeeze2", tiles * (d // nb) * 2, 256, 2, sq_args, True)
    return out


def _pair_fold_fake(x, wi16, wi32, wi64, aug, ws, eps):
    return torch.empty_like(x)


@opaque(fake=_pair_fold_fake, name="transition_small_pairfold_fwd_sm100a")
def _pair_fold_launch(x: torch.Tensor, wi16: torch.Tensor, wi32: torch.Tensor, wi64: torch.Tensor, aug: torch.Tensor, ws: torch.Tensor, eps: float) -> torch.Tensor:
    """x [M, D] -> transition(x) + x (inference) from the folded pair chain's weight forms."""
    if _is_fake(x, wi16):
        return _pair_fold_fake(x, wi16, wi32, wi64, aug, ws, eps)
    return pair_fold_chain(x, wi16, wi32, wi64, aug, ws, eps)


@functools.lru_cache(maxsize=8)
def _pair_load(d: int, n: int):
    ext = _ext()
    for name, path in build_cubins(f"pair_d{d}n{n}", _pair_specs(d, n)).items():
        ext.load_cubin(f"pair_d{d}n{n}_{name}", path)
    return ext


def pair_chain(x, gamma, beta, wa, wb, ws, eps: float, out=None):
    """x bf16 [M, D] (whole tiles of a ``PAIR_CELLS`` cell) -> out = transition(x) + x in three programmatic dependent launches:
    the LayerNorm, the pair expand + SwiGLU, the pair squeeze + residual (gamma / beta fp32, weights bf16)."""
    m, d = x.shape
    h = wa.shape[0]
    tiles = m // ROWS
    tag = f"pair_d{d}n{h // d}"
    ext = _pair_load(d, h // d)
    (uh, ke), (nb, kq) = PAIR_FORCE or _pair_pick(d, h, tiles, num_sms(x.device))
    t = ext.tmap
    xn = torch.empty_like(x)
    hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x) if out is None else out
    st = torch.empty((1,), dtype=torch.float32, device=x.device)
    ln_args = [x, gamma, beta, xn, st, st, m, float(eps), 0, x, x, x, wa, wb, ws, 0]
    ext.launch_kernel(f"{tag}_ln", "transition_sx_ln", (m + 7) // 8, 256, 1, ln_args, True)
    ext.launch_kernel(
        f"{tag}_ex{uh}_{ke}",
        "transition_sx_expand2",
        tiles * (h // uh),
        256 if uh <= 32 else 384,
        2,
        [t(xn, d, m, 64, 64, -1, 0, ke), t(wa, d, h, 64, uh), t(wb, d, h, 64, uh), hid],
        True,
    )
    sq_args = [t(hid, h, m, 64, 64, -1, 0, kq), t(ws, h, d, 64, nb // 2), x, out]
    ext.launch_kernel(f"{tag}_sq{nb}_{kq}", "transition_sx_squeeze2", tiles * (d // nb) * 2, 256, 2, sq_args, True)
    return out


def chain(x, gamma, beta, wa, wb, ws, eps: float, save: bool, cast=None):
    """The three programmatic dependent launches on x [M, D] (whole 128-row tiles; the weights in x's dtype, fp32 on the TF32 grid).
    Returns (out, xn, mean, rstd, ab) as ``fused_generic_sm100a._fwd_launch``: out = transition(x) + x; with ``save`` the row statistics
    and [a | b] [M, 2H] for the generic backward (GATE_AB), else placeholders.
    ``cast`` (round 3, training): the fp32 masters (wa, wb, ws) that the LayerNorm launch's extra blocks round / cast into the forms wa, wb, ws
    this call is given (the per-step weight forms in that launch; the expand then loads its weights after griddepcontrol.wait)."""
    m, d = x.shape
    h = wa.shape[0]
    f32 = x.dtype is torch.float32
    if not save and cast is None and not f32 and pair_takes(m, d, h):
        ph = torch.empty((1,), dtype=torch.float32, device=x.device)
        return pair_chain(x, gamma, beta, wa, wb, ws, eps), x.new_empty((1, d)), ph, ph, x.new_empty((1, 1))
    ext = _load(d, h // d, f32)
    tag = _tag(d, h // d, f32)
    nsm = num_sms(x.device)
    tiles = -(-m // ROWS)  # (part 19) whole tiles, or a ragged last one on the persistent expand (``takes``)
    esz, bke = (4, 32) if f32 else (2, 64)
    t = ext.tmap
    xn = torch.empty_like(x)
    hid = torch.empty((m, h), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    mean = torch.empty((m if save else 1,), dtype=torch.float32, device=x.device)
    rstd = torch.empty_like(mean)
    ab = torch.empty((m, 2 * h), dtype=x.dtype, device=x.device) if save else x.new_empty((1, 1))
    infer16 = not f32 and not save and cast is None
    puh, ptg, psq = (PEX_FORCE if PEX_FORCE is not None else PEX.get((d, h // d, tiles), (0, 0, 0))) if infer16 else (0, 0, 0)
    # (part 19) with the cuBLAS squeeze the LayerNorm launch also copies x into out (save 3), which addmm_ accumulates onto
    lsave, lout = (3, (out, wb, ws)) if psq < 0 else (2 if save else 0, (wa, wb, ws))
    ext.launch_kernel(
        f"{tag}_ln",
        "transition_sx_ln",
        (m + 7) // 8 + (-(-3 * h * d // 4096) if cast else 0),  # (cast: 16 elements per thread)
        256,
        1,
        [x, gamma, beta, xn, rstd, mean, m, float(eps), lsave, *(cast or (x, x, x)), *lout, h * d if cast else 0],
        True,
    )
    if puh:  # (part 19) the persistent pair expand: resident weight panels, the pair loops over its 256-row blocks
        kx = _pex_kbx(d, puh)
        ext.launch_kernel(
            f"{tag}_pex{puh}",
            "transition_sx_expand2p",
            2 * (h // (2 * puh)) * ptg,
            384,
            2,
            [t(xn, d, m, 64, 128, -1, 0, kx), t(wa, d, h, 64, puh), t(wb, d, h, 64, puh), hid, m, ptg],
            True,
        )
    else:
        ch, sk = EXPAND_FORCE or _pick(EXPAND_F32 if f32 else EXPAND_BF16[d], tiles, h, ROWS, d, esz, nsm, 2)
        ext.launch_kernel(
            f"{tag}_expand{ch}_{sk}",
            "transition_sx_expand",
            tiles * (h // ch) * sk,
            384,
            sk,
            [t(xn, d, m, bke, 128), t(wa, d, h, bke, ch), t(wb, d, h, bke, ch), hid, ab, ab, int(save), int(cast is None), 2 * h, h],
            True,
        )
    if psq < 0:  # cuBLAS (one launch, K split inside)
        out.addmm_(hid, ws.t())
        return out, xn, mean, rstd, ab
    # (part 18) the bf16 inference chain's squeeze on a CTA pair (BF16_PAIR_SQ)
    pnb = (BF16_PAIR_SQ_FORCE if BF16_PAIR_SQ_FORCE is not None else BF16_PAIR_SQ.get((d, h // d, tiles), 0)) if infer16 else 0
    if pnb:
        kq = _kbx(h // 64)
        sq_args = [t(hid, h, m, 64, 64, -1, 0, kq), t(ws, h, d, 64, pnb // 2), x, out]
        ext.launch_kernel(f"{tag}_psq{pnb}", "transition_sx_squeeze2", tiles * (d // pnb) * 2, 256, 2, sq_args, True)
        return out, xn, mean, rstd, ab
    nb, sq = SQUEEZE_FORCE or _pick(SQUEEZE_F32[d] if f32 else SQUEEZE, tiles, d, ROWS, h, esz, nsm, 1)
    ext.launch_kernel(
        f"{tag}_sq{nb}_{sq}",
        "transition_sx_squeeze",
        tiles * (d // nb) * sq,
        256,
        sq,
        [t(hid, h, m, bke, 128), t(ws, h, d, bke, nb), x, out],
        True,
    )
    return out, xn, mean, rstd, ab


def chain_h16(x, gamma, beta, wa16, wb16, ws16, sc, eps: float):
    """The fp16-operand chain (inference) on fp32 x [M, D]: ``chain``'s three launches with the ``-DH16=1`` builds and the ``_h16_forms``
    weights / scales ``sc``. Returns out = transition(x) + x (fp32)."""
    m, d = x.shape
    h = wa16.shape[0]
    n = h // d
    ext = _load(d, n, True, True)
    tag = _tag(d, n, True, True)
    nsm = num_sms(x.device)
    tiles = m // ROWS
    t = ext.tmap
    v = lambda w: w.view(torch.bfloat16)  # noqa: E731  (TMA copies bytes: the fp16 operands go through 2-byte maps)
    xn = torch.empty((m, d), dtype=torch.float16, device=x.device)
    hid = torch.empty((m, h), dtype=torch.float16, device=x.device)
    out = torch.empty_like(x)
    ext.launch_kernel(f"{tag}_ln", "transition_sx_ln", (m + 7) // 8, 256, 1, [x, gamma, beta, xn, sc, m, float(eps)], True)
    puh = H16_PAIR_EX_FORCE if H16_PAIR_EX_FORCE is not None else H16_PAIR_EX.get((d, n, tiles), 0)
    if puh:  # the CTA-pair expand (tsx_expand2.cu -DH16): half the tile's xn and half the pair's weights per SM, the whole K
        ke = _kbx(d // 64)
        ext.launch_kernel(
            f"{tag}_pex{puh}",
            "transition_sx_expand2",
            tiles * (h // puh),
            256 if puh <= 32 else 384,
            2,
            [t(v(xn), d, m, 64, 64, -1, 0, ke), t(v(wa16), d, h, 64, puh), t(v(wb16), d, h, 64, puh), hid, sc],
            True,
        )
    else:
        _expand_h16(ext, tag, t, v, xn, wa16, wb16, hid, sc, d, h, m, tiles, nsm)
    _squeeze_h16(ext, tag, t, v, hid, ws16, x, out, sc, d, n, h, m, tiles, nsm)
    return out


def _expand_h16(ext, tag, t, v, xn, wa16, wb16, hid, sc, d, h, m, tiles, nsm):
    ch, sk = EXPAND_FORCE or _pick(EXPAND_H16[d], tiles, h, ROWS, d, 2, nsm, 2)
    ext.launch_kernel(
        f"{tag}_expand{ch}_{sk}",
        "transition_sx_expand",
        tiles * (h // ch) * sk,
        384,
        sk,
        [t(v(xn), d, m, 64, 128), t(v(wa16), d, h, 64, ch), t(v(wb16), d, h, 64, ch), hid, hid, hid, 0, 1, h, 0, sc],
        True,
    )


def _squeeze_h16(ext, tag, t, v, hid, ws16, x, out, sc, d, n, h, m, tiles, nsm):
    pnb = H16_PAIR_FORCE if H16_PAIR_FORCE is not None else H16_PAIR_SQ.get((d, n, tiles), 0)
    if pnb:  # the CTA-pair squeeze (tsx_squeeze2.cu -DH16): the whole hidden width per pair, no partial exchange
        kq = _kbx(h // 64)
        ext.launch_kernel(
            f"{tag}_psq{pnb}",
            "transition_sx_squeeze2",
            tiles * (d // pnb) * 2,
            256,
            2,
            [t(v(hid), h, m, 64, 64, -1, 0, kq), t(v(ws16), h, d, 64, pnb // 2), x, out, sc],
            True,
        )
        return
    nb, sq = SQUEEZE_FORCE or _pick(SQUEEZE_H16[d], tiles, d, ROWS, h, 2, nsm, 1)
    ext.launch_kernel(
        f"{tag}_sq{nb}_{sq}",
        "transition_sx_squeeze",
        tiles * (d // nb) * sq,
        256,
        sq,
        [t(v(hid), h, m, 64, 128), t(v(ws16), h, d, 64, nb), x, out, sc],
        True,
    )


def _h16_fake(x, gamma, beta, wa16, wb16, ws16, sc, eps):
    return torch.empty_like(x)


@opaque(fake=_h16_fake, name="transition_small_h16_fwd_sm100a")
def _h16_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    wa16: torch.Tensor,
    wb16: torch.Tensor,
    ws16: torch.Tensor,
    sc: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if _is_fake(x, wa16):
        return _h16_fake(x, gamma, beta, wa16, wb16, ws16, sc, eps)
    return chain_h16(x, gamma, beta, wa16, wb16, ws16, sc, eps)


def _fwd_fake(x, gamma, beta, wa, wb, ws, eps):
    return torch.empty_like(x)


@opaque(fake=_fwd_fake, name="transition_small_fwd_sm100a")
def _fwd_launch(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, eps: float
) -> torch.Tensor:
    """x [M, D] -> transition(x) + x (inference): ``chain`` without the saves."""
    if _is_fake(x, wa):
        return _fwd_fake(x, gamma, beta, wa, wb, ws, eps)
    return chain(x, gamma, beta, wa, wb, ws, eps, False)[0]


def _forms(wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, f32: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernels' weight forms: fp32 on the TF32 grid (``fused_generic_sm100a._prep``) or bf16 (``fused_wide_sm100a._bf16_forms``)."""
    if f32:
        from .fused_generic_sm100a import _prep

        return _prep(wa, wb, ws, True, False)[:3]
    from .fused_wide_sm100a import _bf16_forms

    return _bf16_forms(wa, wb, ws, torch.bfloat16)


def transition_small_sm100a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry (inference), same signature as ``fused_sm100a.transition_fused_sm100a``. Call ``available()`` first. The
    weight forms are made once per parameter version (``_capture.param_forms``)."""
    from miniworld_engine.kernels import _capture

    shape = x.shape
    f32 = x.dtype is torch.float32
    if _h16(shape[-1], x.numel() // shape[-1], f32):
        from .fused_generic_sm100a import _h16_forms

        g, b, ww = _capture.param_forms(
            "transition_small_h16_sm100a",
            (gamma, beta, wa, wb, ws),
            lambda: (gamma.float().contiguous(), beta.float().contiguous(), _h16_forms(gamma, beta, wa, wb, ws)),
            capture_reads_eager=True,
        )
        return _h16_launch(x.reshape(-1, shape[-1]).contiguous(), g, b, *ww, float(eps)).reshape(shape)
    if not f32 and pair_fold_takes(x.numel() // shape[-1], shape[-1], wa.shape[0]):
        def forms():
            w = _forms(wa, wb, ws, False)
            return (w[2], _pair_fold_forms(w[0], w[1], gamma, beta))

        ws16, (wi16, wi32, wi64, aug) = _capture.param_forms("transition_small_pairfold_sm100a", (gamma, beta, wa, wb, ws), forms, capture_reads_eager=True)
        return _pair_fold_launch(x.reshape(-1, shape[-1]).contiguous(), wi16, wi32, wi64, aug, ws16, float(eps)).reshape(shape)
    g, b, ww = _capture.param_forms(
        f"transition_small_sm100a_{x.dtype}",
        (gamma, beta, wa, wb, ws),
        lambda: (gamma.float().contiguous(), beta.float().contiguous(), _forms(wa, wb, ws, f32)),
        capture_reads_eager=True,
    )
    return _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), g, b, *ww, float(eps)).reshape(shape)


__all__ = ["CELLS", "available", "takes", "transition_small_sm100a"]

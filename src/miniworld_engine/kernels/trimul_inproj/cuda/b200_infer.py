"""B200 (sm_100a) TriMul inference for every width D = 64 / 128 / 256 / 384 / 512, bidirectional or one direction, bf16, B samples
per launch (tokens b-major: M = B L L; every per-token kernel is token-flat, the token mask and the row-dropout scale are read per
sample, the contraction batches over channels x B).

Front (every D): ``k1w`` -- input LN + gated projections + mask -> channel-major planes (tcgen05; D <= 128 computes the LN
statistics in-kernel, D >= 256 takes them from ``k1w_stats``). Contractions: cuBLAS, or ``ctr`` (tcgen05, one programmatic-dependent
launch for both directions) in the small-L cells of ``CTR_CELLS``. Output:
  D <= 128  ``k3g`` -- output LN + output projection + gate + residual in one kernel (the D128 K3 generalised over (D, H)).
  D >= 256  ``k3w`` -- both output GEMMs on the raw operands (t as stored, channel-major; x) with the two LayerNorms folded into
            the weights (``wide_fold_prep``) and undone per token in the epilogue from ``wide_ln_stats`` / ``k1w_stats``;
            gate, dropout and residual in the same epilogue. Nothing between the contraction and y touches HBM but t.
Sources: ``b200_sources/{k1w,k3g,k3w,wide_aux}.cu``. Every call owns its buffers.
"""

import functools
import os
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_SRC = Path(__file__).with_name("b200_sources")
WIDTHS = (64, 128, 256, 384, 512)
EPS = 1e-5
#: k1w computes the input LayerNorm statistics itself up to this width (above it: ``k1w_stats`` first). The sources take it as -DK1W_INSTAT.
K1W_INSTAT = 256
#: extra nvcc defines of the extension (kernel variants; part of its cache key)
DEFS: tuple[str, ...] = ()


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="trimul_b200_v6",
        sources=[
            str(_SRC / f)
            for f in (
                "k1w.cu",
                "k3g.cu",
                "k3w.cu",
                "ctr.cu",
                "k1h.cu",
                "k3h.cu",
                "wide_aux.cu",
                "k1wb.cu",
                "wide_bwd.cu",
                "b1s.cu",
                "b7m.cu",
                "b1g.cu",
                "b7g.cu",
                "bind_b200.cpp",
            )
        ],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            *gencodes("100a"),
            "--expt-relaxed-constexpr",
            "-DNDEBUG",
            f"-I{_SRC}",
            *((f"-DK1W_INSTAT={K1W_INSTAT}",) if K1W_INSTAT != 256 else ()),
            *DEFS,
        ],  # fmt: skip
        extra_cflags=["-std=c++17", f"-I{_SRC}"],
        verbose=False,
    )


#: extra nvcc defines of the L128 one-launch extension (``tf_sm100.cu``)
TF_DEFS: tuple[str, ...] = ()


@functools.lru_cache(maxsize=1)
def _tf_ext():
    """``tf_sm100.cu``: the D64 one-direction L = 128 inference update in one cluster launch (front -> plane-row counters ->
    contraction on the output tokens' CTAs, t exchanged through DSMEM -> LN_out + output projection + gate + residual)."""
    ensure_cuda_home()
    return load_extension(
        name="trimul_b200_l128_v14",
        sources=[str(_SRC / "tf_sm100.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("100a"), "--expt-relaxed-constexpr", "-DNDEBUG", f"-I{_SRC}", *TF_DEFS],
        extra_cflags=["-std=c++17", f"-I{_SRC}"],
        verbose=False,
    )


#: the one-launch L128 path (``_tf_ext``); env TRIMUL_B200_L128=0 runs the three-kernel chain (A/B)
FUSED = os.environ.get("TRIMUL_B200_L128", "1") == "1"


def _fused_ok(x, n, bsz, d, hid, direction, ds, wg, wp, front) -> bool:
    if not (FUSED and n == 128 and bsz == 1 and d == 64 and hid == 64 and direction in (1, 2) and ds is None):
        return False
    ws = (*front, wg, wp)
    if not all(w.dtype == torch.bfloat16 and w.shape == (64, 64) and w.stride(1) == 1 for w in ws):
        return False
    return bool(_tf_ext().tf_supported())


def supports(width: int, length: int, dropout: bool = False, hidden: int | None = None, direction: int = 0) -> bool:
    """k1w / k3g / k3w tile 128 tokens and ln_stats 256, so L is a multiple of 16; a dropout scale on the fused D <= 128 output
    goes through k3g's training epilogue, which groups tiles by column (L a multiple of 128). The hidden (contraction) width is
    the pair width, or -- one direction, D64 / D128 -- twice it (see ``hidden_ok``)."""
    if width not in WIDTHS or length <= 0 or length % 16:
        return False
    if not hidden_ok(width, width if hidden is None else hidden, direction):
        return False
    return not (dropout and width <= 128 and length % 128)


def hidden_ok(width: int, hidden: int, direction: int) -> bool:
    """hidden == width; or hidden == 2 width in one direction at D64 / D128 (Protenix's template TriMul: pair 64, hidden 128); or
    hidden == 2 width bidirectional at D64 (pair 64, hidden 128 per direction).
    One direction at hidden 2 D has, per token, exactly the bidirectional kernels' shapes -- 4 D planes out of k1w, an LN_out over 2 D
    channels into k3g, the same b1s / b1g / b7m / b7g backward -- only the cuBLAS contraction between them pairs the planes differently
    (all 2 D channels in one direction instead of D per direction), so the D64 / D128 kernels serve it unchanged. Bidirectional at
    hidden 2 D (D64 only) is k1w at 8 D = 512 planes and k3g at an LN_out over 256 channels (the D128 bidirectional block's plane and
    LN_out widths at a 64-wide pair); its training backward is ``b200_train._backward_hid_body`` (k3g's H <= 256 bounds D128 at hidden 256)."""
    if hidden == width:
        return True
    if hidden != 2 * width:
        return False
    return width <= 128 if direction != 0 else width == 64


def hidden_of(rows: int, direction: int) -> int:
    """The module's ``d_hidden`` from the row count of its left projection (2 d_hidden bidirectional: [outgoing | incoming])."""
    return rows // 2 if direction == 0 else rows


def planes_of(width: int, hidden: int, direction: int) -> int:
    """k1w's channel-major plane count: [a_out | a_in | b_out | b_in] (hidden each) bidirectional, [a | b] (hidden each) otherwise."""
    return 4 * hidden if direction == 0 else 2 * hidden


def contracted(width: int, hidden: int, direction: int) -> int:
    """Channels per contraction product: the hidden width (per direction bidirectional)."""
    return hidden


def _front(E, wl, wlg, wr, wrg, wp=None, wpp=None):
    """k1w reads the four front matrices in place when their columns are contiguous; otherwise (the D128 bidirectional module
    stores them column-major) ``k1w_prep`` writes row-major copies, in the same launch as k3g's ``wpp`` when that is asked for."""
    ws = (wl, wlg, wr, wrg)
    wc = None if all(w.stride(1) == 1 and w.stride(0) % 8 == 0 for w in ws) else wl.new_empty((4, *wl.shape))
    if wc is not None or wp is not None:
        E.k1w_prep(wl, wlg, wr, wrg, wc, wp, wpp)
    return ws if wc is None else tuple(wc.unbind(0))


#: The contraction through a measured cuBLASLt configuration (2.5.12's H100 technique, ``_h100_bmm.contract``, with a B200 table):
#: key (L, first row-major operand transposed, planes per launch) -> (tile id, custom option, cluster shape id) of cuBLAS's algorithm 66
#: (no split-K, no workspace; outputs bitwise the default pick's). Kernel times (CUPTI, L2 flushed, every operand layout the forward
#: and backward use; ``_tools/ltsweep.py``, B200 2026-11-17) against torch.bmm's pick: L384 x 64 planes 0.95-0.96, L512 x 64 0.93-0.95,
#: L256 x 64 a^T b 0.95, L640 x 128 a^T b 0.91. In the module (graph replay, two processes each): D64 L512 training 0.2945-0.2952 ->
#: 0.2909-0.2911 ms, RF3 0.2918 -> 0.2880, D64 L512 eval 0.0818 -> 0.0798; the L256 / L384 keys measured 0-1.3 % slower in the module
#: (D64 L384 training 0.1845 -> 0.1866, L256 0.1138 -> 0.1148) and are not listed. Anything else: torch.bmm.
LT_B200: dict = {
    (512, False, 64): (23, 3, 6),
    (512, True, 64): (23, 3, 6),
    (640, True, 128): (248, 1, 3),
}
LT = True


def _bmm(a, b, out):
    """``torch.bmm(a, b, out=out)`` for the contraction (a, b: [P, n, n] plane views, one may be transposed), through ``LT_B200``."""
    if LT and (a.shape[-1], a.stride(1) == 1, a.shape[0]) in LT_B200:
        from miniworld_engine.kernels.trimul_inproj.cuda import _h100_bmm

        return _h100_bmm.contract(a, b, out, table=LT_B200)
    return torch.bmm(a, b, out=out)


#: The contraction at small L in inference: ``ctr.cu`` (one launch for both directions, a programmatic dependent of k1w that
#: triggers its own dependents at entry) instead of torch.bmm's cuBLAS launches, in the (L, bidirectional, contracted width) cells
#: where it measured faster in the module (graph replay, two exclusive sweeps each, B200 2026-11-17): bidirectional L128 D64 26.5 ->
#: 22.4 us, D128 32.6 -> 28.6 / 30.6, D256 51.0 -> 49.0, bidirectional D64 L256 44.9 -> 43.0; one direction at 128 contracted
#: channels, L128 (D128 26.5 -> 24.4). Equal or slower elsewhere (one direction D64 L128 18.3 -> 18.4, D64 L256 32.7 -> 34.7, D128 L256
#: 51.0 -> 51.1). Training keeps ``_contract``.
#: + 2026-11-18 pass 15, with k3w XF (ctr triggers k3w after its wait, k3w multiplies x before its own; XF_WIDE): one direction
#: D256 / D384 / D512 L128 and bidirectional D384 L128 (rotated A/B, mean of 4 captures, us: D256 36.15 -> 34.81 / 36.09 -> 34.82, D384
#: 51.48 -> 51.25, D512 74.97 -> 73.72, bidirectional D384 78.53 -> 73.60, RF3 D256 127.8 -> 125.9)
CTR_CELLS = frozenset({(128, True, 64), (128, True, 128), (128, True, 256), (256, True, 64), (128, False, 128), (128, False, 256),
                       (128, False, 384), (128, False, 512), (128, True, 384), (256, False, 128), (256, True, 128)})  # + L256 128-channel cells (2026-11-19, n1-trimul: hidden-128 D64 L256 38.8 -> 36.7 us, bidirectional D128 L256 69.5 -> 65.5 us; D64 one-direction L256 stays cuBLAS: +7 %)
CTR = True
#: S90 part 5: ctr for every D <= 128 contraction up to L512 (env TRIMUL_CTR_ALL = 0: only CTR_CELLS, A/B)
CTR_ALL = os.environ.get("TRIMUL_CTR_ALL", "0") == "1"


def _contract_inf(E, planes, d, direction, trigger=True):
    """``_contract`` for the inference forward: ``ctr_forward`` in the ``CTR_CELLS`` (planes [P, B, n, n] contiguous).
    trigger: ctr lets its dependent launch at its entry -- k3g (D <= 128) waits before reading t; k3w (D >= 256, behind wide_ln_stats)
    does not, so that chain passes False; 2: right after ctr's own wait (k3w XF reads x before its wait: every kernel before the
    contraction must be complete)."""
    n = planes.shape[-1]
    if CTR and ((n, direction == 0, d) in CTR_CELLS or (CTR_ALL and n <= 512 and d in (64, 128) and n % 128 == 0 and os.environ.get("TRIMUL_CTR_ALL", "0") == "1")) and planes.is_contiguous():
        t = planes.new_empty(((2 if direction == 0 else 1) * d, *planes.shape[1:]))
        E.ctr_forward(planes, t, d, direction, 0, 5 if trigger == 2 else (3 if trigger else 1))
        return t
    return _contract(planes, d, direction)


#: M = 64 tcgen05 tiles at small L in inference (``k1h.cu`` front, ``k3h.cu`` output: 64-token tiles on twice the CTAs, two per SM,
#: each stage of a tile's chain about half as long) in the (L, bidirectional, d_pair, hidden) cells listed; k1h serves D64 / D128
#: (planes 128 / 256 at D64, 256 / 512 at D128), k3h D64 at hidden 64 / 128 without dropout. Other cells: k1w / k3g. The cells: where
#: front + output on M = 64 measured faster in the module (mean of 200 L2-flushed graph replays, 3-6 rounds, two processes; B200
#: 2026-11-17, us): hidden 128 L128 19.0-19.2 -> 18.5-18.7, L256 39.4 -> 39.0; bidirectional D64 L128 19.5-20.4 -> 18.5-18.6, L256 41.0
#: -> 39.2; D64 L256 32.7 -> 30.7. Not listed: D64 L128 (18.3-18.5 either way) and D128 (k1h slower: L128 22.6 -> 23.0, the 256-CTA
#: grid takes ~1 us longer to enter than the per-CTA saving).
#: Re-measured 2026-11-17 (pass 6, ab.py: base vs M = 64 alternating, 3 captures each, mean of 200 replays, us): bidirectional D64 L128
#: 22.40 -> 20.44, L256 43.25 -> 42.13; D64 L256 34.4 -> 32.9 (both directions); hidden 128 L128 22.43 -> 21.55; hidden 128 L256 41.6 ->
#: 40.8-41.1 with k3h alone (k1h front + k3h 41.3-41.5: that cell is in M64_OUT only); D128 front on k1h 49.6 -> 59.3 at L256, 24.5 -> 25.8 at L128.
M64_CELLS = frozenset({(128, False, 64, 128), (256, False, 64, 128), (128, True, 64, 64), (256, True, 64, 64), (256, False, 64, 64)})
M64_FRONT: frozenset = M64_CELLS - {(256, False, 64, 128)}  # hidden 128 L256: k3h alone measured best (40.8-41.1 vs 41.3-41.5 us mean with k1h)
M64_OUT: frozenset = M64_CELLS

#: D64 M = 64 output with both LayerNorms folded (``k3h.cu`` k3hf: the MMAs on raw x / t as they land, only the statistics in the
#: LN warps, both warp groups on every tile's epilogue, accumulators and t double buffered) in the M64_OUT cells listed here; needs the
#: forms' cached fold (``integrations.trimul_b200._inference_params``). env TRIMUL_K3H_FOLD=0: k3h (A/B).
K3H_FOLD = os.environ.get("TRIMUL_K3H_FOLD", "1") == "1"
#: Every length: (L, bidirectional, 64, hidden) with hidden 64 / 128 one direction, 64 bidirectional (H 64 / 128). Rotated A/B (pass 15,
#: mean of L2-flushed captures, us, k3h / k3g -> k3hf): hidden 128 L128 20.69 -> 20.36 / 21.23 -> 20.36, bidirectional D64 L128 21.19 -> 20.41,
#: L256 41.76 -> 38.84; D64 L256 32.91 -> 31.46 / 32.78 -> 31.04, hidden 128 L256 40.99 -> 38.66; L384 51.27 -> 49.87, hidden 128 71.27 ->
#: 66.04, bidirectional 75.90 -> 71.22; L512 79.69 -> 75.92 / 115.27 -> 107.52 / 119.72 -> 112.24; L768 165.6 -> 154.9 / 245.9 -> 229.6 /
#: 249.2 -> 232.7; L1024 315.0 -> 297.0, bidirectional 530.3 -> 504.0; L1536 837.7 -> 807.6 / 1465.7 -> 1443.0 / 1478.3 -> 1447.6; L2048
#: 1659 -> 1597; L3968 9050 -> 8860, bidirectional 16925 -> 16717
K3H_FOLD_CELLS = None   # None: every cell k3hf_ok() accepts


def k3hf_ok(cell) -> bool:
    n, bidir, d, hid = cell
    if K3H_FOLD_CELLS is not None:
        return cell in K3H_FOLD_CELLS
    return d == 64 and (hid in (64, 128) if not bidir else hid == 64)


#: D64 front on the LN_in-folded k1w (FOLD, as D128-384) in these (L, bidirectional, d_pair, hidden) cells (forms: front_fold);
#: env TRIMUL_K1W_FOLD64=0: off (A/B). Pass 15 (rotated A/B, us, k1w / k1h -> folded k1w, with k3hf): hidden 128 L256 38.27 -> 38.00 /
#: 38.27 -> 36.90, D64 L256 31.02 -> 30.63 (k1h), L384 49.86 -> 49.03; level at L128 (hidden 128 20.36, bidirectional 20.34) and bidirectional
#: L256 (38.84 -> 38.80); slower at hidden 128 L384 (66.3 -> 66.7), bidirectional L384 (70.4 -> 71.6), L768 (+1.3 to +3 %): not there
K1W_FOLD64 = os.environ.get("TRIMUL_K1W_FOLD64", "1") == "1"
K1W_FOLD64_CELLS: frozenset = frozenset({(256, False, 64, 128), (256, False, 64, 64), (384, False, 64, 64)})


#: D >= 256 inference: k1w reads the front as ONE packed [2P, D] matrix (``pack_front``, made once with the inference forms), one
#: [128][64] TMA box per weight ring slot instead of two [64][64] boxes (a thread issues about one TMA per few hundred clocks whatever
#: the box size; 2026-11-17 pass 8). The same bits.
PACK = os.environ.get("TRIMUL_PACK", "1") == "1"  # env TRIMUL_PACK=0: the two-box front (A/B)


#: D >= 256 inference from L = K3W_OST_MINL on: k3w computes the LN_out statistics (per-token mean / rstd over the H channels of t)
#: from the t stages it streams anyway, in four extra warps, instead of a ``wide_ln_stats`` launch (the H100 K3 normalises in-kernel
#: too; 2026-11-18 pass 12: L384-768 -1.5 to -6 %, D256 L1536 -2 %; at L <= 256 k3w's longer stage hand-off cost as much as the launch
#: saved, D384 L256 +3 %). env TRIMUL_K3W_OST=0: the ln_stats chain everywhere (A/B). K3W_OST_PDL: k3w as a programmatic dependent
#: of the contraction (its producers wait before reading t).
K3W_OST = os.environ.get("TRIMUL_K3W_OST", "1") == "1"
K3W_OST_MINL = int(os.environ.get("TRIMUL_K3W_OST_MINL", "384"))
K3W_OST_PDL = os.environ.get("TRIMUL_K3W_OST_PDL", "1") == "1"
#: D >= 256 inference with at most one 128-token tile per SM (L128, one sample): OST on the tile-grouped schedule (k3w pdl bit 3: a CTA
#: runs the D / 128 channel items of ONE token tile back to back, the statistics computed once from the first item's t stages; grid =
#: the tile count). 2026-11-18 pass 13 (rotated A/B, mean of 4-6 L2-flushed captures, us): bidirectional D256 L128 50.1 -> 47.4, D256
#: 38.7 -> 36.8, D384 55.7 -> 54.5, D512 77.4 -> 73.9, bidirectional D384 83.6 -> 79.9, D512 121.7 -> 118.4, RF3 D256 69.3 -> 67.1. The
#: ungrouped OST at L128 was level (pass 12: every second item re-read a fresh tile); at L256 (3-4 tiles a CTA) grouped OST is slower
#: (D256 92.2 -> 94.8, D384 167.1 -> 169.4), so only one tile a CTA. env TRIMUL_K3W_OST_GROUP=0: off (A/B).
K3W_OST_GROUP = os.environ.get("TRIMUL_K3W_OST_GROUP", "1") == "1"


#: D128 inference output on k3w (env TRIMUL_K3W_D128=0: k3g, A/B): (L, bidirectional) cells
K3W_D128 = os.environ.get("TRIMUL_K3W_D128", "1") == "1"
#: k3w with OST as a programmatic dependent: the x / Wg' stages of a CTA's first item before griddepcontrol.wait, the t stages after
#: (k3w flags bit 4), with ctr triggering its dependent right after its own wait (ctr pdl bit 2) instead of at its exit -- so k3w's
#: set-up, x loads and gate MMAs overlap the contraction (pass 15). env TRIMUL_K3W_XF=0: off (A/B).
XF = os.environ.get("TRIMUL_K3W_XF", "1") == "1"
#: XF where measured (pass 15, rotated A/B, mean of 4 L2-flushed captures, us): the D128 output at L128 (one direction 23.02 -> 22.41 / 24.26 -> 22.44,
#: bidirectional 28.43 -> 27.55; the same bits); not at L256+ (D128 46.97 -> 47.37, bidirectional 70.36 -> 71.14, D256 L384 184.8 ->
#: 186.3); D >= 256 at L128 only with ctr as the contraction (CTR_CELLS; behind cuBLAS level: D256 35.3 -> 35.7 / 36.3 -> 36.2)
XF_MAXL = 128
XF_WIDE = True
K3W_D128_CELLS = frozenset({(128, False), (256, False), (384, False), *((n, True) for n in (128, 256, 384, 512, 640, 768))})


@functools.cache
def _num_sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def pack_front(wl, wlg, wr, wrg):
    """[wl; wr] and [wlg; wrg] ([P, D] each, P the plane count) -> [2P, D]: chunk c (planes 64c .. 64c + 63) is rows 128c .. 128c + 127,
    its 64 gate rows then its 64 projection rows (the k1w weight slot image)."""
    g, p = torch.cat([wlg, wrg]), torch.cat([wl, wr])
    n, d = g.shape
    return torch.stack([g.reshape(n // 64, 64, d), p.reshape(n // 64, 64, d)], 1).reshape(2 * n, d).contiguous()


#: D256 / D384 inference: the LN_in-folded front (k1w FOLD; ``front_fold``, made once with the inference forms). The raw x K-blocks go
#: into TMEM as they land (no normalise pass in front of the MMA, no ``k1w_stats`` launch at D384); the statistics are computed in
#: parallel and enter the accumulator through one K16 MMA per chunk: acc = x.W'^T - mu u + v / rs, plane = bf16(sigmoid(rs acc_g) rs acc_p),
#: W' = bf16(W o gamma_in), u = sum_k W', v = W beta_in. Not the same bits as the normalise-then-MMA front (inside the accuracy
#: standard). Served cells (``fold_cell``; 2026-11-18 pass 14, rotated A/B, mean of 4 L2-flushed captures, us): L128 every D256 / D384
#: (D256 37.8 -> 35.8, D384 54.7 -> 51.1, bidirectional D256 48.7 -> 47.0, D384 80.1 -> 77.9, RF3 D256 123.5 -> 117.0); one direction
#: D384 every length measured (L256 167.6 -> 158.6, L512 633 -> 606, L768 1517 -> 1456), D256 to L512 (L256 92.6 -> 91.1, L512 329 ->
#: 321; L768 810 -> 816 not), bidirectional D384 to L512 (-0.5 to -1.4 %); not bidirectional D256 past L128 (L256 151.4 -> 151.9,
#: L384 302 -> 318). env TRIMUL_K1W_FOLD = 0 / 1 (A/B); FOLD_MAXL caps the length (A/B).
K1W_FOLD = os.environ.get("TRIMUL_K1W_FOLD", "1") == "1"
FOLD_MAXL = int(os.environ.get("TRIMUL_K1W_FOLD_MAXL", "1000000"))
FOLD128_MAXL = int(os.environ.get("TRIMUL_K1W_FOLD128_MAXL", "768"))


def fold_cell(n: int, d: int, direction: int) -> bool:
    """Whether the D >= 256 inference front runs the LN_in-folded k1w at (L, d_pair, direction) (decided by L, not the batch)."""
    if not K1W_FOLD or d not in (128, 256, 384) or n > FOLD_MAXL:
        return False
    if d == 128:  # with the k3w output (K3W_D128_CELLS)
        return n <= FOLD128_MAXL
    if n <= 128:
        return True
    if d == 384:
        return direction != 0 or n <= 512
    return direction != 0 and n <= 512


def _split3(v):
    """fp32 -> three bf16 parts (as fp32) with v ~= a + b + c to ~24 bits."""
    a = v.to(torch.bfloat16).float()
    b = (v - a).to(torch.bfloat16).float()
    c = (v - a - b).to(torch.bfloat16).float()
    return a, b, c


def front_fold(wl, wlg, wr, wrg, gi, bi):
    """The four front matrices (any float dtype, [P / 2, D] each) and LN_in's affine -> (wf [2P, D] bf16 in ``pack_front``'s row order,
    cr [2P, 16] bf16: per row [u1 u2 u1 u3 u2 u1 | v1 v2 v1 v3 v2 v1 | 0 0 0 0], the three-part splits of u = sum_k wf and v = W beta_in,
    against k1w's per-token [a1 a1 a2 a1 a2 a3 | c1 c1 c2 c1 c2 c3] (a = -mu, c = 1 / rstd))."""
    w = pack_front(*(t.float() for t in (wl, wlg, wr, wrg)))  # [2P, D] fp32: chunk c = 64 gate rows, then 64 projection rows
    wf = (w * gi.float()[None, :]).to(torch.bfloat16).contiguous()
    u1, u2, u3 = _split3(wf.float().sum(1))
    v1, v2, v3 = _split3(w @ bi.float())
    z = torch.zeros_like(u1)
    cr = torch.stack([u1, u2, u1, u3, u2, u1, v1, v2, v1, v3, v2, v1, z, z, z, z], 1).to(torch.bfloat16).contiguous()
    return wf, cr


def _fold(x):
    """[c, (B,) n, n] -> [c * B, n, n]: the samples join the channel planes in cuBLAS's batch dim (a view, the planes are contiguous)."""
    n = x.shape[-1]
    return x.view(-1, n, n)


def _contract(planes, d, direction):
    """Planes [P, n, n] or [P, B, n, n] (batch b-major per plane) -> t [(2 d or d), (B,) n, n], per sample the same products as B = 1.

    The ``out=`` buffer is the 3-D folded tensor itself and the caller's shape is a view of it: ``bmm(out=<3-D view of a 4-D buffer>)``
    makes functionalization re-apply the view with the 3-D strides to the 4-D base (``as_strided`` "mismatch in length of strides and
    shape") whenever this body is traced."""
    n = planes.shape[-1]
    channels = (2 if direction == 0 else 1) * d
    samples = planes.numel() // (planes.shape[0] * n * n)
    t = planes.new_empty((channels * samples, n, n))
    pl = lambda a, b: _fold(planes[a:b])
    if direction == 0:
        _bmm(pl(0, d), pl(2 * d, 3 * d).transpose(1, 2), out=t[: d * samples])
        _bmm(pl(d, 2 * d).transpose(1, 2), pl(3 * d, 4 * d), out=t[d * samples :])
    elif direction == 1:
        _bmm(pl(0, d), pl(d, 2 * d).transpose(1, 2), out=t)
    else:
        _bmm(pl(0, d).transpose(1, 2), pl(d, 2 * d), out=t)
    return t.view(channels, *planes.shape[1:])


def _drop_fake(r, p):
    return torch.empty_like(r)


@opaque(fake=_drop_fake, name="trimul_b200_drop_scale")
def drop_scale_op(r: torch.Tensor, p: float) -> torch.Tensor:
    """r (the reference's ``torch.rand`` draw, bf16, contiguous) -> the row-dropout scale ``(r > p).to(bf16) / (1 - p)`` in one launch,
    bitwise the reference's three (``wide_aux.cu`` ``drop_scale_kernel``)."""
    out = torch.empty_like(r)
    with torch.cuda.device(r.device):
        _ext().drop_scale(r, out, float(p))
    return out


def dropout_scale(pair: torch.Tensor, p: float) -> torch.Tensor:
    """``modules.functional.dropout_scale(pair, p, 1)`` (drop_row, [B, 1, L, D]) for a bf16 pair: the same ``torch.rand`` draw (the same
    generator state and bits), the compare / cast / divide as one kernel (2026-11-17: three elementwise launches fewer per training
    call)."""
    if pair.dtype != torch.bfloat16 or (pair.shape[0] * pair.shape[2] * pair.shape[3]) % 8:
        from miniworld_engine.modules.functional import dropout_scale as reference

        return reference(pair, p, 1)
    shape = list(pair.shape)
    shape[1] = 1
    return drop_scale_op(torch.rand(shape, device=pair.device, dtype=pair.dtype), float(p))


def _inference_fake(leaves, mask, ds, direction, rf3=-1.0):
    """y only."""
    return torch.empty_like(leaves[0])


def _fold_fake(params):
    wp, wg = params[0], params[3]
    return [torch.empty(wp.shape, dtype=torch.bfloat16, device=wp.device), torch.empty(wg.shape, dtype=torch.bfloat16, device=wg.device),
            torch.empty((4, wp.shape[0]), dtype=torch.float32, device=wp.device)]  # fmt: skip


@opaque(fake=_fold_fake, name="trimul_b200_fold")
def fold_op(params: list[torch.Tensor]) -> list[torch.Tensor]:
    """[wp, go, bo, wg, gi, bi] (Wp / Wg as stored, fp32 or bf16) -> [wpq, wgq, vec]: k3w's LayerNorm-folded weights (``wide_fold_prep``).
    The D256+ inference forms carry them (``integrations.trimul_b200._inference_params``: made once per parameter set, not per call)."""
    wpq, wgq, vec = _fold_fake(params)
    with torch.cuda.device(wpq.device):
        _ext().wide_fold_prep(*params, wpq, wgq, vec)
    return [wpq, wgq, vec]


@opaque(fake=_inference_fake, name="trimul_b200_inference")
def inference(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor | None, direction: int, rf3: float = -1.0) -> torch.Tensor:
    """leaves = x, wl, wlg, wr, wrg, wg, wp (bf16, [out, in]; the front four may be strided), gi, bi, go, bo (fp32) -- at D64 / D128
    optionally followed by w1, wpp (``b200_train.kernel_forms``: the front four row-major and k3g's permuted Wp made once, no prep launch);
    mask [B * L] bool token mask or None (k1w forms the pair mask); ds [B * L, D] bf16 row-dropout scale or None;
    direction 0 = bidirectional, 1 = outgoing, 2 = incoming. x is [B, L, L, D], the samples b-major (one launch per stage for the
    B samples). The hidden width follows from wl's row count (``hidden_of``, ``hidden_ok``)."""
    return _inference_body(leaves, mask, ds, direction, None, rf3)


def _inference_into_fake(leaves, mask, ds, direction, out):
    """Nothing: y is written into ``out``."""
    return None


@opaque(fake=_inference_into_fake, name="trimul_b200_inference_into", mutates_args=("out",))
def inference_into(
    leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor | None, direction: int, out: torch.Tensor
) -> None:
    """``inference`` writing y into ``out`` (contiguous, x's shape): a chunked batch's launch writes its rows of the batch's output in
    place (``integrations.trimul_b200``)."""
    _inference_body(leaves, mask, ds, direction, out)


def _inference_body(leaves, mask, ds, direction, y, rf3=-1.0):
    """-> y; y None: allocated here (after the planes, as the one-launch path always did)."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves[:11]
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    E = _ext()
    with torch.cuda.device(x.device):
        x2 = x.reshape(m, d)
        hid = hidden_of(wl.shape[0], direction)
        p = planes_of(d, hid, direction)
        planes = x.new_empty((p, bsz, n, n))
        if y is None:
            y = torch.empty_like(x)
        if d <= 128:
            if _fused_ok(x, n, bsz, d, hid, direction, ds, wg, wp, (wl, wlg, wr, wrg)):
                _tf_ext().tf_forward(x2, wl, wlg, wr, wrg, wg, wp, mask, gi, bi, go, bo, planes, y.view(m, d), direction, EPS, rf3)
                return y
            if len(leaves) > 11:  # kernel_forms: row-major front matrices and wpp made once
                wpp, front = leaves[12], (wl, wlg, wr, wrg)
            else:
                wpp = x.new_empty((d, p // 2))  # k3g's output projection, columns in its TMEM K order
                front = _front(E, wl, wlg, wr, wrg, wp, wpp)
            cell = (n, direction == 0, d, hid)
            if K3W_D128 and d == 128 and hid == d and ds is None and len(leaves) >= 19 and (n, direction == 0) in K3W_D128_CELLS:
                # D128 output on k3w (both LayerNorms folded into the weights, the LN_out statistics in-kernel) instead of k3g; the
                # forms' cached k3w fold and folded front (integrations.trimul_b200._inference_params) are the last five leaves
                mean = x.new_empty((m,), dtype=torch.float32)
                rstd = torch.empty_like(mean)
                wf, cr = leaves[-2:]
                if cr.numel() and fold_cell(n, d, direction):
                    _k1w(E, rf3, x2, wf, cr, wf, wf, mask, mean, rstd, gi, bi, planes, -2, EPS)
                else:
                    _k1w(E, rf3, x2, *front, mask, mean, rstd, gi, bi, planes, -2, EPS)
                t = _contract_inf(E, planes, contracted(d, hid, direction), direction, trigger=2 if XF and n <= XF_MAXL else False)
                del planes
                wpq, wgq, vec = leaves[-5:-2]
                grp = K3W_OST_GROUP and n * n // 128 <= _num_sms(x.device.index)
                E.k3w_forward(x2, t.view(t.shape[0], m), wpq, wgq, vec, mean, mean, mean, rstd, None, y.view(m, d), n, 0, None, None,
                              int(K3W_OST_PDL) | 2 | (8 if grp else 0) | (16 if XF and n <= XF_MAXL and K3W_OST_PDL else 0))  # fmt: skip
                return y
            # D64 with the forms' cached fold (the last five leaves: k3hf's Wp' / Wg' / vectors, the LN_in-folded front and its
            # correction rows -- empty for RoseTTAFold3)
            f64 = d == 64 and ds is None and len(leaves) >= 19
            f3 = f64 and K3H_FOLD and k3hf_ok(cell)
            f1 = f64 and K1W_FOLD64 and cell in K1W_FOLD64_CELLS and leaves[-1].numel() > 0
            m64f = cell in M64_FRONT and not f1
            m64o = f3 or (ds is None and cell in M64_OUT and wp.dtype == torch.bfloat16 and wp.is_contiguous() and wg.is_contiguous())
            if f1:
                mean = x.new_empty((m,), dtype=torch.float32)
                _k1w(E, rf3, x2, leaves[-2], leaves[-1], leaves[-2], leaves[-2], mask, mean, torch.empty_like(mean), gi, bi, planes, -2, EPS)
            elif m64f:
                _k1h(E, rf3, x2, *front, mask, gi, bi, planes, 0, EPS)
            else:
                _k1w(E, rf3, x2, *front, mask, None, None, gi, bi, planes, -2, EPS)
            t = _contract_inf(E, planes, contracted(d, hid, direction), direction)
            del planes
            if f3:
                E.k3hf_forward(x2, t, *leaves[-5:-2], y.view(m, d), EPS, 0, 1)
            elif m64o:
                E.k3h_forward(x2, t, wp, wg, gi, bi, go, bo, y.view(m, d), EPS, 0, 1)
            elif ds is None:
                E.k3g_forward(x2, t, wpp, wg, gi, bi, go, bo, y.view(m, d), n, EPS, 0, None, None, None, None, 0)
            else:  # dropout: the training epilogue without the saves (save = 2: y as the training forward's)
                E.k3g_forward(x2, t, wpp, wg, gi, bi, go, bo, y.view(m, d), n, EPS, 2, ds, None, None, None, 0)
            return y
        mean = x.new_empty((m,), dtype=torch.float32)
        rstd = torch.empty_like(mean)
        if len(leaves) > 16 and fold_cell(n, d, direction):  # the forms' LN_in-folded front (front_fold)
            wf, cr = leaves[15], leaves[16]
            _k1w(E, rf3, x2, wf, cr, wf, wf, mask, mean, rstd, gi, bi, planes, -1, EPS)
        else:
            if d > K1W_INSTAT:
                E.k1w_stats(x2, mean, rstd, EPS)
            front = (leaves[14],) * 4 if PACK and len(leaves) > 14 else _front(E, wl, wlg, wr, wrg)   # the forms' packed front
            # grid -1: an inference call (k1w.cu: the third weight issuer at D >= 256 with one tile a CTA; the same bits)
            _k1w(E, rf3, x2, *front, mask, mean, rstd, gi, bi, planes, -1, EPS)
        # one token tile a CTA per sample (decided by L, not by the batch: a batch must give each sample's bits): grouped OST
        grp = K3W_OST and K3W_OST_GROUP and n * n // 128 <= _num_sms(x.device.index)
        ost = (K3W_OST and n >= K3W_OST_MINL) or grp
        xf = XF and XF_WIDE and n <= XF_MAXL and ost and K3W_OST_PDL   # ctr triggers k3w after its wait; k3w loads / multiplies x before its own
        t = _contract_inf(E, planes, d, direction, trigger=2 if xf else False)
        del planes
        if ost:  # k3w computes the LN_out statistics from its own t stages: no wide_ln_stats launch (mo / ro unused)
            mo = ro = mean
        else:
            mo = torch.empty_like(mean)
            ro = torch.empty_like(mean)
            E.wide_ln_stats(t, mo, ro, EPS)
        if len(leaves) > 11:  # the inference forms' cached fold (fold_op)
            wpq, wgq, vec = leaves[11:14]
        else:
            wpq = torch.empty_like(wp, dtype=torch.bfloat16)
            wgq = torch.empty_like(wg, dtype=torch.bfloat16)
            vec = x.new_empty((4, d), dtype=torch.float32)
            E.wide_fold_prep(wp, go, bo, wg, gi, bi, wpq, wgq, vec)
        # k3w as a programmatic dependent of wide_ln_stats (which triggers at entry, after the contraction is complete): its set-up and
        # t / x / weight loads overlap ln_stats; it waits before reading mu_o / rs_o. Only with the cached fold (nothing between them).
        # OST: k3w follows the contraction directly; as a programmatic dependent (K3W_OST_PDL) its producers wait before reading t.
        pdl = int(K3W_OST_PDL) if ost else int(len(leaves) > 11)
        flags = pdl | (2 if ost else 0) | (8 if grp else 0) | (16 if xf else 0)
        E.k3w_forward(x2, t.view(t.shape[0], m), wpq, wgq, vec, mo, ro, mean, rstd, ds, y.view(m, d), n, 0, None, None, flags)
    return y


def _k1w(E, rf3, *args):
    """``k1w_forward``; ``rf3`` >= 0: RoseTTAFold3's length factor on the left projection (0: counted from the token mask in the kernel, > 0: the constant)."""
    return E.k1w_forward_rf3(*args, rf3) if rf3 >= 0.0 else E.k1w_forward(*args)


def _k1h(E, rf3, *args):
    return E.k1h_forward_rf3(*args, rf3) if rf3 >= 0.0 else E.k1h_forward(*args)

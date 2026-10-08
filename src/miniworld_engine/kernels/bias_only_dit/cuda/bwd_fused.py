"""The bf16 training backward's fused kernels (MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED, default on; 0: the per-step launches):

  bo_bwd_tail.cu  ``TailBwd`` (MINIWORLD_BIAS_ONLY_DIT_BWD_TAIL=split3): d out -> the attention core's inputs (res_c_bwd, dh GEMM +
                  SwiGLU backward, dxt GEMM, res_adaln_b_bwd, d(og) GEMM + gate backward) as items over pairs of 128-row tiles on
                  CTA pairs (GEMM items: M 256 cta_group::2 products), three launches
  bo_wgrad.cu     ``Wgrad``: the six weight gradients (squeeze, expand a | b, to_out, value | gate, the AdaLN projections with the
                  cond-LN unfold, the output gates) in ONE launch: one [128][<= 512] output tile per CTA over all the rows
  bo_bwd_mid.cu   ``MidBwd`` (MINIWORLD_BIAS_ONLY_DIT_BWD_MID=1): the dxa / dcg GEMMs, adaln_a_bwd, the dchat GEMM + cond_bwd as three
                  launches on CTA pairs ([G5 + G6], [R3], [G7])
Beside them (other files): bo_pvdpb.cu (pv dV + dbias in one launch, MINIWORLD_BIAS_ONLY_DIT_BWD_PVDPB) and pair_bias_bwd_fin_k in
bias_only_dit_train_rows.cu (pair_bias_bwd + finalize, MINIWORLD_BIAS_ONLY_DIT_BWD_PBFIN).

The tail's tile counters live in a zeroed device buffer per (device, size) that its last CTA zeroes again (graph replays included).
Build failures warn once and keep the per-step launches.
"""

from __future__ import annotations

import os
import warnings

import torch

from miniworld_engine.kernels.bias_only_dit.cuda import _descriptors, _dir

D, DC = 768, 384
SMEM_TAIL = 230912                     # bo_bwd_tail.cu SMEM_BYTES
SMEM_MID = 229376 + 2048 + 1024                # bo_bwd_mid.cu SMEM_BYTES
SMEM_WGRAD = {384: 3 * 65536 + 1024, 512: 2 * 81920 + 1024}   # bo_wgrad.cu SMEM_BYTES by WMAX
T_R1, T_G1, T_G2, T_R2, T_G4 = 0, 1, 2, 3, 4


def bwd_fused_on() -> bool:
    """The fused bf16 training backward (the persistent kernels for the launches they replace) -- the default;
    MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED=0 keeps the per-step launches. Read per call."""
    return os.environ.get("MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED", "1") != "0"


def tail_mode() -> str:
    """How the tail runs (MINIWORLD_BIAS_ONLY_DIT_BWD_TAIL): "split3" (default) -- three launches of bo_bwd_tail.cu on CTA pairs
    ([R1, G1, G2], [R2], [G4]; GEMM items M 256 x N with cta_group::2, 4 operand stages; graph node A/B 270.3 -> 268.8 / 533.4 -> 526.9
    us at L384 / L768 against the seven launches); "off" -- the seven launches (res_c_bwd, mm dh, swiglu_bwd, mm dxt,
    res_adaln_b_bwd, mm dog, gate_bwd). Read per call."""
    return os.environ.get("MINIWORLD_BIAS_ONLY_DIT_BWD_TAIL", "split3")


def pbfin_on() -> bool:
    """pair_bias_bwd and finalize in one launch (bias_only_dit_train_rows.cu pair_bias_bwd_fin_k: a grid barrier between the pair
    LayerNorm backward and finalize's 248 jobs) -- the default: one launch fewer at the same speed (graph node A/B +0.3 / +0.5 us at
    L384 / L768, inside the 0.1-33 us spread); MINIWORLD_BIAS_ONLY_DIT_BWD_PBFIN=0 keeps the two launches. Read per call."""
    return os.environ.get("MINIWORLD_BIAS_ONLY_DIT_BWD_PBFIN", "1") != "0"


def mid_on() -> bool:
    """The dxa GEMM, adaln_a_bwd, the dchat / dcg GEMMs and cond_bwd as three launches of bo_bwd_mid.cu on CTA pairs ([dxa + dcg
    GEMMs], [the AdaLN backward rows], [the dchat GEMM + the cond LayerNorm backward]) with MINIWORLD_BIAS_ONLY_DIT_BWD_MID=1; off by
    default: it loses its graph node A/B (bwdf10: 141.5 -> 148.9 / 278.5 -> 300.7 us at L384 / L768). Its GEMM items run at the SMs'
    TMA intake (~70 GB/s per SM: 0.43 us per G5 / G6 k-block, 0.57 per G7's), so G6's N 192 tiles move more bytes per FLOP than
    cuBLAS's dcg, and G7's cond-LN epilogue (16 us, single TMEM buffer) is not hidden. Read per call (and by the weight pack, which
    then adds the kernel's transposed weights)."""
    return os.environ.get("MINIWORLD_BIAS_ONLY_DIT_BWD_MID", "0") == "1"


def pvdpb_on() -> bool:
    """pv dV and the bias gradient in one launch (bo_pvdpb.cu: the first CTAs pv dV's body, the rest dpb_sm100.cu's; a programmatic
    dependent) where the bias gradient runs on single CTAs (not ``dpb_pair_on``: dpbx2's cta_group::2 MMAs cannot share a function
    with pv dV's cta_group::1 ones) -- the default (bwdf6 node A/B, L384: 37.6 -> 36.0 us); MINIWORLD_BIAS_ONLY_DIT_BWD_PVDPB=0 keeps
    the two launches. Read per call."""
    return os.environ.get("MINIWORLD_BIAS_ONLY_DIT_BWD_PVDPB", "1") != "0"


def pv_pdl_on() -> bool:
    """pv dV after the tail as a programmatic dependent launch (pv_gate_inf.cu's -DPDL_INF build: its setup overlaps the tail's
    last wave, griddepcontrol.wait before it reads do) -- the default; MINIWORLD_BIAS_ONLY_DIT_BWD_PVPDL=0 launches it plainly.
    Read per call."""
    return os.environ.get("MINIWORLD_BIAS_ONLY_DIT_BWD_PVPDL", "1") != "0"


def step_bufs(dev) -> None:
    """Make the fused step's persistent counters (pair_bias_bwd_fin's barrier) before the step allocates anything."""
    pb_bar(dev)


def pb_bar(dev) -> torch.Tensor:
    """The grid-barrier counters of pair_bias_bwd_fin_k (int32 [2], zero; the kernel leaves them zero), one per device, made by an
    eager call."""
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    hit = _BUFS.tab.get(("pbbar", idx))
    if hit is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("pair_bias_bwd_fin: the barrier counters must be made by an eager call first")
        hit = _BUFS.tab[("pbbar", idx)] = torch.zeros(2, dtype=torch.int32, device=dev)
    return hit


def _sm100():
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    return sm100


def ng4(da: int) -> int:
    """Channels per d(og) GEMM item (whole heads): 192 at 768 attention channels, 256 at 1024."""
    return 192 if da == 768 else 256


def tail_launches(T: int, da: int, nsm: int) -> list[tuple[list[int], int, int]]:
    """The tail's three launches on CTA pairs: (item table, DEPMASK, grid). Codes type | n << 3 | pair tile << 8 (pair tile p = tiles
    2p, 2p + 1), phase-major: [every R1, every G1, every G2] (G1 / G2 wait on per-tile counters), [every R2], [every G4]; a pair c of
    the P in the grid takes entries c, c + P, ..."""
    n4 = da // ng4(da)
    tp = (T + 1) // 2
    enc = lambda ty, n, p: ty | (n << 3) | (p << 8)                  # noqa: E731
    a = ([enc(T_R1, 0, p) for p in range(tp)] + [enc(T_G1, j, p) for p in range(tp) for j in range(6)]
         + [enc(T_G2, j, p) for p in range(tp) for j in range(3)])
    b = [enc(T_R2, 0, p) for p in range(tp)]
    c = [enc(T_G4, j, p) for p in range(tp) for j in range(n4)]
    grid = lambda items: 2 * min(nsm // 2, len(items))              # noqa: E731 clusters of 2
    return [(a, (1 << T_G1) | (1 << T_G2), grid(a)), (b, 0, grid(b)), (c, 0, grid(c))]


def wgrad_tiles(da: int) -> list[tuple[int, int, int]]:
    """(g, mt, nt) of every output tile: grad columns / 128 x act columns / N (N 256; 192 for the 384-wide acts)."""
    grads = (768, 3072, 768, 2 * da, 3072, 1536)
    acts = (1536, 768, da, 768, 384, 384)
    out = []
    for g, (ga, ac) in enumerate(zip(grads, acts, strict=True)):
        n = 192 if g >= 4 else 256
        out += [(g, mt, nt) for mt in range(ga // 128) for nt in range(ac // n)]
    return out


def wgrad_plan(da: int, G: int) -> tuple[list[int], int]:
    """bo_wgrad.cu's tiles: each output band (128 grad columns of one gradient) cut into pieces of whole 128-column granules, at most 3
    per piece (4 when 3 would take more CTAs than G), as even as possible. Returns (tile codes g | mt << 3 | granule0 << 8 | granules
    << 12, WMAX = the widest piece in columns: 384 or 512). 768 attention channels: 144 tiles of [128][384]."""
    grads = (768, 3072, 768, 2 * da, 3072, 1536)
    acts = (1536, 768, da, 768, 384, 384)
    for k in (3, 4):
        tiles, wmax = [], 0
        for g, (ga, ac) in enumerate(zip(grads, acts, strict=True)):
            w = ac // 128
            n = -(-w // k)
            sizes = [w // n + (1 if i < w % n else 0) for i in range(n)]
            wmax = max(wmax, max(sizes))
            for mt in range(ga // 128):
                c = 0
                for sz in sizes:
                    tiles.append(g | (mt << 3) | (c << 8) | (sz << 12))
                    c += sz
        if len(tiles) <= G:
            break
    return tiles, 128 * max(wmax, 3)


class _Bufs:
    """Item tables and zeroed counters per (device, key), made outside any capture (the first, eager call) and reused."""

    def __init__(self):
        self.tab: dict = {}

    def get(self, key, items, ncnt, dev):
        hit = self.tab.get(key)
        if hit is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("bo_bwd: the item table / counters must be built by an eager call first")
            hit = self.tab[key] = (torch.tensor(items, dtype=torch.int32, device=dev), torch.zeros(ncnt, dtype=torch.int32, device=dev))
        return hit

    def get2(self, key, tables, ncnt, dev):
        """Two item tables and one counter buffer."""
        hit = self.tab.get(key)
        if hit is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("bo_bwd: the item tables / counters must be built by an eager call first")
            hit = self.tab[key] = (*(torch.tensor(t, dtype=torch.int32, device=dev) for t in tables),
                                   torch.zeros(ncnt, dtype=torch.int32, device=dev))
        return hit


_BUFS = _Bufs()


class _Maps(dict):
    """Encoded tensor maps keyed by (buffers, layouts): the caching allocator hands the step the same buffers again."""

    def make(self, key, build):
        hit = self.get(key)
        if hit is None:
            if len(self) >= 64:
                self.clear()
            hit = self[key] = build()
        return hit


def _key(*ts):
    return tuple((t.data_ptr(), tuple(t.shape), tuple(t.stride())) for t in ts)


class TailBwd:
    """``bo_bwd_tail.cu`` (CTA pairs) for ``nh`` heads of ``dh``. ``__call__`` takes the saved activations and the packed weights,
    writes dz, dGg, dab, dG[:, 2D:], dx1, dy, do, dvg[:, DA:], dd and the bg2 / bs2 / bg1 partials (part[0..2] rows 0 .. 4 T); returns
    4 T."""

    def __init__(self, device_index: int, nh: int, dh: int, trace: bool = False, stages: int | None = None):
        """``stages``: operand stages 3 / 4 / 5 (the GEMM E ring gets the rest: 15 / 11 / 7 slots); default
        MINIWORLD_BIAS_ONLY_DIT_BWD_TAIL_NST, else 4 (bwdf9 graph node A/B: 290.6 / 271.7 / 282.8 us at L384 for 3 / 4 / 5)."""
        stages = stages or int(os.environ.get("MINIWORLD_BIAS_ONLY_DIT_BWD_TAIL_NST", "4"))
        self.idx, self.nh, self.dh, self.da = device_index, nh, dh, nh * dh
        defs = (f"DATT={self.da}", f"NHEAD={nh}", *(() if stages == 4 else (f"TAIL_NST={stages}",)), *(("TRACE",) if trace else ()))
        self.k = _sm100()._sm100_kernel("bo_bwd_tail", "bo_bwd_tail_sm100", device_index, pdl=True, src_dir=str(_dir), defs=defs,
                                        cluster=2, smem=SMEM_TAIL)
        self.qauto, self.qnext = False, 0          # -DTRACE: qauto -- each call's items at the next trace indices (graph copies)
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.maps = _Maps()

    def __call__(self, dout, z, Gg, ab, x, y, G, x1st, og, vg, bg2, bs2, bg1, WsqT, WaT, WbT, WoT, dz, dGg, dab, dG, dx1, dy, do,
                 dvg, dd, part, L):
        M, da, bf = dout.shape[0], self.da, torch.bfloat16
        T = M // 128
        assert M % 128 == 0 and T < (1 << 23)
        for t in (dout, z, x, y, dz, dy):
            assert t.dtype is bf and t.is_contiguous() and t.shape == (M, D), t.shape
        assert Gg.shape == (M, 2 * D) and G.shape == (M, 4 * D) and ab.shape == (M, 4 * D) and og.shape == (M, da)
        assert vg.shape == (M, 2 * da) and dGg.shape == (M, 2 * D) and dab.shape == (M, 4 * D) and dG.shape == (M, 4 * D)
        assert all(t.is_contiguous() and t.dtype is bf for t in (Gg, G, ab, og, vg, dGg, dab, dG, do, dvg, WsqT, WaT, WbT, WoT))
        assert dx1.dtype is torch.float32 and dx1.is_contiguous() and x1st.is_contiguous() and dd.is_contiguous()
        assert part.dim() == 3 and part.shape[2] == D and part.shape[1] >= 4 * T and part.is_contiguous()
        key = ("tail3", self.idx, T, da)
        launches = _BUFS.tab.get(key)
        if launches is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("bo_bwd_tail: the item tables / counters must be built by an eager call first")
            dev = dout.device
            launches = _BUFS.tab[key] = [(torch.tensor(t, dtype=torch.int32, device=dev), len(t), dm, grid)
                                         for t, dm, grid in tail_launches(T, da, self.nsm)]
        cnt = _BUFS.tab.get(("tailcnt", self.idx, T))
        if cnt is None:
            cnt = _BUFS.tab[("tailcnt", self.idx, T)] = torch.zeros(T + 1, dtype=torch.int32, device=dout.device)
        tm, n4 = _sm100()._tm, ng4(da)

        def build():
            row = lambda t: tm(t, [256, 3, M], [512, t.stride(0) * 2], [256, 3, 4], swizzle=0)   # noqa: E731 [4 rows][768] boxes
            box = lambda t, w: tm(t, [w, M], t.stride(0) * 2, [32, 128], swizzle=64)              # noqa: E731
            sbox = lambda t, w: tm(t, [w, M], t.stride(0) * 2, [32, 32], swizzle=64)              # noqa: E731 per-warp-pair stores
            opa = lambda t, w: tm(t, [w, M], t.stride(0) * 2, [64, 128])                          # noqa: E731
            opb = lambda t, k, n, nb: tm(t, [k, n], t.stride(0) * 2, [64, nb])                     # noqa: E731 B halves
            return _descriptors(row(dout), row(z), row(Gg[:, D:]), row(dG[:, 3 * D:]), row(x), row(y), row(Gg), row(G[:, 2 * D:]),
                                box(ab, 4 * D), box(og, da), box(vg, 2 * da), opa(dz, D), opa(dab, 4 * D), opa(dy, D),
                                opb(WsqT, D, 2 * D, 128), opb(WaT, 2 * D, D, 128), opb(WbT, 2 * D, D, 128), opb(WoT, D, da, n4 // 2),
                                sbox(dab, 4 * D), sbox(do, da), sbox(dvg, 2 * da))               # TMA stores of G1 / G4
        maps = self.maps.make(_key(dout, z, Gg, dG, x, y, G, ab, og, vg, dz, dab, dy, WsqT, WaT, WbT, WoT, do, dvg), build)
        qbase = self.qnext if self.qauto else 0
        for items, n, depmask, grid in launches:
            self.k((grid, 1, 1), (384, 1, 1), *maps, items, cnt, x1st, bg2, bs2, bg1, dz, dGg, dab, dG, dx1, dy, do, dvg, dd, part,
                   T, n, L, part.shape[1], depmask, qbase)
            qbase += n                                           # -DTRACE: each launch's items at their own trace indices
        self.qnext = qbase
        return 4 * T


class MidBwd:
    """``bo_bwd_mid.cu`` for ``nh`` heads of ``dh``; ``sf32`` / ``cf32``: x, dx / dc fp32 (else bf16). ``__call__`` writes dxa into
    dG[:, D:2D], ds1 into dG[:, :D], dx, dc, the dcg scratch and the bs1 partials (part3 rows 0 .. 4 T); returns 4 T."""

    def __init__(self, device_index: int, nh: int, dh: int, sf32: bool = False, cf32: bool = False, trace: bool = False):
        self.idx, self.da, self.sf32, self.cf32 = device_index, nh * dh, sf32, cf32
        defs = (f"DATT={self.da}", *(("SF32",) if sf32 else ()), *(("CF32",) if cf32 else ()), *(("TRACE",) if trace else ()))
        self.k = _sm100()._sm100_kernel("bo_bwd_mid", "bo_bwd_mid_sm100", device_index, pdl=True, src_dir=str(_dir), defs=defs,
                                        cluster=2, smem=SMEM_MID)
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.maps = _Maps()
        self.qauto, self.qnext = False, 0          # -DTRACE: qauto -- each call's items at the next trace indices (graph copies)

    def __call__(self, dvg, dGg, dG, G, x, xst, dx1, c, cst, W, bs1, dcg, dx, dc, part3):
        M, da, bf, f32 = dvg.shape[0], self.da, torch.bfloat16, torch.float32
        T = M // 128
        st = f32 if self.sf32 else bf
        assert M % 128 == 0 and T < (1 << 23)
        assert dvg.shape == (M, 2 * da) and dGg.shape == (M, 2 * D) and dG.shape == (M, 4 * D) and G.shape == (M, 4 * D)
        assert all(t.is_contiguous() and t.dtype is bf for t in (dvg, dGg, dG, G, c, dcg)) and c.shape == (M, DC) and dcg.shape == (M, DC)
        assert x.shape == (M, D) and x.is_contiguous() and x.dtype is st and dx.shape == (M, D) and dx.is_contiguous() and dx.dtype is st
        assert dc.shape == (M, DC) and dc.is_contiguous() and dc.dtype is (f32 if self.cf32 else bf)
        assert dx1.dtype is f32 and dx1.is_contiguous() and xst.is_contiguous() and cst.is_contiguous() and bs1.is_contiguous()
        assert part3.dim() == 2 and part3.shape[1] == D and part3.shape[0] >= 4 * T and part3.is_contiguous()
        tm = _sm100()._tm
        ws = (W["WvT"], W["WgtT"], W["WsT0"], W["WsT1"], W["WnT0"], W["WnT1"], W["WnT2"], W["WnT3"])

        def build():
            row = lambda t: tm(t, [256, 3, M], [512, t.stride(0) * 2], [256, 3, 4], swizzle=0)                # noqa: E731 [4][768]
            rowf = lambda t: tm(t, [128, 6, M], [512, t.stride(0) * 4], [128, 3, 4], swizzle=0, dtype="f32")  # noqa: E731 [4][384]
            opa = lambda t, w: tm(t, [w, M], t.stride(0) * 2, [64, 128])                                       # noqa: E731
            opb = lambda t, nb: tm(t, [t.shape[1], t.shape[0]], t.stride(0) * 2, [64, nb])                     # noqa: E731 B halves
            return _descriptors(opa(dvg, 2 * da), opa(dGg, 2 * D), opa(dG, 4 * D), opb(ws[0], 128), opb(ws[1], 128),
                                *(opb(w, 96) for w in ws[2:]), row(dG[:, D:2 * D]), row(G[:, :D]),
                                rowf(x) if self.sf32 else row(x), rowf(dx1),
                                opa(c, DC), opa(dcg, DC))                       # G7's staging: SW128 [128][64] boxes of c, dcg
        maps = self.maps.make(_key(dvg, dGg, dG, G, x, dx1, c, dcg, *ws), build)
        tp = (T + 1) // 2                                        # pair tiles: clusters of 2, pair c takes items c, c + P, ...
        q0 = self.qnext if self.qauto else 0
        for phase, items in ((0, 5 * tp), (1, tp), (2, tp)):
            grid = 2 * min(self.nsm // 2, items)
            self.k((grid, 1, 1), (384, 1, 1), *maps, xst, bs1, cst, c, dcg, dG, dx, dc, part3, T, phase,
                   q0 + (0, 5 * tp, 6 * tp)[phase])
        self.qnext = q0 + 7 * tp
        return 4 * T


_MID: dict = {}
_MID_FAILED: set = set()


def mid_op(index: int, nh: int, dh: int, sf32: bool, cf32: bool):
    """MidBwd for the layout and dtypes, built once; None (after one warning) when its build fails: the five launches run."""
    key = (index, nh, dh, sf32, cf32)
    if key in _MID_FAILED:
        return None
    if key not in _MID:
        try:
            _MID[key] = MidBwd(index, nh, dh, sf32, cf32)
        except Exception as exc:  # noqa: BLE001 -- a toolchain or driver problem keeps the five launches
            _MID_FAILED.add(key)
            warnings.warn(f"bias-only DiT bo_bwd_mid unavailable, keeping the five launches: {exc!r}", RuntimeWarning, stacklevel=2)
            return None
    return _MID[key]


class Wgrad:
    """``bo_wgrad.cu``: the six weight gradients of the bf16 step (and the unfold of the AdaLN projections' one)."""

    def __init__(self, device_index: int, da: int, trace: bool = False):
        self.idx, self.da = device_index, da
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.plan, wmax = wgrad_plan(da, self.nsm)
        self.k = _sm100()._sm100_kernel("bo_wgrad", "bo_wgrad_sm100", device_index, src_dir=str(_dir),
                                        defs=(f"DATT={da}", f"WMAX={wmax}", *(("TRACE",) if trace else ())), smem=SMEM_WGRAD[wmax])
        self.maps = _Maps()

    def __call__(self, grads, acts, outs, dwu, pw, wraw, w1, w2):
        """grads = (dz, dab, dy, dvg, dG, dGg), acts = (h, xt, og, xa, chat, c2) [M, *] bf16 rows; outs = (dWsq, dWab, dWo, dWvg,
        dWgg) in their parameters' dtypes (bf16 / fp32, contiguous); dwu [3072, 384] the unfolded projections' gradient, pw [192, 384]
        fp32 the cond-LN weight partials; wraw [3072, 384], w1 / w2 [384] fp32."""
        M = grads[0].shape[0]
        assert M % 128 == 0 and all(t.shape[0] == M and t.stride(1) == 1 and t.dtype is torch.bfloat16 for t in (*grads, *acts))
        key = ("wgrad", self.idx, self.da)
        tiles = _BUFS.tab.get(key)
        if tiles is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("bo_wgrad: the tile table must be built by an eager call first")
            tiles = _BUFS.tab[key] = torch.tensor(self.plan, dtype=torch.int32, device=grads[0].device)
        tm = _sm100()._tm
        mn = lambda t: tm(t, [t.shape[1], M], t.stride(0) * 2, [64, 64])                    # noqa: E731 MN-major [64 cols][64 rows]
        maps = self.maps.make(_key(*grads, *acts), lambda: _descriptors(*(mn(t) for t in (*grads, *acts))))
        odt = sum(1 << i for i, t in enumerate((*outs[:4], dwu, outs[4])) if t.dtype is torch.float32)
        assert all(t.is_contiguous() for t in (*outs, dwu, pw, wraw, w1, w2))
        self.k((tiles.numel(), 1, 1), (384, 1, 1), *maps, tiles, outs[0], outs[1], outs[2], outs[3], dwu, outs[4], pw, wraw, w1, w2,
               odt, M // 64)


def read_events(kernel, n_items: int):
    """-DTRACE builds: (g_ev [n_items, 16] int64 -- %globaltimer ns per role event, 0 = not recorded; see the .cu header --,
    g_cta0 [1024] int64 the CTAs' start times), after a synchronize."""
    from cuda.bindings import driver as cu
    torch.cuda.synchronize()
    out = []
    for name, n in ((b"g_ev", n_items * 16), (b"g_cta0", 1024)):
        err, dptr, size = cu.cuModuleGetGlobal(kernel.module, name)
        t = torch.zeros(n, dtype=torch.int64, device="cuda")
        if err != cu.CUresult.CUDA_SUCCESS:                      # bo_wgrad.cu has no g_cta0 (its events are per CTA)
            out.append(t)
            continue
        err, = cu.cuMemcpyDtoD(cu.CUdeviceptr(t.data_ptr()), dptr, min(int(size), n * 8))
        assert err == cu.CUresult.CUDA_SUCCESS, err
        out.append(t)
    torch.cuda.synchronize()
    return out[0].view(n_items, 16).cpu(), out[1].cpu()


def clear_events(kernel):
    """Zero g_ev / g_cta0 of a -DTRACE build."""
    from cuda.bindings import driver as cu
    for name in (b"g_ev", b"g_cta0"):
        err, dptr, size = cu.cuModuleGetGlobal(kernel.module, name)
        if err != cu.CUresult.CUDA_SUCCESS:
            continue
        err, = cu.cuMemsetD8(dptr, 0, size)
        assert err == cu.CUresult.CUDA_SUCCESS, err
    torch.cuda.synchronize()


_FAILED = False
_READY: set = set()
_OPS: dict = {}


def bwd_fused_ready(index: int, nh: int, dh: int) -> bool:
    """Build and load both kernels for (nh, dh) on device ``index`` once; False (after one warning) when a toolchain or driver
    problem keeps the per-step launches."""
    global _FAILED
    if _FAILED:
        return False
    key = (index, nh, dh)
    if key in _READY:
        return True
    try:
        ops(index, nh, dh)
    except Exception as exc:  # noqa: BLE001 -- a toolchain or driver problem keeps the per-step launches
        _FAILED = True
        warnings.warn(f"bias-only DiT fused bf16 training backward unavailable, keeping the per-step launches: {exc!r}",
                      RuntimeWarning, stacklevel=2)
        return False
    _READY.add(key)
    return True


def ops(index: int, nh: int, dh: int):
    """(TailBwd, Wgrad) for the layout, built once."""
    key = (index, nh, dh)
    if key not in _OPS:
        _OPS[key] = (TailBwd(index, nh, dh), Wgrad(index, nh * dh))
    return _OPS[key]


__all__ = ["MidBwd", "TailBwd", "Wgrad", "mid_on", "mid_op", "bwd_fused_on", "bwd_fused_ready", "clear_events", "ops", "pb_bar", "pbfin_on", "pv_pdl_on", "pvdpb_on", "read_events",
           "step_bufs",
           "tail_launches", "tail_mode",
           "wgrad_plan", "wgrad_tiles"]

"""The bf16-mixed three-kernel inference step's launchers (the default bf16 step; MINIWORLD_BIAS_ONLY_DIT_INF3_BF16=0 keeps the
12-launch step), the bf16 twin of the fp32 step in ``tf32.py``:

  bo_front_bf16.cu  K1: LN + AdaLN of the block input + the v|g GEMM, a cluster of 8, 6 or 4 CTAs per 128-row tile (``FrontBF16``);
                    x in fp32 or bf16, xa / v / g bf16
  bo_tail2_bf16.cu  K3, the pair form: clusters of 8 = two row tiles x 4 column groups as tcgen05 cta_group::2 pairs (``TailBF16``
                    with ``cl=TAIL_PAIR``), where it takes fewer rounds than CL 8 / 6
  bo_tail_bf16.cu   K3: out GEMM, residual + gate, LN + AdaLN, a|b GEMM, SwiGLU, squeeze GEMM, residual + gate -- one kernel, a
                    cluster of 8 or 6 CTAs per 128-row tile (``TailBF16``); bf16 operands (tcgen05 kind::f16, 64-column k-blocks),
                    fp32 accumulators, LayerNorm and residual; x in fp32 or bf16, out fp32 (the next block's residual) or bf16 (the
                    step's output)
  pv_gate_inf.cu    K2 with -DPDL_INF (``PvGateCore(pdl=True)``)

The cubins build on first use (``inf3_bf16_ready``); a failed build warns once and the runner keeps the 12-launch bf16 step.
"""

from __future__ import annotations

import os
import warnings

import torch

from miniworld_engine.kernels._compile import device_constant
from miniworld_engine.kernels.bias_only_dit.cuda import PTR, _descriptors, _Runs, _dir

TAIL_CLUSTERS = (8, 6)
TAIL_PAIR = 4                                                  # the pair tail's "cluster" code (bo_tail2_bf16.cu)


def tail_rows(M: int, cl: int) -> int:
    """Rows of the tail's XT / H scratch: M, or M padded to whole tile pairs for the pair tail."""
    return -(-M // 256) * 256 if cl == TAIL_PAIR else M


def inf3_bf16_on() -> bool:
    """The bf16-mixed inference step as three kernels per block (front, PDL core, tail) behind the hoisted pre-sigmoided bf16
    conditioning tables -- the default; MINIWORLD_BIAS_ONLY_DIT_INF3_BF16=0 keeps the 12-launch bf16 step. Read per call."""
    return os.environ.get("MINIWORLD_BIAS_ONLY_DIT_INF3_BF16", "1") != "0"


def tail_smem(cl: int) -> int:
    """bo_tail_bf16.cu's SMEM_BYTES (the fp32 tail's layout): CL 8 A ring 6 x 16 KB, five 24-KB W slots, statistics 8 + 2 KB,
    barriers; CL 6 the A ring, three 32-KB W slots, statistics 6 + 2 KB, 16 KB of h transpose tiles, barriers. ``TAIL_PAIR``
    (bo_tail2_bf16.cu): the A ring, four 24-KB W slots, statistics 4 + 2 KB, 16 KB of h transpose tiles, barriers."""
    if cl == TAIL_PAIR:
        return 6 * 16384 + 4 * 24576 + 4096 + 2048 + 16384 + 512
    if cl == 8:
        return 6 * 16384 + 5 * 24576 + 8192 + 2048 + 512
    assert cl == 6, cl
    return 6 * 16384 + 3 * 32768 + 6144 + 2048 + 16384 + 512


def tail_order(cl: int, c: int, nkb: int = 24) -> list[int]:
    """CTA c's z k-block order (bo_tail_bf16.cu kb_h): CL 8 its own 3 h k-blocks first, then the others in order; CL 6 in order."""
    if cl != 8:
        return list(range(nkb))
    ho = 1536 // cl // 64
    return [ho * c + t for t in range(ho)] + [k for k in range(nkb) if not ho * c <= k < ho * c + ho]


def pack_pairs_bf16(w: torch.Tensor, cl: int, z_order: bool = False) -> torch.Tensor:
    """W [768, K] bf16 -> the pair-packed form bo_tail_bf16.cu (cluster ``cl``, NC = 768 / cl output rows per CTA) loads as ONE TMA
    box per two 64-column k-blocks: P [(cl (K / 128) 2) NC, 64], P[((c (K / 128) + p) 2 + h) NC + n, k] = W[NC c + n, 64 o_c(2 p + h)
    + k] with o_c the k-block order (natural, or CTA c's z order: ``tail_order``). Device ops only (capture-safe: slices + cat)."""
    n, k = w.shape
    assert n == 768 and k % 128 == 0 and cl in TAIL_CLUSTERS, (w.shape, cl)
    nc, nkb = 768 // cl, k // 64
    wr = w.reshape(cl, nc, nkb, 64)
    if z_order and cl == 8:
        ho = 1536 // cl // 64
        wr = torch.stack([torch.cat([wr[c, :, ho * c:ho * c + ho], wr[c, :, :ho * c], wr[c, :, ho * c + ho:]], 1) for c in range(cl)], 0)
    return wr.reshape(cl, nc, nkb // 2, 2, 64).permute(0, 2, 3, 1, 4).reshape(-1, 64).contiguous()


def _max_clusters(kernel, cl: int) -> int:
    from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import max_clusters
    return max_clusters(kernel, cl)


class FrontBF16:
    """``bo_front_bf16.cu`` (K1): vg [M, 2 DA] bf16 = (LN(x) s1 + sh1) [Wv; Wg]^T. x [M, 768] fp32 or bf16 contiguous (the block input),
    tab [T, 6, 768] bf16 view (the hoisted tables; row stride a multiple of 8), w [2 DA, 768] bf16 contiguous, vg [M, 2 DA] bf16
    contiguous, xa [M, 768] bf16 scratch (row stride a multiple of 8). A cluster of 8, 6 or 4 CTAs per 128-row tile (``cluster``).
    Bound launches cached per (weights, table, vg, xa, M, T, cluster, x dtype); x by address per call."""

    def __init__(self, device_index: int, da: int, trace: bool = False):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        assert da in (768, 1024), da
        self._sm100, self._tm, self.device_index, self.da, self.trace = sm100, sm100._tm, device_index, da, trace
        self.runs = _Runs()

    def kernel(self, cl: int, xbf: bool = False):
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import front_smem
        return self._sm100._sm100_kernel("bo_front_bf16", "bo_front_bf16_sm100", self.device_index, pdl=True, src_dir=str(_dir),
                                         defs=(f"CL={cl}", f"DATT={self.da}", f"XBF={int(xbf)}", *(("TRACE",) if self.trace else ())),
                                         cluster=cl, smem=front_smem(cl, self.da))       # bo_front_tf32.cu's layout and bytes

    def clusters(self) -> tuple[int, ...]:
        return (8, 6, 4) if self.da == 768 else (8, 4)                # CL 6 splits 2 DA = 1536 columns only

    def cluster(self, n_tiles: int) -> int:
        """CTAs per row tile: the least modelled time over the sizes built (rounds of resident clusters x per-CTA time ~ 1 / CL; the
        fp32 front's model). MINIWORLD_BIAS_ONLY_DIT_INF3_CL forces one."""
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import pick_cluster
        forced = os.environ.get("MINIWORLD_BIAS_ONLY_DIT_INF3_CL")
        if forced:
            assert int(forced) in self.clusters(), (forced, self.da)
            return int(forced)
        return pick_cluster(n_tiles, {cl: (_max_clusters(self.kernel(cl), cl), 8 / cl) for cl in self.clusters()})

    def _bind(self, tab, w, vg, xa, M, T, cl, xbf, serial):
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import _LaunchPDL
        nv = 2 * self.da // cl
        wrows = nv if nv <= 256 else nv // 2
        maps = [self._tm(w, [768, 2 * self.da], 768 * 2, [64, wrows]), self._tm(xa, [768, M], xa.stride(0) * 2, [64, 128]),
                self._tm(tab, [6 * 768, T], tab.stride(0) * 2, [32, 128], swizzle=64)]            # s1 / sh1 boxes
        return _LaunchPDL(self.kernel(cl, xbf), ((M // 128) * cl, 1, 1), (384, 1, 1), *_descriptors(*maps), PTR, vg, xa, int(T),
                          int(xa.stride(0)), 1e-5, serial=serial)

    def __call__(self, x, tab, w, vg, T, xa, cl=None, serial=False):
        """``serial``: launch without the programmatic-serialization attribute (the step's first front where the core takes P
        before its PDL wait: every kernel of the chain then starts after the stream's earlier work, P's writer included, is done)."""
        M, da, bf = x.shape[0], self.da, torch.bfloat16
        xbf = x.dtype is bf
        assert x.dtype in (bf, torch.float32) and x.is_contiguous() and x.shape == (M, 768) and M % 128 == 0 and T % 128 == 0
        assert xa.dtype is bf and xa.dim() == 2 and xa.shape == (M, 768) and xa.stride(1) == 1 and xa.stride(0) % 8 == 0
        assert xa.data_ptr() % 16 == 0
        assert tab.dtype is bf and tab.shape[0] == T and tab[0].numel() == 6 * 768 and tab.stride(-1) == 1
        assert tab.stride(0) % 8 == 0 and tab.data_ptr() % 16 == 0 and (tab.dim() == 2 or tab.stride(1) == 768)
        assert w.dtype is bf and w.is_contiguous() and w.shape == (2 * da, 768)
        assert vg.dtype is bf and vg.is_contiguous() and vg.shape == (M, 2 * da)
        cl = cl or self.cluster(M // 128)
        key = (tab.data_ptr(), tab.stride(0), w.data_ptr(), vg.data_ptr(), xa.data_ptr(), xa.stride(0), M, T, cl, xbf, serial)
        self.runs.bind(key, lambda: self._bind(tab, w, vg, xa, M, T, cl, xbf, serial))(x)
        return vg


class TailBF16:
    """``bo_tail_bf16.cu`` (K3): out [M, 768] = x1 + gate2 (silu(xt Wa^T) (xt Wb^T)) Wsq^T, x1 = x + gate1 a Wo^T, xt = LN(x1) s2 +
    sh2; a [M, DA] bf16 (the core's output), x [M, 768] fp32 or bf16 (the block's residual input), tab [T, 6, 768] bf16 view (the
    hoisted tables, sigmoids applied), wo / wsq = ``pack_pairs_bf16(., cl)`` of Wo [768, DA] / Wsq [768, 1536] (wsq in the z order),
    wab [3072, 768] = [Wa; Wb], all bf16 contiguous; out [M, 768] fp32 or bf16 contiguous (not x); scratch: xt [M, 768] bf16 rows (row
    stride a multiple of 8) and h [M 24, 64] bf16 contiguous, blocked k-block-major. A cluster of 8 or 6 CTAs per 128-row tile
    (``cluster``). Bound launches cached per (a, x, table, weights, scratch, M, T, cluster, dtypes); out by address per call."""

    def __init__(self, device_index: int, da: int, trace: bool = False):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        assert da in (768, 1024), da
        self._sm100, self._tm, self.device_index, self.da, self.trace = sm100, sm100._tm, device_index, da, trace
        self.runs = _Runs()

    def kernel(self, cl: int = 8, xbf: bool = False, obf: bool = False):
        if cl == TAIL_PAIR:
            return self._sm100._sm100_kernel("bo_tail2_bf16", "bo_tail2_bf16_sm100", self.device_index, pdl=True, src_dir=str(_dir),
                                             defs=(f"DATT={self.da}", f"XBF={int(xbf)}", f"OBF={int(obf)}",
                                                   *(("TRACE",) if self.trace else ())), cluster=8, smem=tail_smem(TAIL_PAIR))
        return self._sm100._sm100_kernel("bo_tail_bf16", "bo_tail_bf16_sm100", self.device_index, pdl=True, src_dir=str(_dir),
                                         defs=(f"CL={cl}", f"DATT={self.da}", f"XBF={int(xbf)}", f"OBF={int(obf)}",
                                               *(("TRACE",) if self.trace else ())), cluster=cl, smem=tail_smem(cl))

    def cluster(self, n_tiles: int) -> int:
        """8 or 6 CTAs per row tile: the fewest rounds of resident clusters, a CL 6 CTA costing ~1.4 x a CL 8 one (the fp32 tail's
        model; one CTA per SM either way: the tails take 480-512 TMEM columns); the pair tail (``TAIL_PAIR``) where it takes fewer rounds
        than both (A = 5: L640 / L768). MINIWORLD_BIAS_ONLY_DIT_INF3_TAIL_CL=8 / 6 / 4 forces one."""
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import pick_cluster
        forced = os.environ.get("MINIWORLD_BIAS_ONLY_DIT_INF3_TAIL_CL")
        if forced in ("8", "6", "4"):
            return int(forced)
        n8, n6 = _max_clusters(self.kernel(8), 8), _max_clusters(self.kernel(6), 6)
        rounds = min(-(-n_tiles // max(n8, 1)), -(-n_tiles // max(n6, 1)))
        if -(-n_tiles // max(2 * _max_clusters(self.kernel(TAIL_PAIR), 8), 1)) < rounds:
            return TAIL_PAIR
        return pick_cluster(n_tiles, {8: (n8, 1.0), 6: (n6, 1.4)})

    def _bind(self, a, x, tab, wo, wab, wsq, xt, h, M, T, cl, xbf, obf):
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import _LaunchPDL
        if cl == TAIL_PAIR:
            return self._bind_pair(a, x, tab, wo, wab, wsq, xt, h, M, T, xbf, obf)
        tm, da, nc, nhp = self._tm, self.da, 768 // cl, 1536 // cl // (2 if cl == 6 else 1)
        xmap = (tm(x, [768, M], 768 * 2, [32, 128], swizzle=64) if xbf else
                tm(x, [768, M], 768 * 4, [32, 128], dtype="f32"))
        maps = [tm(a, [da, M], da * 2, [64, 128]), xmap,
                tm(tab, [6 * 768, T], tab.stride(0) * 2, [32, 128], swizzle=64),
                tm(wo, [64, wo.shape[0]], 128, [64, 2 * nc]),
                tm(wab, [768, 3072], 768 * 2, [64, nhp]),
                tm(wsq, [64, wsq.shape[0]], 128, [64, 2 * nc]),
                tm(xt, [768, M], xt.stride(0) * 2, [64, 128]),
                tm(h, [64, h.shape[0]], 128, [64, 128]), tm(h, [64, h.shape[0]], 128, [64, 256])]   # h: one / two k-blocks a box
        return _LaunchPDL(self.kernel(cl, xbf, obf), ((M // 128) * cl, 1, 1), (384, 1, 1), *_descriptors(*maps), PTR, xt, h,
                          int(xt.stride(0)), int(T), 1e-5)

    def _bind_pair(self, a, x, tab, wo, wab, wsq, xt, h, M, T, xbf, obf):
        """bo_tail2_bf16.cu: clusters of 8 = two row tiles x 4 column groups; each CTA's W boxes its half (96 rows of the CL 8 packs,
        natural k order; 128 rows of Wa or Wb); the scratch covers whole tile pairs."""
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import _LaunchPDL
        tm, da = self._tm, self.da
        Mp = tail_rows(M, TAIL_PAIR)
        xmap = (tm(x, [768, M], 768 * 2, [32, 128], swizzle=64) if xbf else
                tm(x, [768, M], 768 * 4, [32, 128], dtype="f32"))
        maps = [tm(a, [da, M], da * 2, [64, 128]), xmap,
                tm(tab, [6 * 768, T], tab.stride(0) * 2, [32, 128], swizzle=64),
                tm(wo, [64, wo.shape[0]], 128, [64, 192]),
                tm(wab, [768, 3072], 768 * 2, [64, 128]),
                tm(wsq, [64, wsq.shape[0]], 128, [64, 192]),
                tm(xt, [768, Mp], xt.stride(0) * 2, [64, 128]),
                tm(h, [64, Mp * 24], 128, [64, 256])]                 # two h k-blocks per box
        return _LaunchPDL(self.kernel(TAIL_PAIR, xbf, obf), ((Mp // 256) * 8, 1, 1), (384, 1, 1), *_descriptors(*maps), PTR, xt, h,
                          int(xt.stride(0)), int(T), int(M // 128), 1e-5)

    def __call__(self, a, x, tab, wo, wab, wsq, out, T, xt, h, cl=None):
        """wo / wsq: ``pack_pairs_bf16(wo, cl)`` / ``pack_pairs_bf16(wsq, cl, z_order=True)`` for ``cl`` (default:
        ``cluster(M / 128)``); the pair tail (``TAIL_PAIR``): ``pack_pairs_bf16(., 8)`` both, natural k order, and xt / h with
        ``tail_rows(M, TAIL_PAIR)`` rows."""
        M, da, bf = x.shape[0], self.da, torch.bfloat16
        cl = cl or self.cluster(M // 128)
        Mp = tail_rows(M, cl)
        xbf, obf = x.dtype is bf, out.dtype is bf
        assert x.dtype in (bf, torch.float32) and out.dtype in (bf, torch.float32), (x.dtype, out.dtype)
        assert xt.dtype is bf and xt.dim() == 2 and xt.shape == (Mp, 768) and xt.stride(1) == 1 and xt.stride(0) % 8 == 0, (xt.shape, xt.stride())
        assert xt.data_ptr() % 16 == 0
        assert h.is_contiguous() and h.shape == (Mp * 24, 64) and h.dtype is bf, h.shape            # blocked k-block-major
        assert a.is_contiguous() and a.shape == (M, da) and a.dtype is bf
        assert x.is_contiguous() and x.shape == (M, 768) and M % 128 == 0 and T % 128 == 0
        assert out.is_contiguous() and out.shape == (M, 768) and out.data_ptr() != x.data_ptr()
        assert tab.dtype is bf and tab.shape[0] == T and tab[0].numel() == 6 * 768 and tab.stride(-1) == 1
        assert tab.stride(0) % 8 == 0 and tab.data_ptr() % 16 == 0 and (tab.dim() == 2 or tab.stride(1) == 768)
        assert wo.shape == (768 * da // 64, 64) and wab.shape == (3072, 768) and wsq.shape == (768 * 24, 64), (wo.shape, wsq.shape)
        assert all(w.is_contiguous() and w.dtype is bf for w in (wo, wab, wsq))
        key = (a.data_ptr(), x.data_ptr(), tab.data_ptr(), tab.stride(0), wo.data_ptr(), wab.data_ptr(), wsq.data_ptr(),
               xt.data_ptr(), xt.stride(0), h.data_ptr(), M, T, cl, xbf, obf)
        self.runs.bind(key, lambda: self._bind(a, x, tab, wo, wab, wsq, xt, h, M, T, cl, xbf, obf))(out)
        return out


_FAILED = False
_READY: set = set()


@device_constant
def inf3_bf16_ready(index: int, nh: int, dh: int) -> bool:
    """Build and load the bf16 step's cubins for (nh, dh) on device ``index`` once (the fronts, the tails and the pair tail for x / out
    fp32 and bf16, a PDL core); False (after one warning) when a toolchain or driver problem keeps the 12-launch bf16 step. A
    constant to ``torch.compile``."""
    global _FAILED
    if _FAILED:
        return False
    key = (index, nh, dh)
    if key in _READY:
        return True
    try:
        from miniworld_engine.kernels.bias_only_dit import cuda as C
        tail, front = TailBF16(index, nh * dh), FrontBF16(index, nh * dh)
        for cl in (*TAIL_CLUSTERS, TAIL_PAIR):
            for xbf, obf in ((True, True), (True, False), (False, False), (False, True)):
                tail.kernel(cl, xbf, obf)
        for cl in front.clusters():
            for xbf in (True, False):
                front.kernel(cl, xbf)
        C.PvGateCore(index, nh=nh, dh=dh, pdl=True, p_early=True).kernel(1, True)   # the step's core (P_EARLY unless switched off)
    except Exception as exc:  # noqa: BLE001 -- a toolchain or driver problem keeps the 12-launch bf16 step
        _FAILED = True
        warnings.warn(f"bias-only DiT bf16 three-kernel inference step unavailable, keeping the 12-launch step: {exc!r}",
                      RuntimeWarning, stacklevel=2)
        return False
    _READY.add(key)
    return True


__all__ = ["FrontBF16", "TAIL_CLUSTERS", "TAIL_PAIR", "TailBF16", "inf3_bf16_on", "inf3_bf16_ready", "pack_pairs_bf16", "tail_order",
           "tail_rows", "tail_smem"]

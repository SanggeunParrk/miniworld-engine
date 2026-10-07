"""The fp32 path of the bias-only token DiT on B200 (sm_100a): TF32 tensor cores, fp32 everything else.

  pv_gate_tf32.cu          the attention core, a = sigmoid(g) (P v) -- or P v (the training backward's dV = P^T dO) -- per head and
                           sample, P fp32 from shared memory as the K-major A operand, v the MN-major B (``PvGateCoreTF32``)
  dpb32_sm100.cu           the bias gradient, dbias = P o (sum_a dO v^T - D): the fp32 port of dpb_sm100.cu (``Dpb32``)
  gemm_glu_tf32.cu         the training transition's expand GEMM with the SwiGLU in its epilogue (h and a | b out) and the dh GEMM
                           with the SwiGLU backward in its epilogue (da | db out, dh never stored) (``GemmGluTF32``)
  bias_only_dit_f32_rows.cu  every fp32 row pass the bf16 extensions do not already take in fp32: SwiGLU and its backward, the
                           softmax (with P^T for training), the pair bias LN(pair) Wf^T and its backward (exact fp32 on the FMA pipe),
                           the training rows (conditioning LN, AdaLN, residual + AdaLN, residual out, their backwards, the gate
                           backward) -- ``rows32()``
  bo_front_tf32.cu         K1 of the three-kernel inference step (the default fp32 inference step; MINIWORLD_BIAS_ONLY_DIT_INF3=0:
                           cuBLAS + rows): LN + AdaLN of the block input and the v|g GEMM, a cluster of 4, 6 or 8 CTAs per 128-row
                           tile (``FrontTF32``)
  bo_tail_tf32.cu          K3 of that step: out GEMM, residual + gate, LN + AdaLN, a|b GEMM, SwiGLU, squeeze GEMM, residual + gate --
                           one kernel, a cluster of 8 or 6 CTAs per 128-row tile (``TailTF32``); K2 is ``pv_gate_tf32.cu -DPDL_INF``.
                           Both exchange their GEMM operands inside the cluster through L2 scratch (TMA store or st.global, one
                           release, TMA-fed GEMM)

General GEMMs are cuBLAS on TF32 tensor cores (``tf32_gemms``: forced whatever the caller's allow_tf32, restored after), as the
token DiT's fp32 path. The cubins and the extension build on first use (``tf32_ready``), never at import; a failed build warns
once and the callers keep the module's PyTorch path.
"""

from __future__ import annotations

import contextlib
import functools
import os
import warnings
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import device_constant
from miniworld_engine.kernels.bias_only_dit.cuda import LAYOUTS, PTR, _descriptors, _Launch, _Runs

_dir = Path(__file__).parent
_SMEM_MAX = 232448


@contextlib.contextmanager
def tf32_gemms(on: bool = True):
    """cuBLAS on TF32 tensor cores for the fp32 path's GEMMs (restored after): the path is the TF32 recipe -- its attention core
    is TF32 too -- and IEEE fp32 GEMMs would dominate the block."""
    if not on:
        yield
        return
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@functools.lru_cache(maxsize=None)
def rows32():
    """The fp32 row kernels (``bias_only_dit_f32_rows.cu``), a torch extension built on first use."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_f32_rows_cuda",
        sources=[str(_dir / "bias_only_dit_f32_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                           *gencodes("80", "90", "100", ptx=("100",))],
        extra_cflags=["-std=c++17"], verbose=False)


def _f32_rows(t: torch.Tensor) -> int:
    """Row stride in bytes of an fp32 [rows, cols] view TMA reads (unit column stride, 16-byte aligned rows)."""
    assert t.dtype is torch.float32 and t.stride(1) == 1 and t.stride(0) % 4 == 0 and t.data_ptr() % 16 == 0, (t.dtype, t.stride())
    return t.stride(0) * 4


# --------------------------------------------------------------------------------------------------- attention core
#: samples per work item the core may be built for (a cubin each): two accumulator sets of SG x DH columns within 512
_PV_GROUPS = {32: (1, 2, 4, 8), 48: (1, 2, 4, 5), 64: (1, 2, 4)}


def pv_batched() -> bool:
    """pv_gate_tf32.cu -DBATCHN: one MMA per K step for all of an item's samples (N = SG x 32-channel atoms) instead of one per
    sample. MINIWORLD_BIAS_ONLY_DIT_PV_BATCH=0: one per sample (the round-2 kernel)."""
    return os.environ.get("MINIWORLD_BIAS_ONLY_DIT_PV_BATCH", "1") != "0"


def pv_groups(dh: int) -> tuple[int, ...]:
    """The sample groups built for head width dh: batched, the N = SG x 32 (dh / 32 rounded up) product within 256 columns and two
    accumulator sets within 512 (48-wide heads: SG <= 4)."""
    if not pv_batched():
        return _PV_GROUPS[dh]
    nw = (dh + 31) // 32 * 32
    return tuple(g for g in _PV_GROUPS[dh] if g * nw <= 256)


def _pv_smem(dh: int, sg: int, nx: int = 3) -> int:
    """pv_gate_tf32.cu's shared memory: NST stages (P chunk + SG v chunks), NX g / a staging tiles, the barriers."""
    na, wb = (dh + 31) // 32, dh - 32
    stb, xg = 128 * 128 + sg * na * 32 * 128, 128 * 128 + 128 * wb * 4
    nst = (_SMEM_MAX - 1024 - nx * xg) // stb
    assert nst >= 2 and 2 * sg * dh <= 512, (dh, sg, nst)
    return nst * stb + nx * xg + 512


def pick_group_tf32(S: int, L: int, nsm: int, nh: int, dh: int) -> int:
    """Samples per work item: the fewest bytes into the busiest SM -- rounds x (the P tile once, SG samples' v tiles, their g in and
    a out). MINIWORLD_BIAS_ONLY_DIT_SG forces one (clamped to the built groups)."""
    groups = [g for g in pv_groups(dh) if g <= max(S, 1)]
    forced = os.environ.get("MINIWORLD_BIAS_ONLY_DIT_SG")
    if forced:
        f = int(forced)
        return max([g for g in groups if g <= f] or [1])
    best = None
    for sg in groups:
        items = nh * (L // 128) * -(-S // sg)
        cost = -(-items // nsm) * (L * 128 * 4 + sg * L * dh * 4 + sg * 128 * dh * 8)
        if best is None or cost < best[0]:
            best = (cost, sg)
    return best[1]


class PvGateCoreTF32:
    """``pv_gate_tf32.cu``: out [S L, nh dh] = sigmoid(g) * (P v) -- or, without g, P v -- per head and sample; P one block's
    [nh L, L] fp32 attention weights (or their per-head transpose; contiguous), v / g / out [S L, nh dh] fp32 views with any
    16-byte aligned row stride. Bound launches (their TMA descriptors carry the pointers) are cached per (buffers, strides, S, L).
    ``pdl``: the three-kernel inference step's build (``-DPDL_INF``: programmatic dependent launch, the gated output rounded to TF32
    for the tail's MMA), launched with the PDL attribute; default off (the training step and the default inference step)."""

    def __init__(self, device_index: int, nh: int = 16, dh: int | None = None, pdl: bool = False):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        self._sm100, self._tm = sm100, sm100._tm
        self.nh, self.dh = nh, dh or 768 // nh
        assert (self.nh, self.dh) in LAYOUTS, (nh, dh)
        self.device_index, self.pdl = device_index, pdl
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs = _Runs()

    def kernel(self, sg: int, gate: bool = True):
        """The cubin for SG samples per item, with or without the gate (built on first use); batched N unless
        MINIWORLD_BIAS_ONLY_DIT_PV_BATCH=0."""
        batch = ("BATCHN",) if pv_batched() else ()
        if self.pdl:
            return self._sm100._sm100_kernel("pv_gate_tf32", "bo_pv_gate_tf32_sm100", self.device_index, pdl=True, src_dir=str(_dir),
                                             defs=(f"SG={sg}", f"GATE={int(gate)}", f"NHEAD={self.nh}", f"DHEAD={self.dh}", *batch,
                                                   "PDL_INF"), smem=_pv_smem(self.dh, sg))
        return self._sm100._sm100_kernel("pv_gate_tf32", "bo_pv_gate_tf32_sm100", self.device_index, src_dir=str(_dir),
                                         defs=(f"SG={sg}", f"GATE={int(gate)}", f"NHEAD={self.nh}", f"DHEAD={self.dh}", *batch),
                                         smem=_pv_smem(self.dh, sg))

    def _halves(self, t, M):
        """TMA maps of a [M, nh dh] fp32 view as the kernel stages it: channels 0-31 of a head (SW128) and 32..dh-1 (SW64 for 16
        channels, SW128 for 32, none for dh 32)."""
        tm, DA, wb, rs = self._tm, self.nh * self.dh, self.dh - 32, _f32_rows(t)
        a = tm(t, [DA, M], rs, [32, 128], dtype="f32")
        b = a if wb == 0 else tm(t, [DA, M], rs, [wb, 128], swizzle=64 if wb == 16 else 128, dtype="f32")
        return a, b

    def _bind(self, v, P, out, S, L, g):
        M, nh, dh = v.shape[0], self.nh, self.dh
        tm = self._tm
        mp = tm(P, [L, nh * L], L * 4, [32, 128], dtype="f32")
        mv = tm(v, [nh * dh, M], _f32_rows(v), [32, 32], swizzle="128a32", dtype="f32")
        ga, gb = self._halves(out if g is None else g, M)
        oa, ob = self._halves(out, M)
        maps = _descriptors(mp, mv, ga, gb, oa, ob)
        sg = pick_group_tf32(S, L, self.nsm, nh, dh)
        k = self.kernel(sg, g is not None)
        grid = (min(self.nsm, nh * (L // 128) * -(-S // sg)), 1, 1)
        return (_LaunchPDL if self.pdl else _Launch)(k, grid, (256, 1, 1), *maps, L, S)

    def __call__(self, v, P, out, S, g=None):
        """v [S L, nh dh] view, P [nh L, L] contiguous, out [S L, nh dh] view, g [S L, nh dh] view or None; all fp32."""
        L = v.shape[0] // S
        assert P.dtype is torch.float32 and P.is_contiguous() and tuple(P.shape) == (self.nh * L, L) and L % 128 == 0
        key = (v.data_ptr(), v.stride(0), None if g is None else (g.data_ptr(), g.stride(0)), P.data_ptr(), out.data_ptr(),
               out.stride(0), S, L, pv_batched())
        self.runs.bind(key, lambda: self._bind(v, P, out, S, L, g))()
        return out


# --------------------------------------------------------------------------------------------------- SwiGLU GEMMs (training)
def glu_smem(bwd: bool) -> int:
    """gemm_glu_tf32.cu's shared memory: NST ring stages of 32 KB, two warpgroups x NB staging buffers of NT 8-KB tiles, barriers."""
    nst, nb, nt = (3, 4, 2) if bwd else (4, 2, 3)
    return nst * 32768 + 2 * nb * nt * 8192 + 512


class GemmGluTF32:
    """``gemm_glu_tf32.cu`` (D = 768, H = 1536, fp32, kind::tf32, 2-CTA clusters). Forward: ``(x, Wab, ab, h)`` -- ab [M, 2H] =
    x [Wa; Wb]^T, h = silu(a) b. Backward (``bwd``): ``(dz, WsqT, ab, dab)`` -- dh = dz WsqT^T in the GEMM, dab = [da | db] of the
    SwiGLU backward from it and ab (dh never stored). All fp32 contiguous; WsqT = Wsq^T [H, D]. Bound launches cached per buffer set."""

    D, H = 768, 1536

    def __init__(self, device_index: int, bwd: bool = False):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        self._tm, self.bwd = sm100._tm, bwd
        self.k = sm100._sm100_kernel("gemm_glu_tf32", "bo_glu_bwd_tf32_sm100" if bwd else "bo_glu_fwd_tf32_sm100", device_index,
                                     src_dir=str(_dir), defs=("BWD",) if bwd else (), cluster=2, smem=glu_smem(bwd))
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs = _Runs()

    def _bind(self, x, w, ab, out):
        M, D, H, tm = x.shape[0], self.D, self.H, self._tm
        f = dict(dtype="f32")
        mx = tm(x, [D, M], D * 4, [32, 64], swizzle=128, **f)
        wmap = lambda t: tm(t, [D, t.shape[0]], D * 4, [32, 128], swizzle=128, **f)                      # noqa: E731
        tile = lambda t: tm(t, [H, M], t.stride(0) * 4, [16, 128], swizzle=64, **f)                     # noqa: E731
        half = lambda t, i: t[:, i * H:(i + 1) * H]                                                      # noqa: E731
        if self.bwd:
            maps = (mx, wmap(w), tile(half(ab, 0)), tile(half(ab, 1)), tile(half(out, 0)), tile(half(out, 1)))
        else:
            maps = (mx, wmap(w[:H]), wmap(w[H:]), tile(out), tile(half(ab, 0)), tile(half(ab, 1)))
        return _Launch(self.k, (self.nsm & ~1, 1, 1), (512, 1, 1), *_descriptors(*maps), (M + 127) // 128)

    def __call__(self, x, w, ab, out):
        M, D, H = x.shape[0], self.D, self.H
        assert x.shape == (M, D) and ab.shape == (M, 2 * H), (x.shape, ab.shape)
        assert w.shape == ((H, D) if self.bwd else (2 * H, D)) and out.shape == ((M, 2 * H) if self.bwd else (M, H)), (w.shape, out.shape)
        assert all(t.dtype is torch.float32 and t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (x, w, ab, out))
        self.runs.bind((x.data_ptr(), w.data_ptr(), ab.data_ptr(), out.data_ptr(), M), lambda: self._bind(x, w, ab, out))()
        return out


_GLU_FAILED: set = set()


def glu_op(device_index: int, bwd: bool):
    """The SwiGLU GEMM (``GemmGluTF32``) for the training step, or None: switched off or its build failed (warned once): cuBLAS + the
    fp32 rows. Backward on by default (MINIWORLD_BIAS_ONLY_DIT_TF32_GLU_BWD=0: off); forward OFF by default
    (MINIWORLD_BIAS_ONLY_DIT_TF32_GLU=1: on) -- its GEMM runs at ~530 TF/s against cuBLAS's ~680, which eats the SwiGLU pass it
    removes (L768 per launch 329 against 254 + 99 us; whole step not better)."""
    var, default = ("MINIWORLD_BIAS_ONLY_DIT_TF32_GLU_BWD", "1") if bwd else ("MINIWORLD_BIAS_ONLY_DIT_TF32_GLU", "0")
    if os.environ.get(var, default) == "0":
        return None
    key = (device_index, bwd)
    if key in _GLU_FAILED:
        return None
    op = _GLU_OPS.get(key)
    if op is None:
        try:
            op = _GLU_OPS[key] = GemmGluTF32(device_index, bwd)
        except Exception as exc:  # noqa: BLE001 -- a toolchain or driver problem keeps cuBLAS + the rows
            _GLU_FAILED.add(key)
            warnings.warn(f"bias-only DiT fp32 SwiGLU GEMM ({'backward' if bwd else 'forward'}) unavailable, keeping cuBLAS + the "
                          f"fp32 rows: {exc!r}", RuntimeWarning, stacklevel=2)
            return None
    return op


_GLU_OPS: dict = {}


# --------------------------------------------------------------------------------------------------- bias gradient
#: dpb32_sm100.cu's layout (NJ = 128: two stages of HP NA (16 KB + 16 KB) beside the P ring and the dbias staging)
DPB32_NJ = 128
DPB32_SMEM = (2 * 16384 + 4 * 3 * 4096) + 2 * 65536 + 256


class Dpb32:
    """``dpb32_sm100.cu`` (the fp32 port of dpb_sm100.cu): dbias [nh L, L] = P o (sum_a do v^T - D) per head, D[h, i] =
    sum_a dd[a, h, i]; do / v [A L, nh dh] fp32 views (16-byte aligned rows), P [nh L, L] fp32 contiguous, dd [A, nh, L] fp32,
    dbias [nh L, L] fp32. Bound launches (their TMA descriptors) cached per buffer set."""

    def __init__(self, device_index: int, nh: int = 16, dh: int | None = None):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        self._sm100, self._tm, self.device_index = sm100, sm100._tm, device_index
        self.nh, self.dh = nh, dh or 768 // nh
        assert (self.nh, self.dh) in LAYOUTS, (nh, dh)
        self.hp = 2 if self.dh == 32 else 1
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs = _Runs()

    def kernel(self):
        return self._sm100._sm100_kernel("dpb32_sm100", "bo_dpb32_sm100", self.device_index, src_dir=str(_dir),
                                         defs=(f"NJ={DPB32_NJ}", f"NHEAD={self.nh}", f"DHEAD={self.dh}"), smem=DPB32_SMEM)

    def _bind(self, do, v, P, out, A, L):
        nh, dh, tm, M = self.nh, self.dh, self._tm, do.shape[0]
        DA, f = nh * dh, dict(dtype="f32", swizzle=128)
        box = lambda t, w, rows: tm(t, [DA, M], _f32_rows(t), [w, rows], **f)                 # noqa: E731
        mdo, mv = box(do, 32, 128), box(v, 32, DPB32_NJ)
        mdo2, mv2 = (box(do, 16, 128), box(v, 16, DPB32_NJ)) if dh == 48 else (mdo, mv)     # 48-wide heads: 32 + 16 channels
        maps = _descriptors(mdo, mdo2, mv, mv2, tm(P, [L, nh * L], L * 4, [32, 128], **f), tm(out, [L, nh * L], L * 4, [32, 32], **f))
        grid = (min(self.nsm, nh // self.hp * (L // 128) * (L // DPB32_NJ)), 1, 1)
        return _Launch(self.kernel(), grid, (256, 1, 1), *maps, PTR, L, A)

    def __call__(self, do, v, P, dd, out, A):
        L = do.shape[0] // A
        assert all(t.dtype is torch.float32 for t in (do, v, P, dd, out)) and dd.is_contiguous() and P.is_contiguous()
        assert out.is_contiguous() and tuple(P.shape) == (self.nh * L, L) and L % 128 == 0
        key = (do.data_ptr(), do.stride(0), v.data_ptr(), v.stride(0), P.data_ptr(), out.data_ptr(), A, L)
        self.runs.bind(key, lambda: self._bind(do, v, P, out, A, L))(dd)
        return out


# --------------------------------------------------------------------------------------------------- three-kernel inference step
#: bo_front_tf32.cu / bo_tail_tf32.cu shared-memory layout constants (mirrored; the sources static_assert theirs)
_SLOT, _NS, _NA = 16384, 3, 4


def inf3_on() -> bool:
    """The fp32 inference step as three kernels per block (front, core, tail) behind the hoisted conditioning tables -- the default;
    MINIWORLD_BIAS_ONLY_DIT_INF3=0 keeps the cuBLAS + rows step (12 launches per block). Read per call."""
    return os.environ.get("MINIWORLD_BIAS_ONLY_DIT_INF3", "1") != "0"


def front_smem(cl: int, da: int) -> int:
    """bo_front_tf32.cu's SMEM_BYTES: a 6-stage A ring of 16 KB, the W ring (as many [WROWS][32] slots as fit), stats exchange, own
    stats, barriers."""
    nv = 2 * da // cl
    wslot = (nv if nv <= 256 else nv // 2) * 128
    o_w = 6 * _SLOT
    misc = cl * 1024 + 1024 + 512
    nws = (_SMEM_MAX - o_w - misc) // wslot
    return o_w + nws * wslot + misc


def tail_smem(cl: int) -> int:
    """bo_tail_tf32.cu's SMEM_BYTES -- CL 8: a 6-stage A ring of 16 KB, five 24-KB W slots, statistics 8 + 2 KB, barriers; CL 6: the
    A ring, three 32-KB W slots, statistics 6 + 2 KB, 16 KB of h transpose tiles, barriers. ``TAIL_PAIR`` (bo_tail2_tf32.cu): the A
    ring, four 24-KB W slots, statistics 4 + 2 KB, 16 KB of h transpose tiles, barriers."""
    if cl == TAIL_PAIR:
        return 6 * 16384 + 4 * 24576 + 4096 + 2048 + 16384 + 512
    if cl == 8:
        return 6 * 16384 + 5 * 24576 + 8192 + 2048 + 512
    assert cl == 6, cl
    return 6 * 16384 + 3 * 32768 + 6144 + 2048 + 16384 + 512


TAIL_CLUSTERS = (8, 6)
#: the pair tail's "cluster" code (bo_tail2_tf32.cu): 4 CTAs per row tile, two tiles per cluster of 8 (round 10, A/B switch below)
TAIL_PAIR = 4


def tail_pair_on() -> bool:
    """The pair tail where it saves a round (the default); MINIWORLD_BIAS_ONLY_DIT_INF3_2CTA=0 (read per call) keeps CL 8 / CL 6."""
    return os.environ.get("MINIWORLD_BIAS_ONLY_DIT_INF3_2CTA", "1") != "0"


def tail_rows(M: int, cl: int) -> int:
    """Rows of the tail's XT / H scratch: M, or M padded to whole tile pairs for the pair tail."""
    return -(-M // 256) * 256 if cl == TAIL_PAIR else M


def pack_pairs(w: torch.Tensor, cl: int = 8) -> torch.Tensor:
    """W [768, K] -> the pair-packed form bo_tail_tf32.cu (cluster ``cl``, NC = 768 / cl output rows per CTA) loads as ONE TMA box
    per two k-blocks: P [(cl (K / 64) 2) NC, 32], P[((c (K / 64) + p) 2 + h) NC + n, k] = W[NC c + n, 64 p + 32 h + k] (c the CTA,
    p the k-block pair). Device ops only (capture-safe)."""
    n, k = w.shape
    assert n == 768 and k % 64 == 0 and cl in TAIL_CLUSTERS, (w.shape, cl)
    return w.reshape(cl, 768 // cl, k // 64, 2, 32).permute(0, 2, 3, 1, 4).reshape(-1, 32).contiguous()


_MAXC: dict = {}


def max_clusters(kernel, cl: int, block: int = 384) -> int:
    """cuOccupancyMaxActiveClusters: how many clusters of ``cl`` CTAs of ``kernel`` can be resident at once (B200, ~227 KB of shared
    memory per CTA: 15 of 8, 22 of 6, 33 of 4, measured). Cached per kernel; 148 // cl if the query fails."""
    key = (id(kernel), cl)
    if key not in _MAXC:
        from cuda.bindings import driver as cu
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = cl * 64, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = kernel.smem
        at = cu.CUlaunchAttribute()
        at.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cl, 1, 1
        cfg.attrs = [at]
        cfg.numAttrs = 1
        err, n = cu.cuOccupancyMaxActiveClusters(kernel.func, cfg)
        _MAXC[key] = int(n) if err == cu.CUresult.CUDA_SUCCESS and n > 0 else 148 // cl
    return _MAXC[key]


def pick_cluster(n_tiles: int, options: dict) -> int:
    """The cluster size with the least modelled time: rounds of resident clusters (ceil(n_tiles / resident)) x the per-CTA time of
    that size (relative); ties to fewer rounds, then to the larger cluster. ``options``: cl -> (resident clusters, relative time)."""
    best = None
    for cl, (res, w) in options.items():
        rounds = -(-n_tiles // max(res, 1))
        key = (rounds * w, rounds, -cl)
        if best is None or key < best[0]:
            best = (key, cl)
    return best[1]


def read_trace(kernel) -> torch.Tensor:
    """g_trace [16 CTAs, 2048 events] int64 (%globaltimer ns, 0 = not recorded) of a -DTRACE cubin, after a synchronize."""
    from cuda.bindings import driver as cu
    torch.cuda.synchronize()
    err, dptr, size = cu.cuModuleGetGlobal(kernel.module, b"g_trace")
    assert err == cu.CUresult.CUDA_SUCCESS, err
    out = torch.zeros(16 * 2048, dtype=torch.int64, device="cuda")
    err, = cu.cuMemcpyDtoD(cu.CUdeviceptr(out.data_ptr()), dptr, size)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    torch.cuda.synchronize()
    return out.view(16, 2048)


def clear_trace(kernel) -> None:
    from cuda.bindings import driver as cu
    err, dptr, size = cu.cuModuleGetGlobal(kernel.module, b"g_trace")
    assert err == cu.CUresult.CUDA_SUCCESS, err
    err, = cu.cuMemsetD8(dptr, 0, size)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    torch.cuda.synchronize()


def front_cluster(n_tiles: int, nsm: int) -> int:
    """The front cluster for ``n_tiles`` without the kernels at hand (estimate: 148-SM B200, 15 / 22 / 33 resident clusters of
    8 / 6 / 4); ``FrontTF32.cluster`` asks the driver. MINIWORLD_BIAS_ONLY_DIT_INF3_CL=4 / 6 / 8 forces one."""
    forced = os.environ.get("MINIWORLD_BIAS_ONLY_DIT_INF3_CL")
    if forced:
        assert forced in ("4", "6", "8"), forced
        return int(forced)
    return pick_cluster(n_tiles, {8: (15 * nsm // 148, 1.0), 6: (22 * nsm // 148, 8 / 6), 4: (33 * nsm // 148, 2.0)})


class _LaunchPDL:
    """A launch with its argument block built once, with programmatic dependent launch (and the kernel's cluster, if any):
    TensorMaps by their 64-B aligned copy, tensors by their address at bind time, PTR slots by the address given per call,
    ints as int32, floats as float32. The driver copies the parameters at launch: safe to reuse under CUDA-graph capture."""

    def __init__(self, kernel, grid, block, *args):
        import ctypes

        from cuda.bindings import driver as cu
        from miniworld_engine.kernels.augmented_attention.cuda.sm100.driver import TensorMap
        self._cu, self.k = cu, kernel
        self.hold, self.dyn, ptrs = [], [], []
        for a in args:
            if isinstance(a, TensorMap):
                self.hold.append(a)
                ptrs.append(a.addr)
                continue
            if a is PTR:
                h = ctypes.c_uint64(0)
                self.dyn.append(h)
            elif isinstance(a, torch.Tensor):
                h = ctypes.c_uint64(a.data_ptr())
            elif isinstance(a, float):
                h = ctypes.c_float(a)
            else:
                h = ctypes.c_int32(int(a))
            self.hold.append(h)
            ptrs.append(ctypes.addressof(h))
        self.arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        self.argp = ctypes.addressof(self.arr)
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block
        cfg.sharedMemBytes = kernel.smem
        attrs = []
        if kernel.cluster:
            at = cu.CUlaunchAttribute()
            at.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = kernel.cluster, 1, 1
            attrs.append(at)
        ap = cu.CUlaunchAttribute()
        ap.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        ap.value.programmaticStreamSerializationAllowed = 1
        attrs.append(ap)
        cfg.attrs = attrs
        cfg.numAttrs = len(attrs)
        self.cfg, self.grid = cfg, grid

    def __call__(self, *tensors):
        cu = self._cu
        for h, t in zip(self.dyn, tensors, strict=True):
            h.value = t.data_ptr()
        self.cfg.hStream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        err, = cu.cuLaunchKernelEx(self.cfg, self.k.func, self.argp, 0)
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernelEx: {err}")


def _check_rows(t: torch.Tensor, cols: int, name: str) -> None:
    assert t.dtype is torch.float32 and t.dim() == 2 and t.shape[1] == cols and t.stride(1) == 1, (name, t.dtype, t.shape, t.stride())
    assert t.data_ptr() % 16 == 0 and t.stride(0) % 4 == 0, (name, t.stride())


class FrontTF32:
    """``bo_front_tf32.cu`` (K1): vg [M, 2 DA] = (LN(x) s1 + sh1) [Wv; Wg]^T, v rounded to TF32. x [M, 768] fp32 contiguous (the
    block input), tab [T, 6, 768] fp32 view (row stride any multiple of 4; the hoisted tables, see ``runner._tables3``), w [2 DA, 768]
    fp32 TF32-rounded contiguous, vg [M, 2 DA] fp32 contiguous, xa [M, 768] fp32 scratch (the cluster's operand exchange); M a
    multiple of 128. A cluster of 8, 6 or 4 CTAs per 128-row tile (``cluster``). ``trace``: the -DTRACE build (``read_trace``).
    Bound launches cached per (weights, table, vg, xa, M, T, cluster); x by address per call."""

    def __init__(self, device_index: int, da: int, trace: bool = False):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        assert da in (768, 1024), da
        self._sm100, self._tm, self.device_index, self.da, self.trace = sm100, sm100._tm, device_index, da, trace
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs = _Runs()

    def kernel(self, cl: int):
        return self._sm100._sm100_kernel("bo_front_tf32", "bo_front_tf32_sm100", self.device_index, pdl=True, src_dir=str(_dir),
                                         defs=(f"CL={cl}", f"DATT={self.da}", *(("TRACE",) if self.trace else ())), cluster=cl,
                                         smem=front_smem(cl, self.da))

    def clusters(self) -> tuple[int, ...]:
        return (8, 6, 4) if self.da == 768 else (8, 4)                # CL 6 splits 2 DA = 1536 columns only

    def cluster(self, n_tiles: int) -> int:
        """CTAs per row tile: the least modelled time over the sizes built (rounds of resident clusters x per-CTA time ~ 1 / CL).
        MINIWORLD_BIAS_ONLY_DIT_INF3_CL forces one."""
        forced = os.environ.get("MINIWORLD_BIAS_ONLY_DIT_INF3_CL")
        if forced:
            assert int(forced) in self.clusters(), (forced, self.da)
            return int(forced)
        return pick_cluster(n_tiles, {cl: (max_clusters(self.kernel(cl), cl), 8 / cl) for cl in self.clusters()})

    def _bind(self, tab, w, vg, xa, M, T, cl):
        nv = 2 * self.da // cl
        wrows = nv if nv <= 256 else nv // 2
        maps = [self._tm(w, [768, 2 * self.da], 768 * 4, [32, wrows], dtype="f32"),
                self._tm(xa, [768, M], xa.stride(0) * 4, [32, 128], dtype="f32")]
        return _LaunchPDL(self.kernel(cl), ((M // 128) * cl, 1, 1), (384, 1, 1), *_descriptors(*maps), PTR, tab, vg, xa, int(T),
                          int(tab.stride(0)), int(xa.stride(0)), 1e-5)

    def __call__(self, x, tab, w, vg, T, xa):
        M, da = x.shape[0], self.da
        _check_rows(xa, 768, "xa")
        assert x.is_contiguous() and x.shape == (M, 768) and x.dtype is torch.float32 and M % 128 == 0 and T % 128 == 0
        assert tab.dtype is torch.float32 and tab.shape[0] == T and tab[0].numel() == 6 * 768 and tab.stride(-1) == 1
        assert tab.stride(0) % 4 == 0 and tab.data_ptr() % 16 == 0 and (tab.dim() == 2 or tab.stride(1) == 768)
        assert w.is_contiguous() and w.shape == (2 * da, 768) and vg.is_contiguous() and vg.shape == (M, 2 * da)
        cl = self.cluster(M // 128)
        key = (tab.data_ptr(), tab.stride(0), w.data_ptr(), vg.data_ptr(), xa.data_ptr(), xa.stride(0), M, T, cl)
        self.runs.bind(key, lambda: self._bind(tab, w, vg, xa, M, T, cl))(x)
        return vg


class TailTF32:
    """``bo_tail_tf32.cu`` (K3): out [M, 768] = x1 + gate2 (silu(xt Wa^T) (xt Wb^T)) Wsq^T, x1 = x + gate1 a Wo^T, xt = LN(x1) s2 + sh2;
    a [M, DA] fp32 (the core's output), x [M, 768] fp32 (the block input), tab [T, 6, 768] fp32 view (the hoisted tables), wo / wsq
    = ``pack_pairs(., cl)`` of Wo [768, DA] / Wsq [768, 1536] for the cluster size used, wab [3072, 768] = [Wa; Wb], all fp32
    TF32-rounded contiguous; out [M, 768] fp32 contiguous (not x); the cluster's operand-exchange scratch: xt [M, 768] fp32 rows (row
    stride any multiple of 4) and h [M 48, 32] fp32 contiguous, BLOCKED k-block-major (k-block kb of row tile t at rows (48 t + kb)
    128). A cluster of 8 or 6 CTAs per 128-row tile (``cluster``), or ``TAIL_PAIR``: ``bo_tail2_tf32.cu``, clusters of 8 holding two
    tiles x 4 column groups as tcgen05 cta_group::2 pairs -- then wo / wsq are the CL 8 packs and the scratch has ``tail_rows(M,
    TAIL_PAIR)`` rows (whole tile pairs). ``trace`` as ``FrontTF32``. Bound launches cached per (a, x, table, weights, scratch, M,
    T, cluster); out by address per call."""

    def __init__(self, device_index: int, da: int, trace: bool = False):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        assert da in (768, 1024), da
        self._sm100, self._tm, self.device_index, self.da, self.trace = sm100, sm100._tm, device_index, da, trace
        self.runs = _Runs()

    def kernel(self, cl: int = 8):
        if cl == TAIL_PAIR:
            return self._sm100._sm100_kernel("bo_tail2_tf32", "bo_tail2_tf32_sm100", self.device_index, pdl=True, src_dir=str(_dir),
                                             defs=(f"DATT={self.da}", *(("TRACE",) if self.trace else ())), cluster=8,
                                             smem=tail_smem(TAIL_PAIR))
        return self._sm100._sm100_kernel("bo_tail_tf32", "bo_tail_tf32_sm100", self.device_index, pdl=True, src_dir=str(_dir),
                                         defs=(f"CL={cl}", f"DATT={self.da}", *(("TRACE",) if self.trace else ())),
                                         cluster=cl, smem=tail_smem(cl))

    def cluster(self, n_tiles: int) -> int:
        """8 or 6 CTAs per row tile: the fewest rounds of resident clusters, a CL 6 CTA costing ~1.4 x a CL 8 one (4/3 of the
        products, xt streamed twice). A = 5: CL 6 at L512 only (20 tiles: one round of 22 against two of 15).
        Unless MINIWORLD_BIAS_ONLY_DIT_INF3_2CTA=0, the pair tail (``TAIL_PAIR``) where it takes fewer rounds than both (A = 5: L640 /
        L768, 25 / 30 tiles in one round of clusters of two tiles). MINIWORLD_BIAS_ONLY_DIT_INF3_TAIL_CL=8 / 6 / 4 forces one."""
        forced = os.environ.get("MINIWORLD_BIAS_ONLY_DIT_INF3_TAIL_CL")
        if forced:
            assert int(forced) in (*TAIL_CLUSTERS, TAIL_PAIR), forced
            return int(forced)
        n8, n6 = max_clusters(self.kernel(8), 8), max_clusters(self.kernel(6), 6)
        if tail_pair_on():
            rounds = min(-(-n_tiles // max(n8, 1)), -(-n_tiles // max(n6, 1)))
            if -(-n_tiles // max(2 * max_clusters(self.kernel(TAIL_PAIR), 8), 1)) < rounds:
                return TAIL_PAIR
        return pick_cluster(n_tiles, {8: (n8, 1.0), 6: (n6, 1.4)})

    def _bind(self, a, x, tab, wo, wab, wsq, xt, h, M, T, cl):
        if cl == TAIL_PAIR:
            return self._bind_pair(a, x, tab, wo, wab, wsq, xt, h, M, T)
        tm, da, f, nc = self._tm, self.da, dict(dtype="f32"), 768 // cl
        maps = [tm(a, [da, M], da * 4, [32, 128], **f), tm(x, [768, M], 768 * 4, [32, 128], **f),
                tm(tab, [6 * 768, T], tab.stride(0) * 4, [32, 128], **f), tm(wo, [32, wo.shape[0]], 32 * 4, [32, 2 * nc], **f),
                tm(wab, [768, 3072], 768 * 4, [32, 1536 // cl // (2 if cl == 6 else 1)], **f),
                tm(wsq, [32, wsq.shape[0]], 32 * 4, [32, 2 * nc], **f),
                tm(xt, [768, M], xt.stride(0) * 4, [32, 128], **f),
                tm(h, [32, h.shape[0]], 128, [32, 256], **f)]             # two h k-blocks per box
        return _LaunchPDL(self.kernel(cl), ((M // 128) * cl, 1, 1), (384, 1, 1), *_descriptors(*maps), PTR, xt, h,
                          int(xt.stride(0)), int(T), 1e-5)

    def _bind_pair(self, a, x, tab, wo, wab, wsq, xt, h, M, T):
        """bo_tail2_tf32.cu: clusters of 8 = two row tiles x 4 column groups; each CTA's W boxes are its half (96 rows of the CL 8
        packs, 128 rows of Wa or Wb); the scratch covers whole tile pairs."""
        tm, da, f, tiles = self._tm, self.da, dict(dtype="f32"), M // 128
        Mp = tail_rows(M, TAIL_PAIR)
        maps = [tm(a, [da, M], da * 4, [32, 128], **f), tm(x, [768, M], 768 * 4, [32, 128], **f),
                tm(tab, [6 * 768, T], tab.stride(0) * 4, [32, 128], **f), tm(wo, [32, wo.shape[0]], 32 * 4, [32, 192], **f),
                tm(wab, [768, 3072], 768 * 4, [32, 128], **f), tm(wsq, [32, wsq.shape[0]], 32 * 4, [32, 192], **f),
                tm(xt, [768, Mp], xt.stride(0) * 4, [32, 128], **f),
                tm(h, [32, Mp * 48], 128, [32, 256], **f)]               # two h k-blocks per box
        return _LaunchPDL(self.kernel(TAIL_PAIR), ((Mp // 256) * 8, 1, 1), (384, 1, 1), *_descriptors(*maps), PTR, xt, h,
                          int(xt.stride(0)), int(T), int(tiles), 1e-5)

    def __call__(self, a, x, tab, wo, wab, wsq, out, T, xt, h, cl=None):
        """wo / wsq: the pair-packed forms (``pack_pairs(., cl)``; the pair tail: ``pack_pairs(., 8)``) for ``cl`` (default:
        ``cluster(M / 128)``); xt / h with ``tail_rows(M, cl)`` rows."""
        M, da = x.shape[0], self.da
        cl = cl or self.cluster(M // 128)
        Mp = tail_rows(M, cl)
        _check_rows(xt, 768, "xt")
        assert xt.shape[0] == Mp, (xt.shape, Mp)
        assert h.is_contiguous() and h.shape == (Mp * 48, 32) and h.dtype is torch.float32, h.shape     # blocked k-block-major
        assert a.is_contiguous() and a.shape == (M, da) and a.dtype is torch.float32
        assert x.is_contiguous() and x.shape == (M, 768) and x.dtype is torch.float32 and M % 128 == 0 and T % 128 == 0
        assert out.is_contiguous() and out.shape == (M, 768) and out.dtype is torch.float32 and out.data_ptr() != x.data_ptr()
        assert tab.dtype is torch.float32 and tab.shape[0] == T and tab[0].numel() == 6 * 768 and tab.stride(-1) == 1
        assert tab.stride(0) % 4 == 0 and tab.data_ptr() % 16 == 0 and (tab.dim() == 2 or tab.stride(1) == 768)
        assert wo.shape == (768 * da // 32, 32) and wab.shape == (3072, 768) and wsq.shape == (768 * 48, 32), (wo.shape, wsq.shape)
        assert all(w.is_contiguous() and w.dtype is torch.float32 for w in (wo, wab, wsq))
        key = (a.data_ptr(), x.data_ptr(), tab.data_ptr(), tab.stride(0), wo.data_ptr(), wab.data_ptr(), wsq.data_ptr(),
               xt.data_ptr(), xt.stride(0), h.data_ptr(), M, T, cl)
        self.runs.bind(key, lambda: self._bind(a, x, tab, wo, wab, wsq, xt, h, M, T, cl))(out)
        return out


def round_tf32(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> fp32 holding the nearest TF32 value, ties away from zero (``cvt.rna.tf32.f32``'s rounding): the packed weights of the
    three-kernel step, so its MMAs read round-to-nearest operands instead of truncating the low 13 mantissa bits."""
    i = t.detach().float().contiguous().view(torch.int32)
    return ((i + 0x1000) & -0x2000).view(torch.float32)


_INF3_FAILED = False
_INF3_READY: set = set()


@device_constant
def inf3_ready(index: int, nh: int, dh: int) -> bool:
    """Build and load the three-kernel step's cubins for (nh, dh) on device ``index`` once (both front clusters, the tail,
    a representative PDL core); False (after one warning) when a toolchain or driver problem keeps the cuBLAS + rows step.
    A constant to ``torch.compile``."""
    global _INF3_FAILED
    if _INF3_FAILED:
        return False
    key = (index, nh, dh)
    if key in _INF3_READY:
        return True
    try:
        da = nh * dh
        front = FrontTF32(index, da)
        for cl in front.clusters():
            front.kernel(cl)
        tail = TailTF32(index, da)
        for cl in TAIL_CLUSTERS:
            tail.kernel(cl)
        PvGateCoreTF32(index, nh, dh, pdl=True).kernel(1, True)
    except Exception as exc:  # noqa: BLE001 -- a toolchain or driver problem keeps the default fp32 step
        _INF3_FAILED = True
        warnings.warn(f"bias-only DiT three-kernel fp32 inference step unavailable, keeping the cuBLAS + rows step: {exc!r}",
                      RuntimeWarning, stacklevel=2)
        return False
    _INF3_READY.add(key)
    return True


# --------------------------------------------------------------------------------------------------- readiness
_FAILED = False
_READY: set = set()


@device_constant
def tf32_ready(index: int, nh: int, dh: int, train: bool) -> bool:
    """Build and load the fp32 path's kernels for (nh, dh) on device ``index`` once -- the fp32 row extension, the bf16 row
    extensions whose dtype-generic passes the fp32 path shares, a representative attention-core cubin (and, for training, the
    ungated core and the bias gradient) -- False (after one warning) when a toolchain or driver problem keeps the module path.
    MINIWORLD_BIAS_ONLY_DIT_TF32=0 turns the fp32 path off. A constant to ``torch.compile``."""
    global _FAILED
    if _FAILED or os.environ.get("MINIWORLD_BIAS_ONLY_DIT_TF32", "1") == "0":
        return False
    key = (index, nh, dh, train)
    if key in _READY:
        return True
    try:
        from miniworld_engine.kernels.bias_only_dit import cuda as C
        rows32()
        C._ext()
        if train:
            from miniworld_engine.kernels.bias_only_dit.cuda.train import ext
            ext()
        core = PvGateCoreTF32(index, nh, dh)
        core.kernel(1, True)
        if train:
            core.kernel(1, False)
            Dpb32(index, nh, dh).kernel()
    except Exception as exc:  # noqa: BLE001 -- a toolchain or driver problem keeps the module path
        _FAILED = True
        warnings.warn(f"bias-only DiT fp32 (TF32) kernels unavailable, keeping the module path: {exc!r}", RuntimeWarning,
                      stacklevel=2)
        return False
    _READY.add(key)
    return True


__all__ = ["Dpb32", "FrontTF32", "GemmGluTF32", "PvGateCoreTF32", "TailTF32", "clear_trace", "front_cluster", "glu_op", "inf3_on",
           "inf3_ready", "max_clusters", "pack_pairs", "pick_cluster", "read_trace", "tail_pair_on", "tail_rows", "tail_smem",
           "pick_group_tf32", "pv_batched", "pv_groups", "round_tf32", "rows32", "tf32_gemms", "tf32_ready"]

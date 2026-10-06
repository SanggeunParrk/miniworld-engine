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
    16-byte aligned row stride. Bound launches (their TMA descriptors carry the pointers) are cached per (buffers, strides, S, L)."""

    def __init__(self, device_index: int, nh: int = 16, dh: int | None = None):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        self._sm100, self._tm = sm100, sm100._tm
        self.nh, self.dh = nh, dh or 768 // nh
        assert (self.nh, self.dh) in LAYOUTS, (nh, dh)
        self.device_index = device_index
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs = _Runs()

    def kernel(self, sg: int, gate: bool = True):
        """The cubin for SG samples per item, with or without the gate (built on first use); batched N unless
        MINIWORLD_BIAS_ONLY_DIT_PV_BATCH=0."""
        batch = ("BATCHN",) if pv_batched() else ()
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
        return _Launch(k, grid, (256, 1, 1), *maps, L, S)

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


__all__ = ["Dpb32", "GemmGluTF32", "PvGateCoreTF32", "glu_op", "pick_group_tf32", "pv_batched", "pv_groups",
           "rows32", "tf32_gemms", "tf32_ready"]

"""EXPERIMENTAL (opt-in, off by default): FP8 (e4m3) storage of the TriMul planes between K1 and the contraction, H100 only.

The contraction ``t[c] = sum_k a[c, i, k] b[c, j, k]`` is per channel, so a per-channel scale of each operand factors out exactly:
``t = sa[c] sb[c] sum_k qa qb``. Scales here are powers of two: ``q * 2^e`` is exact in bf16 and the scaling of ``t`` is exact, so
(``mode="store"``) storage-only error can be measured on the unchanged bf16 contraction, and (``mode="native"``) the contraction
runs as a cuBLASLt batched e4m3 GEMM (bf16 output, scale pointers = 1, no FAST_ACCUM: cuBLAS promotes the tensor-core accumulator
to fp32 periodically) with ``t`` rescaled per channel afterwards.

Nothing here runs unless ``CONFIG`` is set; the bf16 route is bit-identical with it unset.
"""

import ctypes
import os
from dataclasses import dataclass, field
from typing import Callable

import torch

E4M3_MAX = 448.0
SCALE_FLOOR = 2.0**-30


@dataclass
class Fp8Config:
    mode: str = "native"  # "store": quantize + dequantize to bf16, bf16 contraction; "native": cuBLASLt e4m3 contraction
    #: (planes [2h, P, n] bf16) -> per-channel power-of-two scales [2h] fp32 (shape of ``amax``); None = dynamic per-channel amax
    scale_fn: Callable | None = None
    stats: list = field(default_factory=list)  # per call: dict(amax=[2h], scale=[2h], clip=int)
    record: bool = True


CONFIG: Fp8Config | None = None


def pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    """Smallest 2^e with amax / 2^e <= 448 (1 where the channel is all zero), never below 2^-30: a channel whose amax is a denormal-sized
    ~0 (or the 1e-30 clamp) must not get a scale of 2^-99..-107, which would saturate the e4m3 cast on any later input (OpenDDE end-level
    runs). Channels with amax > 448 x 2^-30 (4e-7) are unchanged."""
    s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-30) / E4M3_MAX)))
    return torch.where(amax > 0, s.clamp_min(SCALE_FLOOR), torch.ones_like(s))


def channel_amax(ab: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    return torch.cat([ab[i : i + chunk].abs().amax(dim=(1, 2)).float() for i in range(0, ab.shape[0], chunk)])


def quantize(ab: torch.Tensor, scale: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """e4m3 planes of ``ab / scale`` (saturating), same shape."""
    q = torch.empty(ab.shape, dtype=torch.float8_e4m3fn, device=ab.device)
    for i in range(0, ab.shape[0], chunk):
        q[i : i + chunk] = (
            (ab[i : i + chunk].float() / scale[i : i + chunk, None, None]).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        )
    return q


# ---------------------------------------------------------------------------------------------------------------------- cuBLASLt e4m3
_lt = None
REQUESTED = 8  # heuristic candidates kept (algo sweeps ask for more)
ALGO_INDEX: dict = {}  # (n, batch) -> index of the heuristic candidate used (default 0)
FAST_ACCUM = (
    os.environ.get("MINIWORLD_TRIMUL_FP8_FASTACC") == "1"
)  # cuBLASLt fast accumulation (opt-in: contraction 1.07-1.08x at L>=1024, error vs fp32 +0.03% on top of the 0.17% bf16 output rounding; plans are keyed on it)


class _Heur(ctypes.Structure):
    _fields_ = [
        ("algo", ctypes.c_uint64 * 8),
        ("ws", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


def _lib():
    global _lt
    if _lt is None:
        lib = ctypes.CDLL("libcublasLt.so.12", mode=ctypes.RTLD_GLOBAL)
        _lt = (lib, {}, {})
    return _lt


def _set(fn, obj, attr, value, ct):
    v = ct(value)
    if fn(obj, attr, ctypes.byref(v), ctypes.sizeof(v)):
        raise RuntimeError("cuBLASLt attribute %d" % attr)


class _Plan:
    def __init__(self, dev, n, batch):
        lib, handles, _ = _lib()
        if dev not in handles:
            h = ctypes.c_void_p()
            if lib.cublasLtCreate(ctypes.byref(h)):
                raise RuntimeError("cublasLtCreate")
            handles[dev] = h
        self.handle = handles[dev]
        self.one = torch.ones(1, device=f"cuda:{dev}", dtype=torch.float32)  # keeps the scale pointers alive
        desc = ctypes.c_void_p()
        lib.cublasLtMatmulDescCreate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.cublasLtMatmulDescCreate(ctypes.byref(desc), 68, 0)
        for attr in (3, 4):
            pass
        _set(lib.cublasLtMatmulDescSetAttribute, desc, 3, 1, ctypes.c_int32)  # A (= our b) transposed: K-major
        _set(lib.cublasLtMatmulDescSetAttribute, desc, 4, 0, ctypes.c_int32)
        if FAST_ACCUM:
            _set(lib.cublasLtMatmulDescSetAttribute, desc, 25, 1, ctypes.c_int8)  # CUBLASLT_MATMUL_DESC_FAST_ACCUM
        for attr in (17, 18):  # A_SCALE_POINTER, B_SCALE_POINTER
            _set(lib.cublasLtMatmulDescSetAttribute, desc, attr, self.one.data_ptr(), ctypes.c_void_p)
        self.desc = desc

        def layout(dt, rows, cols, ld, stride):
            lay = ctypes.c_void_p()
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(lay), dt, ctypes.c_uint64(rows), ctypes.c_uint64(cols), ctypes.c_int64(ld))
            _set(lib.cublasLtMatrixLayoutSetAttribute, lay, 5, batch, ctypes.c_int32)
            _set(lib.cublasLtMatrixLayoutSetAttribute, lay, 6, stride, ctypes.c_int64)
            return lay

        self.lq = layout(28, n, n, n, n * n)  # CUDA_R_8F_E4M3, column-major (k x n) == row-major [n, k]
        self.lc = layout(14, n, n, n, n * n)  # CUDA_R_16BF
        pref = ctypes.c_void_p()
        lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref))
        _set(lib.cublasLtMatmulPreferenceSetAttribute, pref, 1, 32 << 20, ctypes.c_uint64)
        self.ws = torch.empty(32 << 20, dtype=torch.uint8, device=f"cuda:{dev}")
        res = (_Heur * REQUESTED)()
        cnt = ctypes.c_int(0)
        r = lib.cublasLtMatmulAlgoGetHeuristic(
            self.handle, desc, self.lq, self.lq, self.lc, self.lc, pref, REQUESTED, res, ctypes.byref(cnt)
        )
        if r or cnt.value == 0:
            raise RuntimeError("no cuBLASLt e4m3 algorithm for n=%d batch=%d" % (n, batch))
        self.algos = [(ctypes.c_uint64 * 8)(*res[i].algo) for i in range(cnt.value)]
        self.algo = self.algos[min(ALGO_INDEX.get((n, batch), 0), cnt.value - 1)]


def contract_e4m3(left: torch.Tensor, right: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[c, i, j] = sum_k left[c, i, k] right[c, j, k] (e4m3 operands, K contiguous; bf16 output, fp32 promotion)."""
    lib, _, plans = _lib()
    dev = left.device.index
    batch, n = left.shape[0], left.shape[-1]
    key = (dev, n, batch, FAST_ACCUM)
    if key not in plans:
        plans[key] = _Plan(dev, n, batch)
    p = plans[key]
    one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
    r = lib.cublasLtMatmul(
        p.handle,
        p.desc,
        ctypes.byref(one),
        ctypes.c_void_p(right.data_ptr()),
        p.lq,
        ctypes.c_void_p(left.data_ptr()),
        p.lq,
        ctypes.byref(zero),
        ctypes.c_void_p(out.data_ptr()),
        p.lc,
        ctypes.c_void_p(out.data_ptr()),
        p.lc,
        p.algo,
        ctypes.c_void_p(p.ws.data_ptr()),
        ctypes.c_size_t(p.ws.numel()),
        ctypes.c_void_p(torch.cuda.current_stream(left.device).cuda_stream),
    )
    if r:
        raise RuntimeError("cublasLtMatmul e4m3 failed: %d" % r)
    return out


def contract(ab: torch.Tensor, d: int, direction: int, bf16_contract) -> torch.Tensor:
    """Planes [2d, P, n] (single direction) -> t [d, P, n] with the FP8 route of ``CONFIG``; ``bf16_contract(planes, d, direction)`` is the
    engine's contraction (used by mode "store")."""
    cfg = CONFIG
    n = ab.shape[-1]
    planes = ab.view(2 * d, -1, n)
    amax = channel_amax(planes)
    scale = pow2_scale(amax) if cfg.scale_fn is None else cfg.scale_fn(planes, amax)
    if cfg.record:
        cfg.stats.append({"amax": amax.cpu(), "scale": scale.cpu(), "n": n, "direction": direction})
    q = quantize(planes, scale)
    if cfg.mode == "store":
        for i in range(0, 2 * d, 8):
            planes[i : i + 8] = (q[i : i + 8].float() * scale[i : i + 8, None, None]).to(planes.dtype)
        return bf16_contract(ab, d, direction)
    samples = planes.shape[1] // n
    a, b = q[:d].view(d * samples, n, n), q[d:].view(d * samples, n, n)
    if direction == 2:  # incoming: contraction over the row index: K-major operands need the transposed planes (K1 would write them so)
        a, b = a.transpose(1, 2).contiguous(), b.transpose(1, 2).contiguous()
    t = torch.empty((d * samples, n, n), dtype=torch.bfloat16, device=ab.device)
    contract_e4m3(a, b, t)
    rescale = (scale[:d] * scale[d:]).repeat_interleave(samples)
    for i in range(0, t.shape[0], 16):
        t[i : i + 16] = (t[i : i + 16].float() * rescale[i : i + 16, None, None]).to(t.dtype)
    return t.view(d, *ab.shape[1:])

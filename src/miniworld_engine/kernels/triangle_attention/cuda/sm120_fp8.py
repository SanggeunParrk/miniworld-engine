"""EXPERIMENTAL, opt-in (never reached by default): the RTX 5090 (sm_120a) TriangleAttention inference route with e4m3 P and V (``sm80/attn_rf8_sm120.cuh``).

The module of the ``triattn`` slot of ``fp8.SM120_ROUTES``: the switches are the H100 module's (``h100_sm90.FP8_PV`` / ``FP8_PV_FUSED`` / ``FP8_PV_UNFUSED``, which
``fp8.enable_triattn`` sets from the model whitelist ``fp8.TRIATTN_MODELS``), read through this module's ``__getattr__`` (as ``b200_fp8`` does); the guard
(``take_overflow`` / ``rescale_all``) and the counters (``STATS`` is the H100 dict) are registered in the H100 module's own registries so that ``fp8.check()`` sees them.

Chain (the rows engine of ``sm80_wide._original_forward_ops``, inference, gated core, head dim 32, L a multiple of 128 from ``MIN_L``)::

    ln_rows, front GEMM, bias_planes        unchanged (q | k | v | g | bias heads)
    vt_quant                                V x vinv[h] -> e4m3 tiles [pair row][head][32-key block][32 channels][32 permuted keys] (one standalone pass; static
                                            per-head power-of-two scale from the first eager call: 4 x the V absmax, saturation flag in ``peak``)
    attn_f8_gate                            the register-resident core with P = 2^(S - m + 8) as e4m3 bytes and V as e4m3: both products of P V on
                                            ``mma.m16n8k32`` (e4m3), row sums from the same bytes, running max per 32-key slot, gate in the epilogue
    out projection                          unchanged

Guard: ``sync`` (eager calls: a saturated V redoes the call after widening the scale) or ``lazy`` (inside a capture / ``MINIWORLD_TRIATTN_SM120_FP8_GUARD=lazy``: the flag is read
by ``fp8.check()`` outside the graph, the scale is widened in place so captured graphs follow); a capture of an uncalibrated site runs the default route.
"""

from __future__ import annotations

import functools
import os
import weakref
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension

from . import h100_sm90 as _hs
from . import sm80_arch as _arch

_SRC = Path(__file__).parent / "sm80"
L2E = 1.4426950408889634
STATS = _hs.STATS  # one counter dict for every arch: fp8.route_stats()["triattn"]
STATS.setdefault("capture_fallbacks", 0)
STATS.setdefault("fallbacks", 0)
STATS.setdefault("retries", 0)
GUARD = os.environ.get("MINIWORLD_TRIATTN_SM120_FP8_GUARD", "sync")
HEADROOM = 4.0  # V scale: |v| * vinv <= 448 / HEADROOM at the calibration absmax
#: smallest L the e4m3 core takes (below it the default core runs): measured same-process A-B, docs/fp8.md "RTX 5090"
MIN_L = int(os.environ.get("MINIWORLD_TRIATTN_SM120_FP8_MINL", "512"))
_DELEGATED = frozenset({"FP8_PV", "FP8_PV_FUSED", "FP8_PV_UNFUSED", "FP8_PV_D64", "FP8_PV_HL", "FP8_VSCALE_AUTO", "FP8_PV_MIN_S", "FP8_PV_FRONT_MIN_S", "FP8_QK"})


def __getattr__(name):
    if name in _DELEGATED:
        return getattr(_hs, name)
    raise AttributeError(name)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="triattn_sm120_f8",
        sources=[str(_SRC / "triattn_f8_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *_arch.gencode(), f"-I{_SRC}"]
        + (["-Xptxas", "-v"] if os.environ.get("MW_F8_PTXAS") else []),
        extra_cflags=["-std=c++17", "-O3"],
        verbose=bool(os.environ.get("MW_F8_PTXAS")),
    )


class _Site:
    __slots__ = ("vinv", "vsc", "ref", "version", "__weakref__")


_SITES: dict = {}  # (device, V weight address, heads) -> _Site (per-head static V scale, calibrated on the first eager call)
_GUARDS: dict = {}  # device -> int32 [2]: [0] running max of |v vinv| (float bits), [1] saturation flag (> 448), written by vt_quant


def _guard(dev: int) -> torch.Tensor:
    g = _GUARDS.get(dev)
    if g is None:
        g = _GUARDS[dev] = torch.zeros(2, dtype=torch.int32, device=f"cuda:{dev}")
        _hs._OVF[("sm120_fp8", dev)] = g[1:]  # fp8.check() -> take_overflow reads and clears the flag
    return g


def overflow_peak() -> float:
    return max((float(g[:1].view(torch.float32).item()) for g in _GUARDS.values()), default=0.0)


def take_overflow() -> bool:
    return _hs.take_overflow()


def rescale_all(factor: float = 4.0) -> None:
    _hs.rescale_all(factor)  # widens the registered V scales in place


def _drop(key, s):
    if _SITES.get(key) is s:
        del _SITES[key]
        _hs._VS_AUTO.pop(("sm120_fp8", *key), None)
        _hs._VINV.pop(("sm120_fp8", *key), None)


def _calibrate(dev: int, site: torch.Tensor, v: torch.Tensor, heads: int) -> _Site:
    s = _Site()
    s.ref, s.version = weakref.ref(site), site._version
    key = (dev, site.data_ptr(), heads)
    weakref.finalize(site, _drop, key, s)
    vv = v.view(-1, heads, 32)
    amax = torch.maximum(vv.amax(dim=(0, 2)), -vv.amin(dim=(0, 2))).float().clamp_min(1e-12)  # eager only: one sync
    s.vinv = torch.exp2(torch.floor(torch.log2(448.0 / (HEADROOM * amax)))).clamp_max(2.0**60).contiguous()
    s.vsc = (1.0 / s.vinv).contiguous()
    _SITES[key] = s
    _hs._VS_AUTO[("sm120_fp8", *key)] = s.vsc  # fp8.check() -> h100_sm90.rescale_all multiplies it and recomputes vinv in place
    _hs._VINV[("sm120_fp8", *key)] = s.vinv
    STATS["calibrations"] += 1
    return s


def kind_of(d_pair: int, heads: int) -> str:
    """The call-site kind of the model whitelist (``fp8.TRIATTN_MODELS``): the d_pair-128 / 4 x 32 trunk site is ``fused``, every other width ``unfused``."""
    return "fused" if (d_pair, heads) == (128, 4) else "unfused"


def serves(x: torch.Tensor, d_pair: int, heads: int, hd: int, length: int) -> bool:
    """Python-level test: does the e4m3 core take this inference call (switch on for the call-site kind, bf16, 32-channel heads, L % 128 == 0, L >= MIN_L)?"""
    if not _hs.FP8_PV or x.dtype != torch.bfloat16 or torch.is_grad_enabled():
        return False
    if not (_hs.FP8_PV_FUSED if kind_of(d_pair, heads) == "fused" else _hs.FP8_PV_UNFUSED):
        return False
    return hd == 32 and heads in (1, 2, 3, 4, 6, 8, 12, 16) and length % 128 == 0 and length >= max(MIN_L, 256)


def route(q, k, v, bias, g, a, mask, site, b: int, length: int, heads: int) -> bool:
    """e4m3 core of one inference call: q / k / v / g token-major views [T, ld] of the front's output (head h = columns [32 h, 32 h + 32)), bias [b, H, L, L], ``a`` [T, dhp]
    the output (bf16(sigmoid(g) o)); ``site`` = the V weight (the scale's key). False = the caller runs the default core (``a`` is overwritten by it)."""
    dev = q.device.index
    capturing = torch.cuda.is_current_stream_capturing()
    key = (dev, site.data_ptr(), heads)
    s = _SITES.get(key)
    if s is not None and (s.ref() is not site or s.version != site._version):
        s = None  # the address belongs to another tensor, or the weights changed in place: stale
    if s is None:
        if capturing:
            STATS["capture_fallbacks"] += 1
            return False
        s = _calibrate(dev, site, v, heads)
    ext = _ext()
    peak = _guard(dev)
    rows = b * length
    v8 = torch.empty((rows * heads * length * 32,), dtype=torch.uint8, device=q.device)
    scl = L2E / 32**0.5
    km = mask if mask.numel() else a.new_empty((0,), dtype=torch.uint8)
    lazy = GUARD in ("lazy", "off") or capturing
    for _ in range(4):
        if not lazy:
            peak.zero_()
        ext.vt_quant(v, s.vinv, v8, peak, length, heads, rows, v.stride(0))
        ext.attn_f8_gate(q, k, v8, s.vsc, bias, g, a, length, heads, b, q.stride(0), k.stride(0), g.stride(0), a.stride(0), scl, km)
        STATS["pv_front"] += 1
        if lazy:
            return True
        if not bool(peak[1].item()):
            return True
        STATS["retries"] += 1
        rescale_all(4.0)
    STATS["fallbacks"] += 1
    return False

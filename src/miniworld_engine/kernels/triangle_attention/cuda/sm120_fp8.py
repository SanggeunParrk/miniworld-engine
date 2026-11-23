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
import hashlib
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
QK_HEADROOM = 2.0  # q / k scales: |q| / sq, |k| sq <= 448 / QK_HEADROOM at the calibration absmax (the H100 FP8_QK_HEADROOM)
#: smallest L the e4m3 core takes (below it the default core runs): measured same-process A-B, docs/fp8.md "RTX 5090"
QK = os.environ.get("MINIWORLD_TRIATTN_SM120_FP8_QK", "1") != "0"  # e4m3 q / k where the model whitelist has them (the fused d_pair-128 kind)
MIN_L = int(os.environ.get("MINIWORLD_TRIATTN_SM120_FP8_MINL", "640"))
#: the PV-only core (no e4m3 q / k: the unfused kinds, QK off, or a site whose q / k saturated) is 0.97-1.05x of the default on the 5090 (same-process A-B, docs/fp8.md
#: "RTX 5090"): off unless MINIWORLD_TRIATTN_SM120_FP8_PVONLY=1; without it such calls run the default core
PV_ONLY = os.environ.get("MINIWORLD_TRIATTN_SM120_FP8_PVONLY", "0") == "1"
_DELEGATED = frozenset({"FP8_PV", "FP8_PV_FUSED", "FP8_PV_UNFUSED", "FP8_PV_D64", "FP8_PV_HL", "FP8_VSCALE_AUTO", "FP8_PV_MIN_S", "FP8_PV_FRONT_MIN_S", "FP8_QK"})


def __getattr__(name):
    if name in _DELEGATED:
        return getattr(_hs, name)
    raise AttributeError(name)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    defs = os.environ.get("MW_F8_DEFS", "").split()  # experiment macros (-DRF8_ST=3, ...): never set in production
    tag = "" if not defs else "_" + hashlib.md5(" ".join(defs).encode()).hexdigest()[:6]
    return load_extension(
        name="triattn_sm120_f8" + tag,
        sources=[str(_SRC / "triattn_f8_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *_arch.gencode(), f"-I{_SRC}", *defs]
        + (["-Xptxas", "-v"] if os.environ.get("MW_F8_PTXAS") else []),
        extra_cflags=["-std=c++17", "-O3"],
        verbose=bool(os.environ.get("MW_F8_PTXAS")),
    )


class _Site:
    __slots__ = ("vinv", "vsc", "qs", "ks", "qk", "ref", "version", "__weakref__")


_SITES: dict = {}  # (device, V weight address, heads) -> _Site (per-head static scales, calibrated on the first eager call)
_GUARDS: dict = {}  # device -> int32 [4]: [0] running max of |v vinv| (float bits), [1] V saturated, [2] q saturated, [3] k saturated (> 448), written by the kernels


def _guard(dev: int) -> torch.Tensor:
    g = _GUARDS.get(dev)
    if g is None:
        g = _GUARDS[dev] = torch.zeros(4, dtype=torch.int32, device=f"cuda:{dev}")
        _hs._OVF[("sm120_fp8", dev)] = g[1:2]  # fp8.check() -> h100_sm90.take_overflow reads and clears the V flag
    return g


def overflow_peak() -> float:
    return max((float(g[:1].view(torch.float32).item()) for g in _GUARDS.values()), default=0.0)


def _drop_qk() -> None:
    """Some q / k value saturated: every site goes back to the bf16 score product (the product-1 q / k scales cannot widen; the H100 FP8_QK does the same). New calls and
    captures run the PV-only core; a graph captured before keeps its e4m3 scores (it is flagged again, ``fp8.check()`` re-runs the forward)."""
    for s in _SITES.values():
        s.qk = False


def take_overflow() -> bool:
    """True when some call saturated a V / q / k value since the last call (clears the flags; synchronizes); the caller redoes the forward (``fp8.check()``)."""
    hit = _hs.take_overflow()
    for g in _GUARDS.values():
        if bool(g[2].item()) or bool(g[3].item()):
            _drop_qk()
            hit = True
        g[2:4].zero_()
    return hit


def rescale_all(factor: float = 4.0) -> None:
    _hs.rescale_all(factor)  # widens the registered V scales in place


def _drop(key, s):
    if _SITES.get(key) is s:
        del _SITES[key]
        _hs._VS_AUTO.pop(("sm120_fp8", *key), None)
        _hs._VINV.pop(("sm120_fp8", *key), None)


def _calibrate(dev: int, site: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, want_qk: bool) -> _Site:
    s = _Site()
    s.ref, s.version = weakref.ref(site), site._version
    key = (dev, site.data_ptr(), heads)
    weakref.finalize(site, _drop, key, s)

    def absmax(t):
        tt = t.view(-1, heads, 32)
        return torch.maximum(tt.amax(dim=(0, 2)), -tt.amin(dim=(0, 2))).float()

    amax = absmax(v).clamp_min(1e-12)  # eager only: one sync per tensor
    s.vinv = torch.exp2(torch.floor(torch.log2(448.0 / (HEADROOM * amax)))).clamp_max(2.0**60).contiguous()
    s.vsc = (1.0 / s.vinv).contiguous()
    s.qk, s.qs, s.ks = False, None, None
    if want_qk:
        sq = _hs.qk_scale_from_amax(absmax(q), absmax(k), QK_HEADROOM)  # per-head power of two, q8 = q / sq, k8 = k sq (None: no room)
        if sq is not None:
            s.qs, s.ks, s.qk = (1.0 / sq).contiguous(), sq.contiguous(), True
            STATS["qk_calibrations"] = STATS.get("qk_calibrations", 0) + 1
    _SITES[key] = s
    _hs._VS_AUTO[("sm120_fp8", *key)] = s.vsc  # fp8.check() -> h100_sm90.rescale_all multiplies it and recomputes vinv in place
    _hs._VINV[("sm120_fp8", *key)] = s.vinv
    STATS["calibrations"] += 1
    return s


def kind_of(d_pair: int, heads: int) -> str:
    """The call-site kind of the model whitelist (``fp8.TRIATTN_MODELS``): the d_pair-128 / 4 x 32 trunk site is ``fused``, every other width ``unfused``."""
    return "fused" if (d_pair, heads) == (128, 4) else "unfused"


def serves(x: torch.Tensor, d_pair: int, heads: int, hd: int, length: int) -> bool:
    """Python-level test: does the e4m3 core take this inference call (switch on for the call-site kind, e4m3 q / k on it unless PV_ONLY, bf16, 32-channel heads, L % 128 == 0, L >= MIN_L)?"""
    if not _hs.FP8_PV or x.dtype != torch.bfloat16 or torch.is_grad_enabled():
        return False
    if not (_hs.FP8_PV_FUSED if kind_of(d_pair, heads) == "fused" else _hs.FP8_PV_UNFUSED):
        return False
    if not PV_ONLY and not (QK and _hs.FP8_QK and kind_of(d_pair, heads) == "fused"):
        return False  # PV-only: no measured gain on the 5090
    return hd == 32 and heads in (1, 2, 3, 4, 6, 8, 12, 16) and length % 128 == 0 and length >= max(MIN_L, 256)


def route(q, k, v, bias, g, a, mask, site, b: int, length: int, heads: int, d_pair: int) -> bool:
    """e4m3 core of one inference call: q / k / v / g token-major views [T, ld] of the front's output (head h = columns [32 h, 32 h + 32)), bias [b, H, L, L], ``a`` [T, dhp]
    the output (bf16(sigmoid(g) o)); ``site`` = the V weight (the scales' key). e4m3 q / k where the model whitelist has them (``h100_sm90.FP8_QK``, the fused kind) and the site is
    calibrated for them. False = the caller runs the default core (``a`` is overwritten by it)."""
    dev = q.device.index
    capturing = torch.cuda.is_current_stream_capturing()
    want_qk = bool(_hs.FP8_QK) and kind_of(d_pair, heads) == "fused" and QK
    key = (dev, site.data_ptr(), heads)
    s = _SITES.get(key)
    if s is not None and (s.ref() is not site or s.version != site._version):
        s = None  # the address belongs to another tensor, or the weights changed in place: stale
    if s is None:
        if capturing:
            STATS["capture_fallbacks"] += 1
            return False
        s = _calibrate(dev, site, q, k, v, heads, want_qk)
    qk = want_qk and s.qk
    if not qk and not PV_ONLY:
        return False  # q / k dropped after a saturation (or no room at calibration): the PV-only core is no gain, the default core runs
    ext = _ext()
    peak = _guard(dev)
    rows = b * length
    v8 = torch.empty((rows * heads * length * 32,), dtype=torch.uint8, device=q.device)
    k8 = torch.empty_like(v8) if qk else v8.new_empty((0,))
    scl = L2E / 32**0.5
    km = mask if mask.numel() else a.new_empty((0,), dtype=torch.uint8)
    lazy = GUARD in ("lazy", "off") or capturing
    nul = v8.new_empty((0,), dtype=torch.float32)
    for _ in range(4):
        if not lazy:
            peak.zero_()
        ext.quant(v, s.vinv, v8, k if qk else v, s.ks if qk else nul, k8, peak, length, heads, rows, v.stride(0), k.stride(0))
        ext.attn_f8_gate(q, k, v8, s.vsc, k8, s.qs if qk else nul, peak, bias, g, a, length, heads, b, q.stride(0), k.stride(0), g.stride(0), a.stride(0), scl, km)
        STATS["pv_front"] += 1
        if qk:
            STATS["qk"] = STATS.get("qk", 0) + 1
        if lazy:
            return True
        flags = peak.tolist()
        if not (flags[1] or flags[2] or flags[3]):
            return True
        STATS["retries"] += 1
        if flags[2] or flags[3]:
            _drop_qk()
            qk = False
            if not PV_ONLY:
                return False  # the default core redoes the call
            k8 = v8.new_empty((0,))
            peak[2:4].zero_()
        if flags[1]:
            rescale_all(4.0)
    STATS["fallbacks"] += 1
    return False

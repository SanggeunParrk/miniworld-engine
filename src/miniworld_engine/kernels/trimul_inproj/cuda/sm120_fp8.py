"""EXPERIMENTAL, opt-in (never reached by default): the RTX 5090 (sm_120a) TriMul inference route with e4m3 planes between the front and the contraction
(bf16 modules, one direction or bidirectional, widths whose front is the in-kernel-statistics ``k1w`` kernel: D128 / D256 / D384; the default route is untouched
and bit-identical with ``POLICY`` None).

Chain (the 5090 wide chain of ``sm80_wide`` with two stages swapped)::

    k1w8 (wide_k1.cuh, k1w_kernel<..., E8>)   front: LN_in folded gated projections, plane value * qinv -> e4m3 bytes (cvt.rn.satfinite), max |scaled value| -> guard
    contraction                               cuBLASLt batched e4m3 GEMM (TN, per-tensor scale pointers sa / sb, fp32 accumulation, bf16 output); the
                                              incoming channels (K-major needs transposed planes) go through one byte transpose first
    k3w (unchanged)                           reads the contraction output at its true scale

Quantisation recipe = the H100 / B200 routes': ONLY the planes a / b are e4m3, one power-of-two scale per tensor (a, b), calibrated on the first eager call of a
module from the exact amax of the plane values (a probe launch of the front with qinv = 1), then HEADROOM x and static (cached per module, dropped with its weight);
guard: ``sync`` (eager: redo an overflowing call after raising the scales) or ``lazy`` (capture / ``MINIWORLD_TRIMUL_FP8_GUARD=lazy``: the peak is read by
:func:`overflow_peak` outside the graph, scales updated in place so captured graphs follow); a capture before any calibration runs the default route.
Same interface as the H100 module of the slot (``POLICY`` / ``GUARD`` / ``GUARD_STATS`` / ``overflow_peak`` / ``rescale_all``).
"""

from __future__ import annotations

import ctypes
import math
import os
import weakref

import torch

from miniworld_engine.fp8 import env as _fp8_env

E4M3_MAX = 448.0
SCALE_FLOOR = 2.0**-30
POLICY = _fp8_env("MINIWORLD_TRIMUL_FP8") or None  # None = off; "calib" = static scales; "dynamic" = exact amax every call
GUARD = os.environ.get("MINIWORLD_TRIMUL_FP8_GUARD", "sync")  # "sync" | "lazy" | "off"
HEADROOM = float(os.environ.get("MINIWORLD_TRIMUL_FP8_HEADROOM", "16"))
_MINL_ENV = os.environ.get("MINIWORLD_TRIMUL_FP8_MINL")
#: smallest L the e4m3 chain takes (below it the default route runs): measured same-process A-B, docs/fp8.md "RTX 5090"
MIN_L = 256
#: d_pair widths of the route (D64: the template stacks, D512: IntelliFold2's trunk, both added on 5090-int 2026-11-23 as on the H100: k1w8 with the
#: statistics pass first); FUSED_WIDTHS: the widths whose k1w8 computes the input statistics in-kernel
WIDTHS = (64, 128, 256, 384, 512)
FUSED_WIDTHS = (128, 256, 384)
#: per-width smallest L (overrides MIN_L; measured same-process A-B, docs/fp8.md "RTX 5090")
#: D64: L256 / 384 one direction 0.88-1.00, L512 1.07, L768 1.08-1.11, L1024 1.18 (bidirectional 1.09-1.16 from L256); D512 1.03-1.04 at L256 / 384, 1.05-1.15 at L512 .. 1024
MIN_L_D: dict = {64: 512}
GUARD_STATS = {"calls": 0, "retries": 0, "fallbacks": 0, "capture_fallbacks": 0, "calibrations": 0}
_CACHES: dict = {}  # per module (key: data_ptr of its to_left weight)
_PEAK: dict = {}  # per device: int32 [2] -- max scaled plane value (fp32 bits) of a / b since the last read
_SC: dict = {}  # per device: float32 [2] -- the scales the contraction reads (copied from the module's state before each launch)
_PLANS: dict = {}
_WS: dict = {}


def pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-30) / E4M3_MAX)))
    return torch.where(amax > 0, s.clamp_min(SCALE_FLOOR), torch.ones_like(s))


def _peak_buffer(device) -> torch.Tensor:
    if device not in _PEAK:
        _PEAK[device] = torch.zeros(2, dtype=torch.int32, device=device)
    return _PEAK[device]


def overflow_peak(reset: bool = True) -> float:
    """Largest scaled plane value since the last read (> 448: some call saturated). One host sync: read once per forward, outside any graph."""
    out = 0.0
    for buf in _PEAK.values():
        out = max(out, float(buf.view(torch.float32).max().item()))
        if reset:
            buf.zero_()
    return out


def _derive(st: dict) -> None:
    st["qinv"].copy_(1.0 / st["s"])


def rescale_all(peak: float) -> float:
    """Raise every calibrated module's scales so a forward whose guard read ``peak`` (> 448) fits with room x2; returns the factor."""
    f = 2.0 ** math.ceil(math.log2(peak / E4M3_MAX * 2.0))
    for cache in _CACHES.values():
        cache["s"].mul_(f)
        _derive(cache)
    return f


# ------------------------------------------------------------------------------------------------------------------------ contraction
def _plan(dev: int, n: int, batch: int):
    """cuBLASLt TN e4m3 -> bf16 plan whose A / B scale pointers read this device's ``_SC`` (one [2] tensor: the two scales of the call being issued)."""
    key = (dev, n, batch)
    if key not in _PLANS:
        from miniworld_engine.kernels.trimul_inproj.cuda import _h100_fp8 as F

        p = F._Plan(dev, n, batch)
        if dev not in _SC:
            _SC[dev] = torch.ones(2, dtype=torch.float32, device=f"cuda:{dev}")
        lib = F._lib()[0]
        base = _SC[dev].data_ptr()
        F._set(lib.cublasLtMatmulDescSetAttribute, p.desc, 17, base, ctypes.c_void_p)  # A_SCALE_POINTER
        F._set(lib.cublasLtMatmulDescSetAttribute, p.desc, 18, base + 4, ctypes.c_void_p)  # B_SCALE_POINTER
        if dev not in _WS:
            _WS[dev] = p.ws
        p.ws = _WS[dev]  # one workspace for every plan (the per-plan 32 MiB is dropped)
        _PLANS[key] = p
    return _PLANS[key]


def _gemm(plan, left: torch.Tensor, right: torch.Tensor, out: torch.Tensor) -> None:
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_fp8 as F

    lib = F._lib()[0]
    one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
    r = lib.cublasLtMatmul(
        plan.handle, plan.desc, ctypes.byref(one),
        ctypes.c_void_p(right.data_ptr()), plan.lq, ctypes.c_void_p(left.data_ptr()), plan.lq, ctypes.byref(zero),
        ctypes.c_void_p(out.data_ptr()), plan.lc, ctypes.c_void_p(out.data_ptr()), plan.lc,
        plan.algo, ctypes.c_void_p(plan.ws.data_ptr()), ctypes.c_size_t(plan.ws.numel()),
        ctypes.c_void_p(torch.cuda.current_stream(left.device).cuda_stream),
    )  # fmt: skip
    if r:
        raise RuntimeError(f"cublasLtMatmul e4m3 failed: {r}")


def prepare(dev: int, n: int, hs: int, direction: int) -> None:
    """Build the cuBLASLt plans of a call (outside any capture)."""
    _plan(dev, n, hs)


def contract(planes8: torch.Tensor, x: torch.Tensor, scales: torch.Tensor, hs: int, n: int, direction: int) -> None:
    """x[c] = sa sb sum_k a[c, i, k] b[c, j, k] from the e4m3 planes [2 hs, n * n] (incoming channels: the front stored them transposed, so this is their A^T B)."""
    dev = planes8.device.index
    if dev not in _SC:
        _SC[dev] = torch.ones(2, dtype=torch.float32, device=planes8.device)
    _SC[dev].copy_(scales)
    a, b = planes8[:hs].view(hs, n, n), planes8[hs:].view(hs, n, n)
    _gemm(_plan(dev, n, hs), a, b, x)  # every channel K-major: the incoming ones were written transposed by the front (k1w8, direction)


# ------------------------------------------------------------------------------------------------------------------------ route
def serves(z: torch.Tensor, hs: int, n: int, grouped: bool) -> bool:
    """Python-level test: does the e4m3 chain take this inference call (bf16, no grouped plane formation, L >= MIN_L)?"""
    if POLICY is None or z.dtype != torch.bfloat16 or grouped or torch.is_grad_enabled():
        return False
    d = z.shape[-1]
    if d not in WIDTHS or hs % 64 or n % 32:
        return False
    return n >= (int(_MINL_ENV) if _MINL_ENV is not None else MIN_L_D.get(d, MIN_L))


def run(E, zf, pk, st_in, mask, x, hs, n, direction, eps_in, key, fuse_z, lfac=None) -> bool:
    """Front + contraction on e4m3 planes into ``x`` ([hs, n, n] bf16), ``st_in`` filled; False = the caller runs the default route."""
    dev = zf.device
    capturing = torch.cuda.is_current_stream_capturing()
    T = n * n
    cache = _CACHES.get(key) if key is not None else None
    peak = _peak_buffer(dev)
    planes8 = zf.new_empty((2 * hs, T), dtype=torch.uint8)

    # the in-kernel input statistics of k1w8 need the ring not to hold the whole K (D128 / 256 / 384); D64 and D512 take the statistics pass first
    fuse_z = fuse_z and zf.shape[-1] in FUSED_WIDTHS
    if not fuse_z:
        E.stats_rows(zf, st_in, eps_in)  # (hs 256 at D128: the statistics kernel, then the plain front)

    def front(qinv):
        E.k1w8(zf, pk["w1"], pk["vs"], pk["vb"], st_in, mask, planes8, n, eps_in, qinv, peak, direction, fuse_z)

    if cache is None or POLICY == "dynamic":
        if capturing:
            GUARD_STATS["capture_fallbacks"] += 1
            return False
        prepare(dev.index, n, hs, direction)
        peak.zero_()
        front(torch.ones(2, dtype=torch.float32, device=dev))
        am = peak.view(torch.float32).clone()
        peak.zero_()
        s = pow2_scale(am) * (HEADROOM if POLICY != "dynamic" else 1.0)
        cache = {"s": s, "qinv": torch.empty(2, dtype=torch.float32, device=dev)}
        _derive(cache)
        GUARD_STATS["calibrations"] += 1
        if key is not None and POLICY != "dynamic":
            _CACHES[key] = cache
    elif not capturing:
        prepare(dev.index, n, hs, direction)
    lazy = GUARD in ("lazy", "off") or capturing
    for _ in range(4):
        if not lazy:
            peak.zero_()
        front(cache["qinv"])
        GUARD_STATS["calls"] += 1
        # RoseTTAFold3 (``lfac``: its 1 / n length factor on W_l, a device scalar): the planes are those of the parameter (one calibration, a stable key),
        # the factor rides on the a-plane scale the contraction reads (x = sa lfac sb sum a b), so a captured graph follows the mask at replay
        contract(planes8, x, cache["s"] if lfac is None else cache["s"] * torch.cat([lfac.reshape(1).float(), cache["s"].new_ones(1)]), hs, n, direction)
        if lazy or POLICY == "dynamic":
            return True
        pv = float(peak.view(torch.float32).max().item())
        if pv <= E4M3_MAX:
            return True
        GUARD_STATS["retries"] += 1
        f = 2.0 ** math.ceil(math.log2(pv / E4M3_MAX * 2.0))
        cache["s"].mul_(f)
        _derive(cache)
    GUARD_STATS["fallbacks"] += 1
    return False


def register_key(w: torch.Tensor):
    """The calibration key of a module (its weight's address); the state is dropped when the weight is freed."""
    key = w.data_ptr()
    if key not in _CACHES and not torch.cuda.is_current_stream_capturing():
        weakref.finalize(w, _CACHES.pop, key, None)
    return key

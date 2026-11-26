"""EXPERIMENTAL, opt-in (never reached by default): the RTX 5090 (sm_120a) OuterProductMean inference with the grouped outer product
``O[(i,c), (j,e)] = sum_s a[s, i, c] b[s, j, e]`` -- the K = S GEMM that is nearly the whole call at depth -- on e4m3 operands (docs/fp8.md "RTX 5090", OPM).

Chain (``opm_sm80.inference`` with the product swapped; prologue, pair counts and epilogue are the default's)::

    prologue (unchanged: LN + both projections + mask -> A, B [S, 32 L] bf16, s-major)
    -> transpose_quant (msa_f8_sm120.cu): A2q, BTq [32 L, Sp] e4m3 = (A, B)^T / scale (K-major: the 5090's cuBLASLt takes e4m3 in the TN form only), saturating
       cast, the max |scaled value| of each operand kept for the guard
    -> cuBLASLt e4m3 TN GEMM (``opm_fp8.product``: fp32 accumulation with cuBLAS's promotion, A / B scale pointers = the dequantisation scales, bf16 O)
    -> the default epilogue (mask counts / count table, / n, W_o, bias, residual)

Scales (``POLICY``, the H100 route's): ``bound`` = from the weights alone (|LN(x)_k| <= sqrt(d_msa - 1) |gamma_k| + |beta_k|: no input can saturate the
cast; no calibration, no host sync) -- the ``safe`` preset's; ``calib`` = the first eager call's max |A|, max |B| (power of two x ``HEADROOM``), guarded
(``sync``: an overflowing eager call raises the scales and re-quantises; ``lazy`` / capture: ``overflow_peak`` / ``rescale_all`` through ``fp8.check()``).
Same interface as ``opm_fp8`` (the H100 module of the slot): ``POLICY``, ``GUARD``, ``GUARD_STATS``, ``overflow_peak``, ``rescale_all``; registered in
``fp8.SM120_ROUTES["opm"]``. Switches: ``FOLDJET_ENGINE_OPM_FP8`` (``bound`` / ``calib``; ``safe`` sets ``bound``), ``FOLDJET_ENGINE_OPM_FP8_GUARD``,
``FOLDJET_ENGINE_OPM_FP8_HEADROOM``, ``FOLDJET_ENGINE_OPM_FP8_SM120_MINS`` (one depth gate replacing :func:`_gate`; 0 = no gate).
"""

from __future__ import annotations

import math
import os
import weakref

import torch

import foldjet_engine._envcompat  # noqa: F401  (old MINIWORLD_* environment names)
from foldjet_engine.fp8 import env as _fp8_env

E4M3_MAX = 448.0
POLICY = _fp8_env("FOLDJET_ENGINE_OPM_FP8") or None
GUARD = os.environ.get("FOLDJET_ENGINE_OPM_FP8_GUARD", "sync")
HEADROOM = float(os.environ.get("FOLDJET_ENGINE_OPM_FP8_HEADROOM", "16"))
MIN_S: int | None = int(os.environ["FOLDJET_ENGINE_OPM_FP8_SM120_MINS"]) if "FOLDJET_ENGINE_OPM_FP8_SM120_MINS" in os.environ else None
GUARD_STATS = {"calls": 0, "retries": 0, "fallbacks": 0, "capture_fallbacks": 0, "calibrations": 0, "declined": 0}
_CACHES: dict = {}  # per module (key: data_ptr of to_left.weight, dropped with it): {"policy", "s" [2], "q" [2], "sab": (s0, s1)}
_LAZY: dict = {}


def _gate(s: int, n: int) -> bool:
    """The (depth, length) cells the route takes; MIN_S replaces it. Depth >= 1024 at every length: the H100 route's served range for accuracy (the
    5090 A-B gains at every measured cell, 1.09-1.79x down to S 128 / 256, but the judge error grows below depth 1024: 1.9e-2 at L 128 S 256 against
    1.1e-2 at S 1024; docs/fp8.md "RTX 5090", OPM)."""
    if MIN_S is not None:
        return s >= MIN_S
    return s >= 1024


def serves(depth: int, length: int) -> bool:
    """Python-level test of an ``opm_sm80.inference`` call (bf16 by contract; the caller checks the row-chunked range)."""
    if POLICY not in ("bound", "calib") or torch.is_grad_enabled():
        return False
    ok = _gate(depth, length)
    if not ok:
        GUARD_STATS["declined"] += 1
    return ok


def _lazy_buffer(device):
    if device not in _LAZY:
        _LAZY[device] = torch.zeros(2, device=device)
    return _LAZY[device]


def overflow_peak(reset: bool = True) -> float:
    """Largest |scaled A / B value| since the last read (> 448: a call saturated). One host sync: once per forward, outside any graph."""
    out = 0.0
    for buf in _LAZY.values():
        out = max(out, float(buf.max().item()))
        if reset:
            buf.zero_()
    return out


def _scale_up(cache, f: float) -> None:
    cache["s"].mul_(f)
    cache["q"].copy_(1.0 / cache["s"])
    cache["sab"][0].copy_(cache["s"][0:1])
    cache["sab"][1].copy_(cache["s"][1:2])


def rescale_all(peak: float) -> float:
    """Raise every calibrated module's scales so a forward whose guard read ``peak`` (> 448) fits with room x2 (in place: captured graphs follow)."""
    f = 2.0 ** math.ceil(math.log2(peak / E4M3_MAX * 2.0))
    for cache in _CACHES.values():
        _scale_up(cache, f)
    return f


def _cache(a, b, leaves):
    key = leaves[2].data_ptr()  # the module's to_left.weight
    c = _CACHES.get(key)
    if c is not None and c["policy"] == POLICY:
        return c
    if torch.cuda.is_current_stream_capturing():
        GUARD_STATS["capture_fallbacks"] += 1
        return None
    from foldjet_engine.kernels.integrations import opm_fp8

    if POLICY == "bound":
        pbias = torch.cat([leaves[6], leaves[7]]) if len(leaves) > 7 else None
        s = opm_fp8.bound_scales(leaves[0], leaves[1], leaves[2], leaves[3], pbias).to(a.device)
    else:  # calib: the bf16 operands of this call
        amax = torch.stack([a.abs().amax(), b.abs().amax()]).float()
        s = opm_fp8.pow2_scale(amax) * HEADROOM
    s = s.float().contiguous()
    c = _CACHES[key] = {"policy": POLICY, "s": s, "q": (1.0 / s).contiguous(), "sab": (s[0:1].clone(), s[1:2].clone())}
    GUARD_STATS["calibrations"] += 1
    weakref.finalize(leaves[2], _CACHES.pop, key, None)
    return c


def outer(a: torch.Tensor, b: torch.Tensor, leaves: list[torch.Tensor], length: int) -> torch.Tensor | None:
    """O [32 L, 32 L] bf16 = A^T B on e4m3 operands from the prologue's A, B [S, 32 L]; None: the caller runs the default product (a capture before
    the module's scales exist, or a call that saturated four times)."""
    from foldjet_engine.kernels.integrations import msa_f8_sm120 as X
    from foldjet_engine.kernels.integrations import opm_fp8

    cache = _cache(a, b, leaves)
    if cache is None:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    lazy = GUARD in ("lazy", "off") or capturing or cache["policy"] == "bound"  # bound: the cast cannot saturate (no host sync)
    for _ in range(4):
        amax = _lazy_buffer(a.device) if lazy else torch.zeros(2, device=a.device)
        a8 = X.transpose_quant(a, cache["q"][0:1], amax[0:1])
        b8 = X.transpose_quant(b, cache["q"][1:2], amax[1:2])
        GUARD_STATS["calls"] += 1
        if not lazy:
            peak = float(amax.max().item())
            if peak > E4M3_MAX:
                GUARD_STATS["retries"] += 1
                _scale_up(cache, 2.0 ** math.ceil(math.log2(peak / E4M3_MAX * 2.0)))
                continue
        return opm_fp8.product(a8, b8, cache["sab"], length)
    GUARD_STATS["fallbacks"] += 1
    return None

"""EXPERIMENTAL, opt-in (off by default): the RTX 5090 (sm_120a) MSA pair-weighted-averaging key contraction ``o_h = w_h v_h`` on e4m3 operands
(docs/fp8.md "RTX 5090", PWA), the H100 routes' recipe (``pwa_fp8``) on the 5090's kernels.

Native widths (``pwa_sm80.inference``: d_msa 64 / 128, 8 heads x 8 / 16 / 32 -- MiniWorld / Boltz-2 / RF3 / Chai-1 8 x 32, AF3 / OpenFold3 / Protenix
64 x 8, Protenix-v2 / OpenDDE 128 x 8, ESMFold2 128 x 16)::

    pair_fwd (default, w bf16) -> w_quant (e4m3 of 256 w) -> ln_vt8 (LN(m) Wv^T written as e4m3 vT [8, S C, N], j contiguous; replaces ln_v)
      -> cuBLASLt e4m3 batched TN GEMM (``pwa_fp8.contract``: o bf16 [8, N, S C], fp32 accumulation) -> gate_out (default, unchanged)

BIG width (``pwa_wide_sm80``: d_msa 256, d_pair 512, 8 x 32 -- IntelliFold2)::

    LN (default) -> g = y Wg^T (the default GEMM tile, g alone) and vt8_y (v = y Wv^T straight into e4m3 vT [8, S 32, N]; Wv held in two halves of
      64 KiB by neighbouring CTAs) -> the same contraction -> gate_fwd / out (default)

Scales: w are probabilities (fixed 2^8, no saturation); v takes a static power-of-two scale (``h100_sm90.vscale_from_amax``, ``HEADROOM`` x the absmax of
the module's first eager call; keyed by the value weight). Both are cuBLASLt scale pointers (device scalars: ``rescale_all`` updates them in place,
captured graphs follow). Guard: the max |v / scale| (``take_overflow`` / ``overflow_peak``); ``GUARD`` ``sync`` redoes a saturating eager call, ``lazy``
leaves it to ``fp8.check()``. Same interface as ``pwa_fp8`` (``ENABLED``, ``GUARD``, ``STATS``, ``take_overflow``, ``rescale_all``); registered in
``fp8.SM120_ROUTES["pwa"]``. Switches: ``MINIWORLD_PWA_FP8=1`` (in ``safe``), ``MINIWORLD_PWA_FP8_GUARD``, ``MINIWORLD_PWA_FP8_HEADROOM``,
``MINIWORLD_PWA_FP8_SM120_MINSN`` (smallest S * N served at the native widths, :data:`MIN_SN`), ``MINIWORLD_PWA_FP8_SM120_BIG_MINSN`` (BIG).
"""

from __future__ import annotations

import os
import weakref

import torch

from foldjet_engine.fp8 import env as _fp8_env

E4M3_MAX = 448.0
W_SCALE = 2.0**-8
ENABLED = _fp8_env("MINIWORLD_PWA_FP8") == "1"
#: ``calls`` = native widths, ``wide_calls`` = the BIG width, ``retries`` = eager saturation redos, ``declined`` = calls the gate sent to the default
STATS = {"calls": 0, "wide_calls": 0, "retries": 0, "declined": 0, "calibrations": 0}
GUARD = os.environ.get("MINIWORLD_PWA_FP8_GUARD", "sync")
HEADROOM = float(os.environ.get("MINIWORLD_PWA_FP8_HEADROOM", "16"))
RESCALE = 4.0


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


#: smallest S * N served per (d_msa, C) at the native widths (same-process A-B, docs/fp8.md "RTX 5090"); 0 = no gate
MIN_SN: dict = {(64, 8): 1 << 15, (64, 16): 1 << 15, (64, 32): 1 << 15, (128, 8): 1 << 15, (128, 16): 1 << 15}  # S128 N128: 0.97 (64 x 8); S128 N256: 1.05
MIN_SN_ALL: int | None = int(os.environ["MINIWORLD_PWA_FP8_SM120_MINSN"]) if os.environ.get("MINIWORLD_PWA_FP8_SM120_MINSN") else None
#: smallest S * N served at the BIG width
BIG_MIN_SN = _env_int("MINIWORLD_PWA_FP8_SM120_BIG_MINSN", 1 << 14)  # the smallest measured cell (S64 N256: 1.03); every measured BIG cell gains
#: ln_vt8 tile (tokens) and resident CTAs per SM, per (d_msa, C)
VT8_CFG: dict = {}

_VS: dict = {}  # key (the value weight's address) -> static scale fp32 [1] (dequant factor of v)
_VINV: dict = {}
_AMAX: dict = {}  # device -> fp32 [1]
_WS: dict = {}  # device -> W_SCALE fp32 [1]


def serves(d_msa: int, c: int, s: int, n: int) -> bool:
    """Does the route take this native-width inference call (the caller checks the switch and the width)?"""
    floor = MIN_SN_ALL if MIN_SN_ALL is not None else MIN_SN.get((d_msa, c), 0)
    ok = n % 16 == 0 and s * n >= floor
    if not ok:
        STATS["declined"] += 1
    return ok


def serves_big(s: int, n: int) -> bool:
    ok = n % 16 == 0 and s * n >= BIG_MIN_SN
    if not ok:
        STATS["declined"] += 1
    return ok


def take_overflow() -> bool:
    """True when a call saturated an e4m3 v value since the last call (clears; one host sync per device)."""
    hit = False
    for a in _AMAX.values():
        hit |= float(a.item()) > E4M3_MAX
        a.zero_()
    return hit


def overflow_peak() -> float:
    """max |v / scale| since the last take_overflow() (does not clear; one host sync)."""
    return max((float(a.item()) for a in _AMAX.values()), default=0.0)


def rescale_all(factor: float = RESCALE) -> None:
    """Multiply every calibrated v scale by ``factor`` in place (a power of two keeps the scales exact; captured graphs follow)."""
    for key, t in _VS.items():
        t.mul_(factor)
        _VINV[key].copy_(1.0 / t)


def _forget(key) -> None:
    _VS.pop(key, None)
    _VINV.pop(key, None)


def _state(dev, key_tensor, probe):
    """(amax, w scale, key); the first eager call of a module calibrates its v scale from ``probe(vinv=1, amax)``. None under a capture before that."""
    amax = _AMAX.get(dev)
    if amax is None:
        amax = _AMAX[dev] = torch.zeros(1, device=dev)
    ws = _WS.get(dev)
    if ws is None:
        ws = _WS[dev] = torch.full((1,), W_SCALE, device=dev)
    key = key_tensor.data_ptr()
    if key not in _VS:
        if torch.cuda.is_current_stream_capturing():
            return None
        from foldjet_engine.kernels.triangle_attention.cuda.h100_sm90 import vscale_from_amax

        am = torch.zeros(1, device=dev)
        probe(torch.ones(1, device=dev), am)
        _VS[key] = vscale_from_amax(am, HEADROOM).float().contiguous()
        _VINV[key] = (1.0 / _VS[key]).contiguous()
        STATS["calibrations"] += 1
        weakref.finalize(key_tensor, _forget, key)
    return amax, ws, key


def _vt8_cfg(d: int, c: int) -> tuple[int, int]:
    return VT8_CFG.get((d, c), (64, 2))


def contract_native(m, w16, lnw, lnb, wv16, eps, key_tensor):
    """o [8, N, S C] bf16 = w v of one native-width inference call (m [S, N, D], w [8, N, N] bf16 from pair_fwd); None: the default runs (a capture
    before the module's v scale exists, or four saturating redos)."""
    from foldjet_engine.kernels.integrations import msa_f8_sm120 as X
    from foldjet_engine.kernels.integrations import pwa_fp8

    d, c = m.shape[2], wv16.shape[0] // 8
    nt, bps = _vt8_cfg(d, c)
    st = _state(m.device, key_tensor, lambda vinv, am: X.ln_vt8(m, lnw, lnb, wv16, eps, vinv, am, nt, bps))
    if st is None:
        return None
    amax, ws, key = st
    STATS["calls"] += 1
    wq = X.w_quant(w16)
    lazy = GUARD in ("lazy", "off") or torch.cuda.is_current_stream_capturing()
    o = torch.empty((8, m.shape[1], m.shape[0] * c), dtype=torch.bfloat16, device=m.device)
    for _ in range(4):
        vt = X.ln_vt8(m, lnw, lnb, wv16, eps, _VINV[key], amax, nt, bps)
        pwa_fp8.contract(wq, vt, o, _VS[key], ws)
        del vt
        if lazy or not take_overflow():
            return o
        STATS["retries"] += 1
        rescale_all()
    return None


def contract_big(y, wv16, w16, key_tensor, depth: int, length: int):
    """o_hm [8, N, S 32] bf16 = w v at the BIG width from its LayerNorm output y [S N, 256] and Wv [256, 256] (``vt8_y``: v straight into the
    e4m3 vT, no bf16 v); None: the caller runs the default v and contraction."""
    from foldjet_engine.kernels.integrations import msa_f8_sm120 as X
    from foldjet_engine.kernels.integrations import pwa_fp8

    st = _state(y.device, key_tensor, lambda vinv, am: X.vt8_y(y, wv16, vinv, am, depth, length))
    if st is None:
        return None
    amax, ws, key = st
    STATS["wide_calls"] += 1
    wq = X.w_quant(w16)
    lazy = GUARD in ("lazy", "off") or torch.cuda.is_current_stream_capturing()
    o = torch.empty((8, length, depth * 32), dtype=torch.bfloat16, device=y.device)
    for _ in range(4):
        vt = X.vt8_y(y, wv16, _VINV[key], amax, depth, length)
        pwa_fp8.contract(wq, vt, o, _VS[key], ws)
        del vt
        if lazy or not take_overflow():
            return o
        STATS["retries"] += 1
        rescale_all()
    return None

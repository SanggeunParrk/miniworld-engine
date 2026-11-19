"""EXPERIMENTAL B200 (sm_100a) e4m3 contraction of MSAPairWeightedAveraging inference (opt-in ``MINIWORLD_PWA_FP8=1``, default OFF; docs/b200/fp8.md "PWA").

o[h] = w[h] . v[h] (K = the key index j, the stage that grows with N^2 S) on e4m3 operands: w (a probability, <= 1) with the fixed scale 2^8 (cannot saturate), v with a
static power-of-two scale from the weights alone (``bound`` policy of the OuterProductMean route: |y_k| <= sqrt(d_msa - 1)|gamma_k| + |beta_k|, so max_c sum_k |Wv_ck| (...) bounds
every v value: no calibration, no guard); fp32 accumulation, bf16 o exactly where cuBLAS bmm writes it, the gate / out pass is the default's. On B200 the head-major v
[H, N, S C] (the default ln_vg output) is used as it is (MN-major B operand of cuBLASLt's e4m3 GEMM: no transpose, unlike Hopper's K-major wgmma); one cast launch each for w and v
(``opm_fp8_b200.cu``'s ``opm_cast1``), 8 head GEMMs (``torch._scaled_mm``). Switch: the H100 module's ``pwa_fp8.ENABLED`` (so ``fp8.enable`` reaches it)."""

from __future__ import annotations

import functools
import math
import weakref

import torch

from . import opm_fp8 as _o
from . import pwa_fp8 as _h

E4M3 = torch.float8_e4m3fn
#: gates from the B200 measurement (docs/b200/fp8.md): head dim 32 from N 1280 (0.95 at 1280, 0.82 at 2048), 16 from 1792 (0.91 at 2048), 8 never (0.95-0.97 at 2048: below the 5 % bar); None = this rule
MIN_N: int | None = None
_CACHES: dict = {}
STATS = _h.STATS


def _gate(n: int, c: int) -> bool:
    if MIN_N is not None:
        return n >= MIN_N
    return n >= (1280 if c >= 32 else 1792 if c >= 16 else 1 << 30)


@functools.lru_cache(maxsize=None)
def _ext():
    from . import opm_fp8_b200 as ob

    return ob._ext()


def prebuild_items():
    from . import opm_fp8_b200 as ob

    yield ob._ext


def _scales(args, device):
    wv = args[5]
    key = wv.data_ptr()
    c = _CACHES.get(key)
    if c is not None:
        return c
    if torch.cuda.is_current_stream_capturing():
        return None
    lnw, lnb = args[3], args[4]
    g = math.sqrt(lnw.numel() - 1) * lnw.detach().double().abs() + lnb.detach().double().abs()
    b = (wv.detach().to(torch.bfloat16).double().abs() @ g).max() * (1.0 + 2.0**-6)
    sv = _o.pow2_scale(b.float().reshape(1)).to(device).reshape(())
    c = _CACHES[key] = {"qv_host": float(1.0 / float(sv)), "sv": sv, "sw": torch.tensor(1.0 / 256.0, device=device), "q": torch.stack([torch.tensor(256.0, device=device), 1.0 / sv]).contiguous()}
    weakref.finalize(wv, _CACHES.pop, key, None)
    return c


def serves(args, w16, v):
    """The scale entry when the route takes this inference contraction (bf16 operands, head-major v [8, N, S C], shape gate), else None."""
    h, n = w16.shape[0], w16.shape[1]
    if w16.dtype != torch.bfloat16 or v.dtype != torch.bfloat16 or h != 8 or not (w16.is_contiguous() and v.is_contiguous()):
        return None
    sc = v.shape[2]
    if not _gate(n, args[5].shape[0] // h):
        return None
    c = _scales(args, v.device)
    if c is None:
        STATS["declined"] += 1
        return None
    return c


def contract(c, w16, v):
    h = w16.shape[0]
    w8 = torch.empty(w16.shape, device=v.device, dtype=E4M3)
    v8 = torch.empty(v.shape, device=v.device, dtype=E4M3)
    ext = _ext()
    ext.opm_cast1(w16, w8, c["q"], 0)
    ext.opm_cast1(v, v8, c["q"], 1)
    o = torch.empty_like(v)
    for k in range(h):
        torch._scaled_mm(w8[k], v8[k], scale_a=c["sw"], scale_b=c["sv"], out_dtype=torch.bfloat16, out=o[k])
    STATS["calls"] += 1
    return o


def set_on(on: bool) -> None:
    """Measurement helper (``_tools/blk3.py``): the route on with every gate removed / off."""
    global MIN_N
    _h.ENABLED = bool(on)
    MIN_N = 0


#: fused form (v written as e4m3 by ``ln_vg_f8``: no cast pass over v): the shape gate (N padded), replaced by the measurement (docs/b200/fp8.md)
MIN_WORK_FUSED: int | None = None  # N^2 S >= 5e8 (L768 S1024 0.86, L512 S2048 0.87, L1536 S384 0.83; L1024 S384 0.96, L512 S1024 0.99 stay bf16)


def _gate_fused(n: int, s: int) -> bool:
    if MIN_N is not None:
        return n >= MIN_N
    return n * n * s >= (MIN_WORK_FUSED if MIN_WORK_FUSED is not None else 500_000_000)


def serves_fused(args, n: int):
    """The scale entry when the fused route (``ln_vg_f8`` + the e4m3 contraction) takes this MiniWorld-side (8 x 32) inference call, else None."""
    if not _gate_fused(n, args[0].shape[1]) or args[0].dtype != torch.bfloat16:
        return None
    c = _scales(args, args[0].device)
    if c is None:
        STATS["declined"] += 1
    return c


def contract8(c, w16, v8):
    """o = sw sv (w8 . v8) per head: v8 is ``ln_vg_f8``'s e4m3 v [H, N, S C]; only w is cast here."""
    h = w16.shape[0]
    w8 = torch.empty(w16.shape, device=v8.device, dtype=E4M3)
    _ext().opm_cast1(w16, w8, c["q"], 0)
    o = torch.empty(v8.shape, device=v8.device, dtype=torch.bfloat16)
    for k in range(h):
        torch._scaled_mm(w8[k], v8[k], scale_a=c["sw"], scale_b=c["sv"], out_dtype=torch.bfloat16, out=o[k])
    STATS["calls"] += 1
    return o

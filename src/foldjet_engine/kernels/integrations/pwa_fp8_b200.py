"""EXPERIMENTAL B200 (sm_100a) e4m3 contraction of MSAPairWeightedAveraging inference (opt-in ``MINIWORLD_PWA_FP8=1``, default OFF; docs/b200/fp8.md "PWA").

o[h] = w[h] . v[h] (K = the key index j, the stage that grows with N^2 S) on e4m3 operands: w (a probability, <= 1) with the fixed scale 2^8 (cannot saturate), v with a
static power-of-two scale from the weights alone (``bound`` policy of the OuterProductMean route: |y_k| <= sqrt(d_msa - 1)|gamma_k| + |beta_k|, so max_c sum_k |Wv_ck| (...) bounds
every v value: no calibration, no guard); fp32 accumulation, bf16 o exactly where cuBLAS bmm writes it, the gate / out pass is the default's. On B200 the head-major v
[H, N, S C] (the default ln_vg output) is used as it is (MN-major B operand of cuBLASLt's e4m3 GEMM: no transpose, unlike Hopper's K-major wgmma); one cast launch each for w and v
(``opm_fp8_b200.cu``'s ``opm_cast1``), 8 head GEMMs (``torch._scaled_mm``). Switch: the H100 module's ``pwa_fp8.ENABLED`` (so ``fp8.enable`` reaches it)."""

from __future__ import annotations

import functools
import math
import os
import weakref
from pathlib import Path

import torch

from . import opm_fp8 as _o
from . import pwa_fp8 as _h

E4M3 = torch.float8_e4m3fn
#: gates from the B200 measurement (docs/b200/fp8.md): head dim 32 from N 1280 (0.95 at 1280, 0.82 at 2048), 16 from 1792 (0.91 at 2048), 8 never (0.95-0.97 at 2048: below the 5 % bar); None = this rule
MIN_N: int | None = None
#: 2026-11-24: OFF unless ``MINIWORLD_PWA_FP8_B200=1`` (on top of ``MINIWORLD_PWA_FP8`` / the ``safe`` preset): FoldForge F3 RoseTTAFold3 5I28 n = 12 / 16
#: (compile + CUDA graphs) gave non-finite coordinates with this route and only with it (every other FP8 route alone finite; ``safe`` without PWA
#: finite, 0.90x bf16); its eager calls stay in range (max |v8| 20 of 448), the fault is in the captured replay, not localized yet
B200_ON: bool = os.environ.get("MINIWORLD_PWA_FP8_B200", "") == "1"
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
    yield _ext_grouped


def _scales(args, device):
    wv, lnw, lnb = args[5], args[3], args[4]
    # (was the pointer of wv alone: a freed weight's address reused by another module's weight at a new version, or an in-place update, read a stale scale)
    key = (wv.data_ptr(), wv._version, tuple(wv.shape), lnw.data_ptr(), lnw._version, lnb.data_ptr(), lnb._version)
    c = _CACHES.get(key)
    if c is not None:
        return c
    if torch.cuda.is_current_stream_capturing():
        return None
    g =math.sqrt(lnw.numel() - 1) * lnw.detach().double().abs() + lnb.detach().double().abs()
    b = (wv.detach().to(torch.bfloat16).double().abs() @ g).max() * (1.0 + 2.0**-6)
    sv = _o.pow2_scale(b.float().reshape(1)).to(device).reshape(())
    c = _CACHES[key] = {"qv_host": float(1.0 / float(sv)), "sv": sv, "dq": (sv.float() / 256.0).reshape(1).contiguous(), "sw": torch.tensor(1.0 / 256.0, device=device), "q": torch.stack([torch.tensor(256.0, device=device), 1.0 / sv]).contiguous()}
    weakref.finalize(wv, _CACHES.pop, key, None)
    return c


def serves(args, w16, v):
    """The scale entry when the route takes this inference contraction (bf16 operands, head-major v [8, N, S C], shape gate), else None."""
    if not B200_ON:
        return None
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
    global MIN_N, B200_ON
    _h.ENABLED = bool(on)
    B200_ON = bool(on)
    MIN_N = 0


#: fused form (v written as e4m3 by ``ln_vg_f8``: no cast pass over v): the shape gate (N padded), replaced by the measurement (docs/b200/fp8.md)
MIN_WORK_FUSED: int | None = None  # None = the (N, S) rule of `_gate_fused`; an int replaces it by padded N^2 S >= MIN_WORK_FUSED


def _gate_fused(n: int, s: int) -> bool:
    if MIN_N is not None:
        return n >= MIN_N
    if MIN_WORK_FUSED is not None:
        return n * n * s >= MIN_WORK_FUSED
    # measured with the grouped contraction (docs/b200/fp8.md, MiniWorld 64 / 128 8 x 32): S 1024 from N 256 (0.916 / 0.931 / 0.909 at N 256 / 384 / 512), S 512 from N 640 (0.939; N 512 0.986, N 384 0.980, N 256 1.017 stay bf16)
    return (s >= 1024 and n >= 256) or (s >= 512 and n >= 640)


def serves_fused(args, n: int):
    """The scale entry when the fused route (``ln_vg_f8`` + the e4m3 contraction) takes this MiniWorld-side (8 x 32) inference call, else None."""
    if not B200_ON or not _gate_fused(n, args[0].shape[1]) or args[0].dtype != torch.bfloat16:
        return None
    c = _scales(args, args[0].device)
    if c is None:
        STATS["declined"] += 1
    return c


#: wide builds (d_msa 64 / 128 / 256, 8 x 8 / 16 / 32; ``ln_vg_w_f8``): the work gate padded N^2 S per head dim C, from the sweep of docs/b200/fp8.md WITH the grouped contraction kernel (every served cell <= 0.95; the smaller cells lose or are flat)
MIN_WORK_WIDE: dict[int, int] = {8: 1_600_000_000, 16: 600_000_000, 32: 600_000_000}


def serves_fused_wide(args, n: int):
    """The scale entry when the fused wide route (``ln_vg_w_f8`` + the e4m3 contraction) takes this inference call (any wide build), else None."""
    h = 8
    c = args[5].shape[0] // h
    if not B200_ON or args[0].dtype != torch.bfloat16 or c not in MIN_WORK_WIDE:
        return None
    if MIN_N is not None:
        ok = n >= MIN_N
    else:
        ok = n * n * args[0].shape[1] >= MIN_WORK_WIDE[c]
    if not ok:
        return None
    sc = _scales(args, args[0].device)
    if sc is None:
        STATS["declined"] += 1
    return sc


#: the eight per-head GEMMs on this many streams (a fork / join of the capture: the head grids fill each other's tails; one equal-flop launch measured 0.64-1.0, two streams 0.88-1.0 of the eight serial heads, docs/b200/fp8.md); 1 = serial
HEAD_STREAMS = 2
_HS: dict = {}


#: ONE grouped tcgen05 e4m3 launch for all eight heads (`pwa8_fp8_b200.cu`: A K-major, B MN-major, per-head planes; docs/b200/fp8.md) instead of the per-head `_scaled_mm`s (HEAD_STREAMS)
GROUPED = True
#: the grouped kernel is the faster of the two up to this per-head work N^2 S C (measured: N256 SC8192 0.41, N768 SC16384 0.80, N1280 SC8192 0.80 of the eight per-head GEMMs; N1536 SC8192 0.98, N768 SC32768 1.04, N2048 SC8192 1.08, N1536 SC32768 1.21 where cuBLAS reaches 2.8 PF/s against 2.4); above it the 2-stream per-head GEMMs run
GROUPED_MAX_WORK = 15_000_000_000


@functools.lru_cache(maxsize=None)
def _ext_grouped():
    from foldjet_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    d = Path(__file__).parent
    b = d.parent / "trimul_inproj" / "cuda" / "b200_sources"
    return load_extension(
        name="pwa8_fp8_b200",
        sources=[str(d / "csrc" / "sm100" / "pwa8_fp8_b200.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("100a"), "--expt-relaxed-constexpr", "-DNDEBUG", f"-I{b}"],
        extra_cflags=["-std=c++17", f"-I{b}"],
        verbose=False,
    )


def _heads(c, w8, v8, o, h):
    if GROUPED and w8.shape[1] % 128 == 0 and v8.shape[2] % 128 == 0 and c.get("dq") is not None and w8.shape[1] * w8.shape[2] * v8.shape[2] <= GROUPED_MAX_WORK:
        _ext_grouped().pwa8_forward(w8, v8, o, c["dq"])
        return
    if HEAD_STREAMS < 2:
        for k in range(h):
            torch._scaled_mm(w8[k], v8[k], scale_a=c["sw"], scale_b=c["sv"], out_dtype=torch.bfloat16, out=o[k])
        return
    dev = v8.device
    main = torch.cuda.current_stream(dev)
    side = _HS.get(dev)
    if side is None:
        side = _HS[dev] = torch.cuda.Stream(dev)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        for k in range(1, h, 2):
            torch._scaled_mm(w8[k], v8[k], scale_a=c["sw"], scale_b=c["sv"], out_dtype=torch.bfloat16, out=o[k])
    for k in range(0, h, 2):
        torch._scaled_mm(w8[k], v8[k], scale_a=c["sw"], scale_b=c["sv"], out_dtype=torch.bfloat16, out=o[k])
    main.wait_stream(side)  # joined before w8 / v8 / o are used or freed on the main stream


def contract8(c, w16, v8):
    """o = sw sv (w8 . v8) per head: v8 is ``ln_vg_f8``'s e4m3 v [H, N, S C]; only w is cast here."""
    h = w16.shape[0]
    w8 = torch.empty(w16.shape, device=v8.device, dtype=E4M3)
    _ext().opm_cast1(w16, w8, c["q"], 0)
    o = torch.empty(v8.shape, device=v8.device, dtype=torch.bfloat16)
    _heads(c, w8, v8, o, h)
    STATS["calls"] += 1
    return o

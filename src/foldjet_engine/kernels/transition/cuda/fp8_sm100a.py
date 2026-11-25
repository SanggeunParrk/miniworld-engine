"""OPT-IN FP8 (e4m3) inference forward of the Transition on B200 (sm_100a): D a multiple of 128 up to 1024, n 2 / 4 (docs/b200/fp8.md).

Off by default: with ``MINIWORLD_TRANSITION_FP8`` unset (and ``MINIWORLD_FP8`` off) nothing here runs and every Transition call takes the B200 default
route (``route_sm100a``), bit-identical to the tree without this module. The switch is the one the H100 route reads (``fp8.PRESETS``), so
``MINIWORLD_FP8=safe`` / ``engine-fp8`` turn this route on at capability (10, 0) as well.

The recipe is the H100 one (``fp8_sm90a.py``, docs/fp8.md "Transition forward"), on tcgen05 kernels. D 128 / 256 (n 2 / 4; ``MINIWORLD_TRANSITION_FP8_FUSED``): ONE fused 2-CTA kernel
(``sm100/fp8/tr8_fwd_sm100.cu``: LayerNorm -> e4m3 xn, expand + SwiGLU with e4m3 h in tensor memory, squeeze from tensor memory, residual; docs/b200/fp8.md section 8); the other widths in three launches (PDL chain):

1. ``tf8_ln`` (``sm100/fp8/tf8_ln.cu``): fp32 LayerNorm, xn quantized straight to e4m3 (one power-of-two scale ``sx`` from the LayerNorm bound
   sqrt(D - 1) max|gamma| + max|beta|: cannot saturate, no calibration);
2. ``tf8_gemm`` MODE 0 (expand): e4m3 xn x e4m3 [Wa | Wb] on ``kind::f8f6f4`` (fp32 accumulators in TMEM), epilogue h = silu(a) b sh with the tanh
   silu, cast ``cvt.rn.satfinite.e4m3x2`` to an e4m3 h [M, n D] (half the bytes of the bf16 round trip), max |h| -> the guard buffer;
3. ``tf8_gemm`` MODE 1 (squeeze): e4m3 h x e4m3 Ws, ``out = x + acc / (sh s2)`` in bf16.

Scales: weights per-tensor power-of-two (Wa and Wb share one), packed once per parameter version and cached; the static h scale ``sh`` is calibrated on the
first eager call of a module (probe call with sh = 1: the pre-cast peak is exact even when it would saturate) with HEADROOM 16, guarded like the TriMul
planes (``overflow_peak`` / ``rescale_all`` / ``fp8.check()``; ``sync`` guard redoes an overflowing eager call, ``lazy`` records the peak, also under
CUDA-graph capture). Scale floors: amax >= 1e-12 (stricter than H100's 1e-30: a scale product must stay finite in fp32).

Environment (read at import): ``MINIWORLD_TRANSITION_FP8=calib``; ``_GUARD`` sync | lazy | off; ``_HEADROOM`` (16); ``_MINROWS`` (replaces ``ROW_GATE``);
``_WIDTHS``; ``_FUSED`` (widths on the fused kernel, default 128,256), ``_FUSED_FLAGS`` (extra nvcc flags of its builds), ``_PAIRB`` (chain GEMMs on 2-CTA clusters sharing B by multicast).
"""

from __future__ import annotations

import functools
import math
import os
import weakref

import torch

from foldjet_engine.fp8 import env as _fp8_env

from ..._compile import opaque
from ..._device import num_sms
from .fused_sm100a import _ext, build_cubins

POLICY = _fp8_env("MINIWORLD_TRANSITION_FP8") or None  # None = off; "calib" = static h scale calibrated on the first eager call
GUARD = os.environ.get("MINIWORLD_TRANSITION_FP8_GUARD", "sync")
HEADROOM = float(os.environ.get("MINIWORLD_TRANSITION_FP8_HEADROOM", "16"))
_MINROWS_ENV = os.environ.get("MINIWORLD_TRANSITION_FP8_MINROWS")
MIN_ROWS: int | None = int(_MINROWS_ENV) if _MINROWS_ENV else None
#: smallest row count routed per (D, n) (below it, and for a (D, n) not listed, the B200 default route runs). Measured on the B200 (docs/b200/fp8.md, same-process
#: interleaved A/B against the default route, graph medians and sustained, idle gap 0.15 s): D 512 n 4 x0.50 at 1024 rows .. 0.77 at 65536+; D 512 n 2 x0.63 at 4096,
#: 0.78 at 8192, 0.89 sustained at 16384 +; D 768 n 4 x1.07 at 1024, 0.90 at 2048, 0.60-0.65 from 4096; D 768 n 2 x1.30 at 1024, 1.16 at 2048, 0.51-0.76 from 4096.
#: D 128 / 256 run the FUSED one-kernel forward (tr8_fwd_sm100.cu, section 8 of docs/b200/fp8.md): D 128 n 4 sustained x1.00 at 16384 rows, 0.95 at 65536, 0.85-0.86 from 147456 (energy
#: x0.83-0.86); D 128 n 2 x0.93 at 65536, 0.87-0.94 from 147456; D 256 n 4 x0.92 at 4096-16384, 0.83-0.84 from 65536 (energy x0.73-0.83).
#: Not routed by default (no gain or a loss, measured with the built kernels): D 128 n 2 below 65536 rows (x1.00, energy x0.88-0.95), D 256 n 2 (sustained x1.00-1.09 to 0.95, cool graph 1.00-1.11),
#: D 384 (chain x0.99-1.05 at 65536-262144 rows n 4, 1.16 n 2, 1.5 at 1024). ``MINIWORLD_TRANSITION_FP8_MINROWS`` + ``_WIDTHS`` switch them on for experiments.
#: D 128 (2026-11-23, re-measured on a B200 through the measure harness, master inference, graph replay, 5 interleaved rounds, two separate
#: sweeps of 16 sizes, rows = L^2 of a [1, L, L, 128] pair; ratio = e4m3 / default): n 2 0.91-0.96 from 65536 to 1327104 rows in both sweeps, 1.00 below
#: 65536, 0.96-1.05 flipping between the sweeps from 1638400; n 4 0.96-0.98 from 147456 to 1048576 (one 1.04 at 409600), 0.98-1.00 below,
#: 1.015-1.09 from 1327104 (4194304: 0.956 / 1.045). Served only inside the window both sweeps agree on (the old gates 16384 / 4096 served the
#: whole range: a loss of up to 9 % at 1.6-2.4 M rows, e.g. AF3 pair transitions of 5I28 n 12).
ROW_GATE = {(512, 4): 1024, (512, 2): 4096, (768, 4): 4096, (768, 2): 4096, (128, 4): 147456, (128, 2): 65536, (256, 4): 1024, (256, 2): 1024}
#: (D, n) cells served only BELOW a row count (the default wins above it): D 256 n 2 (3-launch chain x0.78 at 1024-4096 rows, 1.00 at 8192, 1.39 at 16384; the fused kernel 1.07-1.11)
ROW_MAX = {(256, 2): 8192, (128, 4): 1179648, (128, 2): 1474560}
#: smallest row count at which the FUSED kernel runs a (D, n) (below it the 3-launch chain: the fused kernel has a ~24.6 us floor at D 256 n 4 below ~12000 rows where the chain takes 14.4 us at 1024 rows .. 22.7 us at 8192
#: against the default 26.7 us): D 256 n 4 12288, D 256 n 2 never (chain below ROW_MAX, default above). D 128: fused from its gate.
FUSED_MIN_ROWS = {(256, 4): 12288, (256, 2): 1 << 60}
WIDTHS = tuple(int(w) for w in os.environ.get("MINIWORLD_TRANSITION_FP8_WIDTHS", "128,256,512,768").split(",") if w)
PDL = True
E4M3_MAX = 448.0
AMAX_FLOOR = 1e-12
GUARD_STATS = {"calls": 0, "retries": 0, "fallbacks": 0, "capture_fallbacks": 0, "calibrations": 0}
_CACHES: dict = {}  # per module (key: data_ptr of expand_a's weight; dropped when that tensor is freed)
_LAZY: dict = {}

FUSED_WIDTHS = tuple(int(w) for w in os.environ.get("MINIWORLD_TRANSITION_FP8_FUSED", "128,256").split(",") if w)  # widths run by the fused one-kernel e4m3 forward (tr8_fwd_sm100.cu); the others (and unlisted) the 3-launch chain
_FUSED_FLAGS = tuple(os.environ.get("MINIWORLD_TRANSITION_FP8_FUSED_FLAGS", "").split())  # extra nvcc flags of the fused builds (experiments: -DSIG=1 -DNXS_=3 ...)

PAIRB = os.environ.get("MINIWORLD_TRANSITION_FP8_PAIRB", "0") == "1"  # chain GEMMs on 2-CTA clusters sharing the B tile by multicast (tf8_gemm.cu PAIRB): A/B experiment, off by default

_SPECS = (
    ("f128n4", "fp8/tr8_fwd_sm100.cu", ("-DHID=512", *_FUSED_FLAGS)),
    ("f128n2", "fp8/tr8_fwd_sm100.cu", ("-DHID=256", *_FUSED_FLAGS)),
    ("f256n4", "fp8/tr8_fwd_sm100.cu", ("-DDIM=256", "-DHID=1024", *_FUSED_FLAGS)),
    ("f256n2", "fp8/tr8_fwd_sm100.cu", ("-DDIM=256", "-DHID=512", *_FUSED_FLAGS)),
    ("ln", "fp8/tf8_ln.cu", ()),
    ("swiglu128", "fp8/tf8_gemm.cu", ("-DE4=1", "-DMODE=0", "-DBN=128")),
    ("resid128", "fp8/tf8_gemm.cu", ("-DE4=1", "-DMODE=1", "-DBN=128")),
    ("resid192", "fp8/tf8_gemm.cu", ("-DE4=1", "-DMODE=1", "-DBN=192")),
    ("resid256", "fp8/tf8_gemm.cu", ("-DE4=1", "-DMODE=1", "-DBN=256")),
)

if PAIRB:  # the cluster variants of the chain GEMMs are built (and planned by the prebuild) only when asked for
    _SPECS += (
        ("swiglu128p", "fp8/tf8_gemm.cu", ("-DE4=1", "-DMODE=0", "-DBN=128", "-DPAIRB=1")),
        ("resid128p", "fp8/tf8_gemm.cu", ("-DE4=1", "-DMODE=1", "-DBN=128", "-DPAIRB=1")),
        ("resid192p", "fp8/tf8_gemm.cu", ("-DE4=1", "-DMODE=1", "-DBN=192", "-DPAIRB=1")),
        ("resid256p", "fp8/tf8_gemm.cu", ("-DE4=1", "-DMODE=1", "-DBN=256", "-DPAIRB=1")),
    )


def _specs() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """(name, source, flags) of the route's cubins (``build_cubins``; the sm100 prebuild plan lists them)."""
    return _SPECS


@functools.lru_cache(maxsize=1)
def _load():
    ext = _ext()
    for name, path in build_cubins("fp8", _SPECS).items():
        ext.load_cubin(f"fp8_{name}", path)
    return ext


def serves(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """Does the FP8 route take this (inference) call? bf16 x [..., D] on a B200, D a multiple of 128 (listed in WIDTHS), hidden n D with n 2 / 4,
    at least the row gate of its (D, n) (``ROW_GATE``; ``MIN_ROWS`` replaces it)."""
    d = x.shape[-1]
    if POLICY is None or not x.is_cuda or x.dtype is not torch.bfloat16 or d not in WIDTHS:
        return False
    if wa.dim() != 2 or wa.shape[1] != d or wa.shape[0] not in (2 * d, 4 * d) or ws.shape != (d, wa.shape[0]) or d % 128 or d > 1024:
        return False
    if torch.cuda.get_device_capability(x.device) != (10, 0):
        return False
    rows = x.numel() // d
    gate = MIN_ROWS if MIN_ROWS is not None else ROW_GATE.get((d, wa.shape[0] // d))
    top = None if MIN_ROWS is not None else ROW_MAX.get((d, wa.shape[0] // d))
    return gate is not None and rows >= max(gate, 1) and (top is None or rows < top)


def _pow2_floor(v: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.floor(torch.log2(v)))


def _quant(w: torch.Tensor):
    """Per-tensor power-of-two scale s (amax s <= 448) and the e4m3 bytes of w s. Device ops only (no host sync)."""
    w = w.float()
    s = _pow2_floor(E4M3_MAX / w.abs().amax().clamp_min(AMAX_FLOOR))
    return (w * s).to(torch.float8_e4m3fn).view(torch.uint8).contiguous(), s


def _pack(cache, wa, wb, ws, gamma, beta):
    """The e4m3 weights [Wa8, Wb8 (hid, D), Ws8 (D, hid)] (natural K order: both operands come from shared memory), their scales and the LayerNorm-bound xn
    scale (recomputed when a parameter changed version)."""
    key = tuple((t.data_ptr(), t._version) for t in (wa, wb, ws, gamma, beta))
    if cache.get("key") == key:
        return
    d = wa.shape[1]
    wab = torch.stack([wa.float(), wb.float()])
    s1 = _pow2_floor(E4M3_MAX / wab.abs().amax().clamp_min(AMAX_FLOOR))
    q = (wab * s1).to(torch.float8_e4m3fn).view(torch.uint8)
    cache["wa8"], cache["wb8"] = q[0].contiguous(), q[1].contiguous()
    cache["ws8"], cache["s2"] = _quant(ws)
    bound = math.sqrt(d - 1.0) * gamma.float().abs().amax() + beta.float().abs().amax()
    cache["sx"] = _pow2_floor(E4M3_MAX / bound.clamp_min(AMAX_FLOOR))
    cache["s1"] = s1
    cache["gamma"], cache["beta"] = gamma.float().contiguous(), beta.float().contiguous()
    if "scales" not in cache:
        cache["scales"] = torch.zeros(8, device=wa.device, dtype=torch.float32)
        cache["sh"] = torch.ones((), device=wa.device, dtype=torch.float32)
        cache["calibrated"] = False
    cache["key"] = key
    refresh(cache)


def refresh(cache) -> None:
    """Rebuild the kernels' scale vector [sx, 1 / (sx s1), sh / (sx s1), 1 / (sh s2)] in place from (sx, s1, s2, sh): a captured graph keeps reading the same tensor."""
    sx, s1, s2, sh = cache["sx"], cache["s1"], cache["s2"], cache["sh"]
    ca = 1.0 / (sx * s1)
    cache["scales"][:4].copy_(torch.stack([sx, ca, ca * sh, 1.0 / (sh * s2)]))


def _lazy_buffer(device):
    if device not in _LAZY:
        _LAZY[device] = torch.zeros(1, device=device)
    return _LAZY[device]


def overflow_peak(reset: bool = True) -> float:
    """Largest scaled |h| since the last read (> 448: some call saturated the e4m3 cast). One host sync: once per forward, outside any graph."""
    out = 0.0
    for buf in _LAZY.values():
        out = max(out, float(buf.item()))
        if reset:
            buf.zero_()
    return out


def rescale_all(peak: float) -> float:
    """Lower every calibrated module's h scale so a forward whose guard read ``peak`` (> 448) fits with room x2; returns the factor."""
    if not math.isfinite(peak):  # an inf in the input, not a scale problem: no rescale can fix it (and a 2^inf factor would zero the scale)
        return 1.0
    f = 2.0 ** math.ceil(math.log2(peak / E4M3_MAX * 2.0))
    for cache in _CACHES.values():
        cache["sh"].div_(f)
        refresh(cache)
    return f


def _bn_resid(m: int, d: int, nsm: int) -> int:
    """Squeeze tile width: 256 when D allows it and every SM still gets a tile, else 128."""
    if d % 256 == 0 and -(-m // 128) * (d // 256) >= nsm:
        return 256
    return 192 if d % 192 == 0 else 128


def _flag_hs(flags) -> int:
    return 32 if "-DHS_=32" in flags else 64


def _launch_fused(x2, cache, peak, eps, cubin=None, hs=None):
    """One launch: the fused e4m3 forward (tr8_fwd_sm100.cu, 2-CTA pairs, rows a multiple of 128)."""
    ext = _load()
    m, d = x2.shape
    hid = cache["wa8"].shape[0]
    t, bf = ext.tmap, torch.bfloat16
    hs = hs or _flag_hs(_FUSED_FLAGS)
    out = torch.empty_like(x2)
    tiles = m // 128
    grid = min(num_sms(x2.device), tiles)
    grid = max(grid - grid % 2, 2)
    ext.launch_kernel(cubin or f"fp8_f{d}n{hid // d}", "tr8_fwd_sm100", grid, 512, 2,
                      [t(x2, d, m, 64, 64), t(cache["wa8"].view(bf), d // 2, hid, 64, hs), t(cache["wb8"].view(bf), d // 2, hid, 64, hs),
                       t(cache["ws8"].view(bf), hid // 2, d, 64, d // 2), t(out, d, m, 64, 64), cache["gamma"], cache["beta"], cache["scales"], peak,
                       tiles, float(eps), x2, out], PDL)
    return out


def _launch(x2, cache, peak, eps):
    d = x2.shape[1]
    if d in FUSED_WIDTHS and d in (128, 256) and cache["wa8"].shape[0] in (2 * d, 4 * d) and x2.shape[0] >= FUSED_MIN_ROWS.get((d, cache["wa8"].shape[0] // d), 0):
        m0 = x2.shape[0]
        xp = x2 if m0 % 128 == 0 else torch.nn.functional.pad(x2, (0, 0, 0, -m0 % 128))
        out = _launch_fused(xp, cache, peak, eps)
        return out if m0 == xp.shape[0] else out[:m0]
    return _launch_chain(x2, cache, peak, eps)


def _launch_chain(x2, cache, peak, eps):
    ext = _load()
    m, d = x2.shape
    hid = cache["wa8"].shape[0]
    nsm = num_sms(x2.device)
    sc = cache["scales"]
    t = ext.tmap
    xn8 = torch.empty((m, d), dtype=torch.uint8, device=x2.device)
    ext.launch_kernel("fp8_ln", "tf8_ln", max(1, min(-(-m // 8), nsm * 8)), 256, 1,
                      [x2, cache["gamma"], cache["beta"], xn8, sc, m, d, float(eps)], PDL)
    h8 = torch.empty((m, hid), dtype=torch.uint8, device=x2.device)
    bf = torch.bfloat16
    pb = PAIRB
    ma = t(xn8.view(bf), d // 2, m, 64, 128)
    hb = 64 if pb else 128                                    # B box rows: half of BN under PAIRB (each CTA of the pair loads half and multicasts)
    mb0, mb1 = t(cache["wa8"].view(bf), d // 2, hid, 64, hb), t(cache["wb8"].view(bf), d // 2, hid, 64, hb)
    mt = -(-m // 128)
    if pb:
        tiles = -(-mt // 2) * (hid // 128)
        ext.launch_kernel("fp8_swiglu128p", "tf8_gemm_sm100", 2 * max(1, min(nsm // 2, tiles)), 320, 2,
                          [ma, mb0, mb1, h8, h8, sc, peak, m, hid, d, hid, 0], PDL)
    else:
        tiles = mt * (hid // 128)
        ext.launch_kernel("fp8_swiglu128", "tf8_gemm_sm100", max(1, min(nsm, tiles)), 320, 1,
                          [ma, mb0, mb1, h8, h8, sc, peak, m, hid, d, hid, 0], PDL)
    out = torch.empty_like(x2)
    bn = _bn_resid(m, d, nsm)
    ma = t(h8.view(bf), hid // 2, m, 64, 128)
    mb = t(cache["ws8"].view(bf), hid // 2, d, 64, bn // 2 if pb else bn)
    if pb:
        tiles = -(-mt // 2) * (d // bn)
        ext.launch_kernel(f"fp8_resid{bn}p", "tf8_gemm_sm100", 2 * max(1, min(nsm // 2, tiles)), 320, 2,
                          [ma, mb, mb, out, x2, sc, peak, m, d, hid, d, d], PDL)
    else:
        tiles = mt * (d // bn)
        ext.launch_kernel(f"fp8_resid{bn}", "tf8_gemm_sm100", max(1, min(nsm, tiles)), 320, 1,
                          [ma, mb, mb, out, x2, sc, peak, m, d, hid, d, d], PDL)
    return out


def _default(x, gamma, beta, wa, wb, ws, eps):
    """The B200 default route of the call (the guard's and the capture's fallback)."""
    from . import route_sm100a

    return route_sm100a.transition_sm100a(x, gamma, beta, wa, wb, ws, eps, _fp8=False)


def _fake(x, gamma, beta, wa, wb, ws, eps):
    return torch.empty_like(x)


@opaque(fake=_fake, name="transition_fp8_sm100a_inference")
def inference(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, eps: float
) -> torch.Tensor:
    """y = x + transition(x) through the e4m3 kernels. Eager or under CUDA-graph capture (calibrate eagerly first: a capture with no calibration takes
    the default route)."""
    capturing = torch.cuda.is_current_stream_capturing()
    key = wa.data_ptr()
    cache = _CACHES.get(key)
    if cache is None:
        if capturing:
            GUARD_STATS["capture_fallbacks"] += 1
            return _default(x, gamma, beta, wa, wb, ws, eps)
        cache = _CACHES[key] = {}
        weakref.finalize(wa, _CACHES.pop, key, None)
        _lazy_buffer(x.device)  # (before any capture: a buffer made inside one would live in the graph's pool)
    _pack(cache, wa, wb, ws, gamma, beta)
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    lazy = GUARD in ("lazy", "off") or capturing
    if not cache["calibrated"]:
        if capturing:
            GUARD_STATS["capture_fallbacks"] += 1
            return _default(x, gamma, beta, wa, wb, ws, eps)
        probe = torch.zeros(1, device=x.device)
        cache["sh"].fill_(1.0)
        refresh(cache)
        _launch(x2, cache, probe, eps)  # amax of h (sh = 1): the pre-cast value, saturation does not hide it
        amax = float(probe.item())
        cache["sh"].fill_(2.0 ** math.floor(math.log2(E4M3_MAX / max(amax * HEADROOM, AMAX_FLOOR))))
        refresh(cache)
        cache["calibrated"] = True
        GUARD_STATS["calibrations"] += 1
    if lazy:
        GUARD_STATS["calls"] += 1
        return _launch(x2, cache, _lazy_buffer(x.device), eps).reshape(x.shape)
    peak = torch.zeros(1, device=x.device)
    for _ in range(4):
        peak.zero_()
        y = _launch(x2, cache, peak, eps)
        GUARD_STATS["calls"] += 1
        p = float(peak.item())
        if p <= E4M3_MAX:
            return y.reshape(x.shape)
        if not math.isfinite(p):  # inf activations: the default route decides what the call returns
            break
        GUARD_STATS["retries"] += 1
        cache["sh"].div_(2.0 ** math.ceil(math.log2(p / E4M3_MAX * 2.0)))
        refresh(cache)
    GUARD_STATS["fallbacks"] += 1
    return _default(x, gamma, beta, wa, wb, ws, eps)

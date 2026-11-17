"""OPT-IN FP8 (e4m3) inference forward of the D128 Transition on H100 (engine 3.0.0 line; ``docs/fp8.md``, section Transition).

Off by default: with ``MINIWORLD_TRANSITION_FP8`` unset nothing here runs and every Transition call takes the 2.5.12 route.

The kernel (``transition_fp8_fwd_sm90a.cu``) is the 2.5.12 D128 inference build (``fused_sm90a``, LNFRAG schedule) with both GEMMs on
e4m3 wgmma (fp32 accumulation): xn is quantized in registers straight from the fp32 LayerNorm (scale from the LayerNorm bound, cannot
saturate), the weights are pre-quantized once per parameter version (per-tensor power-of-two scales, K permuted to the e4m3 fragment
order), h = silu(a) b is quantized in the first GEMM's epilogue with a static calibrated scale (``HEADROOM``) and guarded like the TriMul
FP8 planes (``overflow_peak`` / ``rescale_all``, see ``trimul_inproj/cuda/h100_b200style_fp8.py``).

Environment (read at import):

=========================================  ==========================================================================================
``MINIWORLD_TRANSITION_FP8=calib``         route D128 bf16 inference calls (n 2 / 4, rows a multiple of 128) through the FP8 kernel
``MINIWORLD_TRANSITION_FP8_GUARD``         ``sync`` (default, eager: redo an overflowing call) | ``lazy`` (no host sync; CUDA graphs) | ``off``
``MINIWORLD_TRANSITION_FP8_HEADROOM``      h scale headroom over the calibration amax (16)
``MINIWORLD_TRANSITION_FP8_G1``            1 (default): the expand GEMM in e4m3 (xn, Wa, Wb); 0: the bf16 expand
``MINIWORLD_TRANSITION_FP8_G2``            1 (default): the squeeze GEMM in e4m3 (h, Ws); 0: the bf16 squeeze
``MINIWORLD_TRANSITION_FP8_SIGT``          1 (default): sigmoid by one tanh.approx; 0: the 2.5.12 ex2 + rcp form
``MINIWORLD_TRANSITION_FP8_SLOTS``         weight-ring slots (4)
``MINIWORLD_TRANSITION_FP8_MINROWS``       smallest row count routed (below it the default runs)
=========================================  ==========================================================================================
"""

from __future__ import annotations

import functools
import math
import os
import weakref
from pathlib import Path

import torch

from ..._compile import opaque
from ..._device import num_sms
from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_dir = Path(__file__).parent

POLICY = os.environ.get("MINIWORLD_TRANSITION_FP8") or None  # None = off; "calib" = static h scale calibrated on the first eager call
GUARD = os.environ.get("MINIWORLD_TRANSITION_FP8_GUARD", "sync")
HEADROOM = float(os.environ.get("MINIWORLD_TRANSITION_FP8_HEADROOM", "16"))
G1 = int(os.environ.get("MINIWORLD_TRANSITION_FP8_G1", "1"))
G2 = int(os.environ.get("MINIWORLD_TRANSITION_FP8_G2", "1"))
SIGT = int(os.environ.get("MINIWORLD_TRANSITION_FP8_SIGT", "1"))
SLOTS = int(os.environ.get("MINIWORLD_TRANSITION_FP8_SLOTS", "4"))
MIN_ROWS = int(os.environ.get("MINIWORLD_TRANSITION_FP8_MINROWS", "0"))
ABL = int(os.environ.get("MINIWORLD_TRANSITION_FP8_ABL", "0"))  # timing ablation only (wrong output): 3 no chunk loop, 4 no swiglu math

E4M3_MAX = 448.0
GUARD_STATS = {"calls": 0, "retries": 0, "fallbacks": 0, "capture_fallbacks": 0, "calibrations": 0}
_CACHES: dict = {}  # per module (key: data_ptr of expand_a's weight; dropped when that tensor is freed)
_LAZY: dict = {}


@functools.lru_cache(maxsize=16)
def _build(ctas: int, n: int, g1: int, g2: int, sigt: int, slots: int, abl: int):
    ensure_cuda_home()
    return load_extension(
        name=f"transition_fp8_sm90a_c{ctas}_n{n}_f{g1}_g{g2}_t{sigt}_s{slots}_a{abl}_v1",
        sources=[str(_dir / "transition_fp8_fwd_sm90a.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            *gencodes("90a"),
            f"-I{_dir / 'anthropic_v5'}",
            f"-DNCTA={ctas}",
            f"-DMW_N={n}",
            f"-DF8G1={g1}",
            f"-DF8G2={g2}",
            f"-DSIGT={sigt}",
            f"-DWSLOTS={slots}",
            f"-DABL={abl}",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def _cfg():
    return (G1, G2, SIGT, SLOTS, ABL)


def serves(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """Does the FP8 route take this (inference) call? bf16 x [..., 128], hidden 256 / 512, rows a whole number of 128-row tiles, sm_90."""
    if POLICY is None or not x.is_cuda or x.dtype is not torch.bfloat16 or x.shape[-1] != 128:
        return False
    if wa.shape[1] != 128 or wa.shape[0] not in (256, 512) or ws.shape != (128, wa.shape[0]):
        return False
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        return False
    rows = x.numel() // 128
    return rows % 128 == 0 and rows >= max(MIN_ROWS, 128)


def _perm_k(k: int, device) -> torch.Tensor:
    """Physical column of each logical K position: in every 32-wide block, logical 4 t + q (t < 4) reads physical [2t, 2t+1, 8+2t, 9+2t][q],
    logical 16 + 4 t + q reads 16 + the same (the bf16 / fp32 fragment columns of a thread, in the e4m3 k32 A-fragment order)."""
    p32 = []
    for blk in range(2):
        for t in range(4):
            p32 += [16 * blk + c for c in (2 * t, 2 * t + 1, 8 + 2 * t, 9 + 2 * t)]
    base = torch.tensor(p32, dtype=torch.long)
    return (torch.arange(0, k, 32)[:, None] + base[None, :]).reshape(-1).to(device)


def _pow2_floor(v: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.floor(torch.log2(v)))


def _quant(w: torch.Tensor):
    """Per-tensor power-of-two scale s (amax s <= 448) and e4m3 bits of w s (uint8 view). Device ops only (no host sync)."""
    w = w.float()
    s = _pow2_floor(E4M3_MAX / w.abs().amax().clamp_min(1e-30))
    return (w * s).to(torch.float8_e4m3fn).view(torch.uint8).contiguous(), s


def _pack(cache, wa, wb, ws, gamma, beta):
    """The packed e4m3 weights, their scales and the LayerNorm-bound xn scale (recomputed when a parameter changed version)."""
    key = (*((t.data_ptr(), t._version) for t in (wa, wb, ws, gamma, beta)), _cfg())
    if cache.get("key") == key:
        return
    cache["cfg"] = _cfg()
    h = wa.shape[0]
    nch = h // 64
    pk = _perm_k(128, wa.device)
    w1 = torch.stack([wa.float().reshape(nch, 64, 128), wb.float().reshape(nch, 64, 128)], 1).reshape(nch * 128, 128)
    if G1:
        cache["w1"], s1 = _quant(w1[:, pk])
    else:
        cache["w1"], s1 = w1.to(torch.bfloat16).contiguous(), torch.ones((), device=wa.device)
    if G2:
        cache["w2"], s2 = _quant(ws.float()[:, _perm_k(h, ws.device)])
    else:
        cache["w2"], s2 = ws.t().to(torch.bfloat16).contiguous(), torch.ones((), device=ws.device)
    bound = math.sqrt(127.0) * gamma.float().abs().amax() + beta.float().abs().amax()
    cache["sx"] = _pow2_floor(E4M3_MAX / bound.clamp_min(1e-30)) if G1 else torch.ones((), device=wa.device)
    cache["s1"], cache["s2"] = s1, s2
    cache["gamma"], cache["beta"] = gamma.float().contiguous(), beta.float().contiguous()
    if "scales" not in cache:
        cache["scales"] = torch.zeros(8, device=wa.device, dtype=torch.float32)
        cache["sh"] = torch.ones((), device=wa.device, dtype=torch.float32)
        cache["calibrated"] = False
    cache["key"] = key
    refresh(cache)


def refresh(cache) -> None:
    """Rebuild the kernel's scale vector in place from (sx, s1, s2, sh): a captured graph keeps reading the same tensor."""
    sx, s1, s2, sh = cache["sx"], cache["s1"], cache["s2"], cache["sh"]
    ca = 1.0 / (sx * s1)
    if cache["cfg"][1]:
        vals = torch.stack([sx, ca, ca * sh, 1.0 / (sh * s2)])
    else:
        vals = torch.stack([sx, ca, ca, torch.ones_like(sx)])
    cache["scales"][:4].copy_(vals)


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
    """Lower every calibrated module's h scale so a forward whose guard read ``peak`` (> 448) fits with room x2; returns the factor.
    Call outside the graph, then redo the forward."""
    f = 2.0 ** math.ceil(math.log2(peak / E4M3_MAX * 2.0))
    for cache in _CACHES.values():
        cache["sh"].div_(f)
        refresh(cache)
    return f


def _launch(x2, cache, peak, eps):
    ext = _build(num_sms(x2.device), cache["w1"].shape[0] // 256, *cache["cfg"])
    return ext.transition_fp8_fwd(x2, cache["gamma"], cache["beta"], cache["w1"], cache["w2"], cache["scales"], peak, float(eps))


def _default(x, gamma, beta, wa, wb, ws, eps):
    from . import fused_sm90a

    return fused_sm90a.transition_fused_sm90a(x, gamma, beta, wa, wb, ws, eps)


def _fake(x, gamma, beta, wa, wb, ws, eps):
    return torch.empty_like(x)


@opaque(fake=_fake, name="transition_fp8_sm90a_inference")
def inference(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, eps: float
) -> torch.Tensor:
    """y = x + transition(x) through the e4m3 kernel. Eager or under CUDA-graph capture (calibrate eagerly first: a capture with no
    calibration takes the 2.5.12 route)."""
    capturing = torch.cuda.is_current_stream_capturing()
    key = wa.data_ptr()
    cache = _CACHES.get(key)
    if cache is None:
        if capturing:
            GUARD_STATS["capture_fallbacks"] += 1
            return _default(x, gamma, beta, wa, wb, ws, eps)
        cache = _CACHES[key] = {}
        weakref.finalize(wa, _CACHES.pop, key, None)
    _pack(cache, wa, wb, ws, gamma, beta)
    x2 = x.reshape(-1, 128).contiguous()
    lazy = GUARD in ("lazy", "off") or capturing
    if not cache["calibrated"]:
        if capturing:
            GUARD_STATS["capture_fallbacks"] += 1
            return _default(x, gamma, beta, wa, wb, ws, eps)
        probe = torch.zeros(1, device=x.device)
        cache["sh"].fill_(1.0)
        refresh(cache)
        _launch(x2, cache, probe, eps)  # amax of h (sh = 1): the pre-cast value, saturation does not hide it
        amax = float(probe.item()) if cache["cfg"][1] else 1.0
        cache["sh"].fill_(2.0 ** math.floor(math.log2(E4M3_MAX / max(amax * HEADROOM, 1e-30))))
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
        if p <= E4M3_MAX or not cache["cfg"][1]:
            return y.reshape(x.shape)
        GUARD_STATS["retries"] += 1
        cache["sh"].div_(2.0 ** math.ceil(math.log2(p / E4M3_MAX * 2.0)))
        refresh(cache)
    GUARD_STATS["fallbacks"] += 1
    return _default(x, gamma, beta, wa, wb, ws, eps)

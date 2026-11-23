"""STRICTLY OPT-IN e4m3 Transition inference forward on the GeForce RTX 5090 (sm_120a), D 128 / 256 (n 2 / 4); docs/fp8.md "RTX 5090", Transition.

NOT part of ``MINIWORLD_FP8=safe`` / ``engine-fp8`` (unlike the H100 route ``fp8_sm90a``, which is in ``safe``): FoldForge F11 traced the
antibody-antigen accuracy loss of FP8 to e4m3 rounding in the trunk pair Transition, so this route runs only when its own switch is set::

    MINIWORLD_TRANSITION_FP8_SM120=calib python run.py        # environment (read at import), or at run time:
    miniworld_engine.kernels.transition.cuda.fp8_sm120.POLICY = "calib"

``fp8.SM120_OPTIN`` lists it: ``fp8.enable()`` re-reads the variable (a preset never sets it), ``fp8.disable()`` turns it off, ``fp8.check()``
reads its guard and ``fp8.route_stats()["transition"]`` / ``fp8.status()["transition"]`` are this module's on an RTX 5090.

The kernel (``sm80/tr_fwd_f8_sm120.cuh``) is the register-resident one-kernel forward of ``fused_fwd_sm80`` (warp-owned rows across the whole
hidden width, weights streaming through a cp.async ring of 32-hidden chunks) with both GEMMs on ``mma.sync.m16n8k32`` e4m3, fp32 accumulation:
xn is quantised in registers straight from the fp32 LayerNorm (scale from the LayerNorm bound: cannot saturate), the weights once per parameter
version (per-tensor power-of-two scales; GEMM1's K permuted so the thread's own x bytes are its A fragments, Ws's K permuted to the C-fragment
order of h = silu(a) b, as ``fp8_sm90a._perm_k``), h quantised in the GEMM1 epilogue with a static calibrated scale (``HEADROOM``) and guarded
like the H100 route (``overflow_peak`` / ``rescale_all``; ``sync``: an eager call that saturated is redone with a lower scale).

Environment (read at import):

=============================================  ======================================================================================
``MINIWORLD_TRANSITION_FP8_SM120=calib``        route bf16 inference calls at D 128 / 256 (n 2 / 4) through the e4m3 kernel (unset: off)
``MINIWORLD_TRANSITION_FP8_SM120_GUARD``        ``sync`` (default, eager: redo an overflowing call) | ``lazy`` (no host sync; CUDA graphs) | ``off``
``MINIWORLD_TRANSITION_FP8_SM120_HEADROOM``     h scale headroom over the calibration amax (16)
``MINIWORLD_TRANSITION_FP8_SM120_MINROWS``      smallest row count routed, every width (default :data:`ROW_GATE`; 0 = no gate)
``MINIWORLD_TRANSITION_FP8_SM120_WIDTHS``       widths routed (``128,256``)
=============================================  ======================================================================================
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
import weakref
from pathlib import Path

import torch

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from ..._sm120_builds import SM120, is_sm120

_dir = Path(__file__).parent / "sm80"

POLICY = os.environ.get("MINIWORLD_TRANSITION_FP8_SM120") or None  # None = off; "calib" = static h scale from the first eager call (NOT a preset variable)
GUARD = os.environ.get("MINIWORLD_TRANSITION_FP8_SM120_GUARD", "sync")
HEADROOM = float(os.environ.get("MINIWORLD_TRANSITION_FP8_SM120_HEADROOM", "16"))
_MINROWS_ENV = os.environ.get("MINIWORLD_TRANSITION_FP8_SM120_MINROWS")
MIN_ROWS: int | None = int(_MINROWS_ENV) if _MINROWS_ENV else None  # replaces ROW_GATE for every width when set (0 = no gate)
#: smallest row count routed per width (below it the default route runs): same-process A-B, docs/fp8.md "RTX 5090" (Transition)
ROW_GATE = {128: 16384, 256: 16384}
WIDTHS = tuple(int(w) for w in os.environ.get("MINIWORLD_TRANSITION_FP8_SM120_WIDTHS", "128,256").split(",") if w)
#: (D, H) with a kernel build (transition_fwd_f8_sm120.cu)
SHAPES = ((128, 256), (128, 512), (256, 512), (256, 1024))
#: rows per CTA tile (8 warps x 16 rows) and the work-list bound of one launch (tr_fwd_f8_kernel: MAX_ITEMS - 2 tiles a CTA)
BM = 128
_MAX_ITEMS = 253

E4M3_MAX = 448.0
GUARD_STATS = {"calls": 0, "retries": 0, "fallbacks": 0, "capture_fallbacks": 0, "calibrations": 0}
_CACHES: dict = {}  # per module (key: data_ptr of expand_a's weight; dropped when that tensor is freed)
_LAZY: dict = {}


@functools.lru_cache(maxsize=1)
def _ext():
    """The sm_120a extension ``transition_fwd_f8_sm120`` (every (D, H) of :data:`SHAPES`). ``MW_TR8_DEFS``: experiment macros only (tagged name)."""
    ensure_cuda_home()
    defs = os.environ.get("MW_TR8_DEFS", "").split()
    tag = "" if not defs else "_" + hashlib.md5(" ".join(defs).encode()).hexdigest()[:6]
    return load_extension(
        name="transition_fwd_f8_sm120" + tag,
        sources=[str(_dir / "transition_fwd_f8_sm120.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", SM120, f"-I{_dir}", "-DMWE_SM120", *defs]
        + (["-Xptxas", "-v"] if os.environ.get("MW_TR8_PTXAS") else []),
        extra_cflags=["-std=c++17", "-O3"],
        verbose=bool(os.environ.get("MW_TR8_PTXAS")),
    )


def serves(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """Does the e4m3 route take this (inference) call? Switch on, bf16 x [..., D] on an RTX 5090, (D, H) in :data:`SHAPES`, D in :data:`WIDTHS`,
    at least the row gate."""
    if POLICY is None or not x.is_cuda or x.dtype is not torch.bfloat16:
        return False
    d, h = x.shape[-1], wa.shape[0]
    if d not in WIDTHS or (d, h) not in SHAPES or wa.shape != (h, d) or ws.shape != (d, h):
        return False
    if not is_sm120(x.device):
        return False
    rows = x.numel() // d
    return rows >= max(ROW_GATE.get(d, 0) if MIN_ROWS is None else MIN_ROWS, 1)


# ------------------------------------------------------------------------------------------------------------------ weight layouts
def _perm_k1(d: int, device) -> torch.Tensor:
    """Input column of each GEMM1 kernel K position: in k32 step i, position p < 16 = 4 q + j <-> column 32 i + 8 q + j, p = 16 + 4 q + j <->
    32 i + 8 q + 4 + j (thread q's own 16 B of x per row are its A fragment words)."""
    p = torch.arange(32)
    base = 8 * ((p % 16) // 4) + p % 4 + 4 * (p // 16)
    return (torch.arange(0, d, 32)[:, None] + base[None, :]).reshape(-1).to(device)


def _perm_k2(k: int, device) -> torch.Tensor:
    """Hidden unit of each GEMM2 kernel K position (the C-fragment order of h, ``fp8_sm90a._perm_k``): position 4 t + q of each 16 reads
    [2t, 2t+1, 8+2t, 9+2t][q]."""
    p16 = []
    for blk in range(2):
        for t in range(4):
            p16 += [16 * blk + c for c in (2 * t, 2 * t + 1, 8 + 2 * t, 9 + 2 * t)]
    base = torch.tensor(p16, dtype=torch.long)
    return (torch.arange(0, k, 32)[:, None] + base[None, :]).reshape(-1).to(device)


def _o_perm(d: int, device) -> torch.Tensor:
    """Output column of each physical GEMM2 column 8 J + 2 q + e: 32 (J / 4) + 8 q + 2 (J % 4) + e (``fused_fwd_sm80._o_perm``)."""
    pcol = torch.arange(d, device=device)
    jj, q, e = pcol // 8, (pcol % 8) // 2, pcol % 2
    return 32 * (jj // 4) + 8 * q + 2 * (jj % 4) + e


def _gb_cols(d: int, device) -> torch.Tensor:
    """Columns of the LayerNorm affine slots (float4 slot s * 4 + q = columns 32 (s / 2) + 8 q + 4 (s % 2) .. + 3; ``fused_fwd_sm80._tables``)."""
    s = torch.arange(d // 16).view(-1, 1, 1)
    q = torch.arange(4).view(1, -1, 1)
    j = torch.arange(4).view(1, 1, -1)
    return (32 * (s // 2) + 8 * q + 4 * (s % 2) + j).reshape(-1).to(device)


def _pow2_floor(v: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.floor(torch.log2(v)))


def _quant(w: torch.Tensor):
    """Per-tensor power-of-two scale s (amax s <= 448) and the e4m3 bits of w s (uint8). Device ops only."""
    s = _pow2_floor(E4M3_MAX / w.abs().amax().clamp_min(1e-30))
    return (w * s).to(torch.float8_e4m3fn).view(torch.uint8), s


def pack_weights(wa, wb, ws):
    """(w [NCHUNK * SLOT] uint8, s1, s2): per 32-hidden chunk [W1: D / 16 k-granules x (Wa 32 | Wb 32 rows) x 16 B | W2: 2 hidden granules x D
    output rows x 16 B], e4m3 with per-tensor scales s1 (Wa, Wb) and s2 (Ws)."""
    h, d = wa.shape
    nch = h // 32
    w1 = torch.stack([wa.float().reshape(nch, 32, d), wb.float().reshape(nch, 32, d)], 1).reshape(nch, 64, d)[:, :, _perm_k1(d, wa.device)]
    q1, s1 = _quant(w1)
    q1 = q1.reshape(nch, 64, d // 16, 16).transpose(1, 2).reshape(nch, -1)
    w2 = ws.float()[_o_perm(d, ws.device)][:, _perm_k2(h, ws.device)]
    q2, s2 = _quant(w2)
    q2 = q2.reshape(d, nch, 2, 16).permute(1, 2, 0, 3).reshape(nch, -1)
    return torch.cat([q1, q2], 1).contiguous().reshape(-1), s1, s2


def _pack(cache, wa, wb, ws, gamma, beta):
    """The packed weights, their scales and the LayerNorm-bound xn scale (recomputed when a parameter changed version)."""
    key = tuple((t.data_ptr(), t._version) for t in (wa, wb, ws, gamma, beta))
    if cache.get("key") == key:
        return
    d = wa.shape[1]
    cache["w"], cache["s1"], cache["s2"] = pack_weights(wa, wb, ws)
    bound = math.sqrt(d - 1.0) * gamma.float().abs().amax() + beta.float().abs().amax()
    cache["sx"] = _pow2_floor(E4M3_MAX / bound.clamp_min(1e-30))
    cols = _gb_cols(d, wa.device)
    cache["gb"] = torch.cat([gamma.float()[cols], beta.float()[cols]]).contiguous()
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
    vals = torch.stack([sx, 0.5 * ca, ca * sh, sh * s2, 1.0 / (sh * s2)])
    cache["scales"][:5].copy_(vals)


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
    """One launch per block of at most ``_MAX_ITEMS`` tiles a CTA (rows are independent)."""
    ext = _ext()
    d, h = x2.shape[1], cache["w"].numel() // (x2.shape[1] * 96) * 32  # SLOT = 96 D bytes a 32-hidden chunk
    limit = _MAX_ITEMS * torch.cuda.get_device_properties(x2.device).multi_processor_count * BM
    if x2.shape[0] <= limit:
        return ext.fwd8(d, h, x2, cache["w"], cache["gb"], cache["scales"], peak, float(eps))
    out = torch.empty_like(x2)
    for r in range(0, x2.shape[0], limit):
        out[r : r + limit] = ext.fwd8(d, h, x2[r : r + limit], cache["w"], cache["gb"], cache["scales"], peak, float(eps))
    return out


def _default(x, gamma, beta, wa, wb, ws, eps):
    """The default route of the call (the guard's and the capture's fallback)."""
    from . import transition_sm80

    return transition_sm80(x, gamma, beta, wa, wb, ws, eps, fp8=False)


def _fake(x, gamma, beta, wa, wb, ws, eps):
    return torch.empty_like(x)


@opaque(fake=_fake, name="transition_fp8_sm120_inference")
def inference(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, eps: float
) -> torch.Tensor:
    """y = x + transition(x) through the e4m3 kernel. Eager or under CUDA-graph capture (calibrate eagerly first: a capture with no
    calibration takes the default route)."""
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
        if p <= E4M3_MAX:
            return y.reshape(x.shape)
        GUARD_STATS["retries"] += 1
        cache["sh"].div_(2.0 ** math.ceil(math.log2(p / E4M3_MAX * 2.0)))
        refresh(cache)
    GUARD_STATS["fallbacks"] += 1
    return _default(x, gamma, beta, wa, wb, ws, eps)

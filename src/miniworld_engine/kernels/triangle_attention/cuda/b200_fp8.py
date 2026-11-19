"""B200 (sm_100a) TriangleAttention FP8 route: e4m3 P and V on tcgen05 ``kind::f8f6f4`` (the H100 FP8 PV route of docs/fp8.md, redone).

OPT-IN, off by default (docs/b200/fp8.md, section "TriAttn FP8 on the B200").  The module of the ``triattn`` slot of ``fp8.B200_ROUTES`` (FP8_REGISTER.md of agent D; the
route works without the entry too): the switches are the H100 module's (``h100_sm90.FP8_PV`` / ``FP8_PV_FUSED`` / ``FP8_PV_UNFUSED`` / ``FP8_PV_D64``, which ``fp8.enable_triattn`` sets from the
model whitelist ``fp8.TRIATTN_MODELS``), read through this module's ``__getattr__``; the guard (``take_overflow`` / ``rescale_all`` / ``overflow_peak``, also registered in the H100 module's
``_OVF`` / ``_VS_AUTO`` / ``_VINV`` so that ``fp8.check()`` sees it), the counters (``STATS`` is the H100 dict) and the e4m3 core (``b200_sources/triattn_f8_sm100.cu``: head dim 32, and the
``-DTF_D=64`` build whose gate ``MIN_S64`` is closed) are this module's.  Everything else (training, bidirectional, fp32, 16-channel heads, e4m3 q / k) keeps the default bf16 route.
"""

from __future__ import annotations

import functools
import hashlib
import os
import weakref

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, load_extension

from . import b200_triattn as bt
from . import h100_sm90 as _hs

L2E = 1.4426950408889634
STATS = _hs.STATS  # one counter dict for both archs: fp8.route_stats()["triattn"]
STATS.setdefault("capture_fallbacks", 0)  # a capture of an uncalibrated site takes the default route (the Transition route has the same key)
#: the switches ``fp8.enable_triattn`` sets on ``h100_sm90`` (the model whitelist decides): this module reads them from there
_DELEGATED = frozenset(
    {"FP8_PV", "FP8_PV_FUSED", "FP8_PV_UNFUSED", "FP8_PV_D64", "FP8_PV_HL", "FP8_VSCALE_AUTO", "FP8_PV_MIN_S", "FP8_PV_FRONT_MIN_S"}
)
#: the e4m3 q / k of the H100 route is NOT built on the B200 (docs/b200/fp8.md): ``fp8.status()`` reports qk False
FP8_QK = False

#: measured gates (sequence length S of the call, padded): below them the bf16 core runs (docs/b200/fp8.md "TriAttn")
MIN_S = int(os.environ.get("MINIWORLD_TRIATTN_B200_FP8_MINS", "1024"))  # head dim 32, every kind: gap x1.008-1.013, sustained x0.96-0.973, energy x0.95-0.971 at L1024 (docs/b200/fp8.md D.3)
#: the 64-channel-head core (built, measured: module x1.04-1.12 at L512-2048, core + pass 1.01-1.25: NOT served): the gate is closed; MINIWORLD_TRIATTN_B200_FP8_MINS64=<S> opens it (experiments)
MIN_S64 = int(os.environ.get("MINIWORLD_TRIATTN_B200_FP8_MINS64", str(1 << 30)))
HEADROOM = 4.0  # V scale: |v| * vinv <= 448 / HEADROOM at the calibration absmax


def __getattr__(name):
    if name in _DELEGATED:
        return getattr(_hs, name)
    raise AttributeError(name)


@functools.lru_cache(maxsize=8)
def _f8(defs: tuple = ()):
    """The e4m3 core build; ``defs``: experiment macros (-DTF_...) of the source, default ()."""
    defs = (*os.environ.get("MW_F8_DEFS", "").split(), *defs)  # experiment macros (-DTF_...) from the environment: never set in production
    ensure_cuda_home()
    tag = "" if not defs else "_" + hashlib.md5(" ".join(defs).encode()).hexdigest()[:6]
    return load_extension(
        name=f"triattn_b200_f8_v2{tag}",
        sources=[str(bt._SRC / "triattn_f8_sm100.cu")],
        extra_cuda_cflags=[*bt._flags(), *defs, *(("-Xptxas", "-v") if os.environ.get("MW_F8_PTXAS") else ())],
        verbose=bool(os.environ.get("MW_F8_PTXAS")),
    )


def core_forward(q, k, v, bias, scale, vinv, guard, out=None, defs=(), vsc=None):
    """e4m3 P / V core: q, k, v [B, N, S, H, 32] bf16, bias [B, H, S, S] bf16 pre-scaled by log2 e (the HB form) -> (out [B, N, S, H, 32], stat).
    vinv: fp32 [H] per-head V scale (v8 = v * vinv, a power of two), guard: int32 [2] ([0] running max of |v vinv| as float bits, [1] saturation flag)."""
    e = _f8(tuple(defs))
    v8t = e.f8_vtq(v, vinv, guard)
    if bias.dtype != torch.float16:
        bias = bias.to(torch.float16)  # exact: bf16(bias * log2 e) has fewer mantissa bits (the HB bias); -inf stays -inf
    if vsc is None:
        vsc = (1.0 / vinv).contiguous()
    return e.triattn_f8_fwd(q, k, v, v8t, bias, scale, vsc, out if out is not None else q.new_empty(0))


# ------------------------------------------------------------------------------------------------------------ route state / guard
class _Site:
    __slots__ = ("vinv", "vsc", "ref", "version", "__weakref__")


_SITES: dict = {}  # (device, site key, heads) -> _Site (per-head static V scale, calibrated on the first eager call)
_GUARDS: dict = {}  # device -> int32 [2]: [0] running max of |v * vinv| (float bits), [1] saturation flag (> 448), written by f8_vtq


def _guard(dev):
    g = _GUARDS.get(dev)
    if g is None:
        g = _GUARDS[dev] = torch.zeros(2, dtype=torch.int32, device=dev)
        # the H100 module's own guard registry: ``h100_sm90.take_overflow`` (what ``fp8.check`` calls while the ``triattn`` slot is not registered) sees the
        # flag, and its ``rescale_all`` widens the scales registered below in place
        _hs._OVF[("b200_fp8", dev)] = g[1:]
    return g


def overflow_peak() -> float:
    """Largest |v * vinv| any call wrote since the last :func:`take_overflow` (e4m3 saturates at 448); synchronizes."""
    return max((float(g[:1].view(torch.float32).item()) for g in _GUARDS.values()), default=0.0)


def take_overflow() -> bool:
    """True when some call saturated a V value since the last call (clears the guards; synchronizes). The caller then runs :func:`rescale_all`
    and redoes the forward (``fp8.check()``)."""
    hit = False
    for g in _GUARDS.values():
        hit |= bool(g[1].item())
        g.zero_()
    return hit


def rescale_all(factor: float = 4.0) -> None:
    """Widen every calibrated V scale by ``factor`` in place (captured graphs read the scale from this memory and follow)."""
    for s in _SITES.values():
        s.vinv.mul_(1.0 / factor)
        s.vsc.copy_(1.0 / s.vinv)


def _drop(key, s):
    """The weight tensor of a site died: forget its scales (unless the key was meanwhile taken by a newer site)."""
    if _SITES.get(key) is s:
        del _SITES[key]
        _hs._VS_AUTO.pop(("b200_fp8", *key), None)
        _hs._VINV.pop(("b200_fp8", *key), None)


def _calibrate(dev, key, v, H, site=None):
    s = _Site()
    s.ref = weakref.ref(site) if site is not None else None
    s.version = site._version if site is not None else 0
    if site is not None:
        weakref.finalize(site, _drop, (dev, key, H), s)
    amax = torch.maximum(v.amax(dim=(0, 1, 2, 4)), -v.amin(dim=(0, 1, 2, 4))).float().clamp_min(1e-12)  # eager only: one sync (no fp32 / abs copy of V: 22 GB at d_pair 384 L3968)
    s.vinv = torch.exp2(torch.floor(torch.log2(448.0 / (HEADROOM * amax)))).float().clamp_max(2.0**60).contiguous()
    s.vsc = (1.0 / s.vinv).contiguous()
    _SITES[(dev, key, H)] = s
    # the H100 module's rescale_all (fp8.check()) multiplies its vscale by the factor and recomputes 1 / vscale in place: ours are registered there too
    _hs._VS_AUTO[("b200_fp8", dev, key, H)] = s.vsc
    _hs._VINV[("b200_fp8", dev, key, H)] = s.vinv
    STATS["calibrations"] += 1
    return s


def route(kind, q, k, v, bias, scale, qs, site):
    """The e4m3 core for one inference call of the head-dim-32 / 64 core (``b200_triattn._core_forward`` with ``kind`` / ``site`` set; ``site`` =
    the weight tensor the V scale belongs to: the V weight of the fused route, the packed weights of the wide route), or None when this call
    is declined (switch off / model not whitelisted for the kind / below the S gate / uncalibrated capture); the caller then runs the bf16
    core. -> out [B, N, S, H, D]. A site is the weight tensor OBJECT: its scale is dropped when the tensor dies, and recalibrated when the
    address is taken by another tensor or the weights were modified in place (``_version``)."""
    site_key = site.data_ptr() if site is not None else None
    hs = _hs
    sw = {"fused": hs.FP8_PV_FUSED, "unfused": hs.FP8_PV_UNFUSED, "d64": hs.FP8_PV_D64}[kind]
    if not hs.FP8_PV or not sw:
        return None
    B, N, S, H, Dh = q.shape
    if S < (MIN_S64 if Dh == 64 else MIN_S) or Dh not in (32, 64) or H > 16 or q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (10, 0):
        return None
    key = (q.device.index, site_key if site_key is not None else id(None), H)
    s = _SITES.get(key)
    if s is not None and site is not None and (s.ref is None or s.ref() is not site or s.version != site._version):
        s = None  # the address now belongs to another tensor (or the weights changed in place): this scale is stale
    if s is None:
        if torch.cuda.is_current_stream_capturing():
            STATS["capture_fallbacks"] = STATS.get("capture_fallbacks", 0) + 1
            return None
        s = _calibrate(q.device.index, site_key if site_key is not None else id(None), v, H, site)
    STATS["pv_fused" if kind == "fused" else "pv_d64" if kind == "d64" else "pv_front"] += 1
    # a call whose q is pre-scaled by scale * log2 e (the QS build) takes scale = 1 / log2 e (the kernel multiplies by log2 e)
    o, _stat = core_forward(q, k, v, bias, (1.0 / L2E) if qs else scale, s.vinv, _guard(q.device.index), vsc=s.vsc, defs=("-DTF_D=64",) if Dh == 64 else ())
    return o

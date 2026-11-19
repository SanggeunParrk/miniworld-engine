"""EXPERIMENTAL, opt-in (never reached by default): the B200 (sm_100a) TriMul inference route with e4m3 planes between the front and the
contraction (bf16 ``master`` modules; the bf16 route is untouched, and bit-identical with ``POLICY`` None).

Chain (the B200 bf16 chain with two kernels swapped)::

    k1w8 (k1w_f8.cu)  front: LN_in + the four projections + gate, plane value * qinv -> e4m3 bytes (cvt.rn.satfinite), max |scaled value| -> guard
    ctr8 (ctr8.cu)    tcgen05 kind::f8f6f4 contraction on the e4m3 planes, fp32 accumulation, the power-of-two scale sa * sb applied in the epilogue
    k3 (unchanged)    the default output kernels (k3g / k3h / k3hf / k3w) read t at its true scale: no K3 change, no folded operands

Quantisation recipe = the H100 route's (``h100_b200style_fp8``): ONLY the planes a / b are e4m3; one power-of-two scale per tensor (a, b; per direction
in the bidirectional block); scales calibrated on the first eager call of a module from the exact amax of the plane values (a probe launch of the front
with qinv = 1), then HEADROOM (16) x and static (cached per module, dropped with its to_left weight); a lazy guard (``overflow_peak`` / ``rescale_all``,
scales and qinv updated IN PLACE so captured graphs follow); a capture before any calibration takes the bf16 route. Differences forced by tcgen05:
qinv / csc are device arrays read at run time (the H100 pre-scales the packed weights instead), and ``sa sb`` is applied in ctr8's epilogue (the H100
folds it into K3's projection operand because cuBLASLt returns the raw sum), so RoseTTAFold3's length factor f (``rf3``) also rides on csc[0].

Served: bf16, one direction or bidirectional, width D 64 / 128 / 256 / 384 / 512 with the bf16 route's hidden widths (``b200_infer.supports``),
any L (L % 16 != 0: padded on the host with masked tokens as the default does; ctr8 tiles clip at L), B up to what one launch takes, mask None / bool [B, L], no live dropout; L >= MIN_L (below it the bf16
route runs unchanged).
"""

from __future__ import annotations

import functools
import math
import os
import weakref
from pathlib import Path

import torch

from miniworld_engine.fp8 import env as _fp8_env

_SRC = Path(__file__).with_name("b200_sources")
E4M3_MAX = 448.0
SCALE_FLOOR = 2.0**-30

POLICY = _fp8_env("MINIWORLD_TRIMUL_FP8") or None  # None = off; "calib" = static scales (probe on the first eager call); "dynamic" = exact amax every call
GUARD = os.environ.get("MINIWORLD_TRIMUL_FP8_GUARD", "sync")  # "sync" (eager: redo an overflowing call) | "lazy" (capturable) | "off"; under capture always lazy
HEADROOM = float(os.environ.get("MINIWORLD_TRIMUL_FP8_HEADROOM", "16"))
_MINL_ENV = os.environ.get("MINIWORLD_TRIMUL_FP8_MINL")
WIDTHS = (64, 128, 256, 384, 512)
#: smallest L the e4m3 chain takes, per (width, direction: 1 outgoing / 2 incoming / 0 bidirectional); below it the bf16 route runs unchanged
#: (B200 measurements: docs/b200/fp8.md). ``MINIWORLD_TRIMUL_FP8_MINL=<n>`` replaces every gate (0 = none).
MIN_L: dict = {(64, 1): 512, (64, 2): 512, (64, 0): 384, (128, 0): 512, (128, 1): 512, (128, 2): 512, (256, 1): 384, (256, 2): 384, (384, 0): 512, (384, 1): 512, (384, 2): 512, (512, 0): 128, (512, 1): 128, (512, 2): 128}  # (width, direction); other cells: MIN_L_DEFAULT. Served only where the gap-measured FP8 / bf16 ratio is <= 0.99 (docs/b200/fp8.md)
MIN_L_DEFAULT = 256
GUARD_STATS = {"calls": 0, "retries": 0, "fallbacks": 0, "capture_fallbacks": 0, "calibrations": 0}
_CACHES: dict = {}   # calibration state per module (key: data_ptr of its to_left weight; dropped when that tensor is freed)
_PEAK: dict = {}     # per device: int32 [8] -- max scaled plane value (fp32 bits) since the last read (the lazy guard)

#: extra nvcc defines of the extension (part of its cache key)
DEFS: tuple[str, ...] = ()


@functools.lru_cache(maxsize=1)
def _ext8():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="trimul_b200_fp8_v1",
        sources=[str(_SRC / f) for f in ("ctr8.cu", "k1w_f8.cu", "bind_fp8.cpp")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("100a"), "--expt-relaxed-constexpr", "-DNDEBUG", f"-I{_SRC}", *DEFS],
        extra_cflags=["-std=c++17", f"-I{_SRC}"],
        verbose=False,
    )


# ------------------------------------------------------------------------------------------------------------------------ scales / guard
def pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    """Smallest 2^e with amax / 2^e <= 448 (1 where the tensor is all zero), never below 2^-30 (the H100 route's floor: a ~0 amax must not give a
    scale that saturates a later input)."""
    s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-30) / E4M3_MAX)))
    return torch.where(amax > 0, s.clamp_min(SCALE_FLOOR), torch.ones_like(s))


def _peak_buffer(device) -> torch.Tensor:
    if device not in _PEAK:
        _PEAK[device] = torch.zeros(8, dtype=torch.int32, device=device)
    return _PEAK[device]


def overflow_peak(reset: bool = True) -> float:
    """Largest scaled plane value since the last read (> 448: some call saturated the e4m3 cast). One host sync: read it once per forward, outside any graph."""
    out = 0.0
    for buf in _PEAK.values():
        out = max(out, float(buf.view(torch.float32).max().item()))
        if reset:
            buf.zero_()
    return out


def _derive(st: dict) -> None:
    """qinv [P / 64] and csc [2] from the group scales s [ng] (in place: a captured graph reads these tensors at replay)."""
    s, ng = st["s"], st["ng"]
    per = st["chunks"] // ng
    st["qinv"].copy_((1.0 / s).repeat_interleave(per))
    if ng == 4:
        st["csc"].copy_(torch.stack([s[0] * s[2], s[1] * s[3]]))
    else:
        st["csc"].copy_(torch.stack([s[0] * s[1], s[0] * s[1]]))


def refresh(cache: dict) -> None:
    """After ``s`` of a calibrated module changed in place: rebuild its scale-dependent tensors in place."""
    _derive(cache)


def _scale_up(cache: dict, f: float) -> None:
    cache["s"].mul_(f)


def rescale_all(peak: float) -> float:
    """Raise every calibrated module's scales so a forward whose guard read ``peak`` (> 448) fits with room x2; returns the factor. Call outside the
    graph, then redo the forward."""
    f = 2.0 ** math.ceil(math.log2(peak / E4M3_MAX * 2.0))
    for cache in _CACHES.values():
        _scale_up(cache, f)
        refresh(cache)
    return f


# ------------------------------------------------------------------------------------------------------------------------ the chain
def _front8(E8, rf3_unused, x2, wl, wlg, wr, wrg, mask, mean, rstd, gi, bi, planes, grid, eps, qinv, amax, ng):
    return E8.k1w8_forward(x2, wl, wlg, wr, wrg, mask, mean, rstd, gi, bi, planes, grid, eps, qinv, amax, ng)


def _contract8(E8, planes, d, direction, csc, trigger=True):
    n = planes.shape[-1]
    t = planes.new_empty(((2 if direction == 0 else 1) * d, *planes.shape[1:]), dtype=torch.bfloat16)
    E8.ctr8_forward(planes, t, csc, d, direction, 0, 5 if trigger == 2 else (3 if trigger else 1), 0)
    return t


def _body(leaves, mask, direction, st, *, qinv, csc, amax, probe=False):
    """The inference body on e4m3 planes (b200_infer._inference_body's cell logic with k1w8 / ctr8): -> y; probe: the front only, planes discarded."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_infer as BI

    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves[:11]
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    E, E8 = BI._ext(), _ext8()
    EPS = BI.EPS
    x2 = x.reshape(m, d)
    hid = BI.hidden_of(wl.shape[0], direction)
    p = BI.planes_of(d, hid, direction)
    ng = 4 if direction == 0 else 2
    planes = torch.empty((p, bsz, n, n), dtype=torch.uint8, device=x.device)
    y = None if probe else torch.empty_like(x)
    if d <= 128:
        if len(leaves) > 11:  # kernel_forms: row-major front matrices and wpp made once
            wpp, front = leaves[12], (wl, wlg, wr, wrg)
        else:
            wpp = x.new_empty((d, p // 2))
            front = BI._front(E, wl, wlg, wr, wrg, wp, wpp)
        cell = (n, direction == 0, d, hid)
        if BI.K3W_D128 and d == 128 and hid == d and len(leaves) >= 19 and (n, direction == 0) in BI.K3W_D128_CELLS:
            mean = x.new_empty((m,), dtype=torch.float32)
            rstd = torch.empty_like(mean)
            wf, cr = leaves[-2:]
            if cr.numel() and BI.fold_cell(n, d, direction):
                _front8(E8, 0, x2, wf, cr, wf, wf, mask, mean, rstd, gi, bi, planes, -2, EPS, qinv, amax, ng)
            else:
                _front8(E8, 0, x2, *front, mask, mean, rstd, gi, bi, planes, -2, EPS, qinv, amax, ng)
            if probe:
                return None
            t = _contract8(E8, planes, BI.contracted(d, hid, direction), direction, csc, trigger=2 if BI.XF and n <= BI.XF_MAXL else False)
            del planes
            wpq, wgq, vec = leaves[-5:-2]
            grp = BI.K3W_OST_GROUP and n * n // 128 <= BI._num_sms(x.device.index)
            E.k3w_forward(x2, t.view(t.shape[0], m), wpq, wgq, vec, mean, mean, mean, rstd, None, y.view(m, d), n, 0, None, None,
                          int(BI.K3W_OST_PDL) | 2 | (8 if grp else 0) | (16 if BI.XF and n <= BI.XF_MAXL and BI.K3W_OST_PDL else 0))  # fmt: skip
            return y
        f64 = d == 64 and len(leaves) >= 19
        f3 = f64 and BI.K3H_FOLD and BI.k3hf_ok(cell)
        f1 = f64 and BI.K1W_FOLD64 and cell in BI.K1W_FOLD64_CELLS and leaves[-1].numel() > 0
        m64o = f3 or (cell in BI.M64_OUT and wp.dtype == torch.bfloat16 and wp.is_contiguous() and wg.is_contiguous())
        if f1:
            mean = x.new_empty((m,), dtype=torch.float32)
            _front8(E8, 0, x2, leaves[-2], leaves[-1], leaves[-2], leaves[-2], mask, mean, torch.empty_like(mean), gi, bi, planes, -2, EPS, qinv, amax, ng)
        else:
            _front8(E8, 0, x2, *front, mask, None, None, gi, bi, planes, -2, EPS, qinv, amax, ng)
        if probe:
            return None
        t = _contract8(E8, planes, BI.contracted(d, hid, direction), direction, csc)
        del planes
        if f3:
            E.k3hf_forward(x2, t, *leaves[-5:-2], y.view(m, d), EPS, 0, 1)
        elif m64o:
            E.k3h_forward(x2, t, wp, wg, gi, bi, go, bo, y.view(m, d), EPS, 0, 1)
        else:
            E.k3g_forward(x2, t, wpp, wg, gi, bi, go, bo, y.view(m, d), n, EPS, 0, None, None, None, None, 0)
        return y
    mean = x.new_empty((m,), dtype=torch.float32)
    rstd = torch.empty_like(mean)
    if len(leaves) > 16 and BI.fold_cell(n, d, direction):  # the forms' LN_in-folded front
        wf, cr = leaves[15], leaves[16]
        _front8(E8, 0, x2, wf, cr, wf, wf, mask, mean, rstd, gi, bi, planes, -1, EPS, qinv, amax, ng)
    else:
        if d > BI.K1W_INSTAT:
            E.k1w_stats(x2, mean, rstd, EPS)
        front = (leaves[14],) * 4 if BI.PACK and len(leaves) > 14 else BI._front(E, wl, wlg, wr, wrg)
        _front8(E8, 0, x2, *front, mask, mean, rstd, gi, bi, planes, -1, EPS, qinv, amax, ng)
    if probe:
        return None
    grp = BI.K3W_OST and BI.K3W_OST_GROUP and n * n // 128 <= BI._num_sms(x.device.index)
    ost = (BI.K3W_OST and n >= BI.K3W_OST_MINL) or grp
    xf = BI.XF and BI.XF_WIDE and n <= BI.XF_MAXL and ost and BI.K3W_OST_PDL
    t = _contract8(E8, planes, d, direction, csc, trigger=2 if xf else False)
    del planes
    if ost:
        mo = ro = mean
    else:
        mo = torch.empty_like(mean)
        ro = torch.empty_like(mean)
        E.wide_ln_stats(t, mo, ro, EPS)
    if len(leaves) > 11:
        wpq, wgq, vec = leaves[11:14]
    else:
        wpq = torch.empty_like(wp, dtype=torch.bfloat16)
        wgq = torch.empty_like(wg, dtype=torch.bfloat16)
        vec = x.new_empty((4, d), dtype=torch.float32)
        E.wide_fold_prep(wp, go, bo, wg, gi, bi, wpq, wgq, vec)
    pdl = int(BI.K3W_OST_PDL) if ost else int(len(leaves) > 11)
    flags = pdl | (2 if ost else 0) | (8 if grp else 0) | (16 if xf else 0)
    E.k3w_forward(x2, t.view(t.shape[0], m), wpq, wgq, vec, mo, ro, mean, rstd, None, y.view(m, d), n, 0, None, None, flags)
    return y


# ------------------------------------------------------------------------------------------------------------------------ route
def _calibrate(leaves, mask, direction, dev, margin) -> dict:
    """Scales from the exact amax of the plane values (one probe launch of the front with qinv = 1): the state dict of this module."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_infer as BI

    wl = leaves[1]
    hid = BI.hidden_of(wl.shape[0], direction)
    p = BI.planes_of(leaves[0].shape[-1], hid, direction)
    ng = 4 if direction == 0 else 2
    chunks = p // 64
    one = torch.ones(chunks, dtype=torch.float32, device=dev)
    tmp = torch.zeros(8, dtype=torch.int32, device=dev)
    _body(leaves, mask, direction, None, qinv=one, csc=None, amax=tmp, probe=True)
    am = tmp.view(torch.float32)[:ng].clone()
    s = pow2_scale(am) * margin
    st = {"s": s, "ng": ng, "chunks": chunks, "qinv": torch.empty(chunks, dtype=torch.float32, device=dev), "csc": torch.empty(2, dtype=torch.float32, device=dev)}
    _derive(st)
    GUARD_STATS["calibrations"] += 1
    return st


def _lfac(mask, n, bsz, dev):
    """RoseTTAFold3's length factor f = bf16(1 / max(n_real, 1)) as a device scalar (graph-safe: read from the mask at run time)."""
    if mask is None:
        return torch.tensor(float(torch.tensor(1.0 / max(n, 1)).bfloat16().float()), device=dev)
    cnt = mask.reshape(bsz, n).sum().to(torch.float32).clamp_min(1.0)
    return (1.0 / cnt).bfloat16().float()


def inference(leaves: list, mask, direction: int, key, rf3: bool = False):
    """y of the e4m3-plane chain, or None (the caller runs the bf16 route): ``leaves`` as ``b200_infer.inference`` (x, forms...); ``mask``: bool [B * L] or None;
    ``key``: the calibration key (None: no cache -- ``dynamic``)."""
    dev = leaves[0].device
    capturing = torch.cuda.is_current_stream_capturing()
    cache = _CACHES.get(key) if key is not None else None
    if cache is None:
        if capturing:  # no static calibration yet: never calibrate inside a graph
            GUARD_STATS["capture_fallbacks"] += 1
            return None
        cache = _calibrate(leaves, mask, direction, dev, HEADROOM if POLICY != "dynamic" else 1.0)
        if key is not None and POLICY != "dynamic":
            _CACHES[key] = cache
    elif POLICY == "dynamic":
        cache = _calibrate(leaves, mask, direction, dev, 1.0)
    csc = cache["csc"]
    if rf3:
        f = _lfac(mask, leaves[0].shape[1], leaves[0].shape[0], dev).reshape(())
        csc = torch.stack([csc[0] * f, csc[1]])
    lazy = GUARD in ("lazy", "off") or capturing
    peak_buf = _peak_buffer(dev)
    for _ in range(4):
        if not lazy:
            peak_buf.zero_()
        y = _body(leaves, mask, direction, cache, qinv=cache["qinv"], csc=csc, amax=peak_buf)
        GUARD_STATS["calls"] += 1
        if lazy or POLICY == "dynamic":
            return y
        peak = float(peak_buf.view(torch.float32).max().item())
        if peak <= E4M3_MAX:
            return y
        GUARD_STATS["retries"] += 1
        f = 2.0 ** math.ceil(math.log2(peak / E4M3_MAX * 2.0))
        _scale_up(cache, f)
        refresh(cache)
        csc = cache["csc"] if not rf3 else torch.stack([cache["csc"][0] * _lfac(mask, leaves[0].shape[1], leaves[0].shape[0], dev).reshape(()), cache["csc"][1]])
    GUARD_STATS["fallbacks"] += 1
    return None


def serves(module, pair, mask, bidirectional, rf3: bool = False) -> bool:
    """Python-level test: does the e4m3-plane chain take this inference call?"""
    if POLICY is None or pair.ndim != 4 or pair.shape[1] != pair.shape[2] or pair.dtype != torch.bfloat16 or not pair.is_cuda:
        return False
    if torch.cuda.get_device_capability(pair.device) != (10, 0) or not pair.is_contiguous():
        return False
    bsz, n, d = pair.shape[0], pair.shape[1], pair.shape[-1]
    if d not in WIDTHS or bsz < 1:
        return False
    if module.ln_pair.eps != 1e-5 or module.ln_out.eps != 1e-5:
        return False
    if mask is not None and (mask.dtype != torch.bool or mask.numel() != bsz * n):
        return False
    if rf3 and (d > 128 or bsz != 1 or os.environ.get("TRIMUL_B200_RF3", "1") != "1"):
        return False
    from miniworld_engine.kernels.integrations import trimul_b200 as TB
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_infer as BI

    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    pad128 = (n + 127) // 128 * 128  # ctr8 tiles are 128 wide: a length far from a multiple of 128 wastes the last tile row / column (L400: +5 % slower than the bf16 route)
    if n / pad128 < 0.9:
        return False
    npad = TB._padded_length(n)  # L % 16 != 0: padded on the host with masked tokens (the default route's ``_padded_inference``)
    if not BI.supports(d, npad, False, module.d_hidden, direction) or bsz * npad * npad >= 2**31:
        return False
    if TB._plan(module, pair, True, bidirectional)[0] < bsz:  # a call the default splits into several launches: not measured here
        return False
    gate = MIN_L.get((d, direction), MIN_L_DEFAULT)
    return n >= (int(_MINL_ENV) if _MINL_ENV is not None else gate)


def module_route(module, pair, mask, bidirectional, rf3: bool = False):
    """Eager entry (the module hook in ``integrations.trimul_b200.forward``): y of the e4m3-plane chain or None."""
    if not serves(module, pair, mask, bidirectional, rf3):
        return None
    from miniworld_engine.kernels.integrations import trimul_b200 as TB

    bsz, n = pair.shape[0], pair.shape[1]
    npad = TB._padded_length(n)
    if npad != n:  # host pad: zero pair entries, pad tokens masked (every real pair contracts the same terms as without the pad)
        x = pair.new_zeros((bsz, npad, npad, pair.shape[-1]))
        x[:, :n, :n] = pair
        token = torch.zeros((bsz, npad), device=pair.device, dtype=torch.bool)
        token[:, :n] = True if mask is None else (mask.reshape(bsz, n) != 0)
        y = _route_core(module, x, token.reshape(-1), bidirectional, rf3)
        return None if y is None else y[:, :n, :n].contiguous()
    token_mask = None if mask is None else mask.reshape(bsz * n).to(torch.bool).contiguous()
    return _route_core(module, pair, token_mask, bidirectional, rf3)


def _route_core(module, pair, token_mask, bidirectional, rf3):
    from miniworld_engine.kernels.integrations import trimul_b200 as TB

    params = TB._inference_params(module, pair.dtype, None, rf3_inline=bool(rf3))
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    wl = module.to_left.weight
    key = wl.data_ptr()
    if POLICY != "dynamic" and key not in _CACHES and not torch.cuda.is_current_stream_capturing():
        weakref.finalize(wl, _CACHES.pop, key, None)  # drop the state when the weight is freed (a later module at the same address calibrates its own)
    y = inference([pair, *params], token_mask, direction, key if POLICY != "dynamic" else None, rf3)
    return y

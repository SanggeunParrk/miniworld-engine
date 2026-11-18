"""EXPERIMENTAL, opt-in (never reached by default): the one-direction, one-sample TriMul inference route with the planes between K1 and the
contraction stored as e4m3 (``_h100_fp8``; the bf16 route is untouched).

  K1 (``h100_b200style_direct.front(qinv=...)``): the n128 blocks, projection weights pre-scaled per plane channel by a power of two, planes written
  as e4m3 bytes -> cuBLASLt e4m3 batched contraction (bf16 out = the raw sum of the quantised products) -> K3 (``k3.cu`` ``K3_CSCALE``): the
  channel scale ``sa[c] sb[c]`` folded into the projection operand (exact, power of two) and into the LN_out statistics.
"""

import torch

from miniworld_engine.kernels._device import num_sms
from . import _h100_fp8 as F
from . import _h100_runtime as T
from . import h100_b200style as K
from . import h100_uni_wide_inference as I


def dynamic_scales(x, weights, token, gi, bi, h):
    """Per-channel power-of-two scales from the exact amax of the bf16 planes (a bf16 K1 pre-pass: numerics only, not a speed route)."""
    from .h100_b200style_direct import front

    n = x.shape[1]
    npad = -(-n // 16) * 16
    ab = x.new_empty((2 * h, npad, npad))
    xs = x.new_empty((n * n, 2), dtype=torch.float32)
    front(x, weights, token, gi, bi, ab, xs)
    am = F.channel_amax(ab)
    if (
        SCALE_MODE == "tensor"
    ):  # one scale per tensor (a, b): pow2 rounding leaves the e4m3 error unchanged, so a channel-uniform scale costs nothing in accuracy
        am = torch.cat([am[:h].max().expand(h), am[h:].max().expand(h)])
    s = F.pow2_scale(am)
    return s[:h].contiguous(), s[h:].contiguous()


import os as _os0

SCALE_MODE = _os0.environ.get(
    "MINIWORLD_TRIMUL_FP8_SCALE", "tensor"
)  # "tensor" (scalar a / b scales, K3_CSCALE=2) or "channel" (K3_CSCALE=1)
K1_F8C = (
    _os0.environ.get("MINIWORLD_TRIMUL_FP8_K1") == "1"
)  # EXPERIMENTAL: K1 computes its projections in e4m3 too (needs SCALE_MODE tensor)
LAST_AMAX = None  # device scalar of the last call: max |scaled plane value| (> 448 = the e4m3 cast saturated)


def k3_flags(d, flat, bidirectional=False):
    """extra nvcc flags of the FP8 route K3 (the any-L row form for D128; the token-flat D256 K3 takes none). Bidirectional: the channel
    scales differ per direction, so the per-channel form (K3_CSCALE=1) whatever the scale mode."""
    extra = tuple(K.K3_EXTRA.get(d, ()))
    return (f"-DK3_CSCALE={2 if SCALE_MODE == 'tensor' and not bidirectional else 1}",) + extra


def _planes(x, weights, token, gi, bi, xs, d, transposed, sub, scales, margin, amax, record, npad, bidirectional=False):
    """K1 of one direction (hidden = d): the e4m3 planes [2 d, npad, npad] and the channel scale product csc [d]. sub: the calibration state
    of this direction (static scales, prescaled weights) or None."""
    from .h100_b200style_direct import front

    if sub is not None and "sa" in sub:  # static scales (calibrated once): no bf16 pre-pass, prescaled weights and the scale vectors reused
        sa, sb, qinv, csc = sub["sa"], sub["sb"], sub["qinv"], sub["csc"]
    else:
        sa, sb = scales if scales is not None else dynamic_scales(x, weights, token, gi, bi, d)
        if margin != 1.0:  # headroom against inputs larger than the calibration input (powers of two: exact)
            sa, sb = sa * margin, sb * margin
        qinv, csc = (1.0 / torch.cat([sa, sb])).contiguous(), (sa * sb).contiguous()
        if sub is not None:
            sub.update(sa=sa, sb=sb, qinv=qinv, csc=csc)
    if record is not None:
        record.append((sa.cpu(), sb.cpu()))
    abq = torch.empty((2 * d, npad, npad), dtype=torch.float8_e4m3fn, device=x.device)
    if (
        d == 512
    ):  # the packed shared-resident K1 (the bf16 route's D512 front) with the e4m3 plane store: weights pre-scaled and packed once per calibration
        from ._h100_pack import owned_bf16_mask
        from .h100_b200style_ss_regtune import front_f8, prescale_pack

        if sub is not None and "w512" in sub:
            w1 = sub["w512"]
        else:
            w1 = prescale_pack(weights, qinv)
            if sub is not None:
                sub["w512"], sub["w512_src"] = w1, weights
        front_f8(x, w1, owned_bf16_mask(token, x.shape[1], x.device), gi, bi, abq, xs, amax, transposed)
        return abq, csc
    f8c = None
    if K1_F8C and not bidirectional:  # (the FP8-compute K1 is a single-direction experiment)
        from .h100_b200style_direct import f8c_prepare

        if sub is not None and "f8c" in sub:
            f8c = sub["f8c"]
        else:
            f8c = f8c_prepare(weights, gi, bi, qinv, d, d)
            if sub is not None:
                sub["f8c"] = f8c
    front(x, weights, token, gi, bi, abq, xs, qinv=qinv, transposed=transposed, wcache=sub, amax=amax, f8c=f8c)
    return abq, csc


def run(leaves, mask, direction, scales=None, record=None, cache=None, margin=1.0, zero=True, out=None, lfac=None):
    """leaves as ``h100_b200style.trimul`` (bf16 weights); direction 1 outgoing / 2 incoming / 0 both (the bidirectional block: first d hidden
    channels outgoing, second d incoming, one K3 over the 2 d channels); scales = (sa, sb) [h] fp32 powers of two or None (dynamic). Returns y.
    Bidirectional: the two directions are two single-direction K1 + e4m3 contraction launches on the matrix halves (the incoming one on the
    transposed pair), both writing their channels of one ``tri``; ``cache`` holds one calibration state per direction (``cache["bd"]``).
    lfac: a device fp32 scalar [1] (RoseTTAFold3's ``1 / n``) that multiplies the contraction before ``ln_out``: it is folded into the channel scale
    ``csc`` (K3 reads ``csc`` for the LN_out statistics and the host folds it into the projection operand), so the planes, the calibrated
    scales and the cached prescaled weights are those of the plain module (one calibration whatever the length factor)."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    bsz, n, d, h = x.shape[0], x.shape[1], x.shape[-1], wp.shape[1]
    assert bsz == 1 and direction in (0, 1, 2) and h == d * (2 if direction == 0 else 1)
    npad = -(-n // 16) * 16
    with T.native_context(x.device):
        xs = x.new_empty((n * n, 2), dtype=torch.float32)
        vec = x.new_empty((4, d), dtype=torch.float32)
        token = (
            torch.ones(n, device=x.device, dtype=torch.bool)
            if mask is None or mask.numel() == 0
            else mask.reshape(n).to(torch.bool).contiguous()
        )
        global LAST_AMAX
        amax = cache.get("amax") if cache is not None else None
        if amax is None:  # (not ``setdefault(.., torch.zeros(1))``: the argument would allocate and fill a tensor on every call)
            amax = torch.zeros(1, device=x.device)
            if cache is not None:
                cache["amax"] = amax
        if zero:
            amax.zero_()
        LAST_AMAX = amax
        tri = torch.empty((h, npad, npad), dtype=torch.bfloat16, device=x.device)
        cscs = []
        # incoming: K1 runs on the transposed pair (planes come out transposed = K-major for the contraction) and stores the LN stats at the
        # swapped token (K1Params.stats_t), which is where K3 reads them: no host transpose of xs (the bidirectional outgoing K1 stores the
        # same statistics at the natural token)
        for k, dr in enumerate((1, 2) if direction == 0 else (direction,)):
            if direction == 0:
                sl = slice(0, d) if dr == 1 else slice(d, 2 * d)
                weights = [w[sl].contiguous() for w in (wl, wlg, wr, wrg)]
                sub = cache.setdefault("bd", {}).setdefault(dr, {}) if cache is not None else None
                sc = None if scales is None else (scales[0][sl], scales[1][sl])
            else:
                weights = [w.contiguous() for w in (wl, wlg, wr, wrg)]
                sub, sc = cache, scales
            abq, csc = _planes(x, weights, token, gi, bi, xs, d, dr == 2, sub, sc, margin, amax, record, npad, direction == 0)
            F.contract_e4m3(abq[:d], abq[d:], tri[k * d : (k + 1) * d] if direction == 0 else tri)
            del abq
            cscs.append(csc)
        if direction == 0:
            if cache is not None and "csc" in cache:
                csc = cache["csc"]
            else:
                csc = torch.cat(cscs).contiguous()
                if cache is not None:
                    cache["csc"] = csc
        else:
            csc = cscs[0]
        wpf0 = None
        if cache is not None and "k3" in cache:
            wpf, wgf, vec = cache["k3"]
            wpf0 = cache["wpf0"]
        else:
            wpf, wgf = torch.empty_like(wp), torch.empty_like(wg)
            K._aux().fold(wp, go, bo, wg, gi, bi, wpf, wgf, vec)
            wpf0 = wpf.clone() if cache is not None or lfac is not None else None
            if cache is not None:
                cache["wpf0"] = wpf0
            wpf = (wpf.float() * csc[None, :]).to(wpf.dtype)
            if cache is not None:
                cache["k3"] = (wpf, wgf, vec)
        if lfac is not None:  # the length factor rides on the channel scale (never into the cached, calibrated state)
            csc = csc * lfac.reshape(1)
            wpf = (wpf0.float() * csc[None, :]).to(wpf.dtype)
        y = torch.empty_like(x) if out is None else out
        cfg = K._inference_output_config(d)
        from .h100_b200style_static import output_kernel

        # the token-flat K3 (the shipped D256 / D384 inference tile) where it applies, else the any-L row form
        flat = d in (256, 384, 512) and n % 128 == 0  # D512: the token-flat form only (k3reorder.cu has no row form)
        kernel, smem = output_kernel(d, h, cfg, rowpad=not flat, xbuf=K.K3_XBUF.get(d, 1), extra=k3_flags(d, flat, direction == 0))
        bn = cfg[0]
        if flat:
            m = n * n
            kernel.launch(
                (min(m // 128, num_sms(x.device.index)), 1, 1),
                (384, 1, 1),
                [
                    T._launch_module().Struct(
                        [
                            I._map(tri, [64, 64], [m, h], [m * 2]),
                            I._map(x, [64, 128], [d, m], [d * 2]),
                            I._map(wpf, [64, bn], [h, d], [h * 2]),
                            I._map(wgf, [64, bn], [d, d], [d * 2]),
                            I._map(x, [64, 64], [d, m], [d * 2]),
                            I._map(y, [64, 64], [d, m], [d * 2]),
                            vec,
                            xs,
                            csc,
                            m,
                            m // 128,
                            None,
                            None,
                            None,
                            None,
                            None,
                            n,
                        ]
                    )
                ],
                smem,
            )
            return y
        cj = -(-n // 64)
        tiles = -(-(n * cj) // 2)
        fields = [
            I._map(tri, [64, 1, 64], [npad, npad, h], [npad * 2, npad * npad * 2]),
            I._map(x, [64, 64, 1], [d, n, n], [d * 2, n * d * 2]),
            I._map(wpf, [64, bn], [h, d], [h * 2]),
            I._map(wgf, [64, bn], [d, d], [d * 2]),
            I._map(x, [64, 64, 1], [d, n, n], [d * 2, n * d * 2]),
            I._map(y, [64, 64, 1], [d, n, n], [d * 2, n * d * 2]),
            vec,
            xs,
            csc,
            n,
            tiles,
            None,
            None,
            None,
            None,
            None,
            n,
            npad,
            cj,
        ]
        kernel.launch((min(tiles, num_sms(x.device.index)), 1, 1), (384, 1, 1), [T._launch_module().Struct(fields)], smem)
        return y


# ---------------------------------------------------------------------------------------------------------------------- module-level opt-in
import math
import os as _os
import weakref

from miniworld_engine.fp8 import env as _fp8_env
from miniworld_engine.kernels._compile import opaque

POLICY = (
    _fp8_env("MINIWORLD_TRIMUL_FP8") or None
)  # None = off (every route is the bf16 one); "dynamic" = exact amax of the bf16 planes each call (numerics only); "calib" = static scales
STATS: list = []  # (sa, sb, n, direction) per call when ``RECORD``
RECORD = bool(_os.environ.get("MINIWORLD_TRIMUL_FP8_RECORD"))
GUARD = _os.environ.get(
    "MINIWORLD_TRIMUL_FP8_GUARD", "sync"
)  # "sync" (eager only) | "lazy" (no host sync: capturable) | "off". Under CUDA-graph capture the guard is always lazy.
HEADROOM = float(_os.environ.get("MINIWORLD_TRIMUL_FP8_HEADROOM", "16"))
#: smallest L the FP8 planes take, per (width, direction: 1 outgoing / 2 incoming); below it the bf16 default route runs unchanged. Measured on H100 (CUDA-graph replay of the module,
#: same-cubin control): D128 outgoing loses 1-3 % up to L128 and wins from L144 (-5 %), D128 incoming loses 0-7 % up to L176 and wins from L192 (-8 %); D256 shows no GPU gain below
#: L224 outgoing (+-3 %, and +15-20 us of host launch work against the default in eager mode; -6 % from L224), incoming from L144 (flat up to L128, -10 % at L144-176 where the default incoming is slower than its
#: outgoing, flat at L192, -7 % from L224). ``MINIWORLD_TRIMUL_FP8_MINL=<n>`` replaces every gate (0 = none).
#: direction 0 = bidirectional: D128 graph replay 0.95 at L256, 1.00-1.04 at L144-224; D256 1.01-1.04 up to L768, 0.975 at L1024
#: D64 / D384 (n64 K1 blocks, 3.0.0 exploration): D64 +13.5 % at L128, -2 % at L256 (control 1.00: not clear), -10 % from L512; D384 -4 .. -8 % from L128 (graph replay, cool);
#: bidirectional D64 +4 % at L512, L1024 -7 % cool but +8 % sustained, -22 % at L2048; bidirectional D384 -3 % at L512 (not clear), -8 % at L1024
MIN_L = {(128, 1): 144, (128, 2): 192, (256, 1): 224, (256, 2): 144, (128, 0): 256, (256, 0): 1024}
MIN_L.update({(64, 1): 512, (64, 2): 512, (384, 1): 128, (384, 2): 128, (64, 0): 2048, (384, 0): 1024})
#: D512 (IntelliFold2 trunk; one direction, L % 128 == 0, packed shared-resident K1 with the e4m3 plane store + k3reorder K3_CSCALE=2): every length gains (graph replay, FP8 / default 0.76 at L128, 0.87 at L256, 0.89-0.94 at L384-L2048, control 1.00): no gate
MIN_L.update({(512, 1): 0, (512, 2): 0})
#: D512 (IntelliFold2 trunk) is served only inside what was measured: B = 1, L <= MAX_L_512; MINIWORLD_TRIMUL_FP8_D512=0 switches the D512 route off (every other width unchanged)
MAX_L_512 = 2048
D512 = _os.environ.get("MINIWORLD_TRIMUL_FP8_D512", "1") != "0"
#: widths the route serves: D128 / D256 (n128 K1 blocks) and, from the 3.0.0 exploration (branch fp8x/trimul), D64 / D384 (n64 K1 blocks), D512 (the packed K1)
WIDTHS = (64, 128, 256, 384, 512)
#: samples per call the route takes (B > 1 runs the one-sample route per sample into one output; exploration, branch fp8x/trimul)
MAX_SAMPLES = 16
_MINL_ENV = _os.environ.get("MINIWORLD_TRIMUL_FP8_MINL")
GUARD_STATS = {"calls": 0, "retries": 0, "fallbacks": 0, "capture_fallbacks": 0}
_CACHES: dict = {}  # calibration state per module (key: data_ptr of its to_left weight; dropped when that tensor is freed): static device tensors, so a captured graph reads them at replay
_LAZY: dict = {}


def _lazy_buffer(device):
    if device not in _LAZY:
        _LAZY[device] = torch.zeros(1, device=device)
    return _LAZY[device]


def overflow_peak(reset: bool = True) -> float:
    """Largest scaled plane value since the last read (> 448: some call saturated the e4m3 cast). One host sync: read it once per forward, outside any graph."""
    out = 0.0
    for buf in _LAZY.values():
        out = max(out, float(buf.item()))
        if reset:
            buf.zero_()
    return out


def refresh(cache) -> None:
    """After ``sa`` / ``sb`` of a calibrated module changed in place: rebuild its scale-dependent tensors in place (a captured graph keeps reading them).
    A bidirectional module (``cache["bd"]``): the same per direction, then the joint channel scale vector."""
    if "bd" in cache:
        for sub in cache["bd"].values():
            refresh(sub)
        cache["csc"].copy_(torch.cat([cache["bd"][1]["csc"], cache["bd"][2]["csc"]]))
        wpf = cache["k3"][0]
        wpf.copy_((cache["wpf0"].float() * cache["csc"][None, :]).to(wpf.dtype))
        return
    cache["qinv"].copy_(1.0 / torch.cat([cache["sa"], cache["sb"]]))
    cache["csc"].copy_(cache["sa"] * cache["sb"])
    if "w512" in cache:
        from .h100_b200style_ss_regtune import prescale_pack

        with T.native_context(cache["w512"].device):
            prescale_pack(cache["w512_src"], cache["qinv"], out=cache["w512"])
    elif "f8c" in cache:
        from .h100_b200style_direct import f8c_kscale

        f8c_kscale(cache["f8c"], cache["qinv"])
    elif "w" in cache:
        from .h100_b200style_direct import prescale

        for old, new in zip(cache["w"], prescale(cache["w_args"][0], cache["qinv"], *cache["w_args"][1:]), strict=True):
            old.copy_(new)
    if "k3" in cache:
        wpf = cache["k3"][0]
        wpf.copy_((cache["wpf0"].float() * cache["csc"][None, :]).to(wpf.dtype))


def rescale_all(peak: float) -> float:
    """Raise every calibrated module's scales so a forward whose guard read ``peak`` (> 448) fits with room x2; returns the factor. Call outside the graph, then redo the forward."""
    f = 2.0 ** math.ceil(math.log2(peak / 448.0 * 2.0))
    for cache in _CACHES.values():
        _scale_up(cache, f)
        refresh(cache)
    return f


def _scale_up(cache, f) -> None:
    for sub in cache["bd"].values() if "bd" in cache else (cache,):
        sub["sa"].mul_(f)
        sub["sb"].mul_(f)


def serves(module, pair, mask, bidirectional) -> bool:
    """Python-level (Dynamo-traceable) test: does the FP8-plane route take this call? hidden = pair (``WIDTHS``), bf16,
    1 .. ``MAX_SAMPLES`` samples (B > 1: the one-sample route per sample)."""
    if POLICY is None or pair.ndim != 4 or not 1 <= pair.shape[0] <= MAX_SAMPLES or pair.shape[1] != pair.shape[2]:
        return False
    d = pair.shape[-1]
    if d not in WIDTHS or module.d_hidden != d or pair.dtype != torch.bfloat16 or pair.shape[1] < 16:
        return False
    if module.ln_pair.eps != 1e-5 or module.ln_out.eps != 1e-5:
        return False
    if d == 512 and (bidirectional or pair.shape[0] != 1 or pair.shape[1] % 128 or pair.shape[1] > MAX_L_512 or not D512):
        return False  # D512 (measured: one direction, one sample, L % 128 == 0, L <= 2048; token-flat K3 k3reorder.cu K3_CSCALE=2): everything else keeps the bf16 route
    gate = MIN_L.get((d, 0 if bidirectional else 1 if module.outgoing else 2), 256)
    if pair.shape[1] < (int(_MINL_ENV) if _MINL_ENV is not None else gate):
        return False  # below the gate the bf16 default route is as fast or faster (see MIN_L)
    return mask is None or (mask.dtype == torch.bool and mask.numel() == pair.shape[0] * pair.shape[1])


def run_samples(leaves, mask, direction, cache=None, margin=1.0, zero=True, lfac=None):
    """B samples (one direction): the one-sample route per sample, each writing its rows of one output (shared static scales; the guard
    buffer is zeroed before the first sample only, so it reads the peak of the whole call). One sample: ``run``."""
    x = leaves[0]
    if x.shape[0] == 1:
        return run(leaves, mask, direction, cache=cache, margin=margin, zero=zero, lfac=lfac)
    y = torch.empty_like(x)
    for b in range(x.shape[0]):
        mb = None if mask is None else mask.reshape(x.shape[0], -1)[b : b + 1]
        run([x[b : b + 1], *leaves[1:]], mb, direction, cache=cache, margin=margin, zero=zero and b == 0, out=y[b : b + 1], lfac=lfac)
    return y


def _default(pair, mask, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, direction, lfac=None):
    """The bf16 default route of the call (the guard's and the capture's fallback). Single direction: the port; bidirectional: the integration's
    inference body (the packaged K1 / K3 at D128, the port at D256), through a shim holding the module's tensors."""
    if lfac is not None:  # RoseTTAFold3: the default route sees the left weight times the length factor, as its module builds it
        wl = wl * lfac.reshape(()).to(wl.dtype)
    token = mask if mask.numel() else None
    if direction != 0:
        return K.trimul(
            [pair, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo],
            mask if mask.numel() else torch.empty(0, device=pair.device, dtype=torch.bool),
            None,
            direction,
        )
    from types import SimpleNamespace as NS

    from miniworld_engine.kernels.integrations.trimul_h100 import _update_inference_body

    ln = lambda w, b: NS(weight=w, bias=b, eps=1e-5)
    shim = NS(
        to_left=NS(weight=wl),
        to_left_gate=NS(weight=wlg),
        to_right=NS(weight=wr),
        to_right_gate=NS(weight=wrg),
        to_gate=NS(weight=wg),
        to_out=NS(weight=wp),
        ln_pair=ln(gi, bi),
        ln_out=ln(go, bo),
        d_hidden=pair.shape[-1],
    )
    return _update_inference_body(shim, pair, token, True, None)


def _fake(pair, mask, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, direction, lfac=None):
    return torch.empty_like(pair)


@opaque(fake=_fake, name="trimul_h100_fp8_planes_inference")
def inference(
    pair: torch.Tensor,
    mask: torch.Tensor,
    wl: torch.Tensor,
    wlg: torch.Tensor,
    wr: torch.Tensor,
    wrg: torch.Tensor,
    wg: torch.Tensor,
    wp: torch.Tensor,
    gi: torch.Tensor,
    bi: torch.Tensor,
    go: torch.Tensor,
    bo: torch.Tensor,
    direction: int,
    lfac: torch.Tensor | None = None,
) -> torch.Tensor:
    """y of the FP8-plane TriMul for one direction (1 outgoing / 2 incoming), fp32-master or bf16 weights; mask: bool [1, L] or empty. Eager or under CUDA-graph capture
    (calibrate eagerly first: a capture with no calibration takes the bf16 route)."""
    from miniworld_engine.kernels import _capture
    from ._h100_pack import cast_training_weights

    mats = [wl, wlg, wr, wrg, wg, wp]
    if all(w.dtype == torch.bfloat16 for w in mats):
        cast = cast_training_weights(mats)
    else:
        cast = _capture.param_forms("trimul_h100_fp8_weights", mats, lambda: cast_training_weights(mats), capture_reads_eager=True)
    leaves = [pair, *cast, gi, bi, go, bo]
    token = mask if mask.numel() else None
    capturing = torch.cuda.is_current_stream_capturing()
    if POLICY != "calib":
        return run_samples(leaves, token, direction, lfac=lfac)
    key = wl.data_ptr()
    cache = _CACHES.get(key)
    if cache is None:
        if capturing:  # no static calibration yet: never calibrate inside a graph
            GUARD_STATS["capture_fallbacks"] += 1
            return _default(pair, mask, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, direction, lfac)
        cache = _CACHES[key] = {}
        # keyed by the address of to_left's weight: drop the state when that tensor is freed, so another module whose weight later sits at the
        # same address calibrates its own scales (a stale entry served another module's prescaled weights and K3 operands: wrong output, NaN)
        weakref.finalize(wl, _CACHES.pop, key, None)
    lazy = GUARD == "lazy" or capturing or GUARD == "off"
    if GUARD == "lazy" or capturing:
        cache["amax"] = _lazy_buffer(pair.device)
    for _ in range(4):
        y = run_samples(leaves, token, direction, cache=cache, margin=HEADROOM, zero=not (GUARD == "lazy" or capturing), lfac=lfac)
        GUARD_STATS["calls"] += 1
        if lazy:
            return y
        peak = float(LAST_AMAX.item())
        if peak <= 448.0:
            return y
        GUARD_STATS["retries"] += 1
        f = 2.0 ** math.ceil(math.log2(peak / 448.0 * 2.0))
        _scale_up(cache, f)
        refresh(cache)
    GUARD_STATS["fallbacks"] += 1
    return _default(pair, mask, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, direction, lfac)


def module_route(module, pair, mask, bidirectional):
    """Eager entry (kept for the module hook): y of the FP8-plane route or None."""
    if not serves(module, pair, mask, bidirectional):
        return None
    m = mask.reshape(1, -1) if mask is not None else torch.empty(0, device=pair.device, dtype=torch.bool)
    # RoseTTAFold3 (``_LeftScaled``): the module's ``to_left.weight`` is a fresh product ``W_a * (1 / n)`` per call -- no stable address to key
    # the calibration on and a new tensor inside every capture. The route takes the parameter itself and the factor beside it.
    lf = getattr(module, "left_length_factor", None)
    wl = module.left_base_weight if lf is not None else module.to_left.weight
    return inference(
        pair.contiguous(),
        m,
        wl,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
        module.ln_pair.weight,
        module.ln_pair.bias,
        module.ln_out.weight,
        module.ln_out.bias,
        0 if bidirectional else 1 if module.outgoing else 2,
        None if lf is None else lf.float().reshape(1),
    )

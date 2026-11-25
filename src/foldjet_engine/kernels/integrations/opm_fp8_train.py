"""EXPERIMENTAL, opt-in (never reached by default): OuterProductMean TRAINING (forward + backward) on H100 with the three K = S / K = N d_hidden
GEMMs of the module -- the grouped product and its two backward products, ~98 % of the step's FLOPs -- on FP8 operands.

Forward   the FP8 wide prologue (``opm_train._ext(fp8=True)``: LN + both projections + mask, e4m3 A2 / BT [(i,c), s], weight-bound scales
          ``opm_fp8.bound_scales``, recomputed every call from the live weights), the LayerNorm statistics, the bf16 mask and the exact
          count table -> O = A2 . BT^T on cuBLASLt e4m3 (``opm_fp8.product``, bf16 O, kept for dWo) -> the default epilogue.
Backward  ``opm_dgrad_f8``: dO = (dz / n) . Wo as e4m3 (default) or e5m2 bytes in BOTH layouts [(i,c),(j,e)] and [(j,e),(i,c)] from one
          launch (the K-major operands cuBLASLt's TN FP8 form needs), dz/n and dbo as the default dgrad;
          ``opm_t8``: e4m3 A2 / BT -> [s, (i,c)] (byte transposes);
          dA = BT^T . dO^T and dB = A2^T . dO on cuBLASLt (e4m3 x e5m2 / e4m3, fp32 accumulation, bf16 out);
          dWo off the kept O, the prologue backward (LayerNorm, projections' wgrad, dm) and the pair-count division: the default kernels.
Scales    per tensor, powers of two, from bounds (nothing can saturate, no calibration, no history, no host sync, graph-safe):
          A2 / BT: ``opm_fp8.bound_scales`` (weights alone); dO: |dO[(i,c),(j,e)]| <= max_ij(|dz_ij|_2 / n_ij) max_ce |Wo[:, (c,e)]|_2
          (Cauchy-Schwarz), x (1 + 2^-6) for the bf16 rounding of dz / n, to the next power of two over the format's max (448 / 57344).
          The mean normalisation by the pair count stays exact: the count table is the default's fp32 integer table, divided in fp32.

Switch: ``MINIWORLD_OPM_FP8_TRAIN`` = ``e4m3`` (dO e4m3, measured more accurate) | ``e5m2`` (dO e5m2) | unset / empty / ``0`` (off); ``MINIWORLD_FP8_TRAIN=1`` turns it on
as ``e4m3`` unless the per-module variable is set.  ``MINIWORLD_OPM_FP8_TRAIN_MINS=<s>``: the padded-depth gate (default 1024; 0 = none).
Eager autograd only (not under torch.compile); H100 sm_90a, bf16 activations, batch 1, the ``opm_train`` widths; everything else, and the
switch off, runs the default ``opm_train`` path unchanged.
"""

from __future__ import annotations

import os
import weakref

import torch

E4M3_MAX = 448.0
E5M2_MAX = 57344.0


def _mode() -> str | None:
    v = os.environ.get("MINIWORLD_OPM_FP8_TRAIN")
    if v is None and os.environ.get("MINIWORLD_FP8_TRAIN", "").strip().lower() in ("1", "on", "true"):
        v = "e4m3"
    v = (v or "").strip().lower()
    if v in ("", "0", "off", "none"):
        return None
    if v not in ("e5m2", "e4m3"):
        raise ValueError(f"MINIWORLD_OPM_FP8_TRAIN={v!r}: use e5m2, e4m3 or off")
    return v


#: dO format (None: the route is off). Module globals: may be set at run time before a forward.
MODE: str | None = _mode()
MIN_S = int(os.environ.get("MINIWORLD_OPM_FP8_TRAIN_MINS", "1024"))
#: round 2: shape gate. Round 1 lost at L128 (1.15x) and d_pair 384 / 512 (1.09-1.14x); with the scale pass and the transposes gone L128 is 0.91x, d_pair 384 0.98x, d_pair 512 still 1.03x -> d_pair <= 384 served, 0 = no gate
MIN_L = int(os.environ.get("MINIWORLD_OPM_FP8_TRAIN_MINL", "0"))
MAX_CZ = int(os.environ.get("MINIWORLD_OPM_FP8_TRAIN_MAXCZ", "384"))
#: round 3: per-d_pair minimum padded-free length (measured, vs the 3.0.1 default step: d128 wins from L128 (0.90-0.96x), d256 from L256 (0.96x; L128 0.985x = noise), d384 from L384 (0.97x; L256 0.987x, L128 1.04x); d512 never (1.025x)). Ignored when MAX_CZ = 0 (gate off, benchmarks)
MIN_L_BY_CZ = {128: 0, 256: 256, 384: 384}
#: round 2: delayed dO scale (amax history, no pass over dz; first call of a module seeds from the bound); False = round 1's bound every call
DELAYED = os.environ.get("MINIWORLD_OPM_FP8_TRAIN_DELAYED", "1") != "0"
#: round 2: one dgrad pass writes both dO planes (d_pair 128; the transposed plane is staged in shared memory); False = round 1: two launches
SINGLE = os.environ.get("MINIWORLD_OPM_FP8_TRAIN_SINGLE", "1") != "0"
HISTORY = int(os.environ.get("MINIWORLD_OPM_FP8_TRAIN_HISTORY", "16"))
#: round 4: how the product O (the dominant saved tensor: 32N x 32N bf16, ~85 % of the saved bytes at S 1024) is kept for dW_o:
#: "bf16" (rounds 1-3), "fp8" (e4m3 x a power-of-two scale, dequantised in backward), "recompute" (the FP8 product is redone in backward)
OSTORE = os.environ.get("MINIWORLD_OPM_FP8_TRAIN_OSTORE", "bf16")
_OCHUNK = 2048  # rows of O per conversion chunk (O is [32N, 32N])
MARGIN = float(os.environ.get("MINIWORLD_OPM_FP8_TRAIN_MARGIN", "2"))
STATS = {"calls": 0}
_SITES: dict = {}  # (device, wo data_ptr) -> _DSite
_RETIRED: list = []


class _DSite:
    """Delayed dO scale of one module (keyed by its W_o): persistent device state, so CUDA-graph replays update it in place."""

    def __init__(self, device):
        self.hist = torch.zeros(HISTORY, device=device)
        self.st = torch.zeros(5, device=device)  # {sd, 1 / sd (unused), last scaled amax, calls, overflow count}
        self.sd = torch.ones(1, device=device)  # cuBLASLt scale (own allocation)
        self.q = torch.ones(1, device=device)  # dgrad multiplier 1 / sd (own allocation)
        self.amax = torch.zeros(1, device=device)  # max |dO q| of this call (the dgrad's atomic)
        self.seeded = False


def _site(wo: torch.Tensor) -> _DSite:
    key = (wo.device, wo.data_ptr())
    s = _SITES.get(key)
    if s is None:
        s = _SITES[key] = _DSite(wo.device)
        weakref.finalize(wo, lambda k=key: _RETIRED.append(_SITES.pop(k, None)))
    return s


def reset() -> None:
    """Forget every site's history (the next call of each site re-seeds from the bound). Old buffers stay alive (captured graphs)."""
    _RETIRED.extend(_SITES.values())
    _SITES.clear()


def overflow_count() -> float:
    """Calls (all delayed sites) whose dO passed the format max at its scale since the last reset (one host sync)."""
    fmax = E5M2_MAX if MODE == "e5m2" else E4M3_MAX
    return sum(float(s.st[4].item()) + float((s.amax > fmax).item()) for s in _SITES.values())
#: largest |scaled value| seen since the last :func:`peaks` read: [A2, BT, dO] (diagnostics; bounds keep them <= the format max)
_PEAK: dict = {}


def peaks(reset: bool = True) -> list[float]:
    """[A2, BT, dO] max |scaled value| since the last read (one host sync; e4m3 max 448, e5m2 57344)."""
    out = [0.0, 0.0, 0.0]
    for buf in _PEAK.values():
        out = [max(a, float(b)) for a, b in zip(out, buf.tolist(), strict=True)]
        if reset:
            buf.zero_()
    for s in _SITES.values():  # delayed sites: the dO peak of the latest call
        out[2] = max(out[2], float(s.amax))
    return out


def _peak_buffer(device):
    if device not in _PEAK:
        _PEAK[device] = torch.zeros(3, device=device)
    return _PEAK[device]


def serves(msa: torch.Tensor, d_msa: int, d_pair: int, *, interchain: bool = False) -> bool:
    """Does the FP8 training route take this grad-enabled ``opm_train`` call (msa [1, S, N, d_msa] before padding)?"""
    if MODE is None or interchain or torch.compiler.is_compiling():
        return False
    from foldjet_engine.kernels.integrations import opm_train as ot

    if (d_msa, d_pair) not in ot.SHAPES or msa.dim() != 4 or msa.shape[0] != 1 or msa.dtype != torch.bfloat16 or not msa.is_cuda:
        return False
    if torch.cuda.get_device_capability(msa.device) != (9, 0):
        return False
    if (MIN_L and msa.shape[2] < MIN_L) or (MAX_CZ and d_pair > MAX_CZ):  # round 2: the shapes where round 1 lost to the default
        return False
    if MAX_CZ and msa.shape[2] < MIN_L_BY_CZ.get(d_pair, 0):
        return False
    return msa.shape[1] + (-msa.shape[1] % 256) >= MIN_S


class OpmFp8TrainFn(torch.autograd.Function):
    """(msa [1, S, N, cm] padded, mask, lnw, lnb, wa, wb, wo, bo, residual (empty: none), pbias (empty: none), eps, after, dmin, doff) -> z."""

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(ctx, msa, mask, lnw, lnb, wa, wb, wo, bo, residual, pbias, eps, after, dmin, doff):
        from foldjet_engine.kernels.integrations import opm_fp8
        from foldjet_engine.kernels.integrations import opm_train as ot

        bf = torch.bfloat16
        m = msa[0].contiguous()
        mk = mask[0].contiguous()
        if mk.dtype != torch.bool:
            mk = mk != 0
        n, cm, cz = m.shape[1], m.shape[2], wo.shape[0]
        ext = ot._ext(msa.device, native=True, tma=False, cm=cm, cz=cz, fp8=True)
        pb = pbias.detach().float().contiguous() if pbias.numel() else None
        res = residual if residual.numel() else None
        dkw = {} if (dmin, doff) == (1.0, 0.0) else {"dmin": float(dmin), "doff": float(doff)}
        peak = _peak_buffer(m.device)
        gam, bet = lnw.detach().float().contiguous(), lnb.detach().float().contiguous()
        wa16, wb16 = wa.detach().to(bf).contiguous(), wb.detach().to(bf).contiguous()
        # {s_a, s_b, 1 / s_a, 1 / s_b}: opm_fp8.bound_scales, every call (the weights move)
        sc = ext.opm_f8_wscale(gam, bet, wa16, wb16, pb)
        a8, b8, stats, mask16, norm, a8t, b8t = ext.opm_prologue_wide(
            m,
            mk,
            gam,
            bet,
            wa16,
            wb16,
            float(eps),
            True,
            pb,
            True,
            f8q=sc[2:4],
            f8amax=peak,
            f8t=True,
            **dkw,
        )
        sa, sb = sc[0:1].clone(), sc[1:2].clone()  # (separate allocations: cuBLASLt refuses a scale pointer off its alignment)
        o = opm_fp8.product(a8, b8, (sa, sb), n)
        z = ext.opm_epilogue(
            o, norm, wo.detach().to(bf).contiguous(), bo.detach().to(bf).float().contiguous(), n, n, res, int(after), **dkw
        )
        extra: list[torch.Tensor] = []
        if OSTORE == "fp8":  # O kept as e4m3 x 2^k; amax on the device (no host sync, graph safe)
            mn, mx = torch.aminmax(o)
            amax = torch.maximum(mn.abs(), mx).float().clamp_min(1e-30)
            so = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX))).reshape(1)
            o8 = torch.empty(o.shape, device=o.device, dtype=torch.float8_e4m3fn)
            sob = so.to(bf)
            for r0 in range(0, o.shape[0], _OCHUNK):  # chunked: no full-size temporaries
                o8[r0 : r0 + _OCHUNK] = (o[r0 : r0 + _OCHUNK] / sob).to(torch.float8_e4m3fn)
            extra = [o8, so]
        elif OSTORE == "recompute":
            extra = [a8, b8]
        else:
            extra = [o]
        ctx.ostore = OSTORE
        del o
        ctx.save_for_backward(m, mask16, norm, a8t, b8t, stats, lnw, lnb, wa, wb, wo, bo, sa, sb, *extra)
        ctx.eps, ctx.after, ctx.has_bias, ctx.has_residual = eps, bool(after), pb is not None, res is not None
        STATS["calls"] += 1
        return z

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def backward(ctx, dz):
        from foldjet_engine.kernels.integrations import opm_fp8
        from foldjet_engine.kernels.integrations import opm_train as ot

        bf = torch.bfloat16
        m, mask16, norm, a8t, b8t, stats, lnw, lnb, wa, wb, wo, bo, sa, sb, *kept = ctx.saved_tensors
        s, n, cm, cz = m.shape[0], m.shape[1], m.shape[2], wo.shape[0]
        ch = ot.CH
        if ctx.ostore == "fp8":
            o = torch.empty(kept[0].shape, device=kept[0].device, dtype=bf)
            sob = kept[1].to(bf)  # power-of-two scale: exact in bf16
            for r0 in range(0, o.shape[0], _OCHUNK):
                o[r0 : r0 + _OCHUNK] = kept[0][r0 : r0 + _OCHUNK].to(bf) * sob
        elif ctx.ostore == "recompute":
            o = opm_fp8.product(kept[0], kept[1], (sa, sb), n)
        else:
            o = kept[0]
        ext = ot._ext(dz.device, native=True, tma=False, cm=cm, cz=cz, fp8=True)
        bw = wo.detach().t().to(bf).contiguous()  # Wo^T [(c,e), z]
        bw2 = torch.cat([bw, bw.view(ch, ch, cz).transpose(0, 1).reshape(ch * ch, cz)]).contiguous()  # + rows (e, c): the dO^T plane
        fmt = 1 if MODE == "e5m2" else 0
        dz0 = dz[0].contiguous()
        fmax = E5M2_MAX if fmt else E4M3_MAX
        peak = _peak_buffer(dz.device)
        if DELAYED:
            site = _site(wo)
            if not site.seeded:  # first call of this module: the bound (one pass over dz), then the amax history takes over
                if torch.cuda.is_current_stream_capturing():
                    sdq = ext.opm_f8_dscale(dz0, norm, bw, fmax)  # captured first call: stays on the bound pass
                    sd, q, amax = sdq[0:1].clone(), sdq[1:2], peak[2:3]
                else:
                    sdq = ext.opm_f8_dscale(dz0, norm, bw, fmax)
                    ext.opm_f8_dscale_seed(sdq[0:1], site.hist, site.st, site.sd, site.q, site.amax)
                    site.seeded = True
                    sd, q, amax = site.sd, site.q, site.amax
            else:
                ext.opm_f8_dscale_update(site.amax, site.hist, site.st, site.sd, site.q, fmax, MARGIN)
                sd, q, amax = site.sd, site.q, site.amax
        else:
            sdq = ext.opm_f8_dscale(dz0, norm, bw, fmax)  # {sd, 1 / sd}
            sd, q, amax = sdq[0:1].clone(), sdq[1:2], peak[2:3]
        d8, d8t, dzp, dbo = ext.opm_dgrad_f8(dz0, norm, bw2, n, n, int(ctx.after), q, amax, fmt, SINGLE and cz == 128)
        dA = opm_fp8.product(b8t, d8, (sb, sd), n)  # [s, (i,c)] = sum_(j,e) BT[(j,e), s] dO[(i,c), (j,e)]
        dB = opm_fp8.product(a8t, d8t, (sa, sd), n)  # [s, (j,e)] = sum_(i,c) A2[(i,c), s] dO[(i,c), (j,e)]
        del d8, d8t, a8t, b8t
        db = None
        if ctx.has_bias:  # RoseTTAFold3's projection biases: as opm_train._backward_wide
            mk = mask16.reshape(1, -1)
            db = torch.cat(
                [
                    torch.mm(mk, dA.reshape(-1, ch), out_dtype=torch.float32).reshape(-1),
                    torch.mm(mk, dB.reshape(-1, ch), out_dtype=torch.float32).reshape(-1),
                ]
            )
        dWo = ext.opm_dwo(dzp, o, n, n)
        g = lnw.detach().float()[None, :]
        bwp = torch.cat([(wa.detach().float() * g).t(), (wb.detach().float() * g).t()], dim=1).to(bf).contiguous()
        dm, dWa, dWb, dgam, dbet = ext.opm_prologue_bwd(
            dA, dB, m, stats, mask16, lnw.detach().float().contiguous(), lnb.detach().float().contiguous(), float(ctx.eps), bwp, n, s
        )
        return (
            dm.view(1, s, n, cm),
            None,
            dgam.to(lnw.dtype),
            dbet.to(lnb.dtype),
            dWa.to(wa.dtype),
            dWb.to(wb.dtype),
            dWo.to(wo.dtype),
            dbo.to(bo.dtype),
            dz if ctx.has_residual else None,
            db,
            None,
            None,
            None,
            None,
        )


def apply(tensors: list[torch.Tensor], eps: float, after: bool, dmin: float, doff: float) -> torch.Tensor:
    """z for ``opm_train.outer_product_mean``'s padded tensor list (msa, mask, lnw, lnb, wa, wb, wo, bo, residual, pbias)."""
    return OpmFp8TrainFn.apply(*tensors, eps, after, dmin, doff)


__all__ = ["MODE", "OpmFp8TrainFn", "apply", "peaks", "serves"]

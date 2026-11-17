"""EXPERIMENTAL, opt-in (never reached by default): OuterProductMean TRAINING (forward + backward) on H100 with the three K = S / K = N d_hidden
GEMMs of the module -- the grouped product and its two backward products, ~98 % of the step's FLOPs -- on FP8 operands.

Forward   the FP8 wide prologue (``opm_train._ext(fp8=True)``: LN + both projections + mask, e4m3 A2 / BT [(i,c), s], weight-bound scales
          ``opm_fp8.bound_scales``, recomputed every call from the live weights), the LayerNorm statistics, the bf16 mask and the exact
          count table -> O = A2 . BT^T on cuBLASLt e4m3 (``opm_fp8.product``, bf16 O, kept for dWo) -> the default epilogue.
Backward  ``opm_dgrad_f8``: dO = (dz / n) . Wo as e5m2 (default) or e4m3 bytes in BOTH layouts [(i,c),(j,e)] and [(j,e),(i,c)] from one
          launch (the K-major operands cuBLASLt's TN FP8 form needs), dz/n and dbo as the default dgrad;
          ``opm_t8``: e4m3 A2 / BT -> [s, (i,c)] (byte transposes);
          dA = BT^T . dO^T and dB = A2^T . dO on cuBLASLt (e4m3 x e5m2 / e4m3, fp32 accumulation, bf16 out);
          dWo off the kept O, the prologue backward (LayerNorm, projections' wgrad, dm) and the pair-count division: the default kernels.
Scales    per tensor, powers of two, from bounds (nothing can saturate, no calibration, no history, no host sync, graph-safe):
          A2 / BT: ``opm_fp8.bound_scales`` (weights alone); dO: |dO[(i,c),(j,e)]| <= max_ij(|dz_ij|_2 / n_ij) max_ce |Wo[:, (c,e)]|_2
          (Cauchy-Schwarz), x (1 + 2^-6) for the bf16 rounding of dz / n, to the next power of two over the format's max (448 / 57344).
          The mean normalisation by the pair count stays exact: the count table is the default's fp32 integer table, divided in fp32.

Switch: ``MINIWORLD_OPM_FP8_TRAIN`` = ``e5m2`` (dO e5m2) | ``e4m3`` (dO e4m3) | unset / empty / ``0`` (off); ``MINIWORLD_FP8_TRAIN=1`` turns it on
as ``e5m2`` unless the per-module variable is set.  ``MINIWORLD_OPM_FP8_TRAIN_MINS=<s>``: the padded-depth gate (default 1024; 0 = none).
Eager autograd only (not under torch.compile); H100 sm_90a, bf16 activations, batch 1, the ``opm_train`` widths; everything else, and the
switch off, runs the default ``opm_train`` path unchanged.
"""

from __future__ import annotations

import os

import torch

E4M3_MAX = 448.0
E5M2_MAX = 57344.0


def _mode() -> str | None:
    v = os.environ.get("MINIWORLD_OPM_FP8_TRAIN")
    if v is None and os.environ.get("MINIWORLD_FP8_TRAIN", "").strip().lower() in ("1", "on", "true"):
        v = "e5m2"
    v = (v or "").strip().lower()
    if v in ("", "0", "off", "none"):
        return None
    if v not in ("e5m2", "e4m3"):
        raise ValueError(f"MINIWORLD_OPM_FP8_TRAIN={v!r}: use e5m2, e4m3 or off")
    return v


#: dO format (None: the route is off). Module globals: may be set at run time before a forward.
MODE: str | None = _mode()
MIN_S = int(os.environ.get("MINIWORLD_OPM_FP8_TRAIN_MINS", "1024"))
STATS = {"calls": 0}
#: largest |scaled value| seen since the last :func:`peaks` read: [A2, BT, dO] (diagnostics; bounds keep them <= the format max)
_PEAK: dict = {}


def peaks(reset: bool = True) -> list[float]:
    """[A2, BT, dO] max |scaled value| since the last read (one host sync; e4m3 max 448, e5m2 57344)."""
    out = [0.0, 0.0, 0.0]
    for buf in _PEAK.values():
        out = [max(a, float(b)) for a, b in zip(out, buf.tolist(), strict=True)]
        if reset:
            buf.zero_()
    return out


def _peak_buffer(device):
    if device not in _PEAK:
        _PEAK[device] = torch.zeros(3, device=device)
    return _PEAK[device]


def serves(msa: torch.Tensor, d_msa: int, d_pair: int, *, interchain: bool = False) -> bool:
    """Does the FP8 training route take this grad-enabled ``opm_train`` call (msa [1, S, N, d_msa] before padding)?"""
    if MODE is None or interchain or torch.compiler.is_compiling():
        return False
    from miniworld_engine.kernels.integrations import opm_train as ot

    if (d_msa, d_pair) not in ot.SHAPES or msa.dim() != 4 or msa.shape[0] != 1 or msa.dtype != torch.bfloat16 or not msa.is_cuda:
        return False
    if torch.cuda.get_device_capability(msa.device) != (9, 0):
        return False
    return msa.shape[1] + (-msa.shape[1] % 256) >= MIN_S


def _pow2(b: torch.Tensor, fmax: float) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(b.clamp_min(1e-30) / fmax))).clamp_min(2.0**-60)


def _dgrad_scale(dz: torch.Tensor, norm: torch.Tensor, bw: torch.Tensor, fmax: float) -> torch.Tensor:
    """[1] dequantisation scale of dO from the Cauchy-Schwarz bound (device ops, no sync)."""
    r = (torch.linalg.vector_norm(dz, dim=-1, dtype=torch.float32) / norm).amax()
    w = torch.linalg.vector_norm(bw, dim=1, dtype=torch.float32).amax()
    return _pow2(r * w * (1.0 + 2.0**-6), fmax).reshape(1)


class OpmFp8TrainFn(torch.autograd.Function):
    """(msa [1, S, N, cm] padded, mask, lnw, lnb, wa, wb, wo, bo, residual (empty: none), pbias (empty: none), eps, after, dmin, doff) -> z."""

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(ctx, msa, mask, lnw, lnb, wa, wb, wo, bo, residual, pbias, eps, after, dmin, doff):
        from miniworld_engine.kernels.integrations import opm_fp8
        from miniworld_engine.kernels.integrations import opm_train as ot

        bf = torch.bfloat16
        m = msa[0].contiguous()
        mk = mask[0].contiguous()
        if mk.dtype != torch.bool:
            mk = mk != 0
        n, cm, cz = m.shape[1], m.shape[2], wo.shape[0]
        ext = ot._ext(msa.device, native=True, tma=False, cm=cm, cz=cz, fp8=True)
        pb = pbias.detach().float().contiguous() if pbias.numel() else None
        res = residual if residual.numel() else None
        s = opm_fp8.bound_scales(lnw, lnb, wa, wb, pb).to(m.device)  # [2] A2 / BT dequantisation scales (weights change every step)
        dkw = {} if (dmin, doff) == (1.0, 0.0) else {"dmin": float(dmin), "doff": float(doff)}
        peak = _peak_buffer(m.device)
        a8, b8, stats, mask16, norm = ext.opm_prologue_wide(
            m,
            mk,
            lnw.detach().float().contiguous(),
            lnb.detach().float().contiguous(),
            wa.detach().to(bf).contiguous(),
            wb.detach().to(bf).contiguous(),
            float(eps),
            True,
            pb,
            True,
            f8q=(1.0 / s).contiguous(),
            f8amax=peak,
            **dkw,
        )
        sa, sb = s[0:1].clone(), s[1:2].clone()
        o = opm_fp8.product(a8, b8, (sa, sb), n)
        z = ext.opm_epilogue(
            o, norm, wo.detach().to(bf).contiguous(), bo.detach().to(bf).float().contiguous(), n, n, res, int(after), **dkw
        )
        ctx.save_for_backward(m, mask16, norm, a8, b8, stats, lnw, lnb, wa, wb, wo, bo, o, sa, sb)
        ctx.eps, ctx.after, ctx.has_bias, ctx.has_residual = eps, bool(after), pb is not None, res is not None
        STATS["calls"] += 1
        return z

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def backward(ctx, dz):
        from miniworld_engine.kernels.integrations import opm_fp8
        from miniworld_engine.kernels.integrations import opm_train as ot

        bf = torch.bfloat16
        m, mask16, norm, a8, b8, stats, lnw, lnb, wa, wb, wo, bo, o, sa, sb = ctx.saved_tensors
        s, n, cm, cz = m.shape[0], m.shape[1], m.shape[2], wo.shape[0]
        ch = ot.CH
        ext = ot._ext(dz.device, native=True, tma=False, cm=cm, cz=cz, fp8=True)
        bw = wo.detach().t().to(bf).contiguous()  # Wo^T [(c,e), z]
        bw2 = torch.cat([bw, bw.view(ch, ch, cz).transpose(0, 1).reshape(ch * ch, cz)]).contiguous()  # + rows (e, c): the dO^T plane
        fmt = 1 if MODE == "e5m2" else 0
        dz0 = dz[0].contiguous()
        sd = _dgrad_scale(dz0, norm, bw, E5M2_MAX if fmt else E4M3_MAX)
        peak = _peak_buffer(dz.device)
        d8, d8t, dzp, dbo = ext.opm_dgrad_f8(dz0, norm, bw2, n, n, int(ctx.after), (1.0 / sd).contiguous(), peak[2:3], fmt)
        a8t, b8t = ext.opm_t8(a8), ext.opm_t8(b8)  # [s, (i,c)]: K = (i,c) contiguous
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

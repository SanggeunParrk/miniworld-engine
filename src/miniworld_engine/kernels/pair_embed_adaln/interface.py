"""Fused pair builder (kernel family ``pair_embed_adaln``): La-Proteina's initial pair representation

    pair[b,i,j] = pm_ij * ( LN(h_ij) * gamma[b] + beta[b] ),   h_ij = LN_affine( W [onehot_sep(i,j) | onehot_xt(i,j) | onehot_sc(i,j)] )

as a row gather, two LayerNorms and the per-sample adaptive scale / shift in one kernel, instead of three binned fp32 one-hots
(127 + 30 + 30 channels per pair), a concatenation, a Linear, the affine LayerNorm, a pair-wide conditioning path and the adaptive
LayerNorm.  ``gamma`` / ``beta`` are per sample because the conditioning (diffusion times) is constant over (i, j); the caller
computes them with ordinary autograd.  The output is bf16 (what the attention-bias projection consumes); the backward is bitwise
deterministic.  Served: H100 (sm_90a), 256 channels.  :func:`refusal` says why anything else is not served; it never raises.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque

D_PAIR = 256


def refusal(w, ln_weight, ln_bias, gamma, beta, xt, sc, residue_idx, mask, *, n_sep=127, n_xt=30, n_sc=30, **_) -> str | None:
    """None if the fused op can run this call, else why it cannot. Only metadata is read."""
    if not xt.is_cuda:
        return "the input is not on a CUDA device"
    if torch.cuda.get_device_capability(xt.device) != (9, 0):
        return "served on H100 (sm_90a) only"
    nt = n_sep + n_xt + n_sc
    if tuple(w.shape[:1]) != (D_PAIR,) or w.shape[1] < nt:
        return f"w must be [{D_PAIR}, >= {nt}], got {tuple(w.shape)}"
    if nt > 220 or max(n_sep, n_xt, n_sc) > 255:
        return "the kernel packs each bin into 8 bits and keeps 4 copies of the table in shared memory (at most 220 rows)"
    if xt.dim() != 3 or xt.shape[-1] != 3 or residue_idx.shape != xt.shape[:2] or mask.shape != xt.shape[:2]:
        return "xt must be [B, n, 3] and residue_idx / mask [B, n]"
    if sc is not None and sc.shape != xt.shape:
        return "sc must be shaped like xt"
    if gamma.shape != (xt.shape[0], D_PAIR) or beta.shape != gamma.shape:
        return f"gamma / beta must be [B, {D_PAIR}]"
    return None


def _pair_embed_adaln_fwd_fake(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2):
    """[pair bf16 [B, n, n, 256], bins int32 [B, n, n]]: the shapes of the forward (the packed bin codes the backward replays)."""
    B, n = idx.shape
    return [torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device),
            torch.empty((B, n, n), dtype=torch.int32, device=tbl.device)]


@opaque(fake=_pair_embed_adaln_fwd_fake, name="pair_embed_adaln_fwd")
def pair_embed_adaln_fwd(tbl: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                         xt: torch.Tensor, sc: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor, xlim: torch.Tensor,
                         clim: torch.Tensor, has_sc: int, nS: int, nX: int, eps1: float,
                         eps2: float) -> list[torch.Tensor]:
    """[pair [B, n, n, 256] bf16, bins [B, n, n] int32] from the table [NT, 256] fp32 (= W^T), LN affine, gamma / beta [B, 256]."""
    from miniworld_engine.kernels.pair_embed_adaln.cuda import sm90

    return list(sm90.forward(tbl, lnw, lnb, gamma, beta, xt, sc, has_sc, idx, mask, xlim, clim, nS, nX, eps1, eps2))


def _pair_embed_adaln_bwd_fake(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2):
    """[dtbl [NT, 256], dln_w, dln_b [256], dgamma, dbeta [B, 256]], all fp32."""
    B = bins.shape[0]
    f = lambda *s: torch.empty(s, dtype=torch.float32, device=tbl.device)
    return [f(*tbl.shape), f(D_PAIR), f(D_PAIR), f(B, D_PAIR), f(B, D_PAIR)]


@opaque(fake=_pair_embed_adaln_bwd_fake, name="pair_embed_adaln_bwd")
def pair_embed_adaln_bwd(tbl: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, gamma: torch.Tensor, bins: torch.Tensor,
                         dout: torch.Tensor, nS: int, nX: int, eps1: float, eps2: float) -> list[torch.Tensor]:
    """[dtbl [NT, 256], dln_w, dln_b [256], dgamma, dbeta [B, 256]] fp32 from d pair = dout [B, n, n, 256] bf16."""
    from miniworld_engine.kernels.pair_embed_adaln.cuda import sm90

    return list(sm90.backward(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2))


class PairEmbedAdaLNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2):
        ctx.dtypes = (tbl.dtype, lnw.dtype, lnb.dtype, gamma.dtype, beta.dtype)
        f32 = torch.float32
        tbl, lnw, lnb, gamma, beta = (t.to(f32).contiguous() for t in (tbl, lnw, lnb, gamma, beta))
        out, bins = pair_embed_adaln_fwd(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2)
        ctx.save_for_backward(tbl, lnw, lnb, gamma, bins)
        ctx.meta = (nS, nX, eps1, eps2)
        return out

    @staticmethod
    def backward(ctx, g):
        tbl, lnw, lnb, gamma, bins = ctx.saved_tensors
        dtbl, dlw, dlb, dgamma, dbeta = pair_embed_adaln_bwd(tbl, lnw, lnb, gamma, bins, g.to(torch.bfloat16).contiguous(), *ctx.meta)
        outs = [d.to(dt) for d, dt in zip((dtbl, dlw, dlb, dgamma, dbeta), ctx.dtypes, strict=True)]
        return (*outs, *([None] * 11))


def pair_embed_adaln(w, ln_weight, ln_bias, gamma, beta, xt, sc, residue_idx, mask, *, n_sep=127, n_xt=30, n_sc=30,
                     xt_range=(0.1, 3.0), sc_range=(0.1, 3.0), eps1=1e-5, eps2=1e-5) -> torch.Tensor:
    """pair [B, n, n, 256] bf16, differentiable in ``w`` (the Linear weight [256, >= n_sep + n_xt + n_sc], columns in the order
    sep | xt | sc), the LayerNorm's ``ln_weight`` / ``ln_bias`` and ``gamma`` / ``beta`` [B, 256] (``gamma`` after the sigmoid).

    ``xt`` / ``sc`` [B, n, 3] fp32 CA coordinates in nm (``sc`` None when the call has no self-conditioning: its one-hot is zero),
    ``residue_idx`` [B, n] integers, ``mask`` [B, n] bool.  Check :func:`refusal` first."""
    dev = xt.device
    nt = n_sep + n_xt + n_sc
    tbl = w[:, :nt].t().contiguous()
    xlim = torch.linspace(*xt_range, n_xt - 1, device=dev)
    clim = torch.linspace(*sc_range, n_sc - 1, device=dev)
    return PairEmbedAdaLNFunction.apply(tbl, ln_weight, ln_bias, gamma, beta, xt.float().contiguous(),
                                        (sc if sc is not None else xt).float().contiguous(), residue_idx.to(torch.int32).contiguous(),
                                        mask.to(torch.uint8).contiguous(), xlim, clim, int(sc is not None), n_sep, n_xt, eps1, eps2)


__all__ = ["D_PAIR", "refusal", "pair_embed_adaln"]

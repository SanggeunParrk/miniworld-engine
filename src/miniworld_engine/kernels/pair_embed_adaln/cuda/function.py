"""The CUDA backend of ``PairEmbedAdaLN``: the forward / backward launches of ``sm90`` (``pair_embed_adaln.cu``, H100 sm_90a) as opaque ops and an
autograd function. Inputs are the module's prepared tensors (``modules/pair_embed.py``): the table ``tbl`` [NT, 256] (= W^T), ``lnw`` / ``lnb`` [256],
``gamma`` / ``beta`` [B, 256], fp32 coordinates ``xt`` / ``sc`` [B, n, 3] (``sc`` = ``xt`` when there is no self-conditioning, ``has_sc`` 0), int32
``idx`` [B, n], uint8 ``mask`` [B, n] and the fp32 bin limits ``xlim`` / ``clim``. The pair comes out bf16."""
from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque

D_PAIR = 256


def _pair_embed_adaln_fwd_fake(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2):
    """[pair bf16 [B, n, n, 256], bins int32 [B, n, n]]: the shapes of the forward (the packed bin codes the backward replays)."""
    B, n = idx.shape
    return [torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device), torch.empty((B, n, n), dtype=torch.int32, device=tbl.device)]


@opaque(fake=_pair_embed_adaln_fwd_fake, name="pair_embed_adaln_fwd_sm90")
def pair_embed_adaln_fwd(tbl: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, xt: torch.Tensor,
                         sc: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor, xlim: torch.Tensor, clim: torch.Tensor, has_sc: int, nS: int,
                         nX: int, eps1: float, eps2: float) -> list[torch.Tensor]:
    """[pair [B, n, n, 256] bf16, bins [B, n, n] int32] from the table [NT, 256] fp32, the LN affine and gamma / beta [B, 256]."""
    from miniworld_engine.kernels.pair_embed_adaln.cuda import sm90

    return list(sm90.forward(tbl, lnw, lnb, gamma, beta, xt, sc, has_sc, idx, mask, xlim, clim, nS, nX, eps1, eps2))


def _pair_embed_adaln_bwd_fake(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2):
    """[dtbl [NT, 256], dln_w, dln_b [256], dgamma, dbeta [B, 256]], all fp32."""
    B = bins.shape[0]

    def f(*s):
        return torch.empty(s, dtype=torch.float32, device=tbl.device)

    return [f(*tbl.shape), f(D_PAIR), f(D_PAIR), f(B, D_PAIR), f(B, D_PAIR)]


@opaque(fake=_pair_embed_adaln_bwd_fake, name="pair_embed_adaln_bwd_sm90")
def pair_embed_adaln_bwd(tbl: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, gamma: torch.Tensor, bins: torch.Tensor, dout: torch.Tensor,
                         nS: int, nX: int, eps1: float, eps2: float) -> list[torch.Tensor]:
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
        grads = pair_embed_adaln_bwd(tbl, lnw, lnb, gamma, bins, g.to(torch.bfloat16).contiguous(), *ctx.meta)
        outs = [d.to(dt) for d, dt in zip(grads, ctx.dtypes, strict=True)]
        return (*outs, *([None] * 11))


def pair_embed_adaln_cuda(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2) -> torch.Tensor:
    """pair [B, n, n, 256] bf16, differentiable in ``tbl``, ``lnw``, ``lnb``, ``gamma`` and ``beta``."""
    return PairEmbedAdaLNFunction.apply(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2)

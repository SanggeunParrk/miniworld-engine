"""The portable (Triton) backend of ``PairEmbedAdaLN``: the forward / backward kernels of ``main.py`` as an autograd function. The inputs are the
prepared tensors of ``cuda/function.py`` (the same module-side preparation); the pair comes out bf16, the backward is bitwise deterministic."""
from __future__ import annotations

import torch

from miniworld_engine.kernels.pair_embed_adaln.triton.main import portable_backward, portable_forward


class PortablePairEmbedAdaLNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2):
        ctx.dtypes = (tbl.dtype, lnw.dtype, lnb.dtype, gamma.dtype, beta.dtype)
        f32 = torch.float32
        tbl, lnw, lnb, gamma, beta = (t.to(f32).contiguous() for t in (tbl, lnw, lnb, gamma, beta))
        out, bins = portable_forward(tbl, lnw, lnb, gamma, beta, xt, sc, has_sc, idx, mask, xlim, clim, nS, nX, eps1, eps2)
        ctx.save_for_backward(tbl, lnw, lnb, gamma, bins)
        ctx.meta = (nS, nX, eps1, eps2)
        return out

    @staticmethod
    def backward(ctx, g):
        tbl, lnw, lnb, gamma, bins = ctx.saved_tensors
        grads = portable_backward(tbl, lnw, lnb, gamma, bins, g.to(torch.bfloat16).contiguous(), *ctx.meta)
        outs = [d.to(dt) for d, dt in zip(grads, ctx.dtypes, strict=True)]
        return (*outs, *([None] * 11))


def pair_embed_adaln_triton(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2) -> torch.Tensor:
    """pair [B, n, n, 256] bf16, differentiable in ``tbl``, ``lnw``, ``lnb``, ``gamma`` and ``beta``."""
    return PortablePairEmbedAdaLNFunction.apply(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2)

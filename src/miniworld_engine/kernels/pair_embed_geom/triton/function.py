"""The portable (Triton) backend of ``PairEmbedGeom`` / ``PairEmbedCA``: the bins / gather / table-gradient / LayerNorm-backward kernels of ``main.py`` as an
autograd function. The inputs are the prepared tensors of ``cuda/function.py`` (the same module-side preparation): ``eps`` >= 0 folds the attention-bias
LayerNorm into the gather, ``eps`` < 0 returns the pair itself; the pair comes out bf16, the table gradient bf16 (fixed-order partial sums, no atomics)."""

from __future__ import annotations

import torch

from miniworld_engine.kernels.pair_embed_geom.triton.main import (
    portable_backward,
    portable_forward,
    portable_forward_ca,
    portable_ln_backward,
)


def _ln_eps(eps: float) -> float | None:
    return eps if eps >= 0 else None


class PortablePairEmbedFunction(torch.autograd.Function):
    """``mode`` 0: the encoder pair (``a`` = bb [B, n, 4, 3]); 1: the decoder pair (``a`` = CA [B, n, 3]; ``cb`` / ``chains`` / ``alim`` unused)."""

    @staticmethod
    def forward(ctx, tbl, mode, a, ca, cb, mask, idx, chains, dlim, alim, eps):
        if mode == 0:
            out, codes = portable_forward(tbl, a, ca, cb, mask, idx, chains, dlim, alim, with_codes=True, ln_eps=_ln_eps(eps))
        else:
            out, codes = portable_forward_ca(tbl, a, mask, idx, dlim, with_codes=True, ln_eps=_ln_eps(eps))
        ctx.save_for_backward(codes, tbl)
        ctx.mode, ctx.eps = mode, eps
        return out

    @staticmethod
    def backward(ctx, g):
        codes, tbl = ctx.saved_tensors
        g = g.to(torch.bfloat16).contiguous()
        if ctx.eps >= 0:
            g = portable_ln_backward(tbl, codes, g, ctx.mode, ctx.eps)
        return (portable_backward(g, codes, ctx.mode), *([None] * 10))


def pair_embed_geom_triton(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, eps, train: bool) -> torch.Tensor:
    """The encoder pair [B, n, n, 256] bf16 (LayerNorm of it when ``eps`` >= 0); ``train``: differentiable in ``tbl``."""
    if train:
        return PortablePairEmbedFunction.apply(tbl, 0, bb, ca, cb, mask, idx, chains, dlim, alim, eps)
    return portable_forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, ln_eps=_ln_eps(eps))


def pair_embed_ca_triton(tbl, ca_xyz, mask, idx, dlim, eps, train: bool) -> torch.Tensor:
    """The decoder pair [B, n, n, 256] bf16 (LayerNorm of it when ``eps`` >= 0); ``train``: differentiable in ``tbl``."""
    if train:
        return PortablePairEmbedFunction.apply(tbl, 1, ca_xyz, None, None, mask, idx, None, dlim, None, eps)
    return portable_forward_ca(tbl, ca_xyz, mask, idx, dlim, ln_eps=_ln_eps(eps))

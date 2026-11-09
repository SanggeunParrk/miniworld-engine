"""The CUDA backend of ``PairEmbedGeom`` / ``PairEmbedCA``: the launches of ``sm90`` (``pair_embed_geom.cu``, H100 sm_90a) as opaque ops and an autograd
function. Inputs are the modules' prepared tensors (``modules/pair_embed.py``): the bf16 table ``tbl`` ([317, 256] encoder, [157, 256] decoder, = W^T), fp32
coordinates, uint8 masks, int32 ``idx`` / ``chains`` and the fp32 bin limits. ``eps`` >= 0 folds the attention-bias LayerNorm (no affine, that eps) into
the kernel, ``eps`` < 0 returns the pair itself. The pair comes out bf16; the table gradient bf16, accumulated in fp32 in a fixed order (no atomics)."""
from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque

D_PAIR = 256
N_ROWS = {0: 317, 1: 157}

def _ln_eps(eps: float) -> float | None:
    return eps if eps >= 0 else None

def _pair_fake(tbl, idx):
    B, n = idx.shape
    return torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device)

def _pair_codes_fake(tbl, idx):
    """[pair [B, n, n, 256] bf16, codes [B, n, n, 4] int32]."""
    B, n = idx.shape
    return [torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device), torch.empty((B, n, n, 4), dtype=torch.int32, device=tbl.device)]

def _geom_fwd_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, eps):
    return _pair_fake(tbl, idx)

@opaque(fake=_geom_fwd_fake, name="pair_embed_geom_fwd_sm90")
def pair_embed_geom_fwd(tbl: torch.Tensor, bb: torch.Tensor, ca: torch.Tensor, cb: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor,
                        chains: torch.Tensor, dlim: torch.Tensor, alim: torch.Tensor, eps: float) -> torch.Tensor:
    """The encoder pair [B, n, n, 256] bf16 (forward only: no codes written), or its LayerNorm when ``eps`` >= 0."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return sm90.forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, ln_eps=_ln_eps(eps))

def _geom_fwd_train_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, eps):
    return _pair_codes_fake(tbl, idx)

@opaque(fake=_geom_fwd_train_fake, name="pair_embed_geom_fwd_train_sm90")
def pair_embed_geom_fwd_train(tbl: torch.Tensor, bb: torch.Tensor, ca: torch.Tensor, cb: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor,
                              chains: torch.Tensor, dlim: torch.Tensor, alim: torch.Tensor, eps: float) -> list[torch.Tensor]:
    """[pair (or its LayerNorm, ``eps`` >= 0) bf16, codes [B, n, n, 4] int32]: 16 bytes per pair (the packed bins with a live bit, the angle bins, the
    chain product), what the backward replays."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return list(sm90.forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=True, ln_eps=_ln_eps(eps)))

def _ca_fwd_fake(tbl, ca, mask, idx, dlim, eps):
    return _pair_fake(tbl, idx)

@opaque(fake=_ca_fwd_fake, name="pair_embed_ca_fwd_sm90")
def pair_embed_ca_fwd(tbl: torch.Tensor, ca: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor, dlim: torch.Tensor, eps: float) -> torch.Tensor:
    """The decoder pair [B, n, n, 256] bf16 (forward only), or its LayerNorm when ``eps`` >= 0."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return sm90.forward_ca(tbl, ca, mask, idx, dlim, ln_eps=_ln_eps(eps))

def _ca_fwd_train_fake(tbl, ca, mask, idx, dlim, eps):
    return _pair_codes_fake(tbl, idx)

@opaque(fake=_ca_fwd_train_fake, name="pair_embed_ca_fwd_train_sm90")
def pair_embed_ca_fwd_train(tbl: torch.Tensor, ca: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor, dlim: torch.Tensor,
                            eps: float) -> list[torch.Tensor]:
    """[decoder pair (or its LayerNorm) bf16, codes [B, n, n, 4] int32] (two live code words)."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return list(sm90.forward_ca(tbl, ca, mask, idx, dlim, with_codes=True, ln_eps=_ln_eps(eps)))

def _bwd_fake(g, codes, mode):
    return torch.empty((N_ROWS[mode], D_PAIR), dtype=torch.bfloat16, device=g.device)

@opaque(fake=_bwd_fake, name="pair_embed_geom_bwd_sm90")
def pair_embed_geom_bwd(g: torch.Tensor, codes: torch.Tensor, mode: int) -> torch.Tensor:
    """Table gradient [317 (mode 0) or 157 (mode 1), 256] bf16 from d pair [B, n, n, 256] bf16 (bitwise deterministic)."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return sm90.backward(g, codes, mode)

def _ln_bwd_fake(tbl, codes, dh, mode, eps):
    return torch.empty_like(dh)

@opaque(fake=_ln_bwd_fake, name="pair_embed_geom_ln_bwd_sm90")
def pair_embed_geom_ln_bwd(tbl: torch.Tensor, codes: torch.Tensor, dh: torch.Tensor, mode: int, eps: float) -> torch.Tensor:
    """d pair [B, n, n, 256] bf16 from d LayerNorm(pair) (recomputes the pair from the codes)."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return sm90.ln_backward(tbl, codes, dh, mode, eps)

class PairEmbedFunction(torch.autograd.Function):
    """``mode`` 0: the encoder pair (``a`` = bb [B, n, 4, 3]); 1: the decoder pair (``a`` = CA [B, n, 3]; ``cb`` / ``chains`` / ``alim`` unused)."""

    @staticmethod
    def forward(ctx, tbl, mode, a, ca, cb, mask, idx, chains, dlim, alim, eps):
        if mode == 0:
            out, codes = pair_embed_geom_fwd_train(tbl, a, ca, cb, mask, idx, chains, dlim, alim, eps)
        else:
            out, codes = pair_embed_ca_fwd_train(tbl, a, mask, idx, dlim, eps)
        ctx.save_for_backward(codes, tbl)
        ctx.mode, ctx.eps = mode, eps
        return out

    @staticmethod
    def backward(ctx, g):
        codes, tbl = ctx.saved_tensors
        g = g.to(torch.bfloat16).contiguous()
        if ctx.eps >= 0:
            g = pair_embed_geom_ln_bwd(tbl, codes, g, ctx.mode, ctx.eps)
        return (pair_embed_geom_bwd(g, codes, ctx.mode), *([None] * 10))

def pair_embed_geom_cuda(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, eps, train: bool) -> torch.Tensor:
    """The encoder pair [B, n, n, 256] bf16 (LayerNorm of it when ``eps`` >= 0); ``train``: differentiable in ``tbl``."""
    if train:
        return PairEmbedFunction.apply(tbl, 0, bb, ca, cb, mask, idx, chains, dlim, alim, eps)
    return pair_embed_geom_fwd(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, eps)

def pair_embed_ca_cuda(tbl, ca_xyz, mask, idx, dlim, eps, train: bool) -> torch.Tensor:
    """The decoder pair [B, n, n, 256] bf16 (LayerNorm of it when ``eps`` >= 0); ``train``: differentiable in ``tbl``."""
    if train:
        return PairEmbedFunction.apply(tbl, 1, ca_xyz, None, None, mask, idx, None, dlim, None, eps)
    return pair_embed_ca_fwd(tbl, ca_xyz, mask, idx, dlim, eps)

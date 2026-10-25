"""Drivers for the ``pair_embed_geom`` family (the portable kernels of the encoder's backbone pair and the decoder's CA-distance pair).

The activation of these kernels is a pair ``[B, L, L, 256]`` (bf16 out); their inputs are the backbone coordinates (fp32 nm, by definition of the op), the
bf16 autocast table and the per-residue masks. Both values of ``LN`` are real (the attention-bias LayerNorm folded in or not) and are driven; the table-gradient kernel is driven at both modes (encoder table 317 rows, decoder table 157).
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, dev, driver_length, ragged

_B = 2
_L = ragged(driver_length(128))
_EPS = 1e-5


def geom_inputs(b: int = _B, n: int = _L):
    """``(table, bb, ca, cb, mask, idx, chains, dlim, alim)`` for the encoder kernel: a random-walk backbone, ragged lengths, a few missing CA / CB, three chains."""
    d = dev()
    steps = torch.randn(b, n, 3, device=d)
    ca = (steps / steps.norm(dim=-1, keepdim=True) * 0.38).cumsum(1)
    off = torch.randn(b, n, 3, 3, device=d) * 0.12
    bb = torch.stack([ca + off[:, :, 0], ca, ca + off[:, :, 1], ca + off[:, :, 2]], 2).contiguous()
    lengths = torch.randint(max(1, n * 3 // 4), n + 1, (b,), device=d)
    lengths[0] = n
    mask = torch.arange(n, device=d)[None] < lengths[:, None]
    has_ca = mask & (torch.rand(b, n, device=d) > 0.03)
    has_cb = has_ca & (torch.rand(b, n, device=d) > 0.1)
    bb[..., 3, :] = bb[..., 3, :] * has_cb[..., None]
    idx = torch.randint(1, 4, (b, n), device=d).cumsum(1)
    chains = torch.where(mask, torch.randint(0, 3, (b, n), device=d), torch.full((b, n), -1, device=d))
    tbl = (torch.randn(317, 256, device=d) * 0.3).to(torch.bfloat16)
    return (tbl, bb, has_ca.to(torch.uint8), has_cb.to(torch.uint8), mask.to(torch.uint8), idx.to(torch.int32), chains.to(torch.int32),
            torch.linspace(0.1, 2, 20, device=d), torch.linspace(-torch.pi, torch.pi, 20, device=d))


def ca_inputs(b: int = _B, n: int = _L):
    """``(table, ca, mask, idx, dlim)`` for the decoder kernel (table 157 rows)."""
    tbl, bb, _, _, mask, idx, _, _, _ = geom_inputs(b, n)
    d = dev()
    return ((torch.randn(157, 256, device=d) * 0.3).to(torch.bfloat16), (bb[:, :, 1] * mask[..., None]).contiguous(), mask, idx,
            torch.linspace(0.1, 3.0, 29, device=d))


def pair_embed_geom_bins_triton() -> None:
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward

    portable_forward(*geom_inputs(), with_codes=True)


def pair_embed_ca_bins_triton() -> None:
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward_ca

    portable_forward_ca(*ca_inputs(), with_codes=True)


def pair_embed_geom_fwd_gather_triton() -> None:
    """The gather at both tables and both values of ``LN`` (the bins kernels run in front of it)."""
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward, portable_forward_ca

    for args, fn in ((geom_inputs(), portable_forward), (ca_inputs(), portable_forward_ca)):
        fn(*args)
        fn(*args, ln_eps=_EPS)


def _codes_and_grad(mode: int):
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward, portable_forward_ca

    if mode == 0:
        args = geom_inputs()
        _, codes = portable_forward(*args, with_codes=True)
    else:
        args = ca_inputs()
        _, codes = portable_forward_ca(*args, with_codes=True)
    return args[0], codes, torch.randn(_B, _L, _L, 256, device=dev(), dtype=BF16).to(torch.bfloat16)


def pair_embed_geom_dw_triton() -> None:
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_backward

    for mode in (0, 1):
        _, codes, g = _codes_and_grad(mode)
        portable_backward(g, codes, mode)


def pair_embed_geom_bwd_reduce_triton() -> None:
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_backward

    _, codes, g = _codes_and_grad(0)
    portable_backward(g, codes, 0)


def pair_embed_geom_bwd_ln_triton() -> None:
    from miniworld_engine.kernels.pair_embed_geom.triton import portable_ln_backward

    for mode in (0, 1):
        tbl, codes, g = _codes_and_grad(mode)
        portable_ln_backward(tbl, codes, g, mode, _EPS)

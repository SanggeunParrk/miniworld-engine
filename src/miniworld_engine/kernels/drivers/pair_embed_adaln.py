"""Drivers for the ``pair_embed_adaln`` family (portable kernels of La-Proteina's initial pair builder).

The activation is a pair ``[B, L, L, 256]`` (bf16 out); the table, the LayerNorm affine and the per-sample gamma / beta are fp32 parameters (as the engine keeps
them), the coordinates fp32 nm. Both values of ``HAS_SC`` (with / without self-conditioning) are driven.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import dev, driver_length, ragged

_B = 2
_L = ragged(driver_length(128))
_NS, _NX, _NC = 127, 30, 30
_EPS = 1e-5


def adaln_inputs(has_sc: int = 1, b: int = _B, n: int = _L):
    """``(tbl, lnw, lnb, gamma, beta, xt, sc, has_sc, idx, mask, xlim, clim, nS, nX, eps1, eps2)`` of ``forward``."""
    d = dev()
    r = lambda *s: torch.randn(*s, device=d)
    lengths = torch.randint(max(1, n * 3 // 4), n + 1, (b,), device=d)
    lengths[0] = n
    mask = (torch.arange(n, device=d)[None] < lengths[:, None]).to(torch.uint8)
    idx = torch.randint(1, 4, (b, n), device=d).cumsum(1).to(torch.int32)
    return (r(_NS + _NX + _NC, 256) * 0.3, 1 + 0.1 * r(256), 0.1 * r(256), torch.sigmoid(r(b, 256)), 0.2 * r(b, 256),
            torch.cumsum(r(b, n, 3) * 0.2, 1), torch.cumsum(r(b, n, 3) * 0.2, 1), has_sc, idx, mask,
            torch.linspace(0.1, 3.0, _NX - 1, device=d), torch.linspace(0.1, 3.0, _NC - 1, device=d), _NS, _NX, _EPS, _EPS)


def _bwd_args(args, bins):
    return (*args[:5], bins, torch.randn(args[5].shape[0], args[5].shape[1], args[5].shape[1], 256, device=dev()).to(torch.bfloat16), _NS, _NX, _EPS, _EPS)


def pair_embed_adaln_fwd_triton() -> None:
    from miniworld_engine.kernels.pair_embed_adaln.triton import portable_forward

    for has_sc in (1, 0):
        portable_forward(*adaln_inputs(has_sc))


def _backward(has_sc: int = 1):
    from miniworld_engine.kernels.pair_embed_adaln.triton import portable_backward, portable_forward

    args = adaln_inputs(has_sc)
    _, bins = portable_forward(*args)
    tbl, lnw, lnb, gamma, _, _, _, _, _, _, _, _, nS, nX, e1, e2 = args
    g = torch.randn(bins.shape[0], bins.shape[1], bins.shape[1], 256, device=dev()).to(torch.bfloat16)
    return portable_backward(tbl, lnw, lnb, gamma, bins, g, nS, nX, e1, e2)


def pair_embed_adaln_bwd_triton() -> None:
    _backward()


def pair_embed_adaln_dw_triton() -> None:
    _backward()
    _backward(0)


def pair_embed_adaln_bwd_reduce_triton() -> None:
    _backward()

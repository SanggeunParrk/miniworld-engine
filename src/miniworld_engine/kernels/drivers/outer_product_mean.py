"""Driver for the ``outer_product_mean`` family.

``triton_outer_product_mean`` over one MSA stack [1, S, L, d_msa] with a row mask and the pair residual, as the MSA module
calls it. d_msa / d_hidden are the model's 64 / 32 (each spanned by one ``tl.arange`` tile, so they stay aligned); d_pair is
the swept width. S is the MSA depth the trunk runs at; it is not in the autotune key (the S-dependent kernels iterate it).
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import (
    BF16,
    aligned_only,
    dev,
    driver_length,
    driver_width,
    ragged,
)

_L = ragged(driver_length(384))
_S = ragged(1024)
_CM = aligned_only("outer_product_mean.d_msa", 64, "one tl.arange tile spans the MSA channels: a power of two")
_CH = aligned_only("outer_product_mean.d_hidden", 32, "one tl.arange tile spans the hidden channels: a power of two")
_CZ = aligned_only("outer_product_mean.d_pair", driver_width(128), "tiled by the output ladder, whose smallest rung is 32")


def _inputs(dtype=BF16):
    g = torch.Generator(device="cpu").manual_seed(0)
    r = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).to(dev(), dtype)
    msa = r(1, _S, _L, _CM)
    mask = (torch.rand(1, _S, _L, generator=g) > 0.1).to(dev())
    weights = (1.0 + r(_CM, scale=0.1), r(_CM, scale=0.1), r(_CH, _CM, scale=_CM ** -0.5), r(_CH, _CM, scale=_CM ** -0.5),
               r(_CZ, _CH * _CH, scale=_CH ** -1), r(_CZ, scale=0.1))
    residual = r(1, _L, _L, _CZ)
    return msa, mask, weights, residual


def _forward() -> None:
    from miniworld_engine.kernels.outer_product_mean.interface import (
        triton_outer_product_mean,
    )

    msa, mask, weights, residual = _inputs()
    with torch.no_grad():
        triton_outer_product_mean(msa, mask, *weights, residual=residual)


def _backward() -> None:
    from miniworld_engine.kernels.outer_product_mean.interface import (
        triton_outer_product_mean,
    )

    msa, mask, weights, residual = _inputs()
    leaves = [t.requires_grad_() for t in (msa, *weights, residual)]
    out = triton_outer_product_mean(leaves[0], mask, *leaves[1:7], residual=leaves[7])
    out.backward(torch.randn_like(out))


def outer_product_mean_layernorm_gemm_triton() -> None:
    _forward()


def outer_product_mean_epilogue_triton() -> None:
    _forward()


def outer_product_mean_bwd_epilogue_dx_triton() -> None:
    _backward()


def outer_product_mean_bwd_epilogue_dw_triton() -> None:
    _backward()


def outer_product_mean_bwd_layernorm_gemm_triton() -> None:
    _backward()

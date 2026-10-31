"""Driver for the ``outer_product_mean`` family.

``triton_outer_product_mean`` over one MSA stack [1, S, L, d_msa] with a row mask and the pair residual, as the MSA module
calls it. d_msa / d_hidden are the model's 64 / 32 (each spanned by one ``tl.arange`` tile, so they stay aligned); d_pair is
the swept width. S is the MSA depth the trunk runs at; it is not in the autotune key (the S-dependent kernels iterate it).

Three variants are driven: the base call, RoseTTAFold3's projection biases, and Chai-1's 8 groups (d_hidden 64, 512 products, its own
product LayerNorm, no pair-count division). The projection kernels run all three (each is a different constexpr specialisation, so a
different tuning); the epilogue kernels of the base path run the base call and the grouped ones the grouped call.
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
_GROUPS, _CH_GROUPED = 8, 64
_CZ = aligned_only("outer_product_mean.d_pair", driver_width(128), "tiled by the output ladder, whose smallest rung is 32")


def _inputs(dtype=BF16, variant="base"):
    """(msa, mask, weights, residual, options): ``options`` are triton_outer_product_mean's keyword arguments of the variant."""
    g = torch.Generator(device="cpu").manual_seed(0)
    r = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).to(dev(), dtype)
    ch = _CH_GROUPED if variant == "grouped" else _CH
    msa = r(1, _S, _L, _CM)
    mask = (torch.rand(1, _S, _L, generator=g) > 0.1).to(dev())
    nprod = _GROUPS * (ch // _GROUPS) ** 2 if variant == "grouped" else ch * ch
    weights = (1.0 + r(_CM, scale=0.1), r(_CM, scale=0.1), r(ch, _CM, scale=_CM ** -0.5), r(ch, _CM, scale=_CM ** -0.5),
               r(_CZ, nprod, scale=nprod ** -0.5 if variant == "grouped" else _CH ** -1), r(_CZ, scale=0.1))
    residual = r(1, _L, _L, _CZ)
    options = {}
    if variant == "pbias":
        options = {"b_left": r(ch, scale=0.3), "b_right": r(ch, scale=0.3)}
    elif variant == "grouped":
        options = {"groups": _GROUPS, "product_norm_weight": 1.0 + r(nprod, scale=0.1), "product_norm_bias": r(nprod, scale=0.1),
                   "product_norm_eps": 0.1}
    return msa, mask, weights, residual, options


VARIANTS = ("base", "pbias", "grouped")


#: Both sides of the boolean key flags: masked / unmasked rows (HAS_MASK), with / without the pair residual (HAS_RES), and the pair-count
#: division before / after the output projection (NORM_FIRST: ``normalize_before_proj``, RoseTTAFold3's divide-after form). Each side is
#: its own constexpr specialisation and its own cache bucket, so a driver that calls one side leaves the other untuned.
_MASK_SIDES = (True, False)
_EPILOGUE_SIDES = ((True, True), (False, True), (True, False))   # (residual, normalize_before_proj)


def _forward(variants=("base",), masks=(True,), epilogues=((True, True),)) -> None:
    from miniworld_engine.kernels.outer_product_mean.interface import (
        triton_outer_product_mean,
    )

    for variant in variants:
        for use_mask in masks:
            for use_res, norm_first in epilogues:
                msa, mask, weights, residual, options = _inputs(variant=variant)
                if variant != "grouped":
                    options = {**options, "normalize_before_proj": norm_first}
                with torch.no_grad():
                    triton_outer_product_mean(msa, mask if use_mask else None, *weights, residual=residual if use_res else None,
                                              **options)


def _backward(variants=("base",), masks=(True,)) -> None:
    from miniworld_engine.kernels.outer_product_mean.interface import (
        triton_outer_product_mean,
    )

    for variant in variants:
        for use_mask in masks:
            msa, mask, weights, residual, options = _inputs(variant=variant)
            leaves = [t.requires_grad_() for t in (msa, *weights, residual)]
            options = {k: v.requires_grad_() if isinstance(v, torch.Tensor) else v for k, v in options.items()}
            out = triton_outer_product_mean(leaves[0], mask if use_mask else None, *leaves[1:7], residual=leaves[7], **options)
            out.backward(torch.randn_like(out))


def outer_product_mean_layernorm_gemm_triton() -> None:
    _forward(VARIANTS, masks=_MASK_SIDES)


def outer_product_mean_epilogue_triton() -> None:
    _forward(epilogues=_EPILOGUE_SIDES)


def outer_product_mean_bwd_epilogue_dx_triton() -> None:
    _backward()


def outer_product_mean_bwd_epilogue_dw_triton() -> None:
    _backward()


def outer_product_mean_bwd_layernorm_gemm_triton() -> None:
    _backward(VARIANTS, masks=_MASK_SIDES)


def outer_product_mean_epilogue_ln_triton() -> None:
    _forward(("grouped",))


def outer_product_mean_bwd_epilogue_ln_dx_triton() -> None:
    _backward(("grouped",))


def outer_product_mean_bwd_epilogue_ln_dw_triton() -> None:
    _backward(("grouped",))

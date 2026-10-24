"""Numerical checks for the ``outer_product_mean`` family, against `reference.outer_product_mean_reference` in fp32 on the
driver's inputs. Each backward kernel is scored on the gradients it produces. The projection kernels (and the base epilogue's
neighbours) are scored on three variants -- the base call, RoseTTAFold3's projection biases (``_pbias``) and Chai-1's 8 groups
(``_grouped``) -- and the grouped epilogues on the grouped one; the base names keep their historical keys."""
from __future__ import annotations

from miniworld_engine.kernels.checks import _f, _fixed, _grads, _no_tf32
from miniworld_engine.kernels.drivers.outer_product_mean import VARIANTS, _inputs

_NAMES = ("dmsa", "dln_weight", "dln_bias", "dw_left", "dw_right", "dw_out", "db_out", "dresidual")
_OPTION_NAMES = {"pbias": ("db_left", "db_right"), "grouped": ("dproduct_weight", "dproduct_bias")}
_OPTION_KEYS = {"pbias": ("b_left", "b_right"), "grouped": ("product_norm_weight", "product_norm_bias")}


def _suffix(variant: str) -> str:
    return "" if variant == "base" else f"_{variant}"


def _forward_pair(variants=("base",)):
    from miniworld_engine.kernels.outer_product_mean.interface import (
        triton_outer_product_mean,
    )
    from miniworld_engine.kernels.outer_product_mean.reference import (
        outer_product_mean_reference,
    )

    _fixed()
    pairs = {}
    for variant in variants:
        msa, mask, weights, residual, options = _inputs(variant=variant)
        with _no_tf32():
            expected = outer_product_mean_reference(_f(msa), mask, *map(_f, weights), residual=_f(residual),
                                                    **{k: _f(v) if hasattr(v, "dtype") else v for k, v in options.items()})
        pairs["out" + _suffix(variant)] = (triton_outer_product_mean(msa, mask, *weights, residual=residual, **options), expected)
    return pairs


def _backward_pairs(names, variants=("base",), extra=None):
    """``names``: the gradient keys scored; ``extra``: per variant, the keys it adds (db_left / db_right, dproduct_weight / dproduct_bias)."""
    from miniworld_engine.kernels.outer_product_mean.interface import (
        triton_outer_product_mean,
    )
    from miniworld_engine.kernels.outer_product_mean.reference import (
        outer_product_mean_reference,
    )

    _fixed()
    pairs = {}
    for variant in variants:
        msa, mask, weights, residual, options = _inputs(variant=variant)
        keys = _OPTION_KEYS.get(variant, ())
        fixed = {k: v for k, v in options.items() if k not in keys}

        def call(fn, m, w, _mask=mask, _keys=keys, _fixed_opts=fixed):
            return fn(m, _mask, *w[:6], residual=w[6], **dict(zip(_keys, w[7:], strict=True)), **_fixed_opts)

        with _no_tf32():
            got = _grads(lambda m, *w: call(triton_outer_product_mean, m, w), [msa, *weights, residual, *(options[k] for k in keys)],
                         lambda m, *w: call(outer_product_mean_reference, m, w), _NAMES + _OPTION_NAMES.get(variant, ()))
        wanted = tuple(names) + (extra or {}).get(variant, ())
        pairs.update({n + _suffix(variant): got[n] for n in wanted if n in got})
    return pairs


def outer_product_mean_layernorm_gemm_triton():
    return _forward_pair(VARIANTS)


def outer_product_mean_epilogue_triton():
    return _forward_pair()


def outer_product_mean_bwd_epilogue_dx_triton():
    """dO feeds dmsa through the projections; dz / n itself is db_out's sum."""
    return _backward_pairs(("dmsa", "db_out", "dresidual"))


def outer_product_mean_bwd_epilogue_dw_triton():
    return _backward_pairs(("dw_out",))


def outer_product_mean_bwd_layernorm_gemm_triton():
    return _backward_pairs(("dmsa", "dln_weight", "dln_bias", "dw_left", "dw_right"), VARIANTS, {"pbias": ("db_left", "db_right")})


def outer_product_mean_epilogue_ln_triton():
    return _forward_pair(("grouped",))


def outer_product_mean_bwd_epilogue_ln_dx_triton():
    """dO (scored through dmsa), the product LayerNorm's dgamma / dbeta and the residual's pass-through."""
    return _backward_pairs(("dmsa", "db_out", "dresidual", "dproduct_weight", "dproduct_bias"), ("grouped",))


def outer_product_mean_bwd_epilogue_ln_dw_triton():
    return _backward_pairs(("dw_out",), ("grouped",))

"""Numerical checks for the ``outer_product_mean`` family, against `reference.outer_product_mean_reference` in fp32 on the
driver's inputs. Each backward kernel is scored on the gradients it produces."""
from __future__ import annotations

from miniworld_engine.kernels.checks import _f, _fixed, _grads, _no_tf32
from miniworld_engine.kernels.drivers.outer_product_mean import _inputs

_NAMES = ("dmsa", "dln_weight", "dln_bias", "dw_left", "dw_right", "dw_out", "db_out", "dresidual")


def _forward_pair():
    from miniworld_engine.kernels.outer_product_mean.interface import (
        triton_outer_product_mean,
    )
    from miniworld_engine.kernels.outer_product_mean.reference import (
        outer_product_mean_reference,
    )

    _fixed()
    msa, mask, weights, residual = _inputs()
    with _no_tf32():
        expected = outer_product_mean_reference(_f(msa), mask, *map(_f, weights), residual=_f(residual))
    return {"out": (triton_outer_product_mean(msa, mask, *weights, residual=residual), expected)}


def _backward_pairs(*names):
    from miniworld_engine.kernels.outer_product_mean.interface import (
        triton_outer_product_mean,
    )
    from miniworld_engine.kernels.outer_product_mean.reference import (
        outer_product_mean_reference,
    )

    _fixed()
    msa, mask, weights, residual = _inputs()
    with _no_tf32():
        got = _grads(lambda m, *w: triton_outer_product_mean(m, mask, *w[:6], residual=w[6]), [msa, *weights, residual],
                     lambda m, *w: outer_product_mean_reference(m, mask, *w[:6], residual=w[6]), _NAMES)
    return {n: got[n] for n in names}


def outer_product_mean_layernorm_gemm_triton():
    return _forward_pair()


def outer_product_mean_epilogue_triton():
    return _forward_pair()


def outer_product_mean_bwd_epilogue_dx_triton():
    """dO feeds dmsa through the projections; dz / n itself is db_out's sum."""
    return _backward_pairs("dmsa", "db_out", "dresidual")


def outer_product_mean_bwd_epilogue_dw_triton():
    return _backward_pairs("dw_out")


def outer_product_mean_bwd_layernorm_gemm_triton():
    return _backward_pairs("dmsa", "dln_weight", "dln_bias", "dw_left", "dw_right")

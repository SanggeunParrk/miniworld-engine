"""Numerical checks for the ``pair_weighted_averaging`` family, against `reference.pair_weighted_averaging_reference` in fp32 on
the driver's inputs (same keep-mask on both sides). Each backward kernel is scored on the gradients it produces."""
from __future__ import annotations

from miniworld_engine.kernels.checks import _f, _fixed, _grads, _no_tf32
from miniworld_engine.kernels.drivers.pair_weighted_averaging import _P_DROP, _inputs

_NAMES = ("dmsa", "dpair", "dln_msa_weight", "dln_msa_bias", "dw_value", "dw_gate", "dln_pair_weight", "dw_bias",
          "dw_out")


def _forward_pair():
    from miniworld_engine.kernels.pair_weighted_averaging.interface import (
        triton_pair_weighted_averaging,
    )
    from miniworld_engine.kernels.pair_weighted_averaging.reference import (
        pair_weighted_averaging_reference,
    )

    _fixed()
    msa, pair, mask, weights, _keep = _inputs()
    with _no_tf32():
        expected = pair_weighted_averaging_reference(_f(msa), _f(pair), mask, *map(_f, weights))
    return {"out": (triton_pair_weighted_averaging(msa, pair, mask, *weights), expected)}


def _backward_pairs(*names):
    from miniworld_engine.kernels.pair_weighted_averaging.interface import (
        triton_pair_weighted_averaging,
    )
    from miniworld_engine.kernels.pair_weighted_averaging.reference import (
        pair_weighted_averaging_reference,
    )

    _fixed()
    msa, pair, mask, weights, keep = _inputs()
    with _no_tf32():
        got = _grads(lambda m, z, *w: triton_pair_weighted_averaging(m, z, mask, *w, keep=keep, p_drop=_P_DROP),
                     [msa, pair, *weights],
                     lambda m, z, *w: pair_weighted_averaging_reference(m, z, mask, *w, keep=keep, p_drop=_P_DROP), _NAMES)
    return {n: got[n] for n in names}


def pair_weighted_averaging_layernorm_gemm_softmax_triton():
    return _forward_pair()


def pair_weighted_averaging_layernorm_gemm_triton():
    return _forward_pair()


def pair_weighted_averaging_gate_gemm_triton():
    return _forward_pair()


def pair_weighted_averaging_bwd_gate_gemm_triton():
    return _backward_pairs("dmsa", "dw_gate", "dw_out")


def pair_weighted_averaging_bwd_layernorm_gemm_dw_triton():
    return _backward_pairs("dw_value")


def pair_weighted_averaging_bwd_layernorm_gemm_dx_dlnw_triton():
    return _backward_pairs("dmsa", "dln_msa_weight", "dln_msa_bias")


def pair_weighted_averaging_bwd_layernorm_gemm_softmax_triton():
    """dpair, dgamma and dWb are scored. The pair LayerNorm has no offset (it would shift every key's logit of a head by one
    constant, which a softmax over keys cancels)."""
    return _backward_pairs("dpair", "dln_pair_weight", "dw_bias")

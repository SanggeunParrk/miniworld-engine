"""Driver for the ``pair_weighted_averaging`` family.

``triton_pair_weighted_averaging`` over one MSA stack [1, S, L, d_msa] and its pair [1, L, L, d_pair] with a key mask and the
module's row-broadcast dropout, as MSAPairWeightedAveraging calls it in training. d_msa / d_hidden are the model's 64 / 32
(8 heads), each spanned by one ``tl.arange`` tile; d_pair is the swept width (padded to a power of two inside the kernels).
The head width is driven twice: at 32 and at 8, AF3's ``msa_value_dim`` (OpenFold3, Protenix, OpenDDE use it too), which the kernels
zero-pad to a 16-column tile (a different shape bucket, so a different tuning).
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
_D = aligned_only("pair_weighted_averaging.d_msa", 64, "one tl.arange tile spans the MSA channels: a power of two")
_NH = 8
_C = aligned_only("pair_weighted_averaging.d_hidden", 32, "one tl.arange tile spans a head's channels: a power of two")
_DZ = ragged(driver_width(128))
_P_DROP = 0.15
_VALUE_DIMS = (_C, 8)


def _inputs(dtype=BF16, c=_C):
    g = torch.Generator(device="cpu").manual_seed(0)
    r = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).to(dev(), dtype)
    msa = r(1, _S, _L, _D)
    pair = r(1, _L, _L, _DZ)
    mask = (torch.rand(1, _L, generator=g) > 0.1).to(dev())
    hc = _NH * c
    weights = (1.0 + r(_D, scale=0.1), r(_D, scale=0.1), r(hc, _D, scale=_D ** -0.5), r(hc, _D, scale=_D ** -0.5),
               1.0 + r(_DZ, scale=0.1), r(_NH, _DZ, scale=_DZ ** -0.5), r(_D, hc, scale=hc ** -0.5))
    keep = torch.rand(1, _L, _D, generator=g).to(dev()) > _P_DROP
    return msa, pair, mask, weights, keep


#: Both sides of the boolean key flags: a key mask or none (HAS_MASK), dropout or none (HAS_KEEP, eval / p_drop = 0).
_SIDES = (True, False)


def _forward(masks=(True,)) -> None:
    from miniworld_engine.kernels.pair_weighted_averaging.interface import (
        triton_pair_weighted_averaging,
    )

    for c in _VALUE_DIMS:
        for use_mask in masks:
            msa, pair, mask, weights, _keep = _inputs(c=c)
            with torch.no_grad():
                triton_pair_weighted_averaging(msa, pair, mask if use_mask else None, *weights)


def _backward(masks=(True,), keeps=(True,)) -> None:
    from miniworld_engine.kernels.pair_weighted_averaging.interface import (
        triton_pair_weighted_averaging,
    )

    for c in _VALUE_DIMS:
        for use_mask in masks:
            for use_keep in keeps:
                msa, pair, mask, weights, keep = _inputs(c=c)
                leaves = [t.requires_grad_() for t in (msa, pair, *weights)]
                drop = {"keep": keep, "p_drop": _P_DROP} if use_keep else {}
                out = triton_pair_weighted_averaging(leaves[0], leaves[1], mask if use_mask else None, *leaves[2:], **drop)
                out.backward(torch.randn_like(out))


def pair_weighted_averaging_layernorm_gemm_softmax_triton() -> None:
    _forward(masks=_SIDES)


def pair_weighted_averaging_layernorm_gemm_triton() -> None:
    _forward()


def pair_weighted_averaging_gate_gemm_triton() -> None:
    _forward()


def pair_weighted_averaging_bwd_gate_gemm_triton() -> None:
    _backward(keeps=_SIDES)


def pair_weighted_averaging_bwd_layernorm_gemm_dw_triton() -> None:
    _backward()


def pair_weighted_averaging_bwd_layernorm_gemm_dx_dlnw_triton() -> None:
    _backward()


def pair_weighted_averaging_bwd_layernorm_gemm_softmax_triton() -> None:
    _backward(masks=_SIDES)

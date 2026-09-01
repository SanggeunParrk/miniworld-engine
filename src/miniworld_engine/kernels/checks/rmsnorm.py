"""Numerical checks for the ``rmsnorm`` family, against the written-out formula.

The reference is `rmsnorm/reference.py`, not `F.rms_norm`: comparing one implementation against
another checks that they agree, not that either is right.

Widths come from `drivers/rmsnorm.py` rather than being written again here, so a ragged-mode run
checks the same partial tiles it builds, and so the two files cannot drift into checking a shape
nothing tunes. Each checker covers BOTH values of `HAS_WEIGHT` for the same reason the drivers
launch both: they are separate compiled kernels.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import grads_of
from miniworld_engine.kernels.drivers import BF16, dev, vec
from miniworld_engine.kernels.drivers.rmsnorm import _D, _DM, _EPS, _M


def _x(d: int) -> torch.Tensor:
    """PRE-flatten, for the reason `drivers/rmsnorm.py::_rows` gives."""
    return torch.randn(1, _M, d, device=dev(), dtype=BF16)



# ── rmsnorm ──────────────────────────────────────────────────────────────────────────────────


def rmsnorm_fwd_triton():
    """rmsnorm_fwd_kernel at all four (HAS_WEIGHT, HAS_MODULATION) combinations.

    One checker, matching the one driver and the one config ladder: the modulate form is this
    same kernel with a constexpr flipped. See `drivers/rmsnorm.py::rmsnorm_fwd_triton`.
    """
    from miniworld_engine.kernels.rmsnorm.interface import (
        triton_rmsnorm,
        triton_rmsnorm_modulate,
    )
    from miniworld_engine.kernels.rmsnorm.reference import (
        rmsnorm_modulate_reference,
        rmsnorm_reference,
    )

    out = {}
    x = _x(_D)
    for tag, w in (("aff", vec(_D)), ("plain", None)):
        out[f"y_{tag}"] = (triton_rmsnorm(x, w, _EPS), rmsnorm_reference(x, w, _EPS))
    xm, sc, sh = _x(_DM), _x(_DM), _x(_DM)
    for tag, w in (("aff", vec(_DM)), ("plain", None)):
        out[f"ymod_{tag}"] = (triton_rmsnorm_modulate(xm, sc, sh, w, _EPS),
                              rmsnorm_modulate_reference(xm, sc, sh, w, _EPS))
    return out


def rmsnorm_bwd_triton():
    """rmsnorm_bwd_kernel: dx and dweight, against autograd over the reference."""
    from miniworld_engine.kernels.rmsnorm.interface import (
        triton_rmsnorm,
        triton_rmsnorm_modulate,
    )
    from miniworld_engine.kernels.rmsnorm.reference import (
        rmsnorm_modulate_reference,
        rmsnorm_reference,
    )

    out = {}
    x0 = _x(_D)
    da = torch.randn_like(x0)
    for tag, w0 in (("aff", vec(_D)), ("plain", None)):
        ins = [x0] if w0 is None else [x0, w0]
        # The None has to be PASSED, not omitted: `weight` is positional, so dropping it hands
        # `eps` over as the weight.
        pad = (lambda x: (x, None)) if w0 is None else (lambda x, w: (x, w))
        got = grads_of(lambda *t, _p=pad: triton_rmsnorm(*_p(*t), _EPS), ins, da)
        ref = grads_of(lambda *t, _p=pad: rmsnorm_reference(*_p(*t), _EPS), ins, da)
        names = ["dx"] if w0 is None else ["dx", "dweight"]
        for n, g, r in zip(names, got, ref, strict=True):
            out[f"{n}_{tag}"] = (g, r)

    # HAS_MODULATION=True: the same kernel, the same ladder, its own cache bucket. dshift is
    # checked although the kernel never computes it -- it IS dy, returned by reshaping, and a
    # reshape that lost a stride would show up here and nowhere else.
    xm0, sc0, sh0 = _x(_DM), _x(_DM), _x(_DM)
    dm = torch.randn_like(xm0)
    for tag, w0 in (("aff", vec(_DM)), ("plain", None)):
        base = [xm0, sc0, sh0]
        ins = base if w0 is None else [*base, w0]
        order = ((lambda x, s_, h: (x, s_, h, None)) if w0 is None
                 else (lambda x, s_, h, w: (x, s_, h, w)))
        got = grads_of(lambda *t, _o=order: triton_rmsnorm_modulate(*_o(*t), _EPS), ins, dm)
        ref = grads_of(lambda *t, _o=order: rmsnorm_modulate_reference(*_o(*t), _EPS), ins, dm)
        names = ["dxmod", "dscale", "dshift"] + ([] if w0 is None else ["dweightmod"])
        for n, g, r in zip(names, got, ref, strict=True):
            out[f"{n}_{tag}"] = (g, r)
    return out

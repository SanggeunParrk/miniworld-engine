"""Numerical checks for the ``rmsnorm`` family, against the written-out formula.

The reference is `rmsnorm/reference.py`, not `F.rms_norm`: comparing one implementation against
another checks that they agree, not that either is right.

Widths come from `drivers/rmsnorm.py` rather than being written again here, so a ragged-mode run
checks the same partial tiles it builds, and so the two files cannot drift into checking a shape
nothing tunes. Each checker covers BOTH values of `HAS_WEIGHT`, which are separate compiled kernels. The
adaLN modulate that used to live here as a third case is now the `rmsnorm_adamod` family.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import grads_of
from miniworld_engine.kernels.drivers import BF16, dev, vec
from miniworld_engine.kernels.drivers.rmsnorm import _D, _EPS, _M


def _x(d: int) -> torch.Tensor:
    """PRE-flatten, for the reason `drivers/rmsnorm.py::_rows` gives."""
    return torch.randn(1, _M, d, device=dev(), dtype=BF16)



# ── rmsnorm ──────────────────────────────────────────────────────────────────────────────────


def rmsnorm_fwd_triton():
    """rmsnorm_fwd_kernel at both values of HAS_WEIGHT."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_reference

    out = {}
    x = _x(_D)
    for tag, w in (("aff", vec(_D)), ("plain", None)):
        out[f"y_{tag}"] = (triton_rmsnorm(x, w, _EPS), rmsnorm_reference(x, w, _EPS))
    return out


def rmsnorm_bwd_triton():
    """rmsnorm_bwd_kernel: dx and dweight, against autograd over the reference."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_reference

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
    return out

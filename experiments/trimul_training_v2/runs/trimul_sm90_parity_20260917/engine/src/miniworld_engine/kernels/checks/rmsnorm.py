"""Numerical checks for the ``rmsnorm`` family, against the written-out formula.

The reference is `rmsnorm/reference.py`, not `F.rms_norm`: comparing one implementation against
another checks that they agree, not that either is right.

Reference inputs and leaves are FP32 copies of the exact measured values. Do not
round the reference output or its gradients to BF16: two almost identical FP32
results can straddle a BF16 rounding boundary and differ by a full BF16 step.
The registry bands measure kernel error against the unrounded FP32 formula, as
the LayerNorm checkers do. The BF16 forward band is 4e-3: its former 3.2e-3
band was below BF16's 2**-8 rounding bound. FP32 and backward bands are unchanged.

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
        reference = rmsnorm_reference(x.float(), None if w is None else w.float(), _EPS)
        out[f"y_{tag}"] = (triton_rmsnorm(x, w, _EPS), reference)
    return out


def _xw(t: tuple) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(x, weight)`` from the 1- or 2-tensor input list ``grads_of`` hands back.

    The None has to be PASSED, not omitted: ``weight`` is positional in both
    ``triton_rmsnorm`` and ``rmsnorm_reference``, so dropping it hands ``eps`` over as the weight.
    A fixed-width tuple rather than the pair of arity-normalising lambdas this used to build:
    those made the call's argument count unreadable to a checker (and to a reader), which is the
    same thing that let a driver call an op without a required argument for months.
    """
    return t[0], (t[1] if len(t) > 1 else None)


def rmsnorm_bwd_triton():
    """rmsnorm_bwd_kernel: dx and dweight, against autograd over the reference."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_reference

    out = {}
    x0 = _x(_D)
    da = torch.randn_like(x0)
    for tag, w0 in (("aff", vec(_D)), ("plain", None)):
        ins = [x0] if w0 is None else [x0, w0]
        got = grads_of(lambda *t: triton_rmsnorm(*_xw(t), _EPS), ins, da)
        # FP32 leaves matter: promoting only inside the formula still casts the
        # final gradients back to the original BF16 leaves during autograd.
        reference_inputs = [t.float() for t in ins]
        ref = grads_of(lambda *t: rmsnorm_reference(*_xw(t), _EPS), reference_inputs, da.float())
        names = ["dx"] if w0 is None else ["dx", "dweight"]
        for n, g, r in zip(names, got, ref, strict=True):
            out[f"{n}_{tag}"] = (g, r)
    return out

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

from miniworld_engine.kernels.drivers import BF16, dev, vec
from miniworld_engine.kernels.drivers.rmsnorm import _D, _DC, _DM, _EPS, _M


def _x(d: int) -> torch.Tensor:
    """PRE-flatten, for the reason `drivers/rmsnorm.py::_rows` gives."""
    return torch.randn(1, _M, d, device=dev(), dtype=BF16)


def _grads(fn, tensors, da):
    """`fn` applied to clones of `tensors`, backward through `da`, gradients in order."""
    ts = [t.clone().requires_grad_() for t in tensors]
    fn(*ts).backward(da)
    return [t.grad for t in ts]


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
        got = _grads(lambda *t, _p=pad: triton_rmsnorm(*_p(*t), _EPS), ins, da)
        ref = _grads(lambda *t, _p=pad: rmsnorm_reference(*_p(*t), _EPS), ins, da)
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
        got = _grads(lambda *t, _o=order: triton_rmsnorm_modulate(*_o(*t), _EPS), ins, dm)
        ref = _grads(lambda *t, _o=order: rmsnorm_modulate_reference(*_o(*t), _EPS), ins, dm)
        names = ["dxmod", "dscale", "dshift"] + ([] if w0 is None else ["dweightmod"])
        for n, g, r in zip(names, got, ref, strict=True):
            out[f"{n}_{tag}"] = (g, r)
    return out


# ── rmsnorm_adamod ───────────────────────────────────────────────────────────────────────────


def _adamod_inputs():
    """``(q, c, w_scale, w_shift)`` at the DiT block shape.

    The projection weights are scaled 1/sqrt(d_cond) * 0.1: adaLN-Zero starts this projection AT
    zero and a trained block keeps it small, so `scale` lands near 0.1 and `1 + scale` stays far
    from its zero crossing. At the 0.05 a naive init would give, `scale` has unit variance,
    `1 + scale` cancels on some elements, and the relative error this check reports measures that
    cancellation rather than the kernel.
    """
    w = lambda: torch.randn(_DM, _DC, device=dev(), dtype=BF16) * (0.1 / _DC**0.5)
    return _x(_DM), _x(_DC), w(), w()


def rmsnorm_adamod_fwd_triton():
    """rmsnorm_adamod_fwd_kernel: the modulate with its projection folded in."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm_adamod
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_adamod_reference

    q, c, wsc, wsh = _adamod_inputs()
    out = {}
    for tag, w in (("plain", None), ("aff", vec(_DM))):
        out[f"y_{tag}"] = (triton_rmsnorm_adamod(q, c, wsc, wsh, w, _EPS),
                           rmsnorm_adamod_reference(q, c, wsc, wsh, w, _EPS))
    return out


def rmsnorm_adamod_bwd_triton():
    """rmsnorm_adamod_bwd_kernel: every gradient, against autograd over the reference.

    dWsc/dWsh/dc come from cuBLAS calls the kernel feeds rather than from the kernel, but they are
    checked here too -- a wrong dscale shows up in them and nowhere else.
    """
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm_adamod
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_adamod_reference

    q0, c0, wsc0, wsh0 = _adamod_inputs()
    da = torch.randn_like(q0)
    out = {}
    for tag, w0 in (("plain", None), ("aff", vec(_DM))):
        base = [q0, c0, wsc0, wsh0]
        ins = base if w0 is None else [*base, w0]
        pad = (lambda *t: (*t, None)) if w0 is None else (lambda *t: t)
        got = _grads(lambda *t, _p=pad: triton_rmsnorm_adamod(*_p(*t), _EPS), ins, da)
        ref = _grads(lambda *t, _p=pad: rmsnorm_adamod_reference(*_p(*t), _EPS),
                     ins, da)
        names = ["dq", "dc", "dWsc", "dWsh"] + ([] if w0 is None else ["dweight"])
        for n, g, r in zip(names, got, ref, strict=True):
            out[f"{n}_{tag}"] = (g, r)
    return out

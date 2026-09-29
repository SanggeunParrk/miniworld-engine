"""Numerical checks for the ``rmsnorm_adamod`` family, against the written-out formula.

Its own module for the reason ``drivers/rmsnorm_adamod.py`` gives; the shape block and the
gradient helper come from ``checks/rmsnorm.py`` rather than being restated.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, dev, vec
from miniworld_engine.kernels.drivers.rmsnorm import _DC, _DM, _EPS, _M


def _x(d: int) -> torch.Tensor:
    """PRE-flatten, for the reason `drivers/rmsnorm.py::_rows` gives."""
    return torch.randn(1, _M, d, device=dev(), dtype=BF16)


def _adamod_inputs():
    """``(q, c, w_scale, w_shift, w_gate)`` at the DiT block shape.

    The projection weights are scaled 1/sqrt(d_cond) * 0.1: adaLN-Zero starts this projection AT
    zero and a trained block keeps it small, so `scale` lands near 0.1 and `1 + scale` stays far
    from its zero crossing. At the 0.05 a naive init would give, `scale` has unit variance,
    `1 + scale` cancels on some elements, and the relative error this check reports measures that
    cancellation rather than the kernel.
    """
    w = lambda: torch.randn(_DM, _DC, device=dev(), dtype=BF16) * (0.1 / _DC**0.5)
    return _x(_DM), _x(_DC), w(), w(), w()


def rmsnorm_adamod_fwd_triton():
    """rmsnorm_adamod_fwd_kernel: the modulate with its projection folded in."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm_adamod
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_adamod_reference

    q, c, wsc, wsh, wg = _adamod_inputs()
    out = {}
    for tag, w in (("plain", None), ("aff", vec(_DM))):
        y, g = triton_rmsnorm_adamod(q, c, wsc, wsh, wg, w, _EPS)
        ry, rg = rmsnorm_adamod_reference(q, c, wsc, wsh, wg, w, _EPS)
        out[f"y_{tag}"], out[f"gate_{tag}"] = (y, ry), (g, rg)
    return out


def rmsnorm_adamod_bwd_triton():
    """rmsnorm_adamod_bwd_kernel: every gradient, against autograd over the reference.

    dWsc/dWsh/dc come from cuBLAS calls the kernel feeds rather than from the kernel, but they are
    checked here too -- a wrong dscale shows up in them and nowhere else.
    """
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm_adamod
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_adamod_reference

    q0, c0, wsc0, wsh0, wg0 = _adamod_inputs()
    da, dg = torch.randn_like(q0), torch.randn_like(q0)
    out = {}
    for tag, w0 in (("plain", None), ("aff", vec(_DM))):
        base = [q0, c0, wsc0, wsh0, wg0]
        ins = base if w0 is None else [*base, w0]

        def both(fn, _ins=ins, _w=w0):
            """BOTH outputs pushed -- a gate gradient that never reached the stacked buffer
            would leave dWg and dc wrong and nothing else, so it has to be driven."""
            ts = [t.clone().requires_grad_() for t in _ins]
            y, g = fn(*ts, _EPS) if _w is not None else fn(*ts, None, _EPS)
            torch.autograd.backward((y, g), (da, dg))
            return [t.grad for t in ts]

        got, ref = both(triton_rmsnorm_adamod), both(rmsnorm_adamod_reference)
        names = ["dq", "dc", "dWsc", "dWsh", "dWg"] + ([] if w0 is None else ["dweight"])
        for n, g, r in zip(names, got, ref, strict=True):
            out[f"{n}_{tag}"] = (g, r)
    return out

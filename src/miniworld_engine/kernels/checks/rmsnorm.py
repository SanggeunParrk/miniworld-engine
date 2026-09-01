"""Numerical checks for the ``rmsnorm`` family, against the written-out formula.

The reference is `rmsnorm/reference.py`, not `F.rms_norm`: comparing one implementation against
another checks that they agree, not that either is right.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, dev, vec

_D = 32
_M = 4096
_EPS = 1e-5


def _x() -> torch.Tensor:
    """PRE-flatten, for the reason `drivers/rmsnorm.py::_rows` gives."""
    return torch.randn(1, _M, _D, device=dev(), dtype=BF16)


def rmsnorm_fwd_triton():
    """rmsnorm_fwd_kernel: y = x * rstd * w."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_reference

    x, w = _x(), vec(_D)
    return {"y": (triton_rmsnorm(x, w, _EPS), rmsnorm_reference(x, w, _EPS))}


def rmsnorm_bwd_triton():
    """rmsnorm_bwd_kernel: dx and dweight, against autograd over the reference."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_reference

    x0, w0 = _x(), vec(_D)
    da = torch.randn_like(x0)

    x = x0.clone().requires_grad_()
    w = w0.clone().requires_grad_()
    triton_rmsnorm(x, w, _EPS).backward(da)

    xr = x0.clone().requires_grad_()
    wr = w0.clone().requires_grad_()
    rmsnorm_reference(xr, wr, _EPS).backward(da)
    return {"dx": (x.grad, xr.grad), "dweight": (w.grad, wr.grad)}


_DM = 128
_DC = 384


def _adamod_inputs():
    """``(q, c, w_scale, w_shift, weight)`` at the atom-DiT block shape, all leaf-ready."""
    q = torch.randn(1, _M, _DM, device=dev(), dtype=BF16)
    c = torch.randn(1, _M, _DC, device=dev(), dtype=BF16)
    # 1/sqrt(d_cond) * 0.1: adaLN-Zero starts this projection AT zero and a trained block
    # keeps it small, so `scale` lands near 0.1 and `1 + scale` stays far from its zero
    # crossing. At the 0.05 a naive init would give, `scale` has unit variance, `1 + scale`
    # cancels on some elements, and the relative error this check reports measures that
    # cancellation rather than the kernel.
    w = lambda: torch.randn(_DM, _DC, device=dev(), dtype=BF16) * (0.1 / _DC**0.5)
    return q, c, w(), w(), vec(_DM)


def rmsnorm_adamod_fwd_triton():
    """rmsnorm_adamod_fwd_kernel: the modulate with its projection folded in."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm_adamod
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_adamod_reference

    q, c, wsc, wsh, w = _adamod_inputs()
    return {"y": (triton_rmsnorm_adamod(q, c, wsc, wsh, w, _EPS),
                  rmsnorm_adamod_reference(q, c, wsc, wsh, w, _EPS))}


def rmsnorm_adamod_bwd_triton():
    """rmsnorm_adamod_bwd_kernel: every gradient, against autograd over the reference.

    ``dWsc``/``dWsh``/``dc`` come from cuBLAS calls the kernel feeds rather than from the kernel,
    but they are checked here too -- a wrong ``dscale`` shows up in them and nowhere else.
    """
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm_adamod
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_adamod_reference

    q0, c0, wsc0, wsh0, w0 = _adamod_inputs()
    da = torch.randn_like(q0)
    names = ("dq", "dc", "dWsc", "dWsh", "dweight")
    out: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    for fn in (triton_rmsnorm_adamod, rmsnorm_adamod_reference):
        ts = [t.clone().requires_grad_() for t in (q0, c0, wsc0, wsh0, w0)]
        fn(*ts, _EPS).backward(da)
        for name, t in zip(names, ts, strict=True):
            out[name].append(t.grad)
    return {k: (v[0], v[1]) for k, v in out.items()}

"""Numerical check for the ``rope`` family, against the written-out `apply_rotary_emb_3d`."""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import grads_of
from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers.rope import _D, _HALF, _M, _N_HEADS


def _inputs():
    x = torch.randn(1, _M, _N_HEADS, _D, device=dev(), dtype=BF16)
    cos = torch.randn(1, _M, _HALF, device=dev(), dtype=torch.float32)
    sin = torch.randn(1, _M, _HALF, device=dev(), dtype=torch.float32)
    return x, cos, sin


def rope_fwd_triton():
    """rope_3d_kernel: rotated output and dx, against the eager reference.

    dx is checked too -- the backward rotates by the negated angle, and a wrong sign there is
    invisible in the forward.
    """
    from miniworld_engine.kernels.rope.interface import triton_rope_3d
    from miniworld_engine.kernels.rope.reference import rope_3d_reference

    x, cos, sin = _inputs()
    out = {"y": (triton_rope_3d(x, cos, sin), rope_3d_reference(x, cos, sin))}

    da = torch.randn_like(x)
    gk = grads_of(lambda t: triton_rope_3d(t, cos, sin), [x], da)
    gr = grads_of(lambda t: rope_3d_reference(t, cos, sin), [x], da)
    out["dx"] = (gk[0], gr[0])
    return out

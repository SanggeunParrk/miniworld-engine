"""Projection-aware CuTe output backward for bidirectional TriMul training.

For z = xhat*gamma and y = z@W.T + beta@W.T, LayerNorm's row corrections
are c1 = sum(dy*(y-beta@W.T))/K and c2 = sum(dy*sum(W*gamma, K))/K.
Computing those outside the dgrad epilogue allows independent output-column
tiles. No division by gamma is used, including when gamma contains zeros.
"""

import torch
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels._compile import opaque, device_constant
from miniworld_engine.kernels.trimul_inproj.cute.dispatch import _cute_allowed
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_materialize
from miniworld_engine.kernels.layernorm_linear.cute.dgrad_ln_rows import dgrad_ln_rows


@device_constant
def _hopper(device):
    return (
        torch.cuda.get_device_capability(device) == (9, 0)
        and torch.cuda.get_device_name(device) == "NVIDIA H100 80GB HBM3"
    )


def supported(x, w):
    return (
        x.dtype == torch.bfloat16
        and w.dtype == x.dtype
        and x.shape[1] == 256
        and tuple(w.shape) == (128, 256)
        and x.stride(0) == 1
        and x.stride(1) == x.shape[0]
        and x.shape[0] in (384**2, 768**2)
        and _hopper(x.device)
        and _cute_allowed(x.device, x.dtype, "triangle_multiplication_bidirectional")
    )


def _dx_fake(dy, w, xhat, g, rstd, c1, c2):
    """Allocate outputs with the same shape, dtype and strides as _dx."""
    return torch.empty_strided(xhat.shape, (1, xhat.shape[0]), device=xhat.device, dtype=xhat.dtype)


@opaque(fake=_dx_fake, name='trimul_output_rows_dgrad')
def _dx(dy: torch.Tensor, w: torch.Tensor, xhat: torch.Tensor, g: torch.Tensor, rstd: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """Execute  dx behind an opaque compiler boundary."""
    return dgrad_ln_rows(dy, w, xhat, g, rstd, c1, c2)


def forward(x, g, b, w, eps):
    xhat, mean, rstd = _ln_materialize(
        x, torch.ones_like(g), torch.zeros_like(b), eps, shape_key=both_key(x.shape[0]))
    wf, gf, bf = w.float(), g.float(), b.float()
    folded = (wf * gf[None, :]).to(w.dtype)
    bias = (wf * bf[None, :]).sum(1).to(w.dtype)
    y = torch.nn.functional.linear(xhat, folded, bias)
    return y, xhat, mean, rstd


def backward(dy, proj, xhat, rstd, g, b, w):
    wf, gf, bf = w.float(), g.float(), b.float()
    s = (wf * gf[None, :]).sum(1)
    b2 = (wf * bf[None, :]).sum(1)
    c1 = (dy.float() * (proj.float() - b2[None, :])).sum(1) / xhat.shape[1]
    c2 = (dy.float() * s[None, :]).sum(1) / xhat.shape[1]
    dx = _dx(dy, w, xhat, g, rstd, c1, c2)
    t = (dy.t() @ xhat).float()
    db = dy.float().sum(0)
    dw = (gf[None, :] * t + db[:, None] * bf[None, :]).to(w.dtype)
    dg = (wf * t).sum(0).to(g.dtype)
    dbeta = (db @ wf).to(b.dtype)
    return dx, dg, dbeta, dw

"""Linear residual gate for adaLN-Zero: x + gate * branch, with backward.

Registered as ``gated_projection_fwd_residual_triton`` (``_forward``) and ``gated_projection_bwd_residual_triton`` (``_backward``):
both are memory-bound 1-D passes whose one free axis is the block of elements per program. The default config set holds the single
launch they used before they were registered (256 elements, 4 warps, 3 stages: Triton's defaults), so a card with no tuned cache runs
exactly what it ran; the grid set is what a build searches.
"""
import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels._compile import opaque


@triton.autotune(configs=configs_for("gated_projection_fwd_residual_triton"), key=["shape_key"])
@triton.jit
def _forward(X, G, B, Y, N, shape_key, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    x = tl.load(X + i, mask, 0).to(tl.float32)
    g = tl.load(G + i, mask, 0).to(tl.float32)
    b = tl.load(B + i, mask, 0).to(tl.float32)
    # Preserve the eager product rounding before adding the residual.
    product = (g * b).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, x + product, mask)


@triton.autotune(configs=configs_for("gated_projection_bwd_residual_triton"), key=["shape_key"])
@triton.jit
def _backward(DY, G, B, DG, DB, N, shape_key, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dy = tl.load(DY + i, mask, 0).to(tl.float32)
    g = tl.load(G + i, mask, 0).to(tl.float32)
    b = tl.load(B + i, mask, 0).to(tl.float32)
    tl.store(DG + i, dy * b, mask)
    tl.store(DB + i, dy * g, mask)


def residual_key(shape) -> int:
    """The cache key of both launches: the row count (every axis but the channel one), as the level=both kernels key."""
    rows = 1
    for d in tuple(shape)[:-1]:
        rows *= int(d)
    return both_key(rows)


def _fwd_fake(x, gate, branch, shape_key):
    """Return the residual output layout without accessing data."""
    return torch.empty_like(x)


@opaque(fake=_fwd_fake, name="gated_residual_fwd")
def _fwd(x: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor, shape_key: int) -> torch.Tensor:
    """Compute x + gate * branch with eager product rounding."""
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        # no FMA: the product is rounded to the activation dtype before the add, as eager does
        _forward[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
            x, gate, branch, out, n, shape_key=shape_key, enable_fp_fusion=False,
        )
    return out


def _bwd_fake(dy, gate, branch, shape_key):
    """Return gate and branch gradient layouts without accessing data."""
    return torch.empty_like(gate), torch.empty_like(branch)


@opaque(fake=_bwd_fake, name="gated_residual_bwd")
def _bwd(dy: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor, shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gate and branch gradients for the residual product."""
    dg, db = torch.empty_like(gate), torch.empty_like(branch)
    n = dy.numel()
    if n:
        _backward[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](dy, gate, branch, dg, db, n, shape_key=shape_key)
    return dg, db


class _GatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, branch):
        x, gate, branch = x.contiguous(), gate.contiguous(), branch.contiguous()
        ctx.save_for_backward(gate, branch)
        ctx.key = residual_key(x.shape)
        return _fwd(x, gate, branch, ctx.key)

    @staticmethod
    def backward(ctx, dy):
        gate, branch = ctx.saved_tensors
        dg, db = _bwd(dy.contiguous(), gate, branch, ctx.key)
        return dy, dg, db


def gated_residual(x: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """Same-shape, same-dtype residual gate; no sigmoid, normalization, or projection."""
    if x.shape != gate.shape or x.shape != branch.shape:
        raise ValueError("gated_residual requires identical shapes")
    if x.dtype != gate.dtype or x.dtype != branch.dtype:
        raise ValueError("gated_residual requires identical dtypes")
    if x.device != gate.device or x.device != branch.device:
        raise ValueError("gated_residual requires identical devices")
    if not x.is_cuda:
        return x + gate * branch
    return _GatedResidual.apply(x, gate, branch)

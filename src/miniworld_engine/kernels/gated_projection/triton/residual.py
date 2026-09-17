"""Linear residual gate for adaLN-Zero: x + gate * branch, with backward."""
import torch
import triton
import triton.language as tl

from miniworld_engine.kernels._compile import opaque


@triton.jit
def _forward(X, G, B, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    x = tl.load(X + i, mask, 0).to(tl.float32)
    g = tl.load(G + i, mask, 0).to(tl.float32)
    b = tl.load(B + i, mask, 0).to(tl.float32)
    # Preserve the eager product rounding before adding the residual.
    product = (g * b).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, x + product, mask)


@triton.jit
def _backward(DY, G, B, DG, DB, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dy = tl.load(DY + i, mask, 0).to(tl.float32)
    g = tl.load(G + i, mask, 0).to(tl.float32)
    b = tl.load(B + i, mask, 0).to(tl.float32)
    tl.store(DG + i, dy * b, mask)
    tl.store(DB + i, dy * g, mask)


def _fwd_fake(x, gate, branch):
    """Return the residual output layout without accessing data."""
    return torch.empty_like(x)


@opaque(fake=_fwd_fake, name="gated_residual_fwd")
def _fwd(x: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """Compute x + gate * branch with eager product rounding."""
    out = torch.empty_like(x)
    if x.numel():
        _forward[(triton.cdiv(x.numel(), 256),)](
            x, gate, branch, out, x.numel(), BLOCK=256, enable_fp_fusion=False,
        )
    return out


def _bwd_fake(dy, gate, branch):
    """Return gate and branch gradient layouts without accessing data."""
    return torch.empty_like(gate), torch.empty_like(branch)


@opaque(fake=_bwd_fake, name="gated_residual_bwd")
def _bwd(dy: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gate and branch gradients for the residual product."""
    dg, db = torch.empty_like(gate), torch.empty_like(branch)
    if dy.numel():
        _backward[(triton.cdiv(dy.numel(), 256),)](dy, gate, branch, dg, db, dy.numel(), BLOCK=256)
    return dg, db


class _GatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, branch):
        x, gate, branch = x.contiguous(), gate.contiguous(), branch.contiguous()
        ctx.save_for_backward(gate, branch)
        return _fwd(x, gate, branch)

    @staticmethod
    def backward(ctx, dy):
        gate, branch = ctx.saved_tensors
        dg, db = _bwd(dy.contiguous(), gate, branch)
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

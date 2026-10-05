"""A100 (sm_80) hand-CUDA tm2: ``out = sigmoid(x @ W_gate) * (y @ W_out)`` (the trimul output gate + projection + mul), forward and backward, bf16.

The kernels live in ``gated_projection/cuda/sm80`` (one extension for the gated GEMMs).  Forward: two GEMMs into two accumulators of one output tile and the gate
epilogue (one rounding at the store); backward (recompute): ``dB = d s``, ``dA = dB (y W_out)(1 - s)`` with ``s = sigmoid(x W_gate)``, then the four GEMMs of the gradients
on cuBLAS, as the Triton path.  The weights are in ``(D, D)`` matmul form (``x @ W``)."""

import torch

from ..._compile import opaque
from ...gated_projection.cuda import sm80 as _gp


def serves(x: torch.Tensor, y: torch.Tensor, gate_weight: torch.Tensor, out_weight: torch.Tensor) -> bool:
    """sm_80, bf16 operands of one shape and a square weight of a width the 64-column tiles divide."""
    d = x.shape[-1] if x.ndim else 0
    if x.shape != y.shape or gate_weight.shape != (d, d) or out_weight.shape != (d, d) or not _gp.shape_ok(x.dtype, d, d):
        return False
    if y.dtype is not x.dtype or gate_weight.dtype is not x.dtype or out_weight.dtype is not x.dtype:
        return False
    return _gp.serves(d, d, x.device, x.dtype)


def _fwd_fake(x, y, wgt, wot):
    """`out`, shaped and typed like the flat `x`."""
    return torch.empty_like(x)


@opaque(fake=_fwd_fake, name="tm2_sm80_fwd")
def _fwd(x: torch.Tensor, y: torch.Tensor, wgt: torch.Tensor, wot: torch.Tensor) -> torch.Tensor:
    """The fused gate + projection -> ``out``, flat; the weights are [out, in] (the transposes of the matmul form)."""
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _gp._ext().tm2_fwd(x, y, wgt, wot, out)
    return out


def _bwd_fake(x, y, wgt, wot, grad_out):
    """(dA, dB), both shaped and typed like the flat `x`."""
    return torch.empty_like(x), torch.empty_like(x)


@opaque(fake=_bwd_fake, name="tm2_sm80_bwd")
def _bwd(x: torch.Tensor, y: torch.Tensor, wgt: torch.Tensor, wot: torch.Tensor, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate backward -> ``(dA, dB)``; the four GEMMs that consume them stay in the caller."""
    da, db = torch.empty_like(x), torch.empty_like(x)
    with torch.cuda.device(x.device):
        _gp._ext().tm2_bwd(x, y, wgt, wot, grad_out, da, db)
    return da, db


class TM2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, gate_weight, out_weight):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        y2 = y.reshape(-1, shape[-1]).contiguous()
        wgt, wot = gate_weight.t().contiguous(), out_weight.t().contiguous()
        out = _fwd(x2, y2, wgt, wot)
        ctx.save_for_backward(x2, y2, gate_weight, out_weight, wgt, wot)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, grad_out):
        x2, y2, gate_weight, out_weight, wgt, wot = ctx.saved_tensors
        g2 = grad_out.reshape(-1, x2.shape[-1]).to(x2.dtype).contiguous()
        da, db = _bwd(x2, y2, wgt, wot, g2)
        dx = (da @ gate_weight.T).reshape(ctx.shape)
        dy = (db @ out_weight.T).reshape(ctx.shape)
        return dx, dy, torch.matmul(x2.T, da), torch.matmul(y2.T, db)


cuda_tm2 = TM2Function.apply

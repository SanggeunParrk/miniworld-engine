"""A100 (sm_80) hand-CUDA tm1: ``(sigmoid(x@WLg) * (x@WL), sigmoid(x@WRg) * (x@WR))`` (the dual sigmoid-gated projection of the triangle multiplication's front), forward
and backward, bf16.  One GEMM over the four weights packed [gate | projection] by 8 channels, token-major outputs; the kernels live in ``gated_projection/cuda/sm80`` (one
extension for the gated GEMMs).  Backward (recompute): ``dLB = dL s_L``, ``dLA = dLB (x WL)(1 - s_L)``, likewise right; the dgrad / wgrad GEMMs on cuBLAS, as the Triton
path.  The weights are in ``(D, D)`` matmul form (``x @ W``)."""

import torch

from ..._compile import opaque
from ...gated_projection.cuda import sm80 as _gp


def serves(x: torch.Tensor, *weights: torch.Tensor) -> bool:
    """sm_80, bf16 operands and four square weights of a width the 64-column tiles divide."""
    d = x.shape[-1] if x.ndim else 0
    if len(weights) != 4 or any(w.shape != (d, d) or w.dtype is not x.dtype for w in weights) or not _gp.shape_ok(x.dtype, d, d):
        return False
    return _gp.serves(d, d, x.device, x.dtype)


def _pack(wl, wlg, wr, wrg):
    """The four (K, N) matmul-form weights -> [4 D, D] rows: 16 j + 0..7 the gate rows, 16 j + 8..15 the projection rows of channels 8 j .. 8 j + 7 (left channels first)."""
    d = wl.shape[0]
    gate, proj = torch.cat([wlg.t(), wrg.t()]), torch.cat([wl.t(), wr.t()])
    return torch.stack([gate.reshape(-1, 8, d), proj.reshape(-1, 8, d)], 1).reshape(4 * d, d).contiguous()


def _fwd_fake(x, w1):
    """(left, right), both shaped and typed like the flat (M, D) `x`."""
    return torch.empty_like(x), torch.empty_like(x)


@opaque(fake=_fwd_fake, name="tm1_sm80_fwd")
def _fwd(x: torch.Tensor, w1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The dual gated projection -> ``(left, right)``, flat."""
    left, right = torch.empty_like(x), torch.empty_like(x)
    with torch.cuda.device(x.device):
        _gp._ext().tm1_fwd(x, w1, left, right)
    return left, right


def _bwd_fake(x, w1, grad_left, grad_right):
    """(dLA, dLB, dRA, dRB), all shaped and typed like the flat (M, D) `x`."""
    return tuple(torch.empty_like(x) for _ in range(4))


@opaque(fake=_bwd_fake, name="tm1_sm80_bwd")
def _bwd(x: torch.Tensor, w1: torch.Tensor, grad_left: torch.Tensor, grad_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The gate backward -> the four partial gradients ``(dLA, dLB, dRA, dRB)``; the GEMMs that consume them stay in the caller."""
    outs = [torch.empty_like(x) for _ in range(4)]
    with torch.cuda.device(x.device):
        _gp._ext().tm1_bwd(x, w1, grad_left, grad_right, *outs)
    return tuple(outs)


class TM1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, left_weight, left_gate_weight, right_weight, right_gate_weight):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        w1 = _pack(left_weight, left_gate_weight, right_weight, right_gate_weight)
        left, right = _fwd(x2, w1)
        ctx.save_for_backward(x2, w1, left_weight, left_gate_weight, right_weight, right_gate_weight)
        ctx.shape = shape
        return left.reshape(shape), right.reshape(shape)

    @staticmethod
    def backward(ctx, grad_left, grad_right):
        x2, w1, wl, wlg, wr, wrg = ctx.saved_tensors
        d = x2.shape[-1]
        gl = grad_left.reshape(-1, d).to(x2.dtype).contiguous()
        gr = grad_right.reshape(-1, d).to(x2.dtype).contiguous()
        dla, dlb, dra, drb = _bwd(x2, w1, gl, gr)
        dx = dla @ wlg.T + dlb @ wl.T
        dx += dra @ wrg.T + drb @ wr.T
        xt = x2.T
        return dx.reshape(ctx.shape), torch.matmul(xt, dlb), torch.matmul(xt, dla), torch.matmul(xt, drb), torch.matmul(xt, dra)


cuda_tm1 = TM1Function.apply

"""H100 BF16 projections with live FP32 master weights and FP32 weight gradients."""

import torch
import torch.nn.functional as F

from miniworld_engine.kernels._compile import device_constant, opaque


@device_constant
def is_h100(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


def _pack_fake(weights):
    return [torch.empty_like(w, dtype=torch.bfloat16, memory_format=torch.contiguous_format) for w in weights]


@opaque(fake=_pack_fake, name="h100_master_projection_pack")
def _pack(weights: list[torch.Tensor]) -> list[torch.Tensor]:
    # One multi-tensor PyTorch copy (a cast): H100 has no Triton fallback, and this is not worth a kernel of its own.
    outputs = _pack_fake(weights)
    torch._foreach_copy_(outputs, weights)
    return outputs


def pack(weights):
    """Fresh casts on every call/replay; the projection owns the raw parameter gradient."""
    with torch.no_grad():
        return _pack(list(weights))


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, compute_weight, compute_bias):
        ctx.save_for_backward(x, weight, bias, compute_weight, compute_bias)
        with torch.autocast("cuda", enabled=False):
            return F.linear(x, compute_weight, compute_bias)

    @staticmethod
    def backward(ctx, dy):
        x, _weight, bias, compute_weight, _ = ctx.saved_tensors
        x2 = x.reshape(-1, x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        with torch.autocast("cuda", enabled=False):
            dx = torch.mm(dy2, compute_weight).reshape(x.shape)
            dw = torch.mm(dy2.t(), x2, out_dtype=torch.float32)
            db = dy2.sum(0, dtype=torch.float32) if bias is not None else None
        return dx, dw, db, None, None


def linear(x, module, compute_weight, compute_bias=None):
    return _Linear.apply(x, module.weight, module.bias, compute_weight, compute_bias)


def master_linear(module, x):
    """``module(x)`` where a bf16 activation meets fp32 master parameters: bf16 GEMM, fp32 (unrounded) weight / bias gradients.
    Any other combination is the module itself. Used by the PyTorch legs that H100 / B200 run in place of a Triton fallback."""
    w = module.weight
    if x.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.float32:
        b = module.bias
        return linear(x, module, w.to(torch.bfloat16), None if b is None else b.to(torch.bfloat16))
    return module(x)

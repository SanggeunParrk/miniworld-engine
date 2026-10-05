"""H100 BF16 projections with live FP32 master weights and FP32 weight gradients."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from miniworld_engine.kernels._compile import device_constant, opaque


@triton.jit
def _cast_kernel(Src, Dst, Sizes: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    first = 0
    for i in tl.static_range(len(Sizes)):  # ty: ignore[invalid-argument-type]  # Triton constexpr holds a tuple
        count = tl.cdiv(Sizes[i], BLOCK)
        if pid >= first and pid < first + count:
            offsets = (pid - first) * BLOCK + tl.arange(0, BLOCK)
            value = tl.load(Src[i] + offsets, mask=offsets < Sizes[i], other=0)
            tl.store(Dst[i] + offsets, value, mask=offsets < Sizes[i])
        first += count


@triton.jit
def transition_pack_kernel(Wa, Wb, Ws, A, B, S, St, D: tl.constexpr, ROWS: tl.constexpr):
    h = 4 * D
    flat = (tl.program_id(1) * (h // 32) + tl.program_id(0)) * (ROWS * 32) + tl.arange(0, ROWS * 32)
    tl.store(A + flat, tl.load(Wa + flat))
    tl.store(B + flat, tl.load(Wb + flat))
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(0) * 32 + tl.arange(0, 32)
    value = tl.load(Ws + rows[:, None] * h + cols[None, :]).to(tl.bfloat16)
    tl.store(S + rows[:, None] * h + cols[None, :], value)
    tl.store(St + cols[None, :] * D + rows[:, None], value)


@device_constant
def is_h100(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


def _pack_fake(weights):
    return [torch.empty_like(w, dtype=torch.bfloat16, memory_format=torch.contiguous_format) for w in weights]


@opaque(fake=_pack_fake, name="h100_master_projection_pack")
def _pack(weights: list[torch.Tensor]) -> list[torch.Tensor]:
    outputs = _pack_fake(weights)
    if all(w.is_contiguous() for w in weights):
        sizes = tuple(w.numel() for w in weights)
        _cast_kernel[(sum(triton.cdiv(n, 1024) for n in sizes),)](
            tuple(weights), tuple(outputs), sizes, BLOCK=1024, num_warps=4,  # ty: ignore[invalid-argument-type, unknown-argument]  # Triton launch metadata
        )
    else:
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

"""Native-forward dropout with a one-bit-per-element backward mask."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_PACK_BLOCK_BYTES = 256
_BACKWARD_BLOCK_ELEMENTS = 256


@triton.jit
def _pack_bool_kernel(
    mask_ptr,
    packed_ptr,
    n_elements,
    BLOCK_BYTES: tl.constexpr,
):
    byte_offsets = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    element_offsets = byte_offsets * 8
    packed = tl.zeros((BLOCK_BYTES,), dtype=tl.uint32)
    for bit in range(8):
        keep = tl.load(
            mask_ptr + element_offsets + bit,
            mask=element_offsets + bit < n_elements,
            other=0,
        )
        packed += keep.to(tl.uint32) * (1 << bit)
    n_bytes = (n_elements + 7) // 8
    tl.store(
        packed_ptr + byte_offsets,
        packed.to(tl.uint8),
        mask=byte_offsets < n_bytes,
    )


@triton.jit
def _packed_dropout_backward_kernel(
    grad_output_ptr,
    packed_ptr,
    grad_input_ptr,
    n_elements,
    scale,
    BLOCK_ELEMENTS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMENTS + tl.arange(0, BLOCK_ELEMENTS)
    valid = offsets < n_elements
    packed = tl.load(
        packed_ptr + offsets // 8,
        mask=valid,
        other=0,
    ).to(tl.uint32)
    keep = ((packed >> (offsets % 8)) & 1) != 0
    grad_output = tl.load(
        grad_output_ptr + offsets,
        mask=valid,
        other=0.0,
    )
    grad_input = tl.where(keep, grad_output * scale, 0.0)
    tl.store(grad_input_ptr + offsets, grad_input, mask=valid)


@torch.library.custom_op(
    "miniworld_kernels::mpnn_edge_dropout_pack_v1",
    mutates_args=(),
)
def _pack_op(mask: torch.Tensor) -> torch.Tensor:
    if not mask.is_cuda or mask.dtype != torch.bool or not mask.is_contiguous():
        raise ValueError("edge dropout packing requires a contiguous CUDA bool mask")
    packed = torch.empty(
        ((mask.numel() + 7) // 8,),
        device=mask.device,
        dtype=torch.uint8,
    )
    _pack_bool_kernel[(triton.cdiv(packed.numel(), _PACK_BLOCK_BYTES),)](
        mask,
        packed,
        mask.numel(),
        BLOCK_BYTES=_PACK_BLOCK_BYTES,
        num_warps=4,
    )
    return packed


@_pack_op.register_fake
def _(mask: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        ((mask.numel() + 7) // 8,),
        device=mask.device,
        dtype=torch.uint8,
    )


@torch.library.custom_op(
    "miniworld_kernels::mpnn_edge_dropout_backward_v1",
    mutates_args=(),
)
def _backward_op(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if not grad_output.is_cuda or not grad_output.is_contiguous():
        raise ValueError("edge dropout backward requires a contiguous CUDA gradient")
    if (
        not packed.is_cuda
        or packed.device != grad_output.device
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
        or packed.numel() != (grad_output.numel() + 7) // 8
    ):
        raise ValueError("edge dropout backward received an invalid packed mask")
    grad_input = torch.empty_like(grad_output)
    _packed_dropout_backward_kernel[
        (triton.cdiv(grad_output.numel(), _BACKWARD_BLOCK_ELEMENTS),)
    ](
        grad_output,
        packed,
        grad_input,
        grad_output.numel(),
        scale,
        BLOCK_ELEMENTS=_BACKWARD_BLOCK_ELEMENTS,
        num_warps=4,
    )
    return grad_input


@_backward_op.register_fake
def _(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    del packed, scale
    return torch.empty_like(grad_output)


class _BitpackDropout(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        probability: float,
    ) -> torch.Tensor:
        # This is the same native operation reached by F.dropout. Returning its
        # output directly preserves native values and Philox state exactly.
        output, mask = torch.ops.aten.native_dropout.default(
            values,
            probability,
            True,
        )
        ctx.save_for_backward(_pack_op(mask))
        ctx.scale = 1.0 / (1.0 - probability)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (packed,) = ctx.saved_tensors
        # Keep scale a runtime scalar. Making it tl.constexpr changes the BF16
        # rounding boundary relative to native_dropout_backward.
        grad_input = _backward_op(
            grad_output.contiguous(),
            packed,
            ctx.scale,
        )
        return grad_input, None


def edge_dropout_bitpack(
    values: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """Apply native dropout with a packed mask for first-order gradients only."""
    return _BitpackDropout.apply(values, probability)


__all__ = ["edge_dropout_bitpack"]

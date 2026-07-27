"""One-kernel ProteinMPNN encoder edge MLP for A5000 crop shapes.

One CTA owns 128 rows and the complete 128-channel output.  Both exact GELUs and
both tensor-core GEMMs stay in one launch, so only the final update reaches HBM.
The A5000 sweep found a narrow non-spilling regime at 128 rows and eight warps;
smaller row tiles spill or under-utilize the tensor cores because the dependent
second GEMM must retain the complete hidden vector.

The custom autograd boundary deliberately saves only the input and parameters.
Hidden activations are recomputed in backward instead of remaining live across
the rest of the model forward.  The recompute is close but not bitwise: see
`_recompute_projected_op` for the measured reason that reproducing the fused
contraction order exactly is not worth its cost.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_WIDTH = 128


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def _edge_mlp_fwd_kernel(
    preactivation_ptr,
    hidden_weight_ptr,
    hidden_bias_ptr,
    output_weight_ptr,
    output_bias_ptr,
    update_ptr,
    rows,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    output_block = tl.program_id(1)
    row_indices = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    hidden_columns = tl.arange(0, WIDTH)
    output_columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_valid = row_indices < rows
    output_valid = output_columns < WIDTH

    preactivation = tl.load(
        preactivation_ptr + row_indices[:, None] * WIDTH + hidden_columns[None, :],
        mask=row_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    hidden_1 = _gelu(preactivation).to(tl.bfloat16)

    # nn.Linear stores [output, input].  The direct logical [input, output]
    # load is deliberately retained: a coalesced-load/transposed variant
    # spills on sm86 and was slower in the measured sweep.
    hidden_weight = tl.load(
        hidden_weight_ptr + hidden_columns[:, None] + hidden_columns[None, :] * WIDTH,
    ).to(tl.bfloat16)
    projected_accumulator = tl.dot(hidden_1, hidden_weight)
    hidden_bias = tl.load(hidden_bias_ptr + hidden_columns).to(tl.bfloat16)
    projected = (projected_accumulator + hidden_bias[None, :]).to(tl.bfloat16)
    hidden_2 = _gelu(projected.to(tl.float32)).to(tl.bfloat16)

    output_weight = tl.load(
        output_weight_ptr + hidden_columns[:, None] + output_columns[None, :] * WIDTH,
        mask=output_valid[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    update_accumulator = tl.dot(hidden_2, output_weight)
    output_bias = tl.load(
        output_bias_ptr + output_columns,
        mask=output_valid,
        other=0.0,
    ).to(tl.bfloat16)
    update = (update_accumulator + output_bias[None, :]).to(tl.bfloat16)
    tl.store(
        update_ptr + row_indices[:, None] * WIDTH + output_columns[None, :],
        update,
        mask=row_valid[:, None] & output_valid[None, :],
    )


def _forward_impl(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    original_shape = preactivation.shape
    values = preactivation.reshape(-1, _WIDTH)
    rows = values.shape[0]
    update = torch.empty_like(values)
    grid = (
        triton.cdiv(rows, 128),
        1,
    )
    _edge_mlp_fwd_kernel[grid](
        values,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        update,
        rows,
        WIDTH=_WIDTH,
        BLOCK_M=128,
        BLOCK_N=128,
        num_warps=8,
        num_stages=2,
    )
    return update.reshape(original_shape)


@torch.library.custom_op(
    "miniworld_kernels::mpnn_edge_mlp_recompute_projected_v1",
    mutates_args=(),
)
def _recompute_projected_op(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
) -> torch.Tensor:
    """Recompute ``Linear(GELU(x))`` without materializing the GELU output.

    This split-K projection is *not* bitwise identical to the value the fused
    forward held in registers: the fused kernel contracts all 128 channels in one
    ``tl.dot`` while this one accumulates eight 16-wide chunks, so ``projected``
    is reproduced to about ``1.2e-5`` relative and the recomputed update to about
    ``3.8e-5``. Backward therefore differentiates a function very slightly
    different from the one forward evaluated.

    That is a deliberate, measured trade. Reproducing the fused order exactly
    requires a single full-width ``tl.dot``, and a standalone kernel shaped that
    way is pathologically slow on sm_86 -- 1.25 ms against 0.109 ms for this
    launch at crop 2048, per encoder layer -- because a lone ``tl.dot`` whose
    result goes straight to global memory pays a full layout conversion. The
    fused forward avoids that only because its first dot feeds a second one
    (0.180 ms for both of its GEMMs, faster than the two-kernel compute pair at
    0.235 ms). Three encoder layers would pay roughly 3.4 ms per step at B=1,
    about 18% of the step, to remove a 1.2e-5 inconsistency that sits well inside
    this policy's own 4.04e-5 forward error against the PyTorch reference and far
    inside the edge-LayerNorm policy's accepted 2.2e-4 gradient error.
    """
    from miniworld_kernels.kernels.mpnn_message.triton.main import (
        _projection_fwd_kernel,
    )

    rows = preactivation.numel() // _WIDTH
    projected = torch.empty_like(preactivation)
    _projection_fwd_kernel[(triton.cdiv(rows, 128), 1)](
        preactivation,
        hidden_weight,
        hidden_bias,
        projected,
        rows,
        HIDDEN=_WIDTH,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        num_warps=4,
        num_stages=3,
    )
    return projected


@_recompute_projected_op.register_fake
def _(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
) -> torch.Tensor:
    del hidden_weight, hidden_bias
    return torch.empty_like(preactivation)


@torch.library.custom_op(
    "miniworld_kernels::mpnn_edge_mlp_memory_fwd_v2",
    mutates_args=(),
)
def _forward_op(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    return _forward_impl(
        preactivation,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
    )


@_forward_op.register_fake
def _(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    del hidden_weight, hidden_bias, output_weight, output_bias
    return torch.empty_like(preactivation)


def _setup_context(ctx, inputs, output) -> None:
    del output
    preactivation, hidden_weight, hidden_bias, output_weight, output_bias = inputs
    ctx.save_for_backward(
        preactivation,
        hidden_weight,
        hidden_bias,
        output_weight,
    )
    ctx.hidden_bias_dtype = hidden_bias.dtype
    ctx.output_bias_dtype = output_bias.dtype


def _backward(ctx, grad_output):
    preactivation, hidden_weight, hidden_bias, output_weight = ctx.saved_tensors
    grad_output = grad_output.contiguous()

    # Recompute one projection, then reuse the message dX kernel twice.  Each
    # dX launch emits GELU(input) for the following PyTorch weight-gradient
    # GEMM.  At most three full edge tensors are temporary at a time, and none
    # remain live from forward.
    from miniworld_kernels.kernels.mpnn_message.triton.main import (
        _projection_dx_weight_op,
    )

    projected = _recompute_projected_op(
        preactivation,
        hidden_weight,
        hidden_bias,
    )
    # Both dX launches fold their own weight gradient in as they go, so neither
    # GELU activation is ever materialised at full size. The allocator snapshot
    # put three such activations at the backward peak before this change.
    grad_projected, grad_output_weight = _projection_dx_weight_op(
        grad_output,
        output_weight,
        projected,
    )
    # Flattening also covers the public rank-one ``[128]`` input contract.
    # Keep the reduction in the incoming BF16 dtype: CUDA autocast Linear's
    # bias backward has that rounding boundary before the FP32 parameter grad.
    grad_output_bias = grad_output.reshape(-1, _WIDTH).sum(0)
    del projected, grad_output

    grad_preactivation, grad_hidden_weight = _projection_dx_weight_op(
        grad_projected,
        hidden_weight,
        preactivation,
    )
    grad_hidden_bias = grad_projected.reshape(-1, _WIDTH).sum(0)
    return (
        grad_preactivation,
        grad_hidden_weight.to(hidden_weight.dtype),
        grad_hidden_bias.to(ctx.hidden_bias_dtype),
        grad_output_weight.to(output_weight.dtype),
        grad_output_bias.to(ctx.output_bias_dtype),
    )


_forward_op.register_autograd(_backward, setup_context=_setup_context)


def triton_edge_mlp_update(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    """Run the one-kernel, activation-recompute edge MLP."""
    return _forward_op(
        preactivation,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
    )


__all__ = ["triton_edge_mlp_update"]

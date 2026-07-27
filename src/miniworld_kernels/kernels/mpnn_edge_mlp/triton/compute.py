"""Compute-efficient ProteinMPNN encoder edge MLP for A5000 crop shapes.

The memory-efficient implementation in :mod:`.main` keeps both dependent
128x128 projections in one forward kernel and recomputes the first projection
during backward.  This alternative uses two forward kernels and saves the
first projection.  Backward can therefore start immediately with the existing
projection dX kernel; the two global weight gradients and exact BF16 bias
reductions remain PyTorch GEMM/sum operations.

The caller owns dispatch validation.  In particular, flattened element offsets
must fit signed int32 and all operands must satisfy the same contiguous BF16
projection contract as the memory-efficient path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_kernels.kernels.mpnn_message.triton.main import _projection_dx_op


_WIDTH = 128


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def _compute_stage_fwd_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    rows,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    output_block = tl.program_id(1)
    row_indices = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    output_columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_valid = row_indices < rows
    output_valid = output_columns < WIDTH
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for hidden_start in range(0, WIDTH, BLOCK_K):
        hidden_columns = hidden_start + tl.arange(0, BLOCK_K)
        hidden_valid = hidden_columns < WIDTH
        values = tl.load(
            input_ptr + row_indices[:, None] * WIDTH + hidden_columns[None, :],
            mask=row_valid[:, None] & hidden_valid[None, :],
            other=0.0,
        ).to(tl.float32)
        activated = _gelu(values).to(tl.bfloat16)
        # nn.Linear stores [output, input].  The A5000 autotune winner forms
        # the logical [input, output] dot operand directly; the alternative
        # contiguous [output, input] load followed by tl.trans was slower.
        weight = tl.load(
            weight_ptr + output_columns[None, :] * WIDTH + hidden_columns[:, None],
            mask=output_valid[None, :] & hidden_valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        accumulator += tl.dot(activated, weight)

    bias = tl.load(
        bias_ptr + output_columns,
        mask=output_valid,
        other=0.0,
    )
    output = (accumulator + bias.to(tl.bfloat16)).to(tl.bfloat16)
    tl.store(
        output_ptr + row_indices[:, None] * WIDTH + output_columns[None, :],
        output,
        mask=row_valid[:, None] & output_valid[None, :],
    )


def _launch_compute_stage(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    original_shape = inputs.shape
    values = inputs.reshape(-1, _WIDTH)
    rows = values.shape[0]
    output = torch.empty_like(values)
    _compute_stage_fwd_kernel[(triton.cdiv(rows, 128), 1)](
        values,
        weight,
        bias,
        output,
        rows,
        WIDTH=_WIDTH,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        num_warps=4,
        num_stages=3,
    )
    return output.reshape(original_shape)


@torch.library.custom_op(
    "miniworld_kernels::mpnn_edge_mlp_compute_save_projected_fwd_v1",
    mutates_args=(),
)
def _forward_op(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = _launch_compute_stage(
        preactivation,
        hidden_weight,
        hidden_bias,
    )
    update = _launch_compute_stage(
        projected,
        output_weight,
        output_bias,
    )
    return update, projected


@_forward_op.register_fake
def _(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del hidden_weight, hidden_bias, output_weight, output_bias
    return torch.empty_like(preactivation), torch.empty_like(preactivation)


def _setup_context(ctx, inputs, output) -> None:
    preactivation, hidden_weight, hidden_bias, output_weight, output_bias = inputs
    _update, projected = output
    # ``projected`` is an implementation detail, not a second differentiable
    # public output.  Suppress its otherwise full-size materialized zero grad.
    ctx.mark_non_differentiable(projected)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(
        preactivation,
        hidden_weight,
        output_weight,
        projected,
    )
    ctx.hidden_bias_dtype = hidden_bias.dtype
    ctx.output_bias_dtype = output_bias.dtype


def _backward(ctx, grad_update, _grad_projected):
    preactivation, hidden_weight, output_weight, projected = ctx.saved_tensors
    grad_update = grad_update.contiguous()

    # Each dX launch also emits exact GELU(input), which is the right operand
    # required by the following cuBLAS weight-gradient GEMM.
    grad_projected, hidden_2 = _projection_dx_op(
        grad_update,
        output_weight,
        projected,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        grad_output_weight = grad_update.reshape(-1, _WIDTH).T @ hidden_2.reshape(
            -1, _WIDTH
        )
    del projected, hidden_2

    grad_preactivation, hidden_1 = _projection_dx_op(
        grad_projected,
        hidden_weight,
        preactivation,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        grad_hidden_weight = grad_projected.reshape(-1, _WIDTH).T @ hidden_1.reshape(
            -1, _WIDTH
        )

    # Preserve CUDA-autocast Linear's BF16 bias-backward rounding boundary
    # before converting the result to the original FP32 parameter dtype.
    grad_hidden_bias = grad_projected.reshape(-1, _WIDTH).sum(
        dim=0,
        dtype=grad_projected.dtype,
    )
    grad_output_bias = grad_update.reshape(-1, _WIDTH).sum(
        dim=0,
        dtype=grad_update.dtype,
    )
    return (
        grad_preactivation,
        grad_hidden_weight.to(hidden_weight.dtype),
        grad_hidden_bias.to(ctx.hidden_bias_dtype),
        grad_output_weight.to(output_weight.dtype),
        grad_output_bias.to(ctx.output_bias_dtype),
    )


_forward_op.register_autograd(_backward, setup_context=_setup_context)


def triton_edge_mlp_update_compute(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    """Run the projected-save, compute-efficient edge MLP."""
    update, _projected = _forward_op(
        preactivation,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
    )
    return update


__all__ = ["triton_edge_mlp_update_compute"]

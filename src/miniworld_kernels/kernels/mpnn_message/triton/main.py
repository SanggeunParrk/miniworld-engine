"""Two-kernel ProteinMPNN hidden-message reduction for A5000 crop shapes.

The first kernel fuses exact GELU with the 128x128 hidden projection.  The
second fuses the following exact GELU, structural mask, and fixed-K reduction.
Backward keeps the global weight-gradient GEMM in PyTorch. Triton owns the
reduction derivative, the projection input gradient, and bias accumulation. The
compute policy saves the projected activation, while the explicit memory
policy recomputes it once in backward.

Backward is deliberately shape-independent: every shape takes the same
sequence of operations, so gradients never change with the batch size.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._policy import _requires_i64_indexing


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def _gelu_grad(x):
    cdf = 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
    pdf_term = x * 0.3989422804014327 * tl.exp(-0.5 * x * x)
    return cdf + pdf_term


@triton.jit
def _projection_fwd_kernel(
    preactivation_ptr,
    weight_ptr,
    bias_ptr,
    projected_ptr,
    rows,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0).to(tl.int64)
    output_block = tl.program_id(1).to(tl.int64)
    row_indices = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    output_columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_valid = row_indices < rows
    output_valid = output_columns < HIDDEN
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for hidden_start in range(0, HIDDEN, BLOCK_K):
        hidden_columns = hidden_start + tl.arange(0, BLOCK_K)
        hidden_valid = hidden_columns < HIDDEN
        preactivation = tl.load(
            preactivation_ptr + row_indices[:, None] * HIDDEN + hidden_columns[None, :],
            mask=row_valid[:, None] & hidden_valid[None, :],
            other=0.0,
        ).to(tl.float32)
        activated = _gelu(preactivation).to(tl.bfloat16)
        weight = tl.load(
            weight_ptr + output_columns[None, :] * HIDDEN + hidden_columns[:, None],
            mask=hidden_valid[:, None] & output_valid[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        accumulator += tl.dot(activated, weight)

    bias = tl.load(bias_ptr + output_columns, mask=output_valid, other=0.0)
    projected = (accumulator + bias.to(tl.bfloat16)).to(tl.bfloat16)
    tl.store(
        projected_ptr + row_indices[:, None] * HIDDEN + output_columns[None, :],
        projected,
        mask=row_valid[:, None] & output_valid[None, :],
    )


@triton.jit
def _gelu_reduce_fwd_kernel(
    projected_ptr,
    mask_ptr,
    reduced_ptr,
    groups,
    neighbor_scale,
    USE_I64: tl.constexpr,
    HIDDEN: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    group = tl.program_id(0)
    output_block = tl.program_id(1)
    if USE_I64:
        group = group.to(tl.int64)
        output_block = output_block.to(tl.int64)
    output_columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    output_valid = output_columns < HIDDEN
    reduced = tl.zeros((BLOCK_N,), tl.float32)
    # K is exactly 48.  Three 16-neighbor chunks avoid keeping a padded
    # 64xBLOCK_N tile live for the entire epilogue.
    for neighbor_start in tl.static_range(0, NEIGHBORS, 16):
        neighbors = neighbor_start + tl.arange(0, 16)
        offsets = (
            group * NEIGHBORS * HIDDEN
            + neighbors[:, None] * HIDDEN
            + output_columns[None, :]
        )
        projected = tl.load(
            projected_ptr + offsets,
            mask=output_valid[None, :],
            other=0.0,
        ).to(tl.float32)
        hidden = _gelu(projected).to(tl.bfloat16).to(tl.float32)
        edge_weight = tl.load(
            mask_ptr + group * NEIGHBORS + neighbors,
        ).to(tl.float32)
        reduced += tl.sum(hidden * edge_weight[:, None], axis=0)
    reduced /= neighbor_scale
    tl.store(
        reduced_ptr + group * HIDDEN + output_columns,
        reduced,
        mask=(group < groups) & output_valid,
    )


# Multi-group reduction backward without the fused bias accumulation. The shipped
# backward uses `_gelu_reduce_db_bwd_kernel` below; this variant is retained as
# the comparison point for the benchmark forensics under
# `benchmarks/modules/mpnn/profiles/`, which measure the fused versus separate
# bias reduction. It is not reachable from the library's own dispatch.
@triton.jit
def _gelu_reduce_bwd_kernel(
    grad_ptr,
    projected_ptr,
    mask_ptr,
    grad_projected_ptr,
    groups,
    neighbor_scale,
    USE_I64: tl.constexpr,
    HIDDEN: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    GROUPS_PER_CTA: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    group_block = tl.program_id(0)
    output_block = tl.program_id(1)
    if USE_I64:
        group_block = group_block.to(tl.int64)
        output_block = output_block.to(tl.int64)
    output_columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    output_valid = output_columns < HIDDEN

    for segment in tl.static_range(GROUPS_PER_CTA):
        group = group_block * GROUPS_PER_CTA + segment
        group_valid = group < groups
        grad = tl.load(
            grad_ptr + group * HIDDEN + output_columns,
            mask=group_valid & output_valid,
            other=0.0,
        ).to(tl.float32)
        for neighbor_start in tl.static_range(0, NEIGHBORS, 16):
            neighbors = neighbor_start + tl.arange(0, 16)
            rows = group * NEIGHBORS + neighbors
            offsets = rows[:, None] * HIDDEN + output_columns[None, :]
            edge_weight = tl.load(
                mask_ptr + rows,
                mask=group_valid,
                other=0.0,
            ).to(tl.float32)
            grad_hidden = (grad[None, :] * edge_weight[:, None] / neighbor_scale).to(
                tl.bfloat16
            )
            projected = tl.load(
                projected_ptr + offsets,
                mask=group_valid & output_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            grad_projected = grad_hidden.to(tl.float32) * _gelu_grad(projected)
            tl.store(
                grad_projected_ptr + offsets,
                grad_projected,
                mask=group_valid & output_valid[None, :],
            )


@triton.jit
def _gelu_reduce_db_bwd_kernel(
    grad_ptr,
    projected_ptr,
    mask_ptr,
    grad_projected_ptr,
    grad_bias_output_ptr,
    groups,
    neighbor_scale,
    USE_I64: tl.constexpr,
    HIDDEN: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ATOMIC_BIAS: tl.constexpr = False,
):
    group = tl.program_id(0)
    output_block = tl.program_id(1)
    if USE_I64:
        group = group.to(tl.int64)
        output_block = output_block.to(tl.int64)
    output_columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    group_valid = group < groups
    output_valid = output_columns < HIDDEN
    grad = tl.load(
        grad_ptr + group * HIDDEN + output_columns,
        mask=group_valid & output_valid,
        other=0.0,
    ).to(tl.float32)
    grad_bias_partial = tl.zeros((BLOCK_N,), tl.float32)

    # dP is already live here, so reduce its 48-neighbor bias contribution
    # before leaving the CTA. The deterministic path stores one value per
    # group; the default path atomically accumulates the same FP32 partial.
    for neighbor_start in tl.static_range(0, NEIGHBORS, 16):
        neighbors = neighbor_start + tl.arange(0, 16)
        rows = group * NEIGHBORS + neighbors
        offsets = rows[:, None] * HIDDEN + output_columns[None, :]
        edge_weight = tl.load(
            mask_ptr + rows,
            mask=group_valid,
            other=0.0,
        ).to(tl.float32)
        grad_hidden = (grad[None, :] * edge_weight[:, None] / neighbor_scale).to(
            tl.bfloat16
        )
        projected = tl.load(
            projected_ptr + offsets,
            mask=group_valid & output_valid[None, :],
            other=0.0,
        ).to(tl.float32)
        grad_projected = (grad_hidden.to(tl.float32) * _gelu_grad(projected)).to(
            tl.bfloat16
        )
        tl.store(
            grad_projected_ptr + offsets,
            grad_projected,
            mask=group_valid & output_valid[None, :],
        )
        grad_bias_partial += tl.sum(grad_projected.to(tl.float32), axis=0)

    if ATOMIC_BIAS:
        tl.atomic_add(
            grad_bias_output_ptr + output_columns,
            grad_bias_partial,
            mask=group_valid & output_valid,
        )
    else:
        tl.store(
            grad_bias_output_ptr + group * HIDDEN + output_columns,
            grad_bias_partial,
            mask=group_valid & output_valid,
        )


@triton.jit
def _zero_bias_grad_kernel(
    grad_bias_ptr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    columns = tl.arange(0, BLOCK)
    tl.store(grad_bias_ptr + columns, 0.0, mask=columns < HIDDEN)


@triton.jit
def _projection_dx_kernel(
    grad_projected_ptr,
    weight_ptr,
    preactivation_ptr,
    grad_preactivation_ptr,
    activated_ptr,
    rows,
    USE_I64: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    input_block = tl.program_id(1)
    if USE_I64:
        row_block = row_block.to(tl.int64)
        input_block = input_block.to(tl.int64)
    row_indices = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    input_columns = input_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_valid = row_indices < rows
    input_valid = input_columns < HIDDEN
    grad_activated = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for output_start in range(0, HIDDEN, BLOCK_K):
        output_columns = output_start + tl.arange(0, BLOCK_K)
        output_valid = output_columns < HIDDEN
        grad_projected = tl.load(
            grad_projected_ptr
            + row_indices[:, None] * HIDDEN
            + output_columns[None, :],
            mask=row_valid[:, None] & output_valid[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + output_columns[:, None] * HIDDEN + input_columns[None, :],
            mask=output_valid[:, None] & input_valid[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        grad_activated += tl.dot(grad_projected, weight)

    grad_activated = grad_activated.to(tl.bfloat16).to(tl.float32)
    preactivation = tl.load(
        preactivation_ptr + row_indices[:, None] * HIDDEN + input_columns[None, :],
        mask=row_valid[:, None] & input_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    one_plus_erf = 1.0 + tl.erf(preactivation * 0.7071067811865476)
    cdf = 0.5 * one_plus_erf
    pdf_term = (
        preactivation
        * 0.3989422804014327
        * tl.exp(-0.5 * preactivation * preactivation)
    )
    grad_preactivation = grad_activated * (cdf + pdf_term)
    offsets = row_indices[:, None] * HIDDEN + input_columns[None, :]
    valid = row_valid[:, None] & input_valid[None, :]
    tl.store(
        grad_preactivation_ptr + offsets,
        grad_preactivation,
        mask=valid,
    )
    # dW needs GELU(preactivation).  It is effectively free once the dX
    # epilogue has computed the exact GELU CDF, and removes a full-tensor
    # PyTorch GELU launch/read/write from backward.
    tl.store(
        activated_ptr + offsets,
        (0.5 * preactivation * one_plus_erf).to(tl.bfloat16),
        mask=valid,
    )


def _forward_impl(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = preactivation.numel() // (48 * 128)
    rows = groups * 48
    projected = torch.empty_like(preactivation)
    _projection_fwd_kernel[(triton.cdiv(rows, 128), 1)](
        preactivation,
        weight,
        bias,
        projected,
        rows,
        HIDDEN=128,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        num_warps=4,
        num_stages=3,
    )
    reduced = torch.empty(groups, 128, device=preactivation.device, dtype=torch.float32)
    # The exact-K tail is bandwidth/register limited rather than latency
    # limited.  BN64/one warp wins consistently for B=1,2,4,8 on A5000 and
    # avoids a shape bucket whose best-case benefit was below measurement noise.
    tail_block_n = 64
    _gelu_reduce_fwd_kernel[(groups, triton.cdiv(128, tail_block_n))](
        projected,
        edge_mask,
        reduced,
        groups,
        neighbor_scale,
        USE_I64=_requires_i64_indexing(preactivation.numel()),
        HIDDEN=128,
        NEIGHBORS=48,
        BLOCK_N=tail_block_n,
        num_warps=1,
        num_stages=1,
    )
    return reduced.reshape(*preactivation.shape[:-2], 128), projected


@torch.library.custom_op(
    "miniworld_kernels::mpnn_message_training_fwd_v4", mutates_args=()
)
def _forward_op(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _forward_impl(
        preactivation,
        weight,
        bias,
        edge_mask,
        neighbor_scale,
    )


@_forward_op.register_fake
def _(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced = preactivation.new_empty(
        *preactivation.shape[:-2],
        128,
        dtype=torch.float32,
    )
    return reduced, torch.empty_like(preactivation)


@torch.library.custom_op(
    "miniworld_kernels::mpnn_message_gelu_reduce_db_bwd_v2", mutates_args=()
)
def _reduce_backward_op(
    grad_reduced: torch.Tensor,
    projected: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_reduced = grad_reduced.contiguous()
    grad_projected = torch.empty_like(projected)
    elements = projected.numel()
    groups = elements // (48 * 128)
    grad_bias_partial = torch.empty(
        groups,
        128,
        device=projected.device,
        dtype=torch.float32,
    )
    _gelu_reduce_db_bwd_kernel[(groups, 2)](
        grad_reduced,
        projected,
        edge_mask,
        grad_projected,
        grad_bias_partial,
        groups,
        neighbor_scale,
        USE_I64=_requires_i64_indexing(elements),
        HIDDEN=128,
        NEIGHBORS=48,
        BLOCK_N=64,
        ATOMIC_BIAS=False,
        num_warps=2,
        num_stages=1,
    )
    return grad_projected, grad_bias_partial


@_reduce_backward_op.register_fake
def _(
    grad_reduced: torch.Tensor,
    projected: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = projected.numel() // (48 * 128)
    grad_bias_partial = projected.new_empty(groups, 128, dtype=torch.float32)
    return torch.empty_like(projected), grad_bias_partial


@torch.library.custom_op(
    "miniworld_kernels::mpnn_message_gelu_reduce_db_atomic_bwd_v1",
    mutates_args=(),
)
def _reduce_backward_atomic_op(
    grad_reduced: torch.Tensor,
    projected: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_reduced = grad_reduced.contiguous()
    grad_projected = torch.empty_like(projected)
    grad_bias = torch.empty(128, device=projected.device, dtype=torch.float32)
    elements = projected.numel()
    groups = elements // (48 * 128)
    _zero_bias_grad_kernel[(1,)](
        grad_bias,
        HIDDEN=128,
        BLOCK=128,
        num_warps=1,
        num_stages=1,
    )
    _gelu_reduce_db_bwd_kernel[(groups, 2)](
        grad_reduced,
        projected,
        edge_mask,
        grad_projected,
        grad_bias,
        groups,
        neighbor_scale,
        USE_I64=_requires_i64_indexing(elements),
        HIDDEN=128,
        NEIGHBORS=48,
        BLOCK_N=64,
        ATOMIC_BIAS=True,
        num_warps=2,
        num_stages=1,
    )
    return grad_projected, grad_bias


@_reduce_backward_atomic_op.register_fake
def _(
    grad_reduced: torch.Tensor,
    projected: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_bias = projected.new_empty(128, dtype=torch.float32)
    return torch.empty_like(projected), grad_bias


@torch.library.custom_op(
    "miniworld_kernels::mpnn_message_projection_dx_activation_v2", mutates_args=()
)
def _projection_dx_op(
    grad_projected: torch.Tensor,
    weight: torch.Tensor,
    preactivation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = preactivation.numel() // 128
    grad_preactivation = torch.empty_like(preactivation)
    activated = torch.empty_like(preactivation)
    _projection_dx_kernel[(triton.cdiv(rows, 64), 1)](
        grad_projected,
        weight,
        preactivation,
        grad_preactivation,
        activated,
        rows,
        USE_I64=_requires_i64_indexing(preactivation.numel()),
        HIDDEN=128,
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=8,
        num_stages=3,
    )
    return grad_preactivation, activated


_DX_CHUNK_ROWS = 262_144


@torch.library.custom_op(
    "miniworld_kernels::mpnn_projection_dx_weight_v1", mutates_args=()
)
def _projection_dx_weight_op(
    grad_projected: torch.Tensor,
    weight: torch.Tensor,
    preactivation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dX and dW without ever holding a full-size ``GELU(preactivation)``.

    The dX kernel emits the activation the weight-gradient GEMM needs, but that
    activation is dead the moment the GEMM consumes it. Materialising it for every
    row makes a full edge tensor live between the two, and the allocator snapshot
    showed three of those alive at the backward peak -- the single largest term.

    Walking the row axis in blocks keeps one reusable activation buffer instead:
    each block runs the same kernel on a contiguous slice and immediately folds
    its contribution into the weight gradient. dX is written straight into its
    final buffer, so the only extra cost is one launch per block.

    The weight gradient accumulates in FP32 across blocks rather than as a single
    BF16-output GEMM, so it is not bitwise equal to the unchunked form -- it is
    strictly better conditioned, in the same way the atomic bias reduction already
    trades exact reproduction for a smaller footprint.
    """
    rows = preactivation.numel() // 128
    grad_preactivation = torch.empty_like(preactivation)
    grad_weight = torch.zeros(128, 128, device=weight.device, dtype=torch.float32)
    flat_grad_projected = grad_projected.reshape(rows, 128)
    flat_preactivation = preactivation.reshape(rows, 128)
    flat_grad_preactivation = grad_preactivation.reshape(rows, 128)
    block_rows = min(_DX_CHUNK_ROWS, rows)
    activated = torch.empty(
        block_rows, 128, device=preactivation.device, dtype=preactivation.dtype
    )
    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        span = stop - start
        _projection_dx_kernel[(triton.cdiv(span, 64), 1)](
            flat_grad_projected[start:stop],
            weight,
            flat_preactivation[start:stop],
            flat_grad_preactivation[start:stop],
            activated,
            span,
            USE_I64=_requires_i64_indexing(span * 128),
            HIDDEN=128,
            BLOCK_M=64,
            BLOCK_N=128,
            BLOCK_K=32,
            num_warps=8,
            num_stages=3,
        )
        # Keep the per-block GEMM in BF16 -- the same precision the unchunked
        # matmul produced -- and accumulate the small [128, 128] partials in FP32.
        # Casting the operands would allocate a block-sized FP32 pair and undo the
        # saving this function exists for.
        grad_weight += torch.mm(
            flat_grad_projected[start:stop].t(), activated[:span]
        ).to(torch.float32)
    return grad_preactivation, grad_weight


@_projection_dx_weight_op.register_fake
def _(
    grad_projected: torch.Tensor,
    weight: torch.Tensor,
    preactivation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del grad_projected
    return (
        torch.empty_like(preactivation),
        weight.new_empty(128, 128, dtype=torch.float32),
    )


@_projection_dx_op.register_fake
def _(
    grad_projected: torch.Tensor,
    weight: torch.Tensor,
    preactivation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(preactivation), torch.empty_like(preactivation)


def _setup_context(ctx, inputs, output) -> None:
    preactivation, weight, bias, edge_mask, neighbor_scale = inputs
    _reduced, projected = output
    # ``projected`` is an implementation detail saved only for our backward.
    # Marking it non-differentiable prevents autograd from materializing a
    # full-size zero gradient for the discarded auxiliary output.
    ctx.mark_non_differentiable(projected)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(preactivation, weight, projected, edge_mask)
    ctx.neighbor_scale = neighbor_scale
    ctx.bias_dtype = bias.dtype


def _backward_from_projected(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    projected: torch.Tensor,
    edge_mask: torch.Tensor,
    grad_reduced: torch.Tensor,
    neighbor_scale: int,
    bias_dtype: torch.dtype,
):
    if torch.are_deterministic_algorithms_enabled():
        grad_projected, grad_bias_partial = _reduce_backward_op(
            grad_reduced,
            projected,
            edge_mask,
            neighbor_scale,
        )
        grad_bias = grad_bias_partial.sum(0)
    else:
        grad_projected, grad_bias = _reduce_backward_atomic_op(
            grad_reduced,
            projected,
            edge_mask,
            neighbor_scale,
        )

    # One dX launch for every shape. It also emits GELU(preactivation), which is
    # the right operand of the weight-gradient GEMM below. An earlier revision
    # swapped in a PyTorch dX/GELU pair for one calibrated group count; that
    # exception was 22% slower in isolation at its own shape, gave bitwise-equal
    # accuracy, and made backward crash under dynamic-shape compilation, so the
    # single Triton path is now used unconditionally.
    # `projected` is dead once the reduction derivative has consumed it. Dropping
    # it before the dX launch keeps three edge tensors live instead of four; the
    # memory policy also recomputed it here, so this is where its lifetime ends.
    del projected
    grad_preactivation, grad_weight = _projection_dx_weight_op(
        grad_projected,
        weight,
        preactivation,
    )
    del grad_projected
    return (
        grad_preactivation,
        grad_weight.to(weight.dtype),
        grad_bias.to(bias_dtype),
        None,
        None,
    )


def _backward(ctx, grad_reduced, _grad_projected):
    preactivation, weight, projected, edge_mask = ctx.saved_tensors
    return _backward_from_projected(
        preactivation,
        weight,
        projected,
        edge_mask,
        grad_reduced,
        ctx.neighbor_scale,
        ctx.bias_dtype,
    )


_forward_op.register_autograd(_backward, setup_context=_setup_context)


@torch.library.custom_op(
    "miniworld_kernels::mpnn_message_memory_fwd_v1", mutates_args=()
)
def _memory_forward_op(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    reduced, _projected = _forward_impl(
        preactivation,
        weight,
        bias,
        edge_mask,
        neighbor_scale,
    )
    return reduced


@_memory_forward_op.register_fake
def _(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    del weight, bias, edge_mask, neighbor_scale
    return preactivation.new_empty(
        *preactivation.shape[:-2],
        128,
        dtype=torch.float32,
    )


@torch.library.custom_op(
    "miniworld_kernels::mpnn_message_recompute_projected_v1", mutates_args=()
)
def _recompute_projected_op(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    rows = preactivation.numel() // 128
    projected = torch.empty_like(preactivation)
    _projection_fwd_kernel[(triton.cdiv(rows, 128), 1)](
        preactivation,
        weight,
        bias,
        projected,
        rows,
        HIDDEN=128,
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
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    del weight, bias
    return torch.empty_like(preactivation)


def _setup_memory_context(ctx, inputs, output) -> None:
    del output
    preactivation, weight, bias, edge_mask, neighbor_scale = inputs
    ctx.save_for_backward(preactivation, weight, bias, edge_mask)
    ctx.neighbor_scale = neighbor_scale
    ctx.bias_dtype = bias.dtype


def _memory_backward(ctx, grad_reduced):
    preactivation, weight, bias, edge_mask = ctx.saved_tensors
    projected = _recompute_projected_op(preactivation, weight, bias)
    return _backward_from_projected(
        preactivation,
        weight,
        projected,
        edge_mask,
        grad_reduced,
        ctx.neighbor_scale,
        ctx.bias_dtype,
    )


_memory_forward_op.register_autograd(
    _memory_backward,
    setup_context=_setup_memory_context,
)


def triton_message_hidden_reduce(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """Fuse the first four hidden-message lines into two physical kernels."""
    reduced, _projected = _forward_op(
        preactivation,
        weight,
        bias,
        edge_mask,
        neighbor_scale,
    )
    return reduced


def triton_message_hidden_reduce_memory(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """Save no projected activation and recompute it once in backward."""
    return _memory_forward_op(
        preactivation,
        weight,
        bias,
        edge_mask,
        neighbor_scale,
    )


__all__ = [
    "triton_message_hidden_reduce",
    "triton_message_hidden_reduce_memory",
]

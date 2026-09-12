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
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels._tiles import tile_grid, tile_order
import triton.language as tl

from miniworld_engine.kernels.mpnn_message.triton._policy import _requires_i64_indexing


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def _gelu_grad(x):
    cdf = 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
    pdf_term = x * 0.3989422804014327 * tl.exp(-0.5 * x * x)
    return cdf + pdf_term


@triton.autotune(
    configs=configs_for("mpnn_message_fwd_gemm_triton"),
    key=["shape_key"],
)
@triton.jit
def _projection_fwd_kernel(
    preactivation_ptr,
    weight_ptr,
    bias_ptr,
    projected_ptr,
    rows,
    shape_key,
    HIDDEN: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # One 1-D grid and a tuned visit order, not a 2-D grid: CUDA varies axis 0 fastest, so
    # the two-axis form is pinned at the row-first end of the axis `_tiles.py` measures.
    row_block, output_block = tile_order(tl.program_id(0).to(tl.int64),
                          tl.cdiv(rows, BLOCK_M1), tl.cdiv(HIDDEN, BLOCK_N), GROUP_M)
    row_indices = row_block * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    output_columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_valid = row_indices < rows
    output_valid = output_columns < HIDDEN
    accumulator = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)

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


@triton.autotune(
    configs=configs_for("mpnn_message_fwd_gelu_reduce_triton"),
    key=["shape_key"],
)
@triton.jit
def _gelu_reduce_fwd_kernel(
    projected_ptr,
    mask_ptr,
    reduced_ptr,
    groups,
    neighbor_scale,
    shape_key,
    USE_I64: tl.constexpr,
    HIDDEN: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I64:
        pid = pid.to(tl.int64)
    # One 1-D grid and a tuned visit order, not a 2-D grid: CUDA varies axis 0 fastest, so
    # the two-axis form is pinned at the row-first end of the axis `_tiles.py` measures.
    group, output_block = tile_order(pid,
                          groups, tl.cdiv(HIDDEN, BLOCK_N), GROUP_M)
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


@triton.autotune(
    configs=configs_for("mpnn_message_bwd_reduce_dbias_triton"),
    key=["shape_key"],
    reset_to_zero=["grad_bias_output_ptr"],
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
    shape_key,
    USE_I64: tl.constexpr,
    HIDDEN: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    ATOMIC_BIAS: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if USE_I64:
        pid = pid.to(tl.int64)
    # One 1-D grid and a tuned visit order, not a 2-D grid: CUDA varies axis 0 fastest, so
    # the two-axis form is pinned at the row-first end of the axis `_tiles.py` measures.
    group, output_block = tile_order(pid,
                          groups, tl.cdiv(HIDDEN, BLOCK_N), GROUP_M)
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


@triton.autotune(
    configs=configs_for("mpnn_message_bwd_dx_triton"),
    key=["shape_key"],
)
@triton.jit
def _projection_dx_kernel(
    grad_projected_ptr,
    weight_ptr,
    preactivation_ptr,
    grad_preactivation_ptr,
    activated_ptr,
    rows,
    shape_key,
    USE_I64: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I64:
        pid = pid.to(tl.int64)
    # One 1-D grid and a tuned visit order, not a 2-D grid: CUDA varies axis 0 fastest, so
    # the two-axis form is pinned at the row-first end of the axis `_tiles.py` measures.
    row_block, input_block = tile_order(pid,
                          tl.cdiv(rows, BLOCK_M1), tl.cdiv(HIDDEN, BLOCK_N), GROUP_M)
    if USE_I64:
        row_block = row_block.to(tl.int64)
        input_block = input_block.to(tl.int64)
    row_indices = row_block * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    input_columns = input_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_valid = row_indices < rows
    input_valid = input_columns < HIDDEN
    grad_activated = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)

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


def _shape_key(rows: int, **axes: int) -> int:
    """The packed bucket for a launch of `rows` rows.

    `both_key` because these launches are row counts and that is what `BOTH_ROWS` buckets. These
    kernels were not tuned at all before -- every knob was written at the launch site -- so this is
    the key their first cache is written under.
    """
    return both_key(rows, **axes)


def _forward_impl(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    neighbors, hidden = preactivation.shape[-2:]
    groups = preactivation.numel() // (neighbors * hidden)
    rows = groups * neighbors
    projected = torch.empty_like(preactivation)
    _projection_fwd_kernel[lambda meta: tile_grid(rows, hidden, meta["BLOCK_M1"], meta["BLOCK_N"])](
        preactivation,
        weight,
        bias,
        projected,
        rows,
        _shape_key(rows),
        HIDDEN=hidden,
    )
    reduced = torch.empty(groups, hidden, device=preactivation.device, dtype=torch.float32)
    # BLOCK_N=64 and one warp used to be written here, from a measurement on an A5000 at
    # B=1,2,4,8. It is a ladder now: the note that came with it said a shape bucket's best-case
    # benefit was below noise, which is an argument for ONE BUCKET, not for one config on every
    # card the repository ships to.
    _gelu_reduce_fwd_kernel[lambda meta: tile_grid(groups, hidden, 1, meta["BLOCK_N"])](
        projected,
        edge_mask,
        reduced,
        groups,
        neighbor_scale,
        _shape_key(groups, NEIGHBORS=neighbors),
        USE_I64=_requires_i64_indexing(preactivation.numel()),
        HIDDEN=hidden,
        NEIGHBORS=neighbors,
    )
    return reduced.reshape(*preactivation.shape[:-2], hidden), projected


def _forward_op_fake(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The FP32 reduction over the 48 neighbours, and the full-size projection beside it.

    The reduction drops the neighbour axis and is FP32 whatever the input dtype; the projection
    is shaped and typed like `preactivation` and exists only for backward.
    """
    reduced = preactivation.new_empty(
        *preactivation.shape[:-2],
        128,
        dtype=torch.float32,
    )
    return reduced, torch.empty_like(preactivation)


@opaque(fake=_forward_op_fake, name="mpnn_message_training_fwd_v4")
def _forward_op(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project, GELU, and reduce over the 48 neighbours, keeping the projection for backward."""
    return _forward_impl(
        preactivation,
        weight,
        bias,
        edge_mask,
        neighbor_scale,
    )


def _reduce_backward_op_fake(
    grad_reduced: torch.Tensor,
    projected: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The projection's gradient, and the bias gradient as one FP32 partial row per group.

    The caller sums the partials, which is what makes this the deterministic branch: the atomic
    variant below returns the same bias gradient already reduced, in an order the hardware picks.
    """
    neighbors, hidden = projected.shape[-2:]
    groups = projected.numel() // (neighbors * hidden)
    grad_bias_partial = projected.new_empty(groups, hidden, dtype=torch.float32)
    return torch.empty_like(projected), grad_bias_partial


@opaque(fake=_reduce_backward_op_fake, name="mpnn_message_gelu_reduce_db_bwd_v2")
def _reduce_backward_op(
    grad_reduced: torch.Tensor,
    projected: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reduction's backward with the bias gradient left as per-group partials, for a fixed sum order."""
    grad_reduced = grad_reduced.contiguous()
    grad_projected = torch.empty_like(projected)
    elements = projected.numel()
    neighbors, hidden = projected.shape[-2:]
    groups = elements // (neighbors * hidden)
    grad_bias_partial = torch.empty(
        groups,
        hidden,
        device=projected.device,
        dtype=torch.float32,
    )
    _gelu_reduce_db_bwd_kernel[
        lambda meta: tile_grid(groups, hidden, 1, meta["BLOCK_N"])
    ](
        grad_reduced,
        projected,
        edge_mask,
        grad_projected,
        grad_bias_partial,
        groups,
        neighbor_scale,
        _shape_key(groups, NEIGHBORS=neighbors, ATOMIC_BIAS=1),
        USE_I64=_requires_i64_indexing(elements),
        HIDDEN=hidden,
        NEIGHBORS=neighbors,
        ATOMIC_BIAS=False,
    )
    return grad_projected, grad_bias_partial


def _reduce_backward_atomic_op_fake(
    grad_reduced: torch.Tensor,
    projected: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The projection's gradient, and a single `(128,)` FP32 bias gradient.

    Already reduced, unlike the partial-row form above -- the kernel accumulates it atomically,
    so its summation order is whatever the hardware runs.
    """
    grad_bias = projected.new_empty(projected.shape[-1], dtype=torch.float32)
    return torch.empty_like(projected), grad_bias


@opaque(fake=_reduce_backward_atomic_op_fake, name="mpnn_message_gelu_reduce_db_atomic_bwd_v1")
def _reduce_backward_atomic_op(
    grad_reduced: torch.Tensor,
    projected: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reduction's backward with the bias gradient accumulated atomically in one buffer.

    The explicit zero handles cached and single-config launches. ``reset_to_zero``
    also clears the accumulator before every autotune trial and before the final
    selected launch: clearing only once would add every trial into the first dBias.
    """
    grad_reduced = grad_reduced.contiguous()
    grad_projected = torch.empty_like(projected)
    neighbors, hidden = projected.shape[-2:]
    grad_bias = torch.empty(hidden, device=projected.device, dtype=torch.float32)
    elements = projected.numel()
    groups = elements // (neighbors * hidden)
    # The one launch here that stays written out, because there is nothing to search: it fills a
    # `hidden`-element buffer with zeros in a single program. One block, one warp, one stage is
    # not a config that won a measurement -- it is the only shape the work has.
    _zero_bias_grad_kernel[(1,)](
        grad_bias,
        HIDDEN=hidden,
        BLOCK=triton.next_power_of_2(hidden),
        num_warps=1,
        num_stages=1,
    )
    _gelu_reduce_db_bwd_kernel[
        lambda meta: tile_grid(groups, hidden, 1, meta["BLOCK_N"])
    ](
        grad_reduced,
        projected,
        edge_mask,
        grad_projected,
        grad_bias,
        groups,
        neighbor_scale,
        _shape_key(groups, NEIGHBORS=neighbors, ATOMIC_BIAS=2),
        USE_I64=_requires_i64_indexing(elements),
        HIDDEN=hidden,
        NEIGHBORS=neighbors,
        ATOMIC_BIAS=True,
    )
    return grad_projected, grad_bias


def _projection_dx_op_fake(
    grad_projected: torch.Tensor,
    weight: torch.Tensor,
    preactivation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dX and the GELU that follows it, both shaped and typed like `preactivation`.

    The activation is a second output rather than a recompute: the weight-gradient GEMM
    downstream contracts against exactly it, and the kernel already has it in registers.
    """
    return torch.empty_like(preactivation), torch.empty_like(preactivation)


@opaque(fake=_projection_dx_op_fake, name="mpnn_message_projection_dx_activation_v2")
def _projection_dx_op(
    grad_projected: torch.Tensor,
    weight: torch.Tensor,
    preactivation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One dX GEMM that also emits the GELU the following weight-gradient GEMM contracts against."""
    hidden = preactivation.shape[-1]
    rows = preactivation.numel() // hidden
    grad_preactivation = torch.empty_like(preactivation)
    activated = torch.empty_like(preactivation)
    _projection_dx_kernel[
        lambda meta: tile_grid(rows, hidden, meta["BLOCK_M1"], meta["BLOCK_N"])
    ](
        grad_projected,
        weight,
        preactivation,
        grad_preactivation,
        activated,
        rows,
        _shape_key(rows),
        USE_I64=_requires_i64_indexing(preactivation.numel()),
        HIDDEN=hidden,
    )
    return grad_preactivation, activated


_DX_CHUNK_ROWS = 262_144


def _projection_dx_weight_op_fake(
    grad_projected: torch.Tensor,
    weight: torch.Tensor,
    preactivation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dX shaped like `preactivation`, and a `(128, 128)` FP32 weight gradient.

    FP32 because the weight gradient accumulates across row blocks -- see the op for why the
    row axis is walked in blocks at all.
    """
    del grad_projected
    return (
        torch.empty_like(preactivation),
        weight.new_empty(128, 128, dtype=torch.float32),
    )


@opaque(fake=_projection_dx_weight_op_fake, name="mpnn_message_projection_dx_weight_v1")
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
    hidden = preactivation.shape[-1]
    rows = preactivation.numel() // hidden
    grad_preactivation = torch.empty_like(preactivation)
    # Most graphs fit in one block. Its GEMM already supplies the whole dW;
    # allocating/zeroing another matrix and adding it launches two unnecessary
    # GPU operations. Subsequent blocks still accumulate in the same FP32 order.
    grad_weight = None
    flat_grad_projected = grad_projected.reshape(rows, hidden)
    flat_preactivation = preactivation.reshape(rows, hidden)
    flat_grad_preactivation = grad_preactivation.reshape(rows, hidden)
    block_rows = min(_DX_CHUNK_ROWS, rows)
    activated = torch.empty(
        block_rows, hidden, device=preactivation.device, dtype=preactivation.dtype
    )
    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        span = stop - start
        _projection_dx_kernel[
            lambda meta: tile_grid(span, hidden, meta["BLOCK_M1"], meta["BLOCK_N"])
        ](
            flat_grad_projected[start:stop],
            weight,
            flat_preactivation[start:stop],
            flat_grad_preactivation[start:stop],
            activated,
            span,
            _shape_key(span),
            USE_I64=_requires_i64_indexing(span * hidden),
            HIDDEN=hidden,
        )
        # Keep the per-block GEMM in BF16 -- the same precision the unchunked
        # matmul produced -- and accumulate the small [128, 128] partials in FP32.
        # Casting the operands would allocate a block-sized FP32 pair and undo the
        # saving this function exists for.
        partial_weight = torch.mm(
            flat_grad_projected[start:stop].t(), activated[:span]
        ).to(torch.float32)
        if grad_weight is None:
            grad_weight = partial_weight
        else:
            grad_weight.add_(partial_weight)
    assert grad_weight is not None
    return grad_preactivation, grad_weight


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


class _MessageHiddenReduce(torch.autograd.Function):
    """The autograd boundary over the saved-activation forward.

    ``projected`` never leaves this class. It used to be a second output of the op, marked
    non-differentiable so autograd would not materialise a full-size zero gradient for it; a
    ``Function`` can simply save it and return the one tensor the caller wants, so the
    ``mark_non_differentiable``/``set_materialize_grads`` pair that made that safe is gone.
    """

    @staticmethod
    def forward(
        ctx,
        preactivation: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        edge_mask: torch.Tensor,
        neighbor_scale: int,
    ) -> torch.Tensor:
        reduced, projected = _forward_op(
            preactivation,
            weight,
            bias,
            edge_mask,
            neighbor_scale,
        )
        ctx.save_for_backward(preactivation, weight, projected, edge_mask)
        ctx.neighbor_scale = neighbor_scale
        ctx.bias_dtype = bias.dtype
        return reduced

    @staticmethod
    def backward(ctx, grad_reduced):
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


def _memory_forward_op_fake(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """Only the reduction: the leading shape with 128 FP32 channels.

    The projection the saved-activation forward returns beside it is absent by design -- this
    variant keeps nothing and recomputes it in backward.
    """
    del weight, bias, edge_mask, neighbor_scale
    return preactivation.new_empty(
        *preactivation.shape[:-2],
        128,
        dtype=torch.float32,
    )


@opaque(fake=_memory_forward_op_fake, name="mpnn_message_memory_fwd_v1")
def _memory_forward_op(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """The same forward as :func:`_forward_op` with the projection dropped instead of saved."""
    reduced, _projected = _forward_impl(
        preactivation,
        weight,
        bias,
        edge_mask,
        neighbor_scale,
    )
    return reduced


def _recompute_projected_op_fake(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Shaped and typed like `preactivation`: the projection is square in the channel axis."""
    del weight, bias
    return torch.empty_like(preactivation)


@opaque(fake=_recompute_projected_op_fake, name="mpnn_message_recompute_projected_v1")
def _recompute_projected_op(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Recompute the projection backward needs, from the three tensors the memory variant saved."""
    hidden = preactivation.shape[-1]
    rows = preactivation.numel() // hidden
    projected = torch.empty_like(preactivation)
    _projection_fwd_kernel[lambda meta: tile_grid(rows, hidden, meta["BLOCK_M1"], meta["BLOCK_N"])](
        preactivation,
        weight,
        bias,
        projected,
        rows,
        _shape_key(rows),
        HIDDEN=hidden,
    )
    return projected


class _MessageHiddenReduceMemory(torch.autograd.Function):
    """The same boundary for the variant that saves nothing and recomputes.

    It saves `bias` where the other saves `projected`: one `(128,)` vector instead of a full edge
    tensor, and the projection is recomputed in backward from the three it keeps.
    """

    @staticmethod
    def forward(
        ctx,
        preactivation: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        edge_mask: torch.Tensor,
        neighbor_scale: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(preactivation, weight, bias, edge_mask)
        ctx.neighbor_scale = neighbor_scale
        ctx.bias_dtype = bias.dtype
        return _memory_forward_op(
            preactivation,
            weight,
            bias,
            edge_mask,
            neighbor_scale,
        )

    @staticmethod
    def backward(ctx, grad_reduced):
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


def triton_message_hidden_reduce(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """Fuse the first four hidden-message lines into two physical kernels."""
    return _MessageHiddenReduce.apply(
        preactivation,
        weight,
        bias,
        edge_mask,
        neighbor_scale,
    )


def triton_message_hidden_reduce_memory(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """Save no projected activation and recompute it once in backward."""
    return _MessageHiddenReduceMemory.apply(
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

"""One-launch ProteinMPNN encoder node message.

The node half of an encoder layer contracts an edge tensor down to a node tensor::

    preactivation = query + edge @ W1e^T + neighbor[index]
    reduced       = sum_k mask * gelu(W2(gelu(preactivation))) / K

Nothing edge-sized survives that reduction, yet the separate-operation form writes
three edge tensors on the way through and reads them all again in backward.  The
allocator trace at ``B=16, T=8192`` attributed four live 1536 MiB blocks to this
chain -- two reduction gradients, one projection gradient and one BF16 replay copy.

This kernel walks one neighbor group at a time, so the group's whole 48-row window
fits in registers and the reduction happens before anything reaches HBM.  Forward
reads one edge tensor and writes only the node-sized result.  Backward replays the
chain from the same input and writes only the edge gradient the previous layer
needs.

Because each program owns whole groups, the query gradient is an exact per-group row
sum with no atomics.  The neighbor gradient is a scatter and does use atomics, which
is what ``F.embedding``'s backward does too.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_kernels.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of


_WIDTH = 128
# One chunk of the buffered preactivation gradient is 262144 x 128 x 2 bytes =
# 64 MiB, a fixed cost that replaces a full edge tensor at any batch size.
_WEIGHT_CHUNK_ROWS = 262_144


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def _gelu_grad(x):
    cdf = 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
    pdf_term = x * 0.3989422804014327 * tl.exp(-0.5 * x * x)
    return cdf + pdf_term


def _configs() -> list[triton.Config]:
    """Full grid over every tuning knob; nothing is pinned.

    A program that owns one group moves 64 KiB of weights for 12 KiB of data, so
    ``GROUPS`` matters more here than the row tile does elsewhere -- an A5000 sweep
    measured forward at 0.932 ms with one group and 0.658 ms with eight.

    An earlier version of this list fixed ``num_stages=2``, and at two stages *every*
    multi-group configuration fails to compile on shared memory.  The list therefore
    pinned the kernel to its slowest working point and hid a 1.42x win.  Configurations
    that do not fit are skipped by the tuner, so offering them costs compile time only.
    """
    return [
        triton.Config({"GROUPS": groups}, num_warps=warps, num_stages=stages)
        for groups in (1, 2, 4, 8, 16)
        for warps in (4, 8, 16)
        for stages in (1, 2, 3)
    ]


# Narrow the offered grid to this GPU's measured top-K.  Two thirds of this grid cannot
# launch on sm_86 at all -- every multi-group configuration fails on shared memory at
# num_stages >= 2 -- and the cache lets a run skip compiling them to find that out.  It
# narrows rather than pins: Triton still tunes among the cached configs, and a grid
# change invalidates the entry via config_space_hash.  See the same block in the edge
# tail kernel for why the grids are deliberately this wide.
_fwd_prune = make_cache_prune(
    "mpnn_node_message_fwd",
    dtype_of=tensor_dtype_of("edge_ptr"),
    bucket_of=key_bucket_of("WIDTH", "NEIGHBORS"),
)
_replay_prune = make_cache_prune(
    "mpnn_node_message_replay",
    dtype_of=tensor_dtype_of("edge_ptr"),
    bucket_of=key_bucket_of("WIDTH", "NEIGHBORS"),
)
_dx_prune = make_cache_prune(
    "mpnn_node_message_dx",
    dtype_of=tensor_dtype_of("preactivation_ptr"),
    bucket_of=key_bucket_of("WIDTH", "NEIGHBORS"),
)


@triton.autotune(
    configs=_configs(),
    key=["groups_total", "NEIGHBORS"],
    prune_configs_by={"early_config_prune": _fwd_prune},
)
@triton.jit
def _node_message_fwd_kernel(
    edge_ptr,
    query_ptr,
    neighbor_ptr,
    index_ptr,
    edge_weight_ptr,
    hidden_weight_ptr,
    hidden_bias_ptr,
    mask_ptr,
    reduced_ptr,
    groups_total,
    neighbor_scale,
    EDGE_WEIGHT_STRIDE: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUPS: tl.constexpr,
):
    columns = tl.arange(0, WIDTH)
    window = tl.arange(0, BLOCK_M)
    window_valid = window < NEIGHBORS
    edge_weight = tl.load(
        edge_weight_ptr + columns[:, None] + columns[None, :] * EDGE_WEIGHT_STRIDE
    ).to(tl.bfloat16)
    hidden_weight = tl.load(
        hidden_weight_ptr + columns[:, None] + columns[None, :] * WIDTH
    ).to(tl.bfloat16)
    hidden_bias = tl.load(hidden_bias_ptr + columns).to(tl.bfloat16)

    for slot in range(GROUPS):
        group = tl.program_id(0) * GROUPS + slot
        valid = window_valid & (group < groups_total)
        rows = group * NEIGHBORS + window
        offsets = rows[:, None] * WIDTH + columns[None, :]
        edge = tl.load(edge_ptr + offsets, mask=valid[:, None], other=0.0).to(
            tl.bfloat16
        )
        query = tl.load(
            query_ptr + group * WIDTH + columns, mask=group < groups_total, other=0.0
        ).to(tl.float32)
        neighbor_rows = tl.load(index_ptr + rows, mask=valid, other=0)
        neighbor = tl.load(
            neighbor_ptr + neighbor_rows[:, None] * WIDTH + columns[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        # The packed projection applies each block as its own autocast F.linear, so
        # each partial sum is rounded to BF16.
        projected = tl.dot(edge, edge_weight).to(tl.bfloat16).to(tl.float32)
        preactivation = (query[None, :] + projected).to(tl.bfloat16).to(tl.float32)
        preactivation = (preactivation + neighbor).to(tl.bfloat16)
        activated = _gelu(preactivation.to(tl.float32)).to(tl.bfloat16)
        hidden = (tl.dot(activated, hidden_weight) + hidden_bias[None, :]).to(
            tl.bfloat16
        )
        # The separate reduction kernel rounds the second GELU to BF16 before the
        # FP32 accumulation; keep that boundary.
        gated = _gelu(hidden.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        weights = tl.load(mask_ptr + rows, mask=valid, other=0.0).to(tl.float32)
        reduced = tl.sum(
            tl.where(valid[:, None], gated * weights[:, None], 0.0), axis=0
        )
        tl.store(
            reduced_ptr + group * WIDTH + columns,
            reduced / neighbor_scale,
            mask=(group < groups_total) & (columns < WIDTH),
        )


@triton.autotune(
    configs=_configs(),
    key=["groups_total", "NEIGHBORS"],
    reset_to_zero=["grad_hidden_bias_ptr"],
    prune_configs_by={"early_config_prune": _replay_prune},
)
@triton.jit
def _node_message_replay_kernel(
    grad_reduced_ptr,
    edge_ptr,
    query_ptr,
    neighbor_ptr,
    index_ptr,
    edge_weight_ptr,
    hidden_weight_ptr,
    hidden_bias_ptr,
    mask_ptr,
    preactivation_ptr,
    activated_ptr,
    grad_hidden_ptr,
    grad_hidden_bias_ptr,
    groups_total,
    group_offset,
    chunk_groups,
    neighbor_scale,
    EDGE_WEIGHT_STRIDE: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """Replay the message forward, then differentiate the masked reduction.

    Two resident weights in the contraction orientation.  An earlier single-pass
    backward held all four -- both projections in both orientations -- and a measured
    A5000 sweep showed why that does not work: 255 registers with 248 spill bytes,
    6.1 TFLOP/s against the 26 TFLOP/s this same shape reaches in forward, and every
    ``GROUPS > 1`` configuration failing to compile at 149-181 KiB of shared memory
    against a 100 KiB limit.  Being stuck at one group per program also meant the
    weight loads were never amortized at all.
    """
    columns = tl.arange(0, WIDTH)
    window = tl.arange(0, BLOCK_M)
    window_valid = window < NEIGHBORS
    edge_weight = tl.load(
        edge_weight_ptr + columns[:, None] + columns[None, :] * EDGE_WEIGHT_STRIDE
    ).to(tl.bfloat16)
    hidden_weight = tl.load(
        hidden_weight_ptr + columns[:, None] + columns[None, :] * WIDTH
    ).to(tl.bfloat16)
    hidden_bias = tl.load(hidden_bias_ptr + columns).to(tl.bfloat16)
    grad_hidden_bias = tl.zeros((WIDTH,), tl.float32)

    for slot in range(GROUPS):
        local_group = tl.program_id(0) * GROUPS + slot
        group = local_group + group_offset
        alive = (local_group < chunk_groups) & (group < groups_total)
        valid = window_valid & alive
        rows = group * NEIGHBORS + window
        offsets = rows[:, None] * WIDTH + columns[None, :]
        local_offsets = (local_group * NEIGHBORS + window)[:, None] * WIDTH + columns[
            None, :
        ]

        edge = tl.load(edge_ptr + offsets, mask=valid[:, None], other=0.0).to(
            tl.bfloat16
        )
        query = tl.load(query_ptr + group * WIDTH + columns, mask=alive, other=0.0).to(
            tl.float32
        )
        neighbor_rows = tl.load(index_ptr + rows, mask=valid, other=0)
        neighbor = tl.load(
            neighbor_ptr + neighbor_rows[:, None] * WIDTH + columns[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        projected = tl.dot(edge, edge_weight).to(tl.bfloat16).to(tl.float32)
        preactivation = (query[None, :] + projected).to(tl.bfloat16).to(tl.float32)
        preactivation = (preactivation + neighbor).to(tl.bfloat16)
        activated = _gelu(preactivation.to(tl.float32)).to(tl.bfloat16)
        hidden = (tl.dot(activated, hidden_weight) + hidden_bias[None, :]).to(
            tl.bfloat16
        )

        # The separate reduction-backward kernel rounds to BF16 before applying the
        # GELU derivative; keep that boundary.
        grad_reduced = tl.load(
            grad_reduced_ptr + group * WIDTH + columns, mask=alive, other=0.0
        ).to(tl.float32)
        weights = tl.load(mask_ptr + rows, mask=valid, other=0.0).to(tl.float32)
        grad_gated = (grad_reduced[None, :] * weights[:, None] / neighbor_scale).to(
            tl.bfloat16
        )
        grad_hidden = (
            grad_gated.to(tl.float32) * _gelu_grad(hidden.to(tl.float32))
        ).to(tl.bfloat16)
        grad_hidden_bias += tl.sum(
            tl.where(valid[:, None], grad_hidden.to(tl.float32), 0.0), axis=0
        )

        tl.store(preactivation_ptr + local_offsets, preactivation, mask=valid[:, None])
        tl.store(activated_ptr + local_offsets, activated, mask=valid[:, None])
        tl.store(grad_hidden_ptr + local_offsets, grad_hidden, mask=valid[:, None])

    tl.atomic_add(grad_hidden_bias_ptr + columns, grad_hidden_bias)


@triton.autotune(
    configs=_configs(),
    key=["groups_total", "NEIGHBORS"],
    reset_to_zero=["grad_neighbor_ptr"],
    prune_configs_by={"early_config_prune": _dx_prune},
)
@triton.jit
def _node_message_dx_kernel(
    preactivation_ptr,
    grad_hidden_ptr,
    index_ptr,
    edge_weight_ptr,
    hidden_weight_ptr,
    grad_edge_ptr,
    grad_query_ptr,
    grad_neighbor_ptr,
    grad_preactivation_ptr,
    groups_total,
    group_offset,
    chunk_groups,
    EDGE_WEIGHT_STRIDE: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """Both projections backwards, plus the two node-side gradients.

    Two resident weights, in row-major orientation so neither needs a ``tl.trans``.
    Because a program owns whole groups, the broadcast query block's gradient is an
    exact per-group row sum with no atomics; only the neighbour scatter needs them,
    which is what ``F.embedding``'s backward does too.
    """
    columns = tl.arange(0, WIDTH)
    window = tl.arange(0, BLOCK_M)
    window_valid = window < NEIGHBORS
    edge_weight_rows = tl.load(
        edge_weight_ptr + columns[:, None] * EDGE_WEIGHT_STRIDE + columns[None, :]
    ).to(tl.bfloat16)
    hidden_weight_rows = tl.load(
        hidden_weight_ptr + columns[:, None] * WIDTH + columns[None, :]
    ).to(tl.bfloat16)

    for slot in range(GROUPS):
        local_group = tl.program_id(0) * GROUPS + slot
        group = local_group + group_offset
        alive = (local_group < chunk_groups) & (group < groups_total)
        valid = window_valid & alive
        rows = group * NEIGHBORS + window
        offsets = rows[:, None] * WIDTH + columns[None, :]
        local_offsets = (local_group * NEIGHBORS + window)[:, None] * WIDTH + columns[
            None, :
        ]

        grad_hidden = tl.load(
            grad_hidden_ptr + local_offsets, mask=valid[:, None], other=0.0
        ).to(tl.bfloat16)
        preactivation = tl.load(
            preactivation_ptr + local_offsets, mask=valid[:, None], other=0.0
        ).to(tl.float32)
        grad_preactivation = (
            tl.dot(grad_hidden, hidden_weight_rows) * _gelu_grad(preactivation)
        ).to(tl.bfloat16)

        tl.store(
            grad_edge_ptr + offsets,
            tl.dot(grad_preactivation, edge_weight_rows),
            mask=valid[:, None],
        )
        tl.store(
            grad_query_ptr + group * WIDTH + columns,
            tl.sum(
                tl.where(valid[:, None], grad_preactivation.to(tl.float32), 0.0), axis=0
            ),
            mask=alive & (columns < WIDTH),
        )
        neighbor_rows = tl.load(index_ptr + rows, mask=valid, other=0)
        tl.atomic_add(
            grad_neighbor_ptr + neighbor_rows[:, None] * WIDTH + columns[None, :],
            grad_preactivation.to(tl.float32),
            mask=valid[:, None],
        )
        tl.store(
            grad_preactivation_ptr + local_offsets,
            grad_preactivation,
            mask=valid[:, None],
        )


def _block_rows(neighbors: int) -> int:
    block = 16
    while block < neighbors:
        block *= 2
    return block


@torch.library.custom_op("miniworld_kernels::mpnn_node_message_fwd_v1", mutates_args=())
def _forward_op(
    edge_states: torch.Tensor,
    query_projection: torch.Tensor,
    neighbor_projection: torch.Tensor,
    flat_neighbor_indices: torch.Tensor,
    edge_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    neighbors = edge_states.shape[-2]
    groups = edge_states.numel() // (neighbors * _WIDTH)
    reduced = query_projection.new_empty(
        *query_projection.shape[:-1], _WIDTH, dtype=torch.float32
    )
    _node_message_fwd_kernel[lambda meta: (triton.cdiv(groups, meta["GROUPS"]),)](
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        edge_mask,
        reduced,
        groups,
        neighbor_scale,
        EDGE_WEIGHT_STRIDE=edge_weight.stride(0),
        NEIGHBORS=neighbors,
        WIDTH=_WIDTH,
        BLOCK_M=_block_rows(neighbors),
    )
    return reduced


@_forward_op.register_fake
def _(
    edge_states: torch.Tensor,
    query_projection: torch.Tensor,
    neighbor_projection: torch.Tensor,
    flat_neighbor_indices: torch.Tensor,
    edge_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    del edge_states, neighbor_projection, flat_neighbor_indices
    del edge_weight, hidden_weight, hidden_bias, edge_mask, neighbor_scale
    return query_projection.new_empty(
        *query_projection.shape[:-1], _WIDTH, dtype=torch.float32
    )


@torch.library.custom_op("miniworld_kernels::mpnn_node_message_bwd_v1", mutates_args=())
def _backward_op(
    grad_reduced: torch.Tensor,
    edge_states: torch.Tensor,
    query_projection: torch.Tensor,
    neighbor_projection: torch.Tensor,
    flat_neighbor_indices: torch.Tensor,
    edge_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> list[torch.Tensor]:
    neighbors = edge_states.shape[-2]
    rows = edge_states.numel() // _WIDTH
    groups = rows // neighbors
    nodes = neighbor_projection.numel() // _WIDTH
    device = edge_states.device
    float32 = torch.float32

    grad_edge = torch.empty_like(edge_states)
    grad_query = torch.empty(groups, _WIDTH, device=device, dtype=float32)
    grad_neighbor = torch.zeros(nodes, _WIDTH, device=device, dtype=float32)
    grad_hidden_weight = torch.zeros(_WIDTH, _WIDTH, device=device, dtype=float32)
    grad_hidden_bias = torch.zeros(_WIDTH, device=device, dtype=float32)
    grad_edge_weight = torch.zeros(_WIDTH, _WIDTH, device=device, dtype=float32)

    chunk_groups = max(1, min(_WEIGHT_CHUNK_ROWS // neighbors, groups))
    buffer_rows = chunk_groups * neighbors
    preactivation = torch.empty(
        buffer_rows, _WIDTH, device=device, dtype=torch.bfloat16
    )
    activated = torch.empty_like(preactivation)
    grad_hidden = torch.empty_like(preactivation)
    grad_preactivation = torch.empty_like(preactivation)
    flat_edge = edge_states.reshape(rows, _WIDTH)

    for start in range(0, groups, chunk_groups):
        span = min(chunk_groups, groups - start)

        def chunk_grid(meta):
            return (triton.cdiv(span, meta["GROUPS"]),)

        _node_message_replay_kernel[chunk_grid](
            grad_reduced,
            edge_states,
            query_projection,
            neighbor_projection,
            flat_neighbor_indices,
            edge_weight,
            hidden_weight,
            hidden_bias,
            edge_mask,
            preactivation,
            activated,
            grad_hidden,
            grad_hidden_bias,
            groups,
            start,
            span,
            neighbor_scale,
            EDGE_WEIGHT_STRIDE=edge_weight.stride(0),
            NEIGHBORS=neighbors,
            WIDTH=_WIDTH,
            BLOCK_M=_block_rows(neighbors),
        )
        _node_message_dx_kernel[chunk_grid](
            preactivation,
            grad_hidden,
            flat_neighbor_indices,
            edge_weight,
            hidden_weight,
            grad_edge,
            grad_query,
            grad_neighbor,
            grad_preactivation,
            groups,
            start,
            span,
            EDGE_WEIGHT_STRIDE=edge_weight.stride(0),
            NEIGHBORS=neighbors,
            WIDTH=_WIDTH,
            BLOCK_M=_block_rows(neighbors),
        )
        # Both weight gradients reduce over every row.  cuBLAS owns this shape: see the
        # kernel docstring for the sweep that put a Triton replacement at 0.230 ms
        # against 0.197 ms.  The FP32 running sum across chunks is better conditioned
        # than one BF16 GEMM output over all rows.
        span_rows = span * neighbors
        grad_edge_weight += torch.mm(
            grad_preactivation[:span_rows].t(),
            flat_edge[start * neighbors : start * neighbors + span_rows],
        ).to(float32)
        grad_hidden_weight += torch.mm(
            grad_hidden[:span_rows].t(), activated[:span_rows]
        ).to(float32)

    return [
        grad_edge,
        grad_query.view_as(query_projection),
        grad_neighbor.view_as(neighbor_projection),
        grad_edge_weight,
        grad_hidden_weight,
        grad_hidden_bias,
    ]


@_backward_op.register_fake
def _(
    grad_reduced: torch.Tensor,
    edge_states: torch.Tensor,
    query_projection: torch.Tensor,
    neighbor_projection: torch.Tensor,
    flat_neighbor_indices: torch.Tensor,
    edge_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> list[torch.Tensor]:
    del grad_reduced, flat_neighbor_indices, hidden_bias, edge_mask, neighbor_scale
    del edge_weight, hidden_weight
    float32 = torch.float32
    return [
        torch.empty_like(edge_states),
        torch.empty_like(query_projection, dtype=float32),
        torch.empty_like(neighbor_projection, dtype=float32),
        edge_states.new_empty(_WIDTH, _WIDTH, dtype=float32),
        edge_states.new_empty(_WIDTH, _WIDTH, dtype=float32),
        edge_states.new_empty(_WIDTH, dtype=float32),
    ]


def _setup_context(ctx, inputs, output) -> None:
    del output
    (
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        edge_mask,
        neighbor_scale,
    ) = inputs
    ctx.save_for_backward(
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        edge_mask,
    )
    ctx.neighbor_scale = neighbor_scale
    ctx.dtypes = (
        query_projection.dtype,
        neighbor_projection.dtype,
        edge_weight.dtype,
        hidden_weight.dtype,
        hidden_bias.dtype,
    )


def _backward(ctx, grad_reduced):
    (
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        edge_mask,
    ) = ctx.saved_tensors
    query_dtype, neighbor_dtype, edge_dtype, hidden_dtype, bias_dtype = ctx.dtypes
    (
        grad_edge,
        grad_query,
        grad_neighbor,
        grad_edge_weight,
        grad_hidden_weight,
        grad_hidden_bias,
    ) = _backward_op(
        grad_reduced.contiguous(),
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        edge_mask,
        ctx.neighbor_scale,
    )
    return (
        grad_edge,
        grad_query.to(query_dtype),
        grad_neighbor.to(neighbor_dtype),
        None,
        grad_edge_weight.to(edge_dtype),
        grad_hidden_weight.to(hidden_dtype),
        # Autocast's Linear rounds a bias gradient to BF16 before the FP32
        # parameter gradient; keep that boundary.
        grad_hidden_bias.to(torch.bfloat16).to(bias_dtype),
        None,
        None,
    )


_forward_op.register_autograd(_backward, setup_context=_setup_context)


def triton_node_message_reduce(
    edge_states: torch.Tensor,
    query_projection: torch.Tensor,
    neighbor_projection: torch.Tensor,
    flat_neighbor_indices: torch.Tensor,
    edge_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """Run the whole node message as one fused, fully replayed op."""
    return _forward_op(
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        edge_mask,
        neighbor_scale,
    )


__all__ = ["triton_node_message_reduce"]

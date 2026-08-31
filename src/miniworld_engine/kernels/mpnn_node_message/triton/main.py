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

# The per-kernel cache-prune objects that used to sit here are gone with the API that made
# them (`make_cache_prune`, deleted in fcd3c7a). `install_cache_reader` now narrows EVERY
# autotuner to the cached top-K, and `bucket_of_autotuner` reads the bucket from the
# kernel's own `key=[...]` -- so a kernel that keys on `shape_key` is cached without any
# wiring of its own, and a hand-written `bucket_of` could only disagree with it.
import torch
import triton

from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.configs import configs_for
import triton.language as tl


#: NOT used to size a launch any more -- every function below reads the width from the tensor
#: it was handed. It is the width this family's kernels are BUILT for: they load a whole
#: [width, width] weight into registers and hold it across the row tiles, which is what makes
#: them fast and what makes a wider one fail to compile rather than run slowly. `interface.py`
#: asserts it on the way in; this is the number that assertion is about.
_BUILT_FOR_WIDTH = 128
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

def _shape_key(groups: int, **axes: int) -> int:
    """The packed bucket for a node-message launch.

    `both_key` because the launch is `groups` ROWS -- one per node -- and that is what
    `BOTH_ROWS` buckets. The kernels used to key on `groups_total` directly, which is a raw count:
    every distinct node count was its own cache entry, so a 257-residue protein and a 258-residue
    one shared nothing. Floor-clamping into the shared rung set is the whole point of the key.

    `NEIGHBORS` folds in because it changes the work per row and the compiled kernel both -- k is
    a real shape axis here, not a flag. `WIDTH` is 128 and only 128 (`_BUILT_FOR_WIDTH`), so it is not folded
    in: an axis with one value adds a digit that never varies.

    Axes are passed BY NAME, like the edge tail's helper. A positional `neighbors` produced the
    same key, but the key-gap audit reads the fold off the `both_key(...)` call's keywords at the
    launch site -- so a positional one is invisible to it, and `NEIGHBORS` was reported as a
    constexpr outside the key when it had been inside it all along.
    """
    return both_key(groups, **axes)


@triton.autotune(
    configs=configs_for("mpnn_node_message_fwd_gemm_triton"),
    key=["shape_key"],
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
    shape_key,
    neighbor_scale,
    EDGE_WEIGHT_STRIDE: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    GROUPS: tl.constexpr,
):
    columns = tl.arange(0, WIDTH)
    window = tl.arange(0, BLOCK_M1)
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
    configs=configs_for("mpnn_node_message_bwd_recompute_triton"),
    key=["shape_key"],
    reset_to_zero=["grad_hidden_bias_ptr"],
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
    shape_key,
    group_offset,
    chunk_groups,
    neighbor_scale,
    EDGE_WEIGHT_STRIDE: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M1: tl.constexpr,
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
    window = tl.arange(0, BLOCK_M1)
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
    configs=configs_for("mpnn_node_message_bwd_dx_triton"),
    key=["shape_key"],
    reset_to_zero=["grad_neighbor_ptr"],
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
    shape_key,
    group_offset,
    chunk_groups,
    EDGE_WEIGHT_STRIDE: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """Both projections backwards, plus the two node-side gradients.

    Two resident weights, in row-major orientation so neither needs a ``tl.trans``.
    Because a program owns whole groups, the broadcast query block's gradient is an
    exact per-group row sum with no atomics; only the neighbour scatter needs them,
    which is what ``F.embedding``'s backward does too.
    """
    columns = tl.arange(0, WIDTH)
    window = tl.arange(0, BLOCK_M1)
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


def _forward_op_fake(
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
    """One FP32 row per node: the query projection's leading shape with `WIDTH` channels.

    The neighbour axis is gone because it is what the kernel reduces over, and the result is
    FP32 whatever the edge states' dtype -- it is a sum over k terms, and the cast back happens
    at the autograd boundary rather than in the kernel.
    """
    width = edge_states.shape[-1]
    del edge_states, neighbor_projection, flat_neighbor_indices
    del edge_weight, hidden_weight, hidden_bias, edge_mask, neighbor_scale
    return query_projection.new_empty(
        *query_projection.shape[:-1], width, dtype=torch.float32
    )


@opaque(fake=_forward_op_fake, name="mpnn_node_message_fwd_v1")
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
    """Project the edge states, GELU them, and reduce over the k neighbours in one launch."""
    width = edge_states.shape[-1]
    neighbors = edge_states.shape[-2]
    groups = edge_states.numel() // (neighbors * width)
    reduced = query_projection.new_empty(
        *query_projection.shape[:-1], width, dtype=torch.float32
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
        _shape_key(groups, NEIGHBORS=neighbors),
        neighbor_scale,
        EDGE_WEIGHT_STRIDE=edge_weight.stride(0),
        NEIGHBORS=neighbors,
        WIDTH=width,
        BLOCK_M1=_block_rows(neighbors),
    )
    return reduced


def _backward_op_fake(
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
    """The six gradients backward unpacks, in order.

    The edge-state gradient keeps the activation dtype; the other five are FP32 because each is
    a reduction down the row axis and is rounded back at the autograd boundary. The two weights
    are `(WIDTH, WIDTH)` and the bias `(WIDTH,)`, none of which any input's shape carries.
    """
    width = grad_reduced.shape[-1]
    del grad_reduced, flat_neighbor_indices, hidden_bias, edge_mask, neighbor_scale
    del edge_weight, hidden_weight
    float32 = torch.float32
    return [
        torch.empty_like(edge_states),
        torch.empty_like(query_projection, dtype=float32),
        torch.empty_like(neighbor_projection, dtype=float32),
        edge_states.new_empty(width, width, dtype=float32),
        edge_states.new_empty(width, width, dtype=float32),
        edge_states.new_empty(width, dtype=float32),
    ]


@opaque(fake=_backward_op_fake, name="mpnn_node_message_bwd_v1")
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
    """Replay the forward and differentiate it, in blocks down the row axis.

    Nothing from the forward is saved, so the projection and its GELU are recomputed here; the
    row blocking exists so a single reusable activation buffer stands in for a full-size one.
    """
    width = grad_reduced.shape[-1]
    neighbors = edge_states.shape[-2]
    rows = edge_states.numel() // width
    groups = rows // neighbors
    nodes = neighbor_projection.numel() // width
    device = edge_states.device
    float32 = torch.float32

    grad_edge = torch.empty_like(edge_states)
    grad_query = torch.empty(groups, width, device=device, dtype=float32)
    grad_neighbor = torch.zeros(nodes, width, device=device, dtype=float32)
    grad_hidden_weight = torch.zeros(width, width, device=device, dtype=float32)
    grad_hidden_bias = torch.zeros(width, device=device, dtype=float32)
    grad_edge_weight = torch.zeros(width, width, device=device, dtype=float32)

    chunk_groups = max(1, min(_WEIGHT_CHUNK_ROWS // neighbors, groups))
    buffer_rows = chunk_groups * neighbors
    preactivation = torch.empty(
        buffer_rows, width, device=device, dtype=torch.bfloat16
    )
    activated = torch.empty_like(preactivation)
    grad_hidden = torch.empty_like(preactivation)
    grad_preactivation = torch.empty_like(preactivation)
    flat_edge = edge_states.reshape(rows, width)

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
            _shape_key(groups, NEIGHBORS=neighbors),
            start,
            span,
            neighbor_scale,
            EDGE_WEIGHT_STRIDE=edge_weight.stride(0),
            NEIGHBORS=neighbors,
            WIDTH=width,
            BLOCK_M1=_block_rows(neighbors),
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
            _shape_key(groups, NEIGHBORS=neighbors),
            start,
            span,
            EDGE_WEIGHT_STRIDE=edge_weight.stride(0),
            NEIGHBORS=neighbors,
            WIDTH=width,
            BLOCK_M1=_block_rows(neighbors),
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


class _NodeMessageReduce(torch.autograd.Function):
    """The autograd boundary for the fused node message.

    Both launches are opaque ops and this is the ``Function`` over them, which is the one shape
    the whole kernel tree uses. It replaces a ``register_autograd`` on the forward op: that form
    works, and everything its backward needs is a forward input, but two ways of saying the same
    thing cost more than the tidiness of the second one was worth.
    """

    @staticmethod
    def forward(
        ctx,
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

    @staticmethod
    def backward(ctx, grad_reduced):
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
    return _NodeMessageReduce.apply(
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

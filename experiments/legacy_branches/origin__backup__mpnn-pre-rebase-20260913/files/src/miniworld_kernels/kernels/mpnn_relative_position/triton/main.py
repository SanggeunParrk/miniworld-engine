"""Relative-position embedding with a backward written for its actual shape.

ProteinMPNN buckets the sequence offset between an edge's two residues, clamped to
+/-32, plus one bucket for a different chain: 66 classes, embedded to 16 channels. The
index is per *edge*, so at ``B=16, T=8192, K=48`` the backward reduces 6,291,456 rows
into 66 -- around 95,000 rows per bucket, and far worse than that in practice because
every contact beyond the clamp piles into the two end buckets. A measurement of the
distribution the real feature path produces put 33% of all edges in buckets 0 and 64.

That skew is the whole problem. Measured on it, against the uniform index that would
have made the easy case look representative:

| 6,291,456 rows -> [66, 16]      | real index | uniform index |
|---------------------------------|-----------:|--------------:|
| ``F.embedding`` backward        |  18.815 ms |     10.991 ms |
| the same under ``torch.compile``|   7.957 ms |      3.775 ms |
| ``Tensor.index_add_``           |   3.394 ms |      1.596 ms |
| the kernel below                |   3.750 ms |             -- |

``F.embedding``'s backward is slow here for a reason that has nothing to do with the
bucket count: it sorts the index, gathers the gradient into sorted order -- a full
reordered copy, which is where its 768 MiB of temporaries go -- reduces partial
segments, and scatters. Sixteen kernels against one. That is the right algorithm for a
50,000-token vocabulary and the wrong one for a 16-channel row, and the ratio holds at
about 6x across bucket counts from 66 to 262,144, so row width is what decides it.

WHAT THIS BUYS END TO END, WHICH IS ALMOST NOTHING UNDER ``torch.compile``. Measured at
``B=16, T=8192`` on an A5000, against a 1343.41 ms step with the backend off:

| backend      | step         | the reduction | elementwise |
|--------------|-------------:|--------------:|------------:|
| ``off``      |     1343.41  |     ~19 ms    |    40.86 ms |
| ``index_add``|   **1340.04**|      ~3 ms    |    58.18 ms |
| ``triton``   |     1357.41  |      ~4 ms    |    65.21 ms |

The reduction gets five times faster and the step does not move: an autograd boundary
is a fusion boundary, so the bias add and the projection prologue that used to fuse with
their neighbours stop doing so, and that costs as much as the reduction saves. A
``torch.library.custom_op`` was worse still -- opaque rather than merely a boundary --
at +24 ms of elementwise against this +17. The ``triton`` backend ends up 14 ms behind
``off`` because its extra launch splits the graph again and drops the radial-basis
features out of their fusion.

So this is shipped OFF, and enabling it is a judgement about the run, not an
improvement to reach for. It is worth having anyway for three reasons that the table
does not show: in eager execution there is no fusion to lose and the op is a straight
five-fold win; the kernel is more accurate than what it replaces (2.7e-06 against
6.7e-06 relative to FP64) and reproducible run to run, which ``index_add_`` is not; and
it allocates nothing where ``embedding_dense_backward`` allocates 768 MiB.

One thing is deliberately NOT claimed. This pass is worth about 19 ms of the step, not
the 214 ms its profile category reports. Most of that category is ``gather_neighbors``,
a different and much larger scatter -- 1.5 GiB into 131,072 destinations -- which
measures close to its own floor and is left alone. The privatised-table trick does not
transfer there either: a [131072, 128] accumulator is 67 MiB and cannot live in a
program, and at that width ``index_add_`` loses to ``F.embedding``'s sort by 2.5x.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_kernels.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of


def _configs() -> list[triton.Config]:
    """Full grid over every tuning knob; nothing is pinned.

    ``PROGRAMS`` is the interesting one and it trades two costs against each other.
    Each program privatises the whole table, so the global atomics are one flush per
    program rather than one per element -- fewer programs means fewer atomics, more
    programs means more parallelism over a 6-million-row stream. Where that balance
    falls depends on the card, which is what the tuner and the shipped cache are for.
    """
    return [
        triton.Config(
            {"BLOCK_M": block_m, "PROGRAMS": programs},
            num_warps=warps,
            num_stages=stages,
        )
        for block_m in (32, 64, 128, 256)
        for programs in (64, 128, 256, 512, 1024)
        for warps in (4, 8, 16)
        for stages in (1, 2, 3)
    ]


# The largest ``PROGRAMS`` the grid offers.  Kept beside ``_configs`` because the two
# have to agree: the partial buffer is allocated for this many slots.
_MAX_PROGRAMS = 1024

_reduce_prune = make_cache_prune(
    "mpnn_relative_position_bwd",
    dtype_of=tensor_dtype_of("grad_ptr"),
    bucket_of=key_bucket_of("BUCKET_BLOCK", "WIDTH"),
)


# No ``reset_to_zero``: every program plainly overwrites its own slot, so the pass is
# idempotent under the tuner's repeated benchmark runs, and the caller reads back only
# the slots the chosen configuration wrote rather than trusting a hook to have cleared
# the rest.  Depending on that hook instead measured a relative error of 1.42.
@triton.autotune(
    configs=_configs(),
    key=["rows", "buckets", "WIDTH"],
    prune_configs_by={"early_config_prune": _reduce_prune},
)
@triton.jit
def _bucket_reduce_kernel(
    grad_ptr,
    index_ptr,
    partial_table_ptr,
    partial_bias_ptr,
    rows,
    buckets,
    BUCKET_BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    """Reduce one row per edge into the table, and the same rows into the bias.

    The whole table is privatised per program: at 64 buckets and 16 channels that is
    4 KiB of registers, so a program can hold it for the entire stream and write it out
    once. The alternative -- one atomic per element, which is what the scatter this
    replaces does -- pays 6.29M of them into 66 destinations, a third of which land on
    two.

    ``buckets`` is unused in the body: the padded rows are reduced too and the caller
    slices them off. It stays a parameter because it is part of the autotune key -- a
    different table size is a different tuning problem.

    The scatter is expressed as a matmul against the one-hot of the index rather than
    a sort or a shared-memory atomic. ``allow_tf32`` is off deliberately: the left
    operand is exactly 0 or 1, so an FP32 product is an exact sum, while TF32 would
    round the *values* to a 10-bit mantissa and buy nothing.

    The bias gradient is the same rows summed without the one-hot, so it costs one
    more accumulator and no extra traffic. Left to two separate operations it is a
    second full pass over the gradient.
    """
    columns = tl.arange(0, WIDTH)
    bucket_ids = tl.arange(0, BUCKET_BLOCK)
    table_accumulator = tl.zeros((BUCKET_BLOCK, WIDTH), tl.float32)
    bias_accumulator = tl.zeros((WIDTH,), tl.float32)

    start = tl.program_id(0) * BLOCK_M
    stride = PROGRAMS * BLOCK_M
    for base in range(start, rows, stride):
        offsets = base + tl.arange(0, BLOCK_M)
        valid = offsets < rows
        # ``other=-1`` so a masked-off row matches no bucket and contributes nothing,
        # which keeps the one-hot correct without a second mask on the product.
        index = tl.load(index_ptr + offsets, mask=valid, other=-1)
        values = tl.load(
            grad_ptr + offsets[:, None] * WIDTH + columns[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        selected = (index[None, :] == bucket_ids[:, None]).to(tl.float32)
        table_accumulator += tl.dot(selected, values, allow_tf32=False)
        bias_accumulator += tl.sum(values, axis=0)

    # Each program *stores* its partial rather than adding it in, and a fixed-order
    # sum combines them.  Combining by atomic would be a little faster and would make
    # the gradient irreproducible run to run, because the order the programs land in
    # is scheduling order.  It also measures less accurate: the two end buckets take
    # about a million rows each, and a flat FP32 chain over a million terms is worse
    # conditioned than P partial sums combined in a tree.  Measured on the target
    # shape: 2.720e-06 relative to an FP64 reduction and bit-for-bit reproducible,
    # against 2.789e-06 and irreproducible for the atomic combine, and 2.157e-05 and
    # irreproducible for ``index_add_``.
    tl.store(
        partial_table_ptr
        + tl.program_id(0) * BUCKET_BLOCK * WIDTH
        + bucket_ids[:, None] * WIDTH
        + columns[None, :],
        table_accumulator,
    )
    tl.store(partial_bias_ptr + tl.program_id(0) * WIDTH + columns, bias_accumulator)


def triton_bucket_reduce(
    grad_output: torch.Tensor,
    bucket: torch.Tensor,
    buckets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grad_table, grad_bias)`` for a flattened gradient and index.

    Two stages, so the result is reproducible: the kernel writes one partial per
    program and the sum below combines them in program order.  The buffer is
    ``PROGRAMS x BUCKET_BLOCK x WIDTH`` floats -- 1 MiB at the shipped table size,
    against the 768 MiB that ``F.embedding``'s backward allocates for the same
    reduction, because it materialises a reordered copy of the whole gradient.
    """
    width = grad_output.shape[-1]
    rows = grad_output.numel() // width
    padded = triton.next_power_of_2(buckets)
    grad_output = grad_output.contiguous()
    # Sized for the largest program count the grid offers, because the buffer has to
    # exist before the tuner has chosen one.
    partial_table = grad_output.new_empty(
        (_MAX_PROGRAMS, padded, width), dtype=torch.float32
    )
    partial_bias = grad_output.new_empty((_MAX_PROGRAMS, width), dtype=torch.float32)
    _bucket_reduce_kernel[lambda meta: (meta["PROGRAMS"],)](
        grad_output.reshape(rows, width),
        bucket.reshape(rows),
        partial_table,
        partial_bias,
        rows,
        buckets,
        BUCKET_BLOCK=padded,
        WIDTH=width,
    )
    # Read back exactly the slots this launch wrote.  An earlier version zeroed the
    # whole buffer with ``reset_to_zero`` and summed all of it, which measured a
    # relative error of 1.42 -- the tuner benchmarks configurations with different
    # program counts against the same allocation, and a wider one's partials were
    # still sitting in the tail when a narrower one ran.  Slicing to the chosen count
    # makes the pass correct by construction rather than by a hook firing at the right
    # moment, and it drops a memset per call.
    programs = _bucket_reduce_kernel.best_config.kwargs["PROGRAMS"]
    # Summed in program order, so the result is the same on every run.
    return (
        partial_table[:programs].sum(dim=0)[:buckets],
        partial_bias[:programs].sum(dim=0),
    )


def _index_add_bucket_reduce(
    grad_output: torch.Tensor,
    bucket: torch.Tensor,
    buckets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same reduction through ``Tensor.index_add_``.

    Not a fallback.  On the index distribution the model actually produces this beat
    the first hand-written kernel written for this shape, and it beats ``F.embedding``'s
    own backward by more than five times.  Most of the win available here comes from
    the boundary rather than from the reduction, so the cheap implementation is a real
    option and is measured against the kernel rather than assumed to lose.
    """
    width = grad_output.shape[-1]
    # ``contiguous`` first: an incoming gradient need not be, and both this reduction
    # and the kernel below address the rows as a plain row-major block.
    flat = grad_output.contiguous().reshape(-1, width).float()
    grad_table = torch.zeros(
        buckets, width, device=grad_output.device, dtype=torch.float32
    )
    grad_table.index_add_(0, bucket.reshape(-1), flat)
    return grad_table, flat.sum(dim=0)


class _RelativePositionEmbed(torch.autograd.Function):
    """Replace only the backward, and let the compiler keep the forward.

    An earlier version of this was a ``torch.library.custom_op``, which is opaque to
    Inductor by construction.  That cost more than the reduction saved: measured at
    ``B=16``, the boundary cut the relative-position reduction from about 19 ms to
    about 3, and added 24 ms of elementwise work, because the bias add and the
    downstream projection's prologue stopped fusing with their neighbours.  Net effect
    on the step was 1.0 ms out of 1358.

    ``torch.autograd.Function`` is the right boundary for this: Dynamo traces the
    forward *into* the graph, so it fuses exactly as ``F.embedding(bucket, table) +
    bias`` did, while backward stays ours.
    """

    @staticmethod
    def forward(ctx, bucket, table, bias, use_triton):
        ctx.save_for_backward(bucket)
        ctx.buckets = table.shape[0]
        ctx.dtypes = (table.dtype, bias.dtype)
        # Carried through the call rather than read from module state, so two models
        # with different backends cannot race each other through a shared global.
        ctx.use_triton = use_triton
        return torch.nn.functional.embedding(bucket, table) + bias

    @staticmethod
    def backward(ctx, grad_output):
        (bucket,) = ctx.saved_tensors
        table_dtype, bias_dtype = ctx.dtypes
        reduce = triton_bucket_reduce if ctx.use_triton else _index_add_bucket_reduce
        grad_table, grad_bias = reduce(grad_output, bucket, ctx.buckets)
        return None, grad_table.to(table_dtype), grad_bias.to(bias_dtype), None


def relative_position_embed_op(
    bucket: torch.Tensor,
    table: torch.Tensor,
    bias: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """Look up one row per edge, with a backward written for the shape it reduces.

    Left as plain ``F.embedding``, the reduction that PyTorch and Inductor choose for
    66 destinations and 16 channels costs about 19 ms per step at ``B=16`` against
    3 ms for either replacement here -- and 768 MiB of temporaries against under one,
    because ``embedding_dense_backward`` sorts the index and materialises a reordered
    copy of the whole gradient to do it.
    """
    return _RelativePositionEmbed.apply(bucket, table, bias, backend == "triton")


__all__ = ["relative_position_embed_op", "triton_bucket_reduce"]

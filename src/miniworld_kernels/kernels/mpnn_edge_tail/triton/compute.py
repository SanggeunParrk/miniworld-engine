"""Compute-efficient ProteinMPNN encoder edge tail: GEMM outputs saved, not replayed.

The memory-oriented path in :mod:`.main` chains the projections inside one kernel and
recomputes the whole chain during backward, which is right when a 1536 MiB edge tensor
per layer is the binding constraint.  With memory handled by gradient accumulation it is
the wrong trade: chaining the three GEMMs into one kernel costs 94.4 ms, because holding
the chain in registers caps the tile at [64, 128] while cuBLAS runs [128, 256] at 42.7
TFLOP/s.

Measured on an A5000 at 3,145,728 edge rows (B=8, T=8192, k=48), backward only, all ten
gradients, BF16 with FP32 accumulation, compile=true cudagraph=disabled:

    compiled PyTorch    48.2 ms
    this kernel         39.3 ms    1.23x      (2048: 1.29x, 4096: 1.28x)

An earlier revision of this note put compiled PyTorch at 70.5 ms.  It does not reproduce,
and the reason is worth keeping: no kernel bench called ``torch.compile`` at all, so every
row the CSV labelled ``compiled=True`` ran eager.  Eager on the previous revision of this
bench measured 89.2 ms -- not like-for-like, since that bench scored three gradients where
this one scores ten, but it is the figure the apparent 2x rested on.  Compiling the
baseline is worth more than this kernel is.

Accuracy, against an FP32 reference: every one of the ten gradients lands closer than the
BF16 PyTorch chain does, and the forward matches ``edge_tail_update_pytorch`` to 4.08e-3,
the same distance the PyTorch row sits at.

So one GEMM per kernel, properly tiled, with the neighbouring elementwise folded into the
prologue and epilogue instead.  Forward saves the two GEMM outputs the derivative needs
plus the pre-norm values and the dropout mask; backward reads them and recomputes only
the GELUs, which is the cheap direction.

The rule that produced every win here: store a GEMM output, recompute an elementwise
result.  The rule that produced every loss: assuming less traffic means less time.  Five
separate traffic reductions measured slower -- chaining the GEMMs, dropping the
contraction loop in the last dX pass, applying GELU on load instead of storing the
activation, moving a bias reduction out of the kernel, and merging the first two backward
stages.  All five lost to pipelining or register pressure.  The two changes that did pay
were algorithmic: a BF16 residual-gradient buffer, and handing the neighbour scatter to
ATen's sort-and-segment reduction, which beats one atomic per element by more than four
times at this width.

Six kernels, three each way, and every one of them is the same shape: a [BLOCK_M, WIDTH]
row tile, one GEMM against a [WIDTH, WIDTH] weight, elementwise work in the prologue and
the epilogue.  :func:`_row_gemm` is that GEMM; the kernels below are its prologues and
epilogues.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

WIDTH = 128


# ---- GELU, split so the derivative can borrow the forward's transcendental ------------
@triton.jit
def _gelu_erf(x):
    """The one transcendental GELU and its derivative share."""
    return tl.erf(x * 0.7071067811865476)


@triton.jit
def _gelu_from_erf(x, erf_term):
    return 0.5 * x * (1.0 + erf_term)


@triton.jit
def _gelu_grad_from_erf(x, erf_term):
    return 0.5 * (1.0 + erf_term) + x * 0.3989422804014327 * tl.exp(-0.5 * x * x)


@triton.jit
def _gelu(x):
    return _gelu_from_erf(x, _gelu_erf(x))


# ---- the tile every kernel in this file is built around ------------------------------
@triton.jit
def _row_gemm(
    x_ptr, w_ptr, row_block, valid, columns,
    WIDTH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    CONTRACT_OUT: tl.constexpr,
):
    """FP32 [BLOCK_M, WIDTH] accumulator for ``x @ w``, contracted in BLOCK_K chunks.

    ``w`` is always stored [out, in], the orientation ``nn.Linear`` keeps.  Forward
    contracts over ``in`` and backward over ``out``, so the two directions differ only
    in which of the two indices strides -- the transpose ``F.linear`` applies is done by
    addressing, never by materialising a tensor.
    """
    accumulator = tl.zeros((BLOCK_M, WIDTH), tl.float32)
    for start in range(0, WIDTH, BLOCK_K):
        contraction = start + tl.arange(0, BLOCK_K)
        left = tl.load(
            x_ptr + row_block[:, None] * WIDTH + contraction[None, :],
            mask=valid[:, None], other=0.0,
        )
        if CONTRACT_OUT:
            right = tl.load(w_ptr + contraction[:, None] * WIDTH + columns[None, :])
        else:
            right = tl.load(w_ptr + contraction[:, None] + columns[None, :] * WIDTH)
        accumulator += tl.dot(left, right.to(tl.bfloat16))
    return accumulator


def _configs():
    return [
        triton.Config({"BLOCK_M": m, "BLOCK_K": k}, num_warps=w, num_stages=s)
        for m in (64, 128, 256)
        for k in (32, 64, 128)
        for w in (4, 8)
        for s in (1, 2, 3)
    ]


def _elementwise_configs():
    return [
        triton.Config({"BLOCK_M": m}, num_warps=w, num_stages=s)
        for m in (64, 128, 256, 512)
        for w in (4, 8)
        for s in (1, 2, 3)
    ]


# ---- forward -------------------------------------------------------------------------
@triton.autotune(configs=_configs(), key=["rows"])
@triton.jit
def _project_edge(
    edge_ptr, query_ptr, table_ptr, index_ptr, w1_ptr,
    preactivation_ptr, activated_ptr,
    rows, NEIGHBORS: tl.constexpr, WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """``edge @ W1 + query[group] + table[index]``, then GELU.

    The two gathers are the prologue's whole job: one broadcast over the neighbour axis,
    one indexed, both folded in rather than materialised as separate tensors.
    """
    row_block = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row_block < rows
    columns = tl.arange(0, WIDTH)
    accumulator = _row_gemm(
        edge_ptr, w1_ptr, row_block, valid, columns,
        WIDTH=WIDTH, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, CONTRACT_OUT=False,
    )

    offsets = row_block[:, None] * WIDTH + columns[None, :]
    groups = row_block // NEIGHBORS
    query = tl.load(
        query_ptr + groups[:, None] * WIDTH + columns[None, :],
        mask=valid[:, None], other=0.0,
    ).to(tl.float32)
    neighbor_rows = tl.load(index_ptr + row_block, mask=valid, other=0)
    neighbor = tl.load(
        table_ptr + neighbor_rows[:, None] * WIDTH + columns[None, :],
        mask=valid[:, None], other=0.0,
    ).to(tl.float32)
    preactivation = (accumulator + query + neighbor).to(tl.bfloat16)
    tl.store(preactivation_ptr + offsets, preactivation, mask=valid[:, None])
    # Stored rather than recomputed in the next stage: applying the GELU on load puts a
    # tl.erf between the load and the dot, Triton stops pipelining the loop, and the
    # 3.2 GB it saves costs 4.2 ms.
    tl.store(
        activated_ptr + offsets,
        _gelu(preactivation.to(tl.float32)).to(tl.bfloat16),
        mask=valid[:, None],
    )


@triton.autotune(configs=_configs(), key=["rows"])
@triton.jit
def _project_hidden(
    activated_ptr, w2_ptr, b2_ptr, hidden_ptr, activated_hidden_ptr,
    rows, WIDTH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """``gelu(preactivation) @ W2 + b2``, then GELU."""
    row_block = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row_block < rows
    columns = tl.arange(0, WIDTH)
    accumulator = _row_gemm(
        activated_ptr, w2_ptr, row_block, valid, columns,
        WIDTH=WIDTH, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, CONTRACT_OUT=False,
    )

    offsets = row_block[:, None] * WIDTH + columns[None, :]
    hidden = (accumulator + tl.load(b2_ptr + columns)[None, :]).to(tl.bfloat16)
    tl.store(hidden_ptr + offsets, hidden, mask=valid[:, None])
    tl.store(
        activated_hidden_ptr + offsets,
        _gelu(hidden.to(tl.float32)).to(tl.bfloat16),
        mask=valid[:, None],
    )


@triton.autotune(configs=_configs(), key=["rows", "DROPOUT"])
@triton.jit
def _project_output(
    activated_hidden_ptr, w3_ptr, b3_ptr, edge_ptr, gamma_ptr, beta_ptr, seed_ptr,
    out_ptr, values_ptr, keep_ptr,
    rows, keep_probability, dropout_scale, eps,
    WIDTH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    DROPOUT: tl.constexpr,
):
    """``gelu(hidden) @ W3 + b3``, then dropout, the residual add, and LayerNorm."""
    row_block = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row_block < rows
    columns = tl.arange(0, WIDTH)
    accumulator = _row_gemm(
        activated_hidden_ptr, w3_ptr, row_block, valid, columns,
        WIDTH=WIDTH, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, CONTRACT_OUT=False,
    )

    offsets = row_block[:, None] * WIDTH + columns[None, :]
    update = accumulator + tl.load(b3_ptr + columns)[None, :]
    if DROPOUT:
        keep = tl.rand(tl.load(seed_ptr), offsets, n_rounds=7) < keep_probability
        update = tl.where(keep, update * dropout_scale, 0.0)
        tl.store(keep_ptr + offsets, keep.to(tl.int8), mask=valid[:, None])
    edge = tl.load(edge_ptr + offsets, mask=valid[:, None], other=0.0).to(tl.float32)
    values = (edge + update).to(tl.bfloat16)
    tl.store(values_ptr + offsets, values, mask=valid[:, None])

    # The tile spans the whole width, so LayerNorm's row reduction lands in the
    # epilogue rather than in two extra passes over the tensor.
    values_f32 = values.to(tl.float32)
    mean = tl.sum(values_f32, axis=1) / WIDTH
    centered = values_f32 - mean[:, None]
    rstd = 1.0 / tl.sqrt(tl.sum(centered * centered, axis=1) / WIDTH + eps)
    normalized = centered * rstd[:, None]
    tl.store(
        out_ptr + offsets,
        (normalized * tl.load(gamma_ptr + columns)[None, :]
         + tl.load(beta_ptr + columns)[None, :]).to(tl.bfloat16),
        mask=valid[:, None],
    )


# ---- backward ------------------------------------------------------------------------
@triton.autotune(
    configs=_elementwise_configs(),
    key=["rows", "DROPOUT"],
    reset_to_zero=[
        "grad_norm_weight_ptr", "grad_norm_bias_ptr", "grad_output_bias_ptr"
    ],
)
@triton.jit
def _norm_backward(
    grad_out_ptr, values_ptr, keep_ptr, gamma_ptr,
    grad_values_ptr, grad_update_ptr,
    grad_norm_weight_ptr, grad_norm_bias_ptr, grad_output_bias_ptr,
    rows, dropout_scale, eps,
    WIDTH: tl.constexpr, BLOCK_M: tl.constexpr, DROPOUT: tl.constexpr,
):
    """LayerNorm and dropout backward, statistics recomputed from the saved values.

    Two row reductions to avoid saving mean and rstd; both are cheaper than the
    bandwidth another pair of tensors would cost.  The three parameter gradients each
    get their own accumulator -- see the note in :func:`_launch_backward` for why they
    cannot share one.
    """
    row_block = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row_block < rows
    columns = tl.arange(0, WIDTH)
    offsets = row_block[:, None] * WIDTH + columns[None, :]

    values = tl.load(values_ptr + offsets, mask=valid[:, None], other=0.0).to(
        tl.float32
    )
    mean = tl.sum(values, axis=1) / WIDTH
    centered = values - mean[:, None]
    rstd = 1.0 / tl.sqrt(tl.sum(centered * centered, axis=1) / WIDTH + eps)
    normalized = centered * rstd[:, None]

    grad_out = tl.load(grad_out_ptr + offsets, mask=valid[:, None], other=0.0).to(
        tl.float32
    )
    scaled = grad_out * tl.load(gamma_ptr + columns)[None, :]
    grad_values = rstd[:, None] * (
        scaled
        - (tl.sum(scaled, axis=1) / WIDTH)[:, None]
        - normalized * (tl.sum(scaled * normalized, axis=1) / WIDTH)[:, None]
    )
    if DROPOUT:
        keep = tl.load(keep_ptr + offsets, mask=valid[:, None], other=0).to(tl.int1)
        grad_update = tl.where(keep, grad_values * dropout_scale, 0.0)
    else:
        grad_update = grad_values

    tl.store(
        grad_values_ptr + offsets, grad_values.to(tl.bfloat16), mask=valid[:, None]
    )
    tl.store(
        grad_update_ptr + offsets, grad_update.to(tl.bfloat16), mask=valid[:, None]
    )
    tl.atomic_add(
        grad_norm_weight_ptr + columns,
        tl.sum(tl.where(valid[:, None], grad_out * normalized, 0.0), axis=0),
    )
    tl.atomic_add(
        grad_norm_bias_ptr + columns,
        tl.sum(tl.where(valid[:, None], grad_out, 0.0), axis=0),
    )
    tl.atomic_add(
        grad_output_bias_ptr + columns,
        tl.sum(tl.where(valid[:, None], grad_update, 0.0), axis=0),
    )


@triton.autotune(
    configs=_configs(), key=["rows", "EMIT_BIAS"], reset_to_zero=["grad_bias_ptr"]
)
@triton.jit
def _project_backward(
    grad_out_ptr, weight_ptr, preactivation_ptr, grad_in_ptr, activated_ptr,
    grad_bias_ptr,
    rows, WIDTH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    EMIT_BIAS: tl.constexpr,
):
    """One dX GEMM with the GELU derivative and the cuBLAS operand in the epilogue.

    Runs twice, once per inner projection.  ``preactivation_ptr`` is the GEMM output
    this call differentiates through -- ``hidden`` for the third projection, the first
    projection's ``preactivation`` for the second -- and the GELU and its derivative
    share their single ``erf`` over it.  ``activated_ptr`` receives that GELU because
    the weight gradient contracts against it on cuBLAS afterwards.

    ``EMIT_BIAS`` belongs to the call that PRODUCES a bias gradient -- the third
    projection's dX, which yields ``grad_b2`` -- not the second's.
    """
    row_block = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row_block < rows
    columns = tl.arange(0, WIDTH)
    accumulator = _row_gemm(
        grad_out_ptr, weight_ptr, row_block, valid, columns,
        WIDTH=WIDTH, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, CONTRACT_OUT=True,
    )

    offsets = row_block[:, None] * WIDTH + columns[None, :]
    preactivation = tl.load(
        preactivation_ptr + offsets, mask=valid[:, None], other=0.0
    ).to(tl.float32)
    erf_term = _gelu_erf(preactivation)
    grad_in = accumulator * _gelu_grad_from_erf(preactivation, erf_term)
    tl.store(grad_in_ptr + offsets, grad_in.to(tl.bfloat16), mask=valid[:, None])
    tl.store(
        activated_ptr + offsets,
        _gelu_from_erf(preactivation, erf_term).to(tl.bfloat16),
        mask=valid[:, None],
    )
    if EMIT_BIAS:
        tl.atomic_add(
            grad_bias_ptr + columns,
            tl.sum(tl.where(valid[:, None], grad_in, 0.0), axis=0),
        )


@triton.autotune(configs=_configs(), key=["rows"], reset_to_zero=["grad_query_ptr"])
@triton.jit
def _edge_backward(
    grad_preactivation_ptr, w1_ptr, grad_values_ptr,
    grad_edge_ptr, grad_query_ptr,
    rows, groups_total,
    NEIGHBORS: tl.constexpr, WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """The last dX GEMM, the residual add, and the query gradient.

    The neighbour gradient is deliberately NOT here.  A tile holds one query's k
    nearest and those are distinct by construction, so an in-tile scatter has nothing
    to coalesce and degenerates to one FP32 atomic per element -- measured at 16 ms of
    this pass's 21, against 7.9 ms for the sort-and-segment reduction ATen already has
    for exactly this shape.  It runs outside as ``embedding_dense_backward``.

    The query gradient stays, because it is the opposite case: it is broadcast over the
    neighbour axis, so the tile reduces it to a handful of atomics before touching
    memory.  Same data, seven times cheaper.
    """
    row_block = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = row_block < rows
    columns = tl.arange(0, WIDTH)
    offsets = row_block[:, None] * WIDTH + columns[None, :]
    accumulator = _row_gemm(
        grad_preactivation_ptr, w1_ptr, row_block, valid, columns,
        WIDTH=WIDTH, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, CONTRACT_OUT=True,
    )

    residual = tl.load(
        grad_values_ptr + offsets, mask=valid[:, None], other=0.0
    ).to(tl.float32)
    tl.store(
        grad_edge_ptr + offsets, (residual + accumulator).to(tl.bfloat16),
        mask=valid[:, None],
    )

    # Re-read rather than reuse the contraction's operand: the loop slices it into
    # BLOCK_K chunks and the scatter below wants the whole row.
    grad_preactivation = tl.load(
        grad_preactivation_ptr + offsets, mask=valid[:, None], other=0.0
    ).to(tl.float32)
    groups = row_block // NEIGHBORS
    first_group = (tl.program_id(0) * BLOCK_M) // NEIGHBORS
    for span in tl.static_range((BLOCK_M + NEIGHBORS - 1) // NEIGHBORS + 1):
        group = first_group + span
        selected = valid & (groups == group)
        tl.atomic_add(
            grad_query_ptr + group * WIDTH + columns,
            tl.sum(tl.where(selected[:, None], grad_preactivation, 0.0), axis=0),
            mask=group < groups_total,
        )


# ---- launchers -----------------------------------------------------------------------
def _neighbors_of(rows: int, nodes: int) -> int:
    if rows % nodes:
        raise ValueError(f"{rows} edge rows do not divide into {nodes} nodes")
    return rows // nodes


def _grid(rows):
    return lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)


def _launch_forward(edge, query, table, index, w1, w2, b2, w3, b3, gamma, beta,
                    seed, eps, dropout_probability):
    """Returns ``(out, saved)``, where ``saved`` is exactly what backward reads back."""
    rows = edge.shape[0]
    dropout = dropout_probability > 0.0
    empty = lambda: torch.empty_like(edge)
    out, preactivation, activated = empty(), empty(), empty()
    hidden, activated_hidden, values = empty(), empty(), empty()
    # One byte per element rather than a redraw: the Philox draw measured 12% of a
    # comparable kernel and a byte read is well under that.  With dropout off nothing
    # touches it -- both accesses sit behind the DROPOUT constexpr -- so it shrinks to a
    # placeholder instead of 0.4 GiB of untouched memory at the sweep point.
    keep = torch.empty(
        rows * WIDTH if dropout else 1, device=edge.device, dtype=torch.int8
    )

    grid = _grid(rows)
    _project_edge[grid](
        edge, query, table, index, w1, preactivation, activated,
        rows, NEIGHBORS=_neighbors_of(rows, query.shape[0]), WIDTH=WIDTH,
    )
    _project_hidden[grid](
        activated, w2, b2, hidden, activated_hidden, rows, WIDTH=WIDTH,
    )
    _project_output[grid](
        activated_hidden, w3, b3, edge, gamma, beta, seed,
        out, values, keep, rows,
        1.0 - dropout_probability,
        1.0 / (1.0 - dropout_probability) if dropout else 1.0, eps,
        WIDTH=WIDTH, DROPOUT=dropout,
    )
    # activated and activated_hidden are not saved: backward recomputes both GELUs from
    # the GEMM outputs above, which is the cheap direction and drops 1.6 GiB of live
    # tensors across the step.
    return out, (preactivation, hidden, values, keep)


def _launch_backward(grad_out, saved, edge, index, w1, w2, w3, gamma, nodes,
                     eps, dropout_probability):
    preactivation, hidden, values, keep = saved
    rows = edge.shape[0]
    dropout = dropout_probability > 0.0
    scale = 1.0 / (1.0 - dropout_probability) if dropout else 1.0
    empty = lambda: torch.empty_like(edge)
    zeros = lambda n: torch.zeros(n, device=edge.device, dtype=torch.float32)

    # BF16, not FP32: this is written by the norm pass and read by the edge pass, and
    # its value is rounded into a BF16 grad_edge one instruction later.  5.6 ms cheaper.
    grad_values = torch.empty(rows, WIDTH, device=edge.device, dtype=torch.bfloat16)
    grad_update, grad_hidden, grad_preactivation = empty(), empty(), empty()
    activated, activated_hidden, grad_edge = empty(), empty(), empty()
    # One accumulator per LAUNCH -- not per kernel, and not per gradient.  reset_to_zero
    # clears whatever tensor it is handed, and Triton applies it whenever that launch
    # AUTOTUNES, so any launch sharing a buffer with an earlier one wipes what the
    # earlier one accumulated.  This zeroed both LayerNorm gradients once, and then
    # zeroed grad_b2 the moment EMIT_BIAS entered the autotune key: the second dX launch
    # stopped hitting the first one's cache entry, started tuning, and reset the buffer
    # the first had just filled.  It emits no bias gradient, so it gets its own scratch
    # and the sharing cannot come back.
    grad_norm_weight, grad_norm_bias = zeros(WIDTH), zeros(WIDTH)
    grad_output_bias, grad_hidden_bias = zeros(WIDTH), zeros(WIDTH)
    grad_bias_scratch = zeros(WIDTH)
    grad_query = zeros(nodes * WIDTH).view(nodes, WIDTH)

    grid = _grid(rows)
    _norm_backward[grid](
        grad_out, values, keep, gamma, grad_values, grad_update,
        grad_norm_weight, grad_norm_bias, grad_output_bias, rows, scale, eps,
        WIDTH=WIDTH, DROPOUT=dropout,
    )
    _project_backward[grid](
        grad_update, w3, hidden, grad_hidden, activated_hidden,
        grad_hidden_bias, rows, WIDTH=WIDTH, EMIT_BIAS=True,
    )
    _project_backward[grid](
        grad_hidden, w2, preactivation, grad_preactivation, activated,
        grad_bias_scratch, rows, WIDTH=WIDTH, EMIT_BIAS=False,
    )
    _edge_backward[grid](
        grad_preactivation, w1, grad_values, grad_edge, grad_query, rows, nodes,
        NEIGHBORS=_neighbors_of(rows, nodes), WIDTH=WIDTH,
    )
    # ATen's sort-and-segment reduction rather than an in-kernel scatter: a row tile
    # holds one query's k nearest and those are distinct by construction, so there is
    # nothing to coalesce and it degenerates to one FP32 atomic per element -- measured
    # at 16 ms against 3.3 for this call.
    grad_neighbor = torch.ops.aten.embedding_dense_backward(
        grad_preactivation, index, nodes, -1, False
    )
    # Weight gradients on cuBLAS: a direct sweep put the best reachable Triton
    # reduction at 0.230 ms against cuBLAS's 0.197 for the same contraction.
    return dict(
        grad_edge=grad_edge,
        grad_query=grad_query,
        grad_neighbor=grad_neighbor,
        grad_w1=grad_preactivation.t() @ edge,
        grad_w2=grad_hidden.t() @ activated,
        grad_w3=grad_update.t() @ activated_hidden,
        grad_b2=grad_hidden_bias,
        grad_b3=grad_output_bias,
        grad_gamma=grad_norm_weight,
        grad_beta=grad_norm_bias,
    )


class _EdgeTailCompute(torch.autograd.Function):
    """Autograd boundary that keeps forward visible to the compiler.

    A ``torch.library.custom_op`` would be opaque to Inductor, which costs more than the
    kernels save; ``autograd.Function`` lets Dynamo trace the forward into the graph
    while the backward stays ours.
    """

    @staticmethod
    def forward(ctx, edge, query, table, index, w1, w2, b2, w3, b3, gamma, beta,
                seed, eps, dropout_probability):
        out, saved = _launch_forward(
            edge, query, table, index, w1, w2, b2, w3, b3, gamma, beta, seed, eps,
            dropout_probability,
        )
        ctx.save_for_backward(edge, index, w1, w2, w3, gamma, *saved)
        ctx.nodes = query.shape[0]
        ctx.eps = eps
        ctx.dropout_probability = dropout_probability
        ctx.dtypes = (w1.dtype, b2.dtype, gamma.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        edge, index, w1, w2, w3, gamma, *saved = ctx.saved_tensors
        weight_dtype, bias_dtype, norm_dtype = ctx.dtypes
        grads = _launch_backward(
            grad_out.contiguous(), saved, edge, index, w1, w2, w3, gamma,
            ctx.nodes, ctx.eps, ctx.dropout_probability,
        )
        return (
            grads["grad_edge"],
            grads["grad_query"].to(edge.dtype),
            grads["grad_neighbor"].to(edge.dtype),
            None,
            grads["grad_w1"].to(weight_dtype),
            grads["grad_w2"].to(weight_dtype),
            grads["grad_b2"].to(bias_dtype),
            grads["grad_w3"].to(weight_dtype),
            grads["grad_b3"].to(bias_dtype),
            grads["grad_gamma"].to(norm_dtype),
            grads["grad_beta"].to(norm_dtype),
            None, None, None,
        )


def edge_tail_compute(
    edge_states: torch.Tensor,
    query_projection: torch.Tensor,
    neighbor_projection: torch.Tensor,
    flat_neighbor_indices: torch.Tensor,
    edge_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float = 1e-5,
    dropout_probability: float = 0.0,
    seed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the encoder edge tail on flattened ``[rows, 128]`` operands."""
    if seed is None:
        seed = torch.zeros(1, device=edge_states.device, dtype=torch.int64)
    return _EdgeTailCompute.apply(
        edge_states, query_projection, neighbor_projection, flat_neighbor_indices,
        edge_weight, hidden_weight, hidden_bias, output_weight, output_bias,
        norm_weight, norm_bias, seed, eps, dropout_probability,
    )


__all__ = ["edge_tail_compute"]

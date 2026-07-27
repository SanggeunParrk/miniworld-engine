"""One-launch ProteinMPNN encoder edge tail.

The encoder's edge update is a chain of edge-sized operations::

    preactivation = query + edge @ W1e^T + neighbor[index]
    update        = W3(gelu(W2(gelu(preactivation))))
    edge_out      = layer_norm(edge + dropout(update))

Run as separate operations each link writes a ``[B, T, K, 128]`` tensor to HBM, and
backward has to keep or re-derive several of them.  The allocator trace at
``B=16, T=8192`` attributed seven live 1536 MiB blocks to exactly this chain.

This kernel keeps every link in registers.  Forward reads one edge tensor and
writes one edge tensor, and its autograd boundary saves no edge-sized tensor other
than the output the next layer needs anyway.  Backward replays the whole chain from
that same input, so none of the three GEMM activations is ever materialized.

Two consequences are deliberate and each has a dedicated test:

* The output is BF16, not the FP32 that autocast's ``layer_norm`` promotion returns.
  The value is a LayerNorm output feeding an autocast ``F.linear`` that casts it
  straight back to BF16, and the previous memory policy already stored a BF16 copy
  of the same quantity for backward.
* Dropout draws from Triton's Philox stream keyed by an explicit seed tensor instead
  of ``aten::native_dropout``'s stream, because backward has to reproduce the mask
  without storing it.  The keep probability and the ``1 / (1 - p)`` scale are
  identical; only the particular draw differs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_kernels.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of


_WIDTH = 128
# Backward stages its per-row operands so the weight-gradient pass can reduce them
# without keeping anything edge-sized alive.  One chunk of 262144 rows is 64 MiB per
# buffer, so the eight of them are a fixed 576 MiB regardless of batch size.
_WEIGHT_CHUNK_ROWS = 262_144
# Philox rounds for the dropout draw.  Triton defaults to ten; seven is the documented
# minimum that clears BigCrush, and the mask only has to be a good Bernoulli draw.
# Forward and backward MUST use the same value: they do not store the mask, they redraw
# it, so a mismatch would leave forward plausible and every gradient wrong.  A measured
# A5000 sweep put the draw at 43% of the one-GEMM normalization kernel (1.316 ms against
# 0.747 ms with dropout off), and the cost is register pressure -- four live uint32 of
# Philox state per element -- rather than the arithmetic itself.
# ``tl.constexpr`` rather than a plain int: a Triton kernel cannot read a non-constexpr
# module global.
_PHILOX_ROUNDS = tl.constexpr(7)


# GELU and its derivative share their only transcendental: the derivative's cdf term is
# exactly the factor GELU multiplies by.  The dX pass needs both at the same point, so it
# calls ``_gelu_erf`` once and feeds the two ``_from_erf`` forms.
#
# Only the ``tl.erf`` call is hoisted -- the arithmetic around it is character for
# character what it was, in the same order.  Sharing the *cdf* instead would have
# reassociated ``0.5 * x * (1 + e)`` into ``x * (0.5 * (1 + e))``, and while both scalings
# by a half are exact for normal inputs, "should round the same" is not a claim worth
# making when the cheaper hoist gives the same saving with none of the doubt.
#
# ``_gelu`` and ``_gelu_grad`` are defined in terms of the same pieces rather than
# duplicating the expression, so the shared and unshared paths cannot drift apart.
@triton.jit
def _gelu_erf(x):
    return tl.erf(x * 0.7071067811865476)


@triton.jit
def _gelu_from_erf(x, erf_term):
    return 0.5 * x * (1.0 + erf_term)


@triton.jit
def _gelu_grad_from_erf(x, erf_term):
    cdf = 0.5 * (1.0 + erf_term)
    pdf_term = x * 0.3989422804014327 * tl.exp(-0.5 * x * x)
    return cdf + pdf_term


@triton.jit
def _gelu(x):
    return _gelu_from_erf(x, _gelu_erf(x))


@triton.jit
def _gelu_grad(x):
    return _gelu_grad_from_erf(x, _gelu_erf(x))


def _configs(
    block_sizes: tuple[int, ...],
    tile_counts: tuple[int, ...] = (1, 2, 4),
    warp_counts: tuple[int, ...] = (4, 8, 16),
    stage_counts: tuple[int, ...] = (1, 2, 3),
    contraction_sizes: tuple[int, ...] | None = None,
) -> list[triton.Config]:
    """Build a full grid over every tuning knob.

    ``contraction_sizes`` is opt-in: a kernel that contracts the full width in one dot
    takes no ``BLOCK_K`` argument, and Triton rejects a configuration carrying a key the
    kernel does not declare.  Only the kernels that loop over the contraction ask for it.

    No knob is ever pinned to a single value.  Three separate times in this file's
    history a knob was fixed on the strength of one measurement, and all three times it
    hid the winner from the autotuner, which can only choose from the list it is given:

    * a ``BLOCK_M * TILES >= 64`` floor hid the backward winner -- 9.73 against 19.21 ms
    * ``triton.Config``'s default of three stages hid the weight-gradient winner --
      1.13 against 7.15 ms
    * ``num_stages=2`` hid the node message's winner, because at two stages every
      multi-group configuration fails to compile at all -- 0.658 against 0.932 ms

    Configurations that exceed shared memory or registers are skipped by the tuner, so
    offering them costs first-call compile time and nothing else.  That cost is what
    the repository's autotune cache exists for; see docs/operations/dispatch-cache.md.
    """
    return [
        triton.Config(
            {"BLOCK_M": block_m, "TILES": tiles}
            | ({} if block_k is None else {"BLOCK_K": block_k}),
            num_warps=warps,
            num_stages=stages,
        )
        for block_m in block_sizes
        for tiles in tile_counts
        for warps in warp_counts
        for stages in stage_counts
        for block_k in (contraction_sizes or (None,))
    ]


def _project_configs() -> list[triton.Config]:
    """Row-tile grid for the two-GEMM projection half of forward."""
    return _configs((32, 64, 128))


def _norm_configs() -> list[triton.Config]:
    """Row-tile and contraction grid for the dropout/residual/LayerNorm half.

    This is the one pass measured to be shared-memory bound rather than register
    bound: an A5000 run reported 65536 bytes of staging, which is one block per SM on
    a 100 KiB card, about 17% occupancy, and 153 GB/s on a 768 GB/s part.  A full-width
    contraction stages a 128x128 operand; splitting it into ``BLOCK_K`` slices stages
    only ``BLOCK_K x 128``, so 32 cuts the per-stage footprint about fourfold and buys
    occupancy directly.  128 is kept in the list and reproduces today's single dot
    exactly, so the tuner can always decline the loop.

    The kernel is the only consumer of this grid, so its ``BLOCK_K`` choice is free:
    nothing else has to reproduce its arithmetic bit for bit.
    """
    return _configs((32, 64, 128, 256), contraction_sizes=(32, 64, 128))


def _backward_configs() -> list[triton.Config]:
    """Row-tile grid for the two backward passes.

    Each holds three weight tiles in a single orientation, so neither needs a
    ``tl.trans``.  Three weights and no accumulator does not spill badly, which is why
    these two passes are not split further: halving each into a two-weight and a
    one-weight pass measured slower end to end -- 665.06 against 604.50 ms at B=8 --
    and raised reserved memory from 20736 to 28608 MiB at B=16, past what the policy
    exists to fit.
    """
    return _configs((16, 32, 64, 128))


# Narrow the offered grid to the top-K this GPU already measured, exactly as every other
# kernel family in the package does.  Two reasons this pass needs it more than most:
#
#  * the grids are large on purpose -- 324 for the norm pass alone -- because pinning a
#    knob has hidden the winner three times here.  Paying 324 compiles on every first
#    call is the price of that, and the cache is where the price is meant to be paid.
#  * an sm_86 card cannot launch a good fraction of these configurations at all (the
#    node message sweep had two thirds of them fail on shared memory).  A committed
#    cache skips them instead of compiling each one to find out.
#
# The cache NARROWS, it does not pin: Triton still autotunes among the cached top-K, a
# stale cache is detected by config_space_hash and falls back to the full grid, and
# MINIWORLD_RUN_AUTOTUNE=1 ignores it entirely.  Config choice is performance-only --
# every candidate computes the same arithmetic -- so a missing or wrong cache can only
# ever be slower, never incorrect.
#
# Buckets deliberately exclude ``rows``: the row count is what BLOCK_M/TILES exist to
# absorb, and bucketing on it would mean a separate cache entry per batch size.
_project_prune = make_cache_prune(
    "mpnn_edge_tail_project",
    dtype_of=tensor_dtype_of("edge_ptr"),
    bucket_of=key_bucket_of("WIDTH", "NEIGHBORS"),
)
_norm_prune = make_cache_prune(
    "mpnn_edge_tail_norm",
    dtype_of=tensor_dtype_of("edge_ptr"),
    bucket_of=key_bucket_of("WIDTH", "DROPOUT"),
)
_replay_prune = make_cache_prune(
    "mpnn_edge_tail_replay",
    dtype_of=tensor_dtype_of("edge_ptr"),
    bucket_of=key_bucket_of("WIDTH", "NEIGHBORS", "DROPOUT"),
)
_dx_prune = make_cache_prune(
    "mpnn_edge_tail_dx",
    dtype_of=tensor_dtype_of("preactivation_ptr"),
    bucket_of=key_bucket_of("WIDTH", "NEIGHBORS"),
)


@triton.autotune(
    configs=_project_configs(),
    key=["rows", "NEIGHBORS"],
    prune_configs_by={"early_config_prune": _project_prune},
)
@triton.jit
def _edge_tail_project_kernel(
    edge_ptr,
    query_ptr,
    neighbor_ptr,
    index_ptr,
    edge_weight_ptr,
    hidden_weight_ptr,
    hidden_bias_ptr,
    hidden_ptr,
    rows,
    EDGE_WEIGHT_STRIDE: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILES: tl.constexpr,
):
    """Packed W1 edge block, first GELU, and the hidden projection.

    Forward's first half. Backward does not reuse it: its replay pass needs the
    preactivation as well, and folding both GEMMs plus the output projection into one
    replay measured faster than splitting the replay to share this kernel.
    """
    columns = tl.arange(0, WIDTH)
    # nn.Linear stores [output, input] and tl.dot contracts [input, output], so each
    # weight is loaded once in the contraction orientation.
    edge_weight = tl.load(
        edge_weight_ptr + columns[:, None] + columns[None, :] * EDGE_WEIGHT_STRIDE
    ).to(tl.bfloat16)
    hidden_weight = tl.load(
        hidden_weight_ptr + columns[:, None] + columns[None, :] * WIDTH
    ).to(tl.bfloat16)
    hidden_bias = tl.load(hidden_bias_ptr + columns).to(tl.bfloat16)

    first_tile = tl.program_id(0) * TILES
    for tile in range(TILES):
        row_indices = (first_tile + tile) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_valid = row_indices < rows
        offsets = row_indices[:, None] * WIDTH + columns[None, :]
        edge = tl.load(edge_ptr + offsets, mask=row_valid[:, None], other=0.0).to(
            tl.bfloat16
        )
        groups = row_indices // NEIGHBORS
        query = tl.load(
            query_ptr + groups[:, None] * WIDTH + columns[None, :],
            mask=row_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        neighbor_rows = tl.load(index_ptr + row_indices, mask=row_valid, other=0)
        neighbor = tl.load(
            neighbor_ptr + neighbor_rows[:, None] * WIDTH + columns[None, :],
            mask=row_valid[:, None],
            other=0.0,
        ).to(tl.float32)

        # Reproduce the packed projection's addition order: each block is its own
        # autocast F.linear, so each partial sum is rounded to BF16.
        projected = tl.dot(edge, edge_weight).to(tl.bfloat16).to(tl.float32)
        preactivation = (query + projected).to(tl.bfloat16).to(tl.float32)
        preactivation = (preactivation + neighbor).to(tl.bfloat16)
        activated = _gelu(preactivation.to(tl.float32)).to(tl.bfloat16)
        hidden = (tl.dot(activated, hidden_weight) + hidden_bias[None, :]).to(
            tl.bfloat16
        )
        tl.store(hidden_ptr + offsets, hidden, mask=row_valid[:, None])


@triton.autotune(
    configs=_norm_configs(),
    key=["rows", "DROPOUT"],
    prune_configs_by={"early_config_prune": _norm_prune},
)
@triton.jit
def _edge_tail_norm_kernel(
    hidden_ptr,
    edge_ptr,
    output_weight_ptr,
    output_bias_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    out_ptr,
    seed_ptr,
    rows,
    keep_probability,
    dropout_scale,
    eps,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DROPOUT: tl.constexpr,
):
    """Second GELU, output projection, dropout, residual add and LayerNorm.

    The projection contracts in ``BLOCK_K`` slices rather than one full-width dot.
    That is an occupancy decision, not an arithmetic one: staging a whole 128x128
    operand put this kernel at one block per SM.
    """
    columns = tl.arange(0, WIDTH)
    output_bias = tl.load(output_bias_ptr + columns).to(tl.bfloat16)
    norm_weight = tl.load(norm_weight_ptr + columns).to(tl.float32)
    norm_bias = tl.load(norm_bias_ptr + columns).to(tl.float32)
    seed = tl.load(seed_ptr)

    first_tile = tl.program_id(0) * TILES
    for tile in range(TILES):
        row_indices = (first_tile + tile) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_valid = row_indices < rows
        offsets = row_indices[:, None] * WIDTH + columns[None, :]
        edge = tl.load(edge_ptr + offsets, mask=row_valid[:, None], other=0.0).to(
            tl.bfloat16
        )
        # Contract in BLOCK_K slices, applying the second GELU per slice as it is
        # loaded.  At BLOCK_K == WIDTH this is one full-width dot and reproduces the
        # unlooped kernel exactly; below that it stages a smaller operand and buys
        # occupancy.  The accumulation is FP32 throughout either way, so the split does
        # not change the result the way a BF16 partial sum would.
        accumulator = tl.zeros((BLOCK_M, WIDTH), tl.float32)
        for contraction in range(0, WIDTH, BLOCK_K):
            slice_columns = contraction + tl.arange(0, BLOCK_K)
            hidden_slice = tl.load(
                hidden_ptr + row_indices[:, None] * WIDTH + slice_columns[None, :],
                mask=row_valid[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.dot(
                _gelu(hidden_slice).to(tl.bfloat16),
                tl.load(
                    output_weight_ptr
                    + slice_columns[:, None]
                    + columns[None, :] * WIDTH
                ).to(tl.bfloat16),
            )
        update = (accumulator + output_bias[None, :]).to(tl.bfloat16)
        if DROPOUT:
            keep = tl.rand(seed, offsets, n_rounds=_PHILOX_ROUNDS) < keep_probability
            dropped = tl.where(keep, update.to(tl.float32) * dropout_scale, 0.0).to(
                tl.bfloat16
            )
        else:
            dropped = update
        values = (edge.to(tl.float32) + dropped.to(tl.float32)).to(tl.bfloat16)
        values = values.to(tl.float32)
        mean = tl.sum(values, axis=1) / WIDTH
        centered = values - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / WIDTH
        rstd = 1.0 / tl.sqrt(variance + eps)
        tl.store(
            out_ptr + offsets,
            centered * rstd[:, None] * norm_weight[None, :] + norm_bias[None, :],
            mask=row_valid[:, None],
        )


# The three weight gradients are ``[128, rows] x [rows, 128]`` reductions.  They stay
# on cuBLAS: a direct sweep of a Triton replacement on a 262144-row chunk found its
# time almost exactly linear in program count, because each program ends in a
# ``tl.atomic_add`` of a 128x128 FP32 block -- 16384 CTAs took 3.49 ms, 1024 took
# 0.385 ms, and the best configuration reachable at all (128 CTAs, 16 warps) was
# 0.230 ms against cuBLAS's 0.197 ms at 43.6 TFLOP/s and 681 GB/s.  An earlier note
# here claimed the opposite; it came from a profile bucket that had mixed several GEMM
# shapes together, and the direct measurement overturned it.


@triton.autotune(
    configs=_backward_configs(),
    key=["rows", "NEIGHBORS", "DROPOUT"],
    reset_to_zero=[
        "grad_output_bias_ptr",
        "grad_norm_weight_ptr",
        "grad_norm_bias_ptr",
    ],
    prune_configs_by={"early_config_prune": _replay_prune},
)
@triton.jit
def _edge_tail_replay_kernel(
    grad_out_ptr,
    edge_ptr,
    query_ptr,
    neighbor_ptr,
    index_ptr,
    edge_weight_ptr,
    hidden_weight_ptr,
    hidden_bias_ptr,
    output_weight_ptr,
    output_bias_ptr,
    norm_weight_ptr,
    preactivation_ptr,
    hidden_ptr,
    grad_update_ptr,
    grad_values_ptr,
    grad_output_bias_ptr,
    grad_norm_weight_ptr,
    grad_norm_bias_ptr,
    seed_ptr,
    rows,
    row_offset,
    chunk_rows,
    keep_probability,
    dropout_scale,
    eps,
    EDGE_WEIGHT_STRIDE: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILES: tl.constexpr,
    DROPOUT: tl.constexpr,
):
    """Replay the forward chain, then differentiate LayerNorm, dropout and residual.

    Holds the three weights in their contraction orientation and nothing else: no
    gradient accumulator and no transpose.  Adding one 128x128 FP32 accumulator on top
    measured 7.45 ms per 262144-row chunk against 2.09 ms without it.
    """
    columns = tl.arange(0, WIDTH)
    edge_weight = tl.load(
        edge_weight_ptr + columns[:, None] + columns[None, :] * EDGE_WEIGHT_STRIDE
    ).to(tl.bfloat16)
    hidden_weight = tl.load(
        hidden_weight_ptr + columns[:, None] + columns[None, :] * WIDTH
    ).to(tl.bfloat16)
    output_weight = tl.load(
        output_weight_ptr + columns[:, None] + columns[None, :] * WIDTH
    ).to(tl.bfloat16)
    hidden_bias = tl.load(hidden_bias_ptr + columns).to(tl.bfloat16)
    output_bias = tl.load(output_bias_ptr + columns).to(tl.bfloat16)
    norm_weight = tl.load(norm_weight_ptr + columns).to(tl.float32)
    seed = tl.load(seed_ptr)

    grad_output_bias = tl.zeros((WIDTH,), tl.float32)
    grad_norm_weight = tl.zeros((WIDTH,), tl.float32)
    grad_norm_bias = tl.zeros((WIDTH,), tl.float32)

    first_tile = tl.program_id(0) * TILES
    for tile in range(TILES):
        local_rows = (first_tile + tile) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_indices = local_rows + row_offset
        row_valid = (local_rows < chunk_rows) & (row_indices < rows)
        offsets = row_indices[:, None] * WIDTH + columns[None, :]
        local_offsets = local_rows[:, None] * WIDTH + columns[None, :]

        # --- replay forward from the saved inputs -----------------------------
        edge = tl.load(edge_ptr + offsets, mask=row_valid[:, None], other=0.0).to(
            tl.bfloat16
        )
        groups = row_indices // NEIGHBORS
        query = tl.load(
            query_ptr + groups[:, None] * WIDTH + columns[None, :],
            mask=row_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        neighbor_rows = tl.load(index_ptr + row_indices, mask=row_valid, other=0)
        neighbor = tl.load(
            neighbor_ptr + neighbor_rows[:, None] * WIDTH + columns[None, :],
            mask=row_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        projected = tl.dot(edge, edge_weight).to(tl.bfloat16).to(tl.float32)
        preactivation = (query + projected).to(tl.bfloat16).to(tl.float32)
        preactivation = (preactivation + neighbor).to(tl.bfloat16)
        activated = _gelu(preactivation.to(tl.float32)).to(tl.bfloat16)
        hidden = (tl.dot(activated, hidden_weight) + hidden_bias[None, :]).to(
            tl.bfloat16
        )
        activated_hidden = _gelu(hidden.to(tl.float32)).to(tl.bfloat16)
        update = (tl.dot(activated_hidden, output_weight) + output_bias[None, :]).to(
            tl.bfloat16
        )
        if DROPOUT:
            keep = tl.rand(seed, offsets, n_rounds=_PHILOX_ROUNDS) < keep_probability
            dropped = tl.where(keep, update.to(tl.float32) * dropout_scale, 0.0).to(
                tl.bfloat16
            )
        else:
            dropped = update
        values = (edge.to(tl.float32) + dropped.to(tl.float32)).to(tl.bfloat16)
        values = values.to(tl.float32)
        mean = tl.sum(values, axis=1) / WIDTH
        centered = values - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / WIDTH
        rstd = 1.0 / tl.sqrt(variance + eps)
        normalized = centered * rstd[:, None]

        # --- LayerNorm, dropout, residual ------------------------------------
        grad_out = tl.load(
            grad_out_ptr + offsets, mask=row_valid[:, None], other=0.0
        ).to(tl.float32)
        grad_norm_weight += tl.sum(grad_out * normalized, axis=0)
        grad_norm_bias += tl.sum(grad_out, axis=0)
        scaled = grad_out * norm_weight[None, :]
        grad_values = rstd[:, None] * (
            scaled
            - (tl.sum(scaled, axis=1) / WIDTH)[:, None]
            - normalized * (tl.sum(scaled * normalized, axis=1) / WIDTH)[:, None]
        )
        if DROPOUT:
            grad_update = tl.where(keep, grad_values * dropout_scale, 0.0)
        else:
            grad_update = grad_values
        grad_output_bias += tl.sum(grad_update, axis=0)

        # The residual branch's gradient is dv itself.  It goes to an FP32 chunk buffer
        # rather than to the final gradient: the dX pass has to add the W1 block's
        # contribution, and a read-modify-write there would not be idempotent, so the
        # autotuner's repeated benchmark runs would accumulate into it.  A measured run
        # with that bug reported a relative error of 57 on this gradient alone.
        tl.store(grad_values_ptr + local_offsets, grad_values, mask=row_valid[:, None])
        tl.store(
            preactivation_ptr + local_offsets, preactivation, mask=row_valid[:, None]
        )
        tl.store(hidden_ptr + local_offsets, hidden, mask=row_valid[:, None])
        tl.store(
            grad_update_ptr + local_offsets,
            grad_update.to(tl.bfloat16),
            mask=row_valid[:, None],
        )

    tl.atomic_add(grad_output_bias_ptr + columns, grad_output_bias)
    tl.atomic_add(grad_norm_weight_ptr + columns, grad_norm_weight)
    tl.atomic_add(grad_norm_bias_ptr + columns, grad_norm_bias)


@triton.autotune(
    configs=_backward_configs(),
    key=["rows", "NEIGHBORS"],
    reset_to_zero=["grad_query_ptr", "grad_neighbor_ptr", "grad_hidden_bias_ptr"],
    prune_configs_by={"early_config_prune": _dx_prune},
)
@triton.jit
def _edge_tail_dx_kernel(
    preactivation_ptr,
    hidden_ptr,
    grad_update_ptr,
    grad_values_ptr,
    index_ptr,
    edge_weight_ptr,
    hidden_weight_ptr,
    output_weight_ptr,
    grad_edge_ptr,
    grad_preactivation_ptr,
    grad_hidden_ptr,
    activated_ptr,
    activated_hidden_ptr,
    grad_query_ptr,
    grad_neighbor_ptr,
    grad_hidden_bias_ptr,
    rows,
    groups_total,
    row_offset,
    chunk_rows,
    EDGE_WEIGHT_STRIDE: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILES: tl.constexpr,
):
    """Walk the chain backwards from ``grad_update`` to ``grad_edge``.

    Holds the same three weights in their row-major orientation, which is a plain load
    rather than a ``tl.trans``.  Avoiding the transpose is what keeps this inside the
    shared-memory budget: three transposes plus three contraction tiles measured
    278 KiB of staging against a 100 KiB limit on sm_86.
    """
    columns = tl.arange(0, WIDTH)
    edge_weight = tl.load(
        edge_weight_ptr + columns[:, None] * EDGE_WEIGHT_STRIDE + columns[None, :]
    ).to(tl.bfloat16)
    hidden_weight = tl.load(
        hidden_weight_ptr + columns[:, None] * WIDTH + columns[None, :]
    ).to(tl.bfloat16)
    output_weight = tl.load(
        output_weight_ptr + columns[:, None] * WIDTH + columns[None, :]
    ).to(tl.bfloat16)
    grad_hidden_bias = tl.zeros((WIDTH,), tl.float32)

    first_tile = tl.program_id(0) * TILES
    for tile in range(TILES):
        local_rows = (first_tile + tile) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_indices = local_rows + row_offset
        row_valid = (local_rows < chunk_rows) & (row_indices < rows)
        offsets = row_indices[:, None] * WIDTH + columns[None, :]
        local_offsets = local_rows[:, None] * WIDTH + columns[None, :]

        grad_update = tl.load(
            grad_update_ptr + local_offsets, mask=row_valid[:, None], other=0.0
        ).to(tl.bfloat16)
        hidden = tl.load(
            hidden_ptr + local_offsets, mask=row_valid[:, None], other=0.0
        ).to(tl.float32)
        # This pass wants both GELU and its derivative at ``hidden``, and again at
        # ``preactivation`` below -- four calls over two distinct arguments, so half the
        # erf evaluations were recomputation.  Hoisting the shared term is bit for bit
        # identical; see the note on the helpers.
        hidden_erf = _gelu_erf(hidden)
        grad_hidden = tl.dot(grad_update, output_weight) * _gelu_grad_from_erf(
            hidden, hidden_erf
        )
        grad_hidden_bias += tl.sum(grad_hidden, axis=0)
        grad_hidden_bf16 = grad_hidden.to(tl.bfloat16)
        # Both MLP weight gradients contract against GELU of a preactivation this pass
        # has already loaded for its own derivative.  Emitting them here costs two
        # stores; making cuBLAS's operands any other way costs a separate elementwise
        # pass over the same rows.  Each is stored the moment its argument's erf is in
        # hand rather than together at the end of the body: the pass is register bound,
        # and keeping two [BLOCK_M, WIDTH] erf blocks live at once is how a saving this
        # size gets paid straight back in spill.
        tl.store(
            activated_hidden_ptr + local_offsets,
            _gelu_from_erf(hidden, hidden_erf).to(tl.bfloat16),
            mask=row_valid[:, None],
        )

        preactivation = tl.load(
            preactivation_ptr + local_offsets, mask=row_valid[:, None], other=0.0
        ).to(tl.float32)
        preactivation_erf = _gelu_erf(preactivation)
        grad_preactivation = (
            tl.dot(grad_hidden_bf16, hidden_weight)
            * _gelu_grad_from_erf(preactivation, preactivation_erf)
        ).to(tl.bfloat16)
        tl.store(
            activated_ptr + local_offsets,
            _gelu_from_erf(preactivation, preactivation_erf).to(tl.bfloat16),
            mask=row_valid[:, None],
        )

        # The residual branch's gradient arrives through an FP32 chunk buffer, so this
        # is a plain overwrite: the pass stays idempotent and the two contributions
        # round to BF16 exactly once, here.
        residual = tl.load(
            grad_values_ptr + local_offsets, mask=row_valid[:, None], other=0.0
        ).to(tl.float32)
        tl.store(
            grad_edge_ptr + offsets,
            residual + tl.dot(grad_preactivation, edge_weight),
            mask=row_valid[:, None],
        )
        tl.store(
            grad_hidden_ptr + local_offsets,
            grad_hidden_bf16,
            mask=row_valid[:, None],
        )
        tl.store(
            grad_preactivation_ptr + local_offsets,
            grad_preactivation,
            mask=row_valid[:, None],
        )
        neighbor_rows = tl.load(index_ptr + row_indices, mask=row_valid, other=0)
        tl.atomic_add(
            grad_neighbor_ptr + neighbor_rows[:, None] * WIDTH + columns[None, :],
            grad_preactivation.to(tl.float32),
            mask=row_valid[:, None],
        )
        # The query block is broadcast over the neighbor axis, so its gradient is a
        # per-group row sum.  A row tile touches only a handful of groups; reducing
        # inside the tile turns one atomic per row into a few per tile.
        groups = row_indices // NEIGHBORS
        first_group = ((first_tile + tile) * BLOCK_M + row_offset) // NEIGHBORS
        for span in tl.static_range((BLOCK_M + NEIGHBORS - 1) // NEIGHBORS + 1):
            group = first_group + span
            selected = row_valid & (groups == group)
            tl.atomic_add(
                grad_query_ptr + group * WIDTH + columns,
                tl.sum(tl.where(selected[:, None], grad_preactivation, 0.0), axis=0),
                mask=(columns < WIDTH) & (group < groups_total),
            )

    tl.atomic_add(grad_hidden_bias_ptr + columns, grad_hidden_bias)


@torch.library.custom_op("miniworld_kernels::mpnn_edge_tail_fwd_v1", mutates_args=())
def _forward_op(
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
    seed: torch.Tensor,
    eps: float,
    dropout_probability: float,
) -> torch.Tensor:
    neighbors = edge_states.shape[-2]
    rows = edge_states.numel() // _WIDTH
    out = torch.empty_like(edge_states)
    dropout = dropout_probability > 0.0

    def grid(meta):
        return (triton.cdiv(triton.cdiv(rows, meta["BLOCK_M"]), meta["TILES"]),)

    # A full-size intermediate rather than a chunk buffer: the compiled peak is in
    # backward, so a transient that only exists during forward is free.  Backward does
    # chunk, which is why it carries chunk buffers instead.
    hidden = torch.empty_like(edge_states)
    _edge_tail_project_kernel[grid](
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        hidden,
        rows,
        EDGE_WEIGHT_STRIDE=edge_weight.stride(0),
        NEIGHBORS=neighbors,
        WIDTH=_WIDTH,
    )
    _edge_tail_norm_kernel[grid](
        hidden,
        edge_states,
        output_weight,
        output_bias,
        norm_weight,
        norm_bias,
        out,
        seed,
        rows,
        1.0 - dropout_probability,
        1.0 / (1.0 - dropout_probability) if dropout else 1.0,
        eps,
        WIDTH=_WIDTH,
        DROPOUT=dropout,
    )
    return out


@_forward_op.register_fake
def _(
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
    seed: torch.Tensor,
    eps: float,
    dropout_probability: float,
) -> torch.Tensor:
    del query_projection, neighbor_projection, flat_neighbor_indices
    del edge_weight, hidden_weight, hidden_bias, output_weight, output_bias
    del norm_weight, norm_bias, seed, eps, dropout_probability
    return torch.empty_like(edge_states)


@torch.library.custom_op("miniworld_kernels::mpnn_edge_tail_bwd_v1", mutates_args=())
def _backward_op(
    grad_out: torch.Tensor,
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
    seed: torch.Tensor,
    eps: float,
    dropout_probability: float,
) -> list[torch.Tensor]:
    """Three passes over one row chunk at a time.

    The passes are split by what has to stay resident, not by what is conceptually
    separate.  Each of the three 128x128 weights costs 32 registers per thread at
    eight warps in one orientation, and each 128x128 FP32 gradient accumulator costs
    64.  A single kernel holding both orientations and one accumulator measured 7.4 ms
    per 262144-row chunk against a 1.5 ms budget; split this way, no pass carries more
    than three weight tiles or three accumulators, and none needs a ``tl.trans`` of a
    weight.

    The eight chunk buffers are a fixed 576 MiB at 262144 rows, independent of batch
    size, and they replace what would otherwise be eight full edge tensors.
    """
    neighbors = edge_states.shape[-2]
    rows = edge_states.numel() // _WIDTH
    nodes = neighbor_projection.numel() // _WIDTH
    groups = query_projection.numel() // _WIDTH
    device = edge_states.device
    float32 = torch.float32

    grad_edge = torch.empty_like(edge_states)
    grad_query = torch.zeros(groups, _WIDTH, device=device, dtype=float32)
    grad_neighbor = torch.zeros(nodes, _WIDTH, device=device, dtype=float32)
    grad_edge_weight = torch.zeros(_WIDTH, _WIDTH, device=device, dtype=float32)
    grad_hidden_weight = torch.zeros(_WIDTH, _WIDTH, device=device, dtype=float32)
    grad_output_weight = torch.zeros(_WIDTH, _WIDTH, device=device, dtype=float32)
    grad_hidden_bias = torch.zeros(_WIDTH, device=device, dtype=float32)
    grad_output_bias = torch.zeros(_WIDTH, device=device, dtype=float32)
    grad_norm_weight = torch.zeros(_WIDTH, device=device, dtype=float32)
    grad_norm_bias = torch.zeros(_WIDTH, device=device, dtype=float32)

    chunk_rows = min(_WEIGHT_CHUNK_ROWS, rows)
    preactivation = torch.empty(chunk_rows, _WIDTH, device=device, dtype=torch.bfloat16)
    hidden = torch.empty_like(preactivation)
    chunk_grad_update = torch.empty_like(preactivation)
    chunk_grad_hidden = torch.empty_like(preactivation)
    chunk_grad_preactivation = torch.empty_like(preactivation)
    chunk_activated = torch.empty_like(preactivation)
    chunk_activated_hidden = torch.empty_like(preactivation)
    chunk_grad_values = torch.empty(chunk_rows, _WIDTH, device=device, dtype=float32)
    flat_edge = edge_states.reshape(rows, _WIDTH)

    dropout = dropout_probability > 0.0
    keep_probability = 1.0 - dropout_probability
    scale = 1.0 / keep_probability if dropout else 1.0

    def chunk_grid(span):
        return lambda meta: (
            triton.cdiv(triton.cdiv(span, meta["BLOCK_M"]), meta["TILES"]),
        )

    for start in range(0, rows, chunk_rows):
        span = min(chunk_rows, rows - start)
        _edge_tail_replay_kernel[chunk_grid(span)](
            grad_out,
            edge_states,
            query_projection,
            neighbor_projection,
            flat_neighbor_indices,
            edge_weight,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
            norm_weight,
            preactivation,
            hidden,
            chunk_grad_update,
            chunk_grad_values,
            grad_output_bias,
            grad_norm_weight,
            grad_norm_bias,
            seed,
            rows,
            start,
            span,
            keep_probability,
            scale,
            eps,
            EDGE_WEIGHT_STRIDE=edge_weight.stride(0),
            NEIGHBORS=neighbors,
            WIDTH=_WIDTH,
            DROPOUT=dropout,
        )
        _edge_tail_dx_kernel[chunk_grid(span)](
            preactivation,
            hidden,
            chunk_grad_update,
            chunk_grad_values,
            flat_neighbor_indices,
            edge_weight,
            hidden_weight,
            output_weight,
            grad_edge,
            chunk_grad_preactivation,
            chunk_grad_hidden,
            chunk_activated,
            chunk_activated_hidden,
            grad_query,
            grad_neighbor,
            grad_hidden_bias,
            rows,
            groups,
            start,
            span,
            EDGE_WEIGHT_STRIDE=edge_weight.stride(0),
            NEIGHBORS=neighbors,
            WIDTH=_WIDTH,
        )
        # Each weight gradient reduces over every row.  cuBLAS owns these: see the note
        # above the dX pass for the sweep that put a Triton replacement at 0.230 ms
        # against 0.197 ms here.  The FP32 running sum across chunks is better
        # conditioned than one BF16 GEMM output over all rows.
        grad_edge_weight += torch.mm(
            chunk_grad_preactivation[:span].t(), flat_edge[start : start + span]
        ).to(float32)
        grad_hidden_weight += torch.mm(
            chunk_grad_hidden[:span].t(), chunk_activated[:span]
        ).to(float32)
        grad_output_weight += torch.mm(
            chunk_grad_update[:span].t(), chunk_activated_hidden[:span]
        ).to(float32)

    return [
        grad_edge,
        grad_query.view_as(query_projection),
        grad_neighbor.view_as(neighbor_projection),
        grad_edge_weight,
        grad_hidden_weight,
        grad_output_weight,
        grad_hidden_bias,
        grad_output_bias,
        grad_norm_weight,
        grad_norm_bias,
    ]


@_backward_op.register_fake
def _(
    grad_out: torch.Tensor,
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
    seed: torch.Tensor,
    eps: float,
    dropout_probability: float,
) -> list[torch.Tensor]:
    del grad_out, flat_neighbor_indices, seed, eps, dropout_probability
    del hidden_bias, output_bias
    float32 = torch.float32
    return [
        torch.empty_like(edge_states),
        torch.empty_like(query_projection, dtype=float32),
        torch.empty_like(neighbor_projection, dtype=float32),
        edge_states.new_empty(_WIDTH, _WIDTH, dtype=float32),
        edge_states.new_empty(_WIDTH, _WIDTH, dtype=float32),
        edge_states.new_empty(_WIDTH, _WIDTH, dtype=float32),
        edge_states.new_empty(_WIDTH, dtype=float32),
        edge_states.new_empty(_WIDTH, dtype=float32),
        edge_states.new_empty(_WIDTH, dtype=float32),
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
        output_weight,
        output_bias,
        norm_weight,
        norm_bias,
        seed,
        eps,
        dropout_probability,
    ) = inputs
    ctx.save_for_backward(
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        norm_weight,
        seed,
    )
    ctx.eps = eps
    ctx.dropout_probability = dropout_probability
    ctx.dtypes = (
        query_projection.dtype,
        neighbor_projection.dtype,
        edge_weight.dtype,
        hidden_weight.dtype,
        hidden_bias.dtype,
        output_weight.dtype,
        output_bias.dtype,
        norm_weight.dtype,
        norm_bias.dtype,
    )


def _backward(ctx, grad_out):
    (
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        norm_weight,
        seed,
    ) = ctx.saved_tensors
    (
        query_dtype,
        neighbor_dtype,
        edge_weight_dtype,
        hidden_weight_dtype,
        hidden_bias_dtype,
        output_weight_dtype,
        output_bias_dtype,
        norm_weight_dtype,
        norm_bias_dtype,
    ) = ctx.dtypes
    (
        grad_edge,
        grad_query,
        grad_neighbor,
        grad_edge_weight,
        grad_hidden_weight,
        grad_output_weight,
        grad_hidden_bias,
        grad_output_bias,
        grad_norm_weight,
        grad_norm_bias,
    ) = _backward_op(
        grad_out.contiguous(),
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        norm_weight,
        seed,
        ctx.eps,
        ctx.dropout_probability,
    )
    # Autocast's Linear rounds a bias gradient to BF16 before the FP32 parameter
    # gradient; keep that boundary rather than returning the FP32 partial sum.
    return (
        grad_edge,
        grad_query.to(query_dtype),
        grad_neighbor.to(neighbor_dtype),
        None,
        grad_edge_weight.to(edge_weight_dtype),
        grad_hidden_weight.to(hidden_weight_dtype),
        grad_hidden_bias.to(torch.bfloat16).to(hidden_bias_dtype),
        grad_output_weight.to(output_weight_dtype),
        grad_output_bias.to(torch.bfloat16).to(output_bias_dtype),
        grad_norm_weight.to(norm_weight_dtype),
        grad_norm_bias.to(norm_bias_dtype),
        None,
        None,
        None,
    )


_forward_op.register_autograd(_backward, setup_context=_setup_context)


def triton_edge_tail_update(
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
    seed: torch.Tensor,
    eps: float,
    dropout_probability: float,
) -> torch.Tensor:
    """Run the whole encoder edge tail as one fused, fully replayed op."""
    return _forward_op(
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        norm_weight,
        norm_bias,
        seed,
        eps,
        dropout_probability,
    )


__all__ = ["triton_edge_tail_update"]

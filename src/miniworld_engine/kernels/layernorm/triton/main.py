
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/layernorm.py
from miniworld_engine.autotune.configs import configs_for
import torch
import triton
import triton.language as tl


from miniworld_engine.autotune.shape_key import both_key, length_of, rows_of, pack


def get_seq_group(length) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_mixed
    return bucket_mixed(length)


# (The hand-CUDA warp-per-row LN backward is NOT here: it lives in kernels/layernorm/cuda/masked_bwd.py and is chosen by the caller.)

# The two shipped kernels (`layer_norm_fwd_fused`, `layer_norm_bwd_dx_fused`) tile the FEATURE
# axis too. It used to arrive as BLOCK_N = triton.next_power_of_2(N) from every launch site — the
# whole row, a caller-computed constant the tuner never saw, and the reason BLOCK_M1 was the only
# thing being swept. It is the REDUCE axis (mean/var in the forward; the c1/c2 row sums and the
# dw/db column sums in the backward), so it is a CSV tile: N is
# d_hidden/d_pair at 128..1024 while BLOCK_N stops at 256, and a sweep that cannot express a whole
# row would turn "tiled" into "always two passes over X", which the tuner could not decline.
# still takes BLOCK_N from its caller.


#: Programs per SM in the persistent backward (`layer_norm_bwd_dx_fused`). 8 is within 1-5% of the
#: best of {4, 8, 16} at every shape measured (bf16/fp32 pair LayerNorm, D 128 and 256, L 384-768);
#: the launch is capped at the tile count.
_PERSIST_WAVES = 8


def _persistent_programs(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count * _PERSIST_WAVES


# fmt: off


# `eps` is constexpr but deliberately NOT keyed: it only appears in `1 / sqrt(var + eps)`, so it
# branches nothing and shifts no work -- keying it would just multiply the bucket count.
@triton.autotune(configs=configs_for("layernorm_fwd_saveact_triton"),
                 key=['shape_key', 'HAS_ROWSCALE'])
@triton.jit
def layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd, Rowscale,
    stride_r, stride_c,
    M, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_ROWSCALE: tl.constexpr, STORE_STATS: tl.constexpr = True,
):
    # STORE_STATS=False (inference: nothing will run a backward) drops the two (M,) fp32 stores. At
    # x[1, L, L, 128] bf16 that is 8 of every 520 bytes moved per row, 1.5% of a kernel that sits
    # on the HBM floor. Not part of the autotune key: it changes no tile choice.
    # Map the program id to the rows of X and Y it should compute.
    row = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    row_mask = rows < M

    # Keep a covering row tile in registers. For smaller feature tiles, combine
    # centred tile moments with Welford's formula, then re-read X to normalize.
    # E[x*x] - E[x]*E[x] loses variance on large-offset inputs (even in FP32).
    # Both paths mask tail columns out of the statistics.
    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                    mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / N
        xc = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / N
        rstd = tl.math.rsqrt(var + eps)

        if STORE_STATS:
            tl.store(Mean + rows, mean, mask=row_mask)
            tl.store(Rstd + rows, rstd, mask=row_mask)

        w = tl.load(W + cols, mask=col_mask, other=0.0)
        b = tl.load(B + cols, mask=col_mask, other=0.0)
        y = xc * rstd[:, None] * w + b
        if HAS_ROWSCALE:  # fold a per-row scale (e.g. AF pair-mask) into the LN epilogue — free
            rs = tl.load(Rowscale + rows, mask=row_mask, other=0.0).to(tl.float32)
            y = y * rs[:, None]
        tl.store(Y + rows[:, None] * stride_r + cols[None, :] * stride_c, y, mask=mask)
    else:
        mean = tl.zeros([BLOCK_M1], dtype=tl.float32)
        m2 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        count = 0
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            mask = row_mask[:, None] & (cols[None, :] < N)
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0.0).to(tl.float32)
            tile_count = tl.minimum(BLOCK_K, N - n0)
            tile_mean = tl.sum(x, axis=1) / tile_count
            centered = tl.where(mask, x - tile_mean[:, None], 0.0)
            tile_m2 = tl.sum(centered * centered, axis=1)
            next_count = count + tile_count
            delta = tile_mean - mean
            m2 += tile_m2 + delta * delta * (count * tile_count / next_count)
            mean += delta * (tile_count / next_count)
            count = next_count
        var = m2 / N
        rstd = tl.math.rsqrt(var + eps)

        # Write mean / rstd
        if STORE_STATS:
            tl.store(Mean + rows, mean, mask=row_mask)
            tl.store(Rstd + rows, rstd, mask=row_mask)

        if HAS_ROWSCALE:  # fold a per-row scale (e.g. AF pair-mask) into the LN epilogue — free
            rs = tl.load(Rowscale + rows, mask=row_mask, other=0.0).to(tl.float32)

        # Normalize and apply linear transformation
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=col_mask, other=0.0)
            b = tl.load(B + cols, mask=col_mask, other=0.0)
            x_hat = (x - mean[:, None]) * rstd[:, None]
            y = x_hat * w + b
            if HAS_ROWSCALE:
                y = y * rs[:, None]
            tl.store(Y + rows[:, None] * stride_r + cols[None, :] * stride_c, y, mask=mask)
# fmt: on


# fmt: off


# fmt: off


# HAS_ROWSCALE is keyed: it is constexpr, so the two variants already compile separately, but the
# autotune cache is keyed only on `key=[...]` -- without it the tile measured on the dense path is
# reused by the masked one, which does strictly more work per row.
@triton.autotune(configs=configs_for("layernorm_bwd_atomic_triton"),
                 key=['shape_key', 'HAS_ROWSCALE'],
                 reset_to_zero=['DW', 'DB'])
@triton.jit
def layer_norm_bwd_dx_fused(
    DX, DY, DW, DB,
    X, W, Mean, Rstd, Rowscale,
    stride_wc, stride_bc, stride_r, stride_c,
    M, N: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_ROWSCALE: tl.constexpr,
):
    # PERSISTENT: a fixed number of programs, each walking the row tiles pid, pid + nprog, ... .
    # The dw/db column sums stay in register vectors across every tile a program visits and are
    # flushed with ONE atomic_add each per program, instead of two per row tile; the tile loop is
    # software pipelined so the next tile's x/dy loads overlap this tile's reductions. Measured
    # (H100, bf16 x[1, L, L, 128]) the backward went 60 -> 47 us at L=384 and 193 -> 162 us at
    # L=768 (docs/records/portable_norms_ladder_20261025.md).
    pid = tl.program_id(0).to(tl.int64)
    nprog = tl.num_programs(0)
    tiles = tl.cdiv(M, BLOCK_M1)

    # TWO-PASS over the N tiles when the feature axis is tiled. dx needs c1/c2, which reduce over
    # the WHOLE row, so once the feature axis is tiled the (x, dy) tiles must be visited twice. dw/db
    # do NOT depend on c1/c2 (they reduce over M, per column), so their atomics are issued in pass 1
    # and pass 2 only re-reads x/dy and writes dx. dw/db partials are fp32 (w is cast to fp32 before
    # use); `reset_to_zero=[DW, DB]` is kept because the autotuner re-runs every candidate against
    # these accumulators.
    #
    # COVERING TILE (BLOCK_K >= N): one trip over the row, register-held dw/db accumulators. N and
    # BLOCK_K are both tl.constexpr, so this guard is resolved at TRACE time and only one branch is
    # emitted.
    if BLOCK_K >= N:
        # Software-pipeline the tile loop only while the staged x/dy tiles fit shared memory:
        # 3 stages x 2 tensors x 8192 fp32 elements is 196 KB of the 227 KB an H100 offers.
        PIPE: tl.constexpr = 3 if BLOCK_M1 * BLOCK_K <= 8192 else 1
        cols = tl.arange(0, BLOCK_K)
        col_mask = cols < N
        w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
        dw_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        db_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for t in tl.range(pid, tiles, nprog, num_stages=PIPE):
            rows = tl.arange(0, BLOCK_M1) + t * BLOCK_M1
            row_mask = rows < M
            mask = row_mask[:, None] & col_mask[None, :]
            mean = tl.load(Mean + rows, mask=row_mask, other=0.0).to(tl.float32)
            rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)
            x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                        mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                         mask=mask, other=0).to(tl.float32)
            if HAS_ROWSCALE:  # bwd of y = LN(x)*rs: scale incoming grad by rs (then dx/dw/db all follow)
                rs = tl.load(Rowscale + rows, mask=row_mask, other=0.0).to(tl.float32)
                dy = dy * rs[:, None]
            xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
            wdy = tl.where(mask, w[None, :] * dy, 0.0)
            c1 = tl.sum(xhat * wdy, axis=1) / N
            c2 = tl.sum(wdy, axis=1) / N
            dw_acc += tl.sum(dy * xhat, axis=0)
            db_acc += tl.sum(dy, axis=0)
            dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
            tl.store(DX + rows[:, None] * stride_r + cols[None, :] * stride_c, dx, mask=mask)
        # These atomics only accumulate parameter gradients; no CTA reads their result.
        # The following kernel/stream dependency supplies visibility. Keep GPU-wide
        # atomicity without acquire/release ordering for unrelated activation memory.
        tl.atomic_add(DW + cols, dw_acc, mask=col_mask, sem="relaxed")
        tl.atomic_add(DB + cols, db_acc, mask=col_mask, sem="relaxed")
    else:
        for t in range(pid, tiles, nprog):
            rows = tl.arange(0, BLOCK_M1) + t * BLOCK_M1
            row_mask = rows < M
            mean = tl.load(Mean + rows, mask=row_mask, other=0.0).to(tl.float32)
            rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)
            if HAS_ROWSCALE:  # bwd of y = LN(x)*rs: scale incoming grad by rs (then dx/dw/db all follow)
                rs = tl.load(Rowscale + rows, mask=row_mask, other=0.0).to(tl.float32)
            c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
            c2 = tl.zeros([BLOCK_M1], dtype=tl.float32)
            for n0 in range(0, N, BLOCK_K):
                cols = n0 + tl.arange(0, BLOCK_K)
                col_mask = cols < N
                mask = row_mask[:, None] & col_mask[None, :]
                x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                            mask=mask, other=0).to(tl.float32)
                dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                             mask=mask, other=0).to(tl.float32)
                if HAS_ROWSCALE:
                    dy = dy * rs[:, None]
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
                wdy = tl.where(mask, w[None, :] * dy, 0.0)
                c1 += tl.sum(xhat * wdy, axis=1)
                c2 += tl.sum(wdy, axis=1)
                # Accumulate partial sums for dw/db (column reduction over this row tile)
                tl.atomic_add(DW + cols, tl.sum(dy * xhat, axis=0), mask=col_mask, sem="relaxed")
                tl.atomic_add(DB + cols, tl.sum(dy, axis=0), mask=col_mask, sem="relaxed")
            c1 = c1 / N
            c2 = c2 / N

            # Compute + write dx
            for n0 in range(0, N, BLOCK_K):
                cols = n0 + tl.arange(0, BLOCK_K)
                col_mask = cols < N
                mask = row_mask[:, None] & col_mask[None, :]
                x = tl.load(X + rows[:, None] * stride_r + cols[None, :] * stride_c,
                            mask=mask, other=0).to(tl.float32)
                dy = tl.load(DY + rows[:, None] * stride_r + cols[None, :] * stride_c,
                             mask=mask, other=0).to(tl.float32)
                if HAS_ROWSCALE:
                    dy = dy * rs[:, None]
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
                wdy = tl.where(mask, w[None, :] * dy, 0.0)
                dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
                tl.store(DX + rows[:, None] * stride_r + cols[None, :] * stride_c, dx, mask=mask)
# fmt: on


def _ln_fwd_fake(x_2d, weight, bias, rs, eps, has_rs, shape_key, store_stats=True):
    """``y`` like ``x_2d``, plus ``mean`` and ``rstd`` as (M,) -- fp32, not the bf16 of the
    activation: the kernel reduces and stores the row statistics in fp32 and the backward reloads
    them as such. Without ``store_stats`` (inference) the two are empty (0,) placeholders."""
    m = x_2d.shape[0] if store_stats else 0
    return (
        torch.empty_like(x_2d),
        x_2d.new_empty((m,), dtype=torch.float32),   # mean
        x_2d.new_empty((m,), dtype=torch.float32),   # rstd
    )


@opaque(fake=_ln_fwd_fake, name="layernorm_fwd")
def _ln_fwd(
    x_2d: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    rs: torch.Tensor | None,
    eps: float,
    has_rs: bool,
    shape_key: int,
    store_stats: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The launch: ``y = LN(x) * rs`` -> ``(y, mean, rstd)``. ``x_2d`` arrives flat and contiguous.

    Split out of ``TritonLayerNormFunction.forward`` so the reshape, the ``row_scale`` flattening
    and ``save_for_backward`` stay traceable -- see ``kernels._compile``. ``rs`` is None on the
    dense path; the kernel wants a real pointer there, so ``rstd`` stands in as the placeholder it
    never reads (``HAS_ROWSCALE=False``).

    ``store_stats=False`` is the no-grad forward: the statistics are not written and ``mean`` /
    ``rstd`` come back as empty (0,) tensors that nothing reads.
    """
    M, N = x_2d.shape
    y_2d = torch.empty_like(x_2d)
    mean = torch.empty(M if store_stats else 0, dtype=torch.float32, device=x_2d.device)
    rstd = torch.empty(M if store_stats else 0, dtype=torch.float32, device=x_2d.device)
    # fmt: off
    grid = lambda META: [triton.cdiv(M, META["BLOCK_M1"])]
    layer_norm_fwd_fused[grid](
        x_2d, y_2d, weight, bias, mean, rstd, rs if has_rs else rstd,
        x_2d.stride(0), x_2d.stride(1),
        M, N, eps,
        shape_key=pack(shape_key, N=N), HAS_ROWSCALE=has_rs, STORE_STATS=store_stats,
    )
    # fmt: on
    return y_2d, mean, rstd


class TritonLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
        row_scale: torch.Tensor | None = None,
    ):
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()

        has_rs = row_scale is not None
        # per-row scale (e.g. AF pair-mask) folded into the LN epilogue — y = LN(x)*rs, FREE
        # (no extra (M,N) multiply / HBM round-trip). rs reshaped to [M], broadcast over N.
        rs = row_scale.reshape(-1).to(x_2d.dtype).contiguous() if has_rs else None
        y_2d, mean, rstd = _ln_fwd(
            x_2d, weight, bias, rs, eps, has_rs,
            # L = x.shape[-2], read BEFORE the reshape above -- one rule for pair
            # (B, L, L, D) and token/atom (B, L, D). Never the row count M.
            both_key(rows_of(x.shape)),
            # Nothing runs a backward when no input takes a gradient (inference): skip the stats.
            store_stats=any(ctx.needs_input_grad[:3]),
        )

        ctx.save_for_backward(
            # x at its OWN dtype. This used to be `.to(torch.bfloat16)`, halving the saved
            # activation -- and a layernorm backward recomputed from a bf16 copy of x is the
            # narrowest precision in the whole normalisation. Every load in these kernels already
            # widens with `.to(tl.float32)` and mean/rstd are stored fp32; saving x narrow put the
            # one value that cannot be recovered back at bf16. The cast is also a no-op when x
            # really is bf16, so removing it costs nothing there and keeps fp32 where it is asked
            # for.
            x_2d,
            weight,
            mean,
            rstd,
            rs if has_rs else None,
        )
        ctx.has_rowscale = has_rs
        ctx.input_shape = x.shape
        return y_2d.view_as(x)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, mean, rstd, rs = ctx.saved_tensors
        dx, dw, db = _ln_bwd(
            dy.reshape(-1, dy.shape[-1]).contiguous(), x, weight, mean, rstd, rs,
            ctx.has_rowscale, list(ctx.input_shape),
            # ctx.input_shape is the forward's PRE-flatten shape; L = its [-2].
            both_key(rows_of(ctx.input_shape)),
        )
        return dx, dw, db, None, None   # eps, row_scale take no gradient


def _ln_bwd_fake(dy_2d, x, weight, mean, rstd, rs, has_rs, input_shape, shape_key):
    """``dx`` at the forward's PRE-flatten ``input_shape`` (not ``dy_2d``'s flat (M, N)), plus
    ``dweight`` and ``dbias`` as (N,) fp32.

    The Triton body allocates fp32 accumulators; the hand-CUDA backward in ``cuda/masked_bwd.py`` shares this fake and
    calls ``.float()`` on its results.
    """
    n = x.shape[-1]
    return (
        dy_2d.new_empty(tuple(input_shape)),
        x.new_empty((n,), dtype=torch.float32),
        x.new_empty((n,), dtype=torch.float32),
    )


@opaque(fake=_ln_bwd_fake, name="layernorm_bwd")
def _ln_bwd(
    dy_2d: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    rs: torch.Tensor | None,
    has_rs: bool,
    input_shape: list[int],
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward launches -> ``(dx, dweight, dbias)``.

    Split out of ``TritonLayerNormFunction.backward`` (see ``kernels._compile``). It returns only
    the three real gradients -- a ``torch.library`` schema cannot return ``None``, so the caller
    re-adds the two ``None`` slots that ``eps`` and ``row_scale`` need. The Triton backward only.
    """
    x = x.to(dy_2d.dtype)
    M, N = x.shape

    # allocate output
    dx_2d = torch.empty_like(dy_2d)
    dw = torch.zeros(N, dtype=torch.float32, device=x.device)
    db = torch.zeros(N, dtype=torch.float32, device=x.device)

    # fmt: off
    programs = _persistent_programs(x.device)
    grid = lambda META: [min(triton.cdiv(M, META["BLOCK_M1"]), programs)]
    layer_norm_bwd_dx_fused[grid](
        dx_2d, dy_2d, dw, db,
        x, weight, mean, rstd, rs if has_rs else rstd,  # rs folds the mask grad in (free)
        dw.stride(0), db.stride(0), x.stride(0), x.stride(1),
        M, N,
        shape_key=pack(shape_key, N=N), HAS_ROWSCALE=has_rs,
    )
    # fmt: on

    return dx_2d.view(tuple(input_shape)), dw, db


def triton_layernorm(x, weight, bias, eps, row_scale=None):
    """LayerNorm (autograd). Optional `row_scale` [M] folds a per-row scale into the LN epilogue
    (fwd) and the grad into the LN backward (bwd) — y = LN(x)*rs, the AF pair-mask applied FREE
    (no separate (M,N) multiply). rs=None -> plain LN."""
    # Absent affine parameters are constants, not trainable synthetic parameters.
    # Construct outside Function.apply so autograd never returns a gradient for None.
    if weight is None:
        weight = torch.ones(x.shape[-1], device=x.device, dtype=torch.float32)
    if bias is None:
        bias = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float32)
    return TritonLayerNormFunction.apply(x, weight, bias, eps, row_scale)


def _triton_layernorm_masked_fake(x, weight, bias, eps, row_scale):
    """Same shape and dtype as x -- row_scale is a per-row factor, it does not reshape."""
    return torch.empty_like(x)


@opaque(fake=_triton_layernorm_masked_fake, name="layernorm_masked_fwd")
def triton_layernorm_masked(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    row_scale: torch.Tensor,
) -> torch.Tensor:
    """Forward-only LN with a per-row scale folded into the epilogue (free) — for the
    AF triangle pair-mask: y = LN(x) * row_scale[row]. row_scale is [M] (or anything
    reshapeable to [M]); broadcast over the feature dim. No autograd (bench/inference)."""
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    y_2d = torch.empty_like(x_2d)
    M, N = x_2d.shape
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    rs = row_scale.reshape(-1).to(x_2d.dtype).contiguous()
    # fmt: off
    grid = lambda META: [triton.cdiv(M, META["BLOCK_M1"])]
    layer_norm_fwd_fused[grid](
        x_2d, y_2d, weight, bias, mean, rstd, rs,
        x_2d.stride(0), x_2d.stride(1),
        M, N, eps,
        shape_key=both_key(rows_of(x.shape), N=N), HAS_ROWSCALE=True,
    )
    # fmt: on
    return y_2d.view_as(x)

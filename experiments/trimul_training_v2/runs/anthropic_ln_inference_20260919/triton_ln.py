import triton
import triton.language as tl
@triton.jit
def layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd, Rowscale,
    stride_r, stride_c,
    M, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, HAS_ROWSCALE: tl.constexpr,
):
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
        rstd = 1 / tl.sqrt(var + eps)


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
        rstd = 1 / tl.sqrt(var + eps)

        # Write mean / rstd

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

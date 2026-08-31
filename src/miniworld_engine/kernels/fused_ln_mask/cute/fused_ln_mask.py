"""Fused LayerNorm + per-row mask multiply (Triton kernel).

Replaces

    x_normed = cuequiv_ln(x)             # ~0.20 ms at L=1024, D=128
    x_normed = x_normed * mask_2d        # ~0.35 ms

with one HBM pass:

    fused_ln_mask(x, weight, bias, mask) # B200: ~0.090 ms at L=1024

``x`` is (B, L, L, D); ``mask`` is a 2D pair mask shaped (B, L, L) (one scalar
per output row). Math:

    row_norm = LayerNorm(x[b, r, c, :], over D axis; weight, bias, eps)
    out[b, r, c, :] = row_norm * mask[b, r, c]

The kernel is bandwidth-bound (read x + write out, ~512 MB at L=1024). On B200
the tile is autotuned; the winning shape across L is BLOCK_M1=8 with a single
warp (i.e. ~8 rows / warp, two 128-bit bf16 loads per row), which saturates
HBM at ~6 TB/s. Configs are kept and the reduction stays in fp32.
"""

from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for

import torch

from miniworld_engine.kernels._compile import opaque
import triton
import triton.language as tl



from miniworld_engine.autotune.buckets import bucket_mixed as _bucket
from miniworld_engine.autotune.shape_key import length_of, token_key


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)



# BLOCK_K tiles the D reduction (mean/var over D), so D need not be a power of two; a row that
# sets it >= D keeps the single-pass schedule. BLOCK_M1 is the row tile. Both come from the CSV.
# EPS is constexpr but deliberately NOT keyed: it only appears in `rsqrt(var + EPS)`, so it
# branches nothing and shifts no work -- keying it would just multiply the bucket count.
@triton.autotune(configs=configs_for("layernorm_fwd_rowscale_triton"), key=['shape_key'])
@triton.jit
def _fused_ln_mask_kernel(
    x_ptr,
    mask_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    D: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
    shape_key,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    mask_m = offs_m < M

    # TWO-PASS (not Welford): pass 1 accumulates Σx and Σx² over the D tiles in fp32 — both are
    # plain sums, so they are exact across tiles — and pass 2 re-reads x to normalize. LayerNorm
    # re-uses the row after reducing it, so the row must either be re-read or carried through a
    # Welford state; the re-read is the cheaper and far simpler of the two, and it disappears when
    # the tuner picks BLOCK_K >= D (one trip per loop, x hot in L1/L2).
    #
    # COVERING TILE (BLOCK_K >= D): both loops are single-trip, but the two tl.loads of x are NOT
    # CSE'd (a tl.load of mask_ptr and the loop structure sit between them and Triton cannot prove
    # the raw pointers do not alias), so the covering config read x twice — 3 HBM passes on a
    # kernel whose entire reason for existing is "one HBM pass". D and BLOCK_K are both
    # tl.constexpr, so the guard is resolved at TRACE time and only ONE branch is emitted. The fast
    # path uses the CENTRED variance Σ(x-mean)²/D (numerically stabler, and x is already live); the
    # uncentered Σx²/D - mean² is kept in the tiled branch, where it is what lets each tile be read
    # exactly once.
    if BLOCK_K >= D:
        offs_d = tl.arange(0, BLOCK_K)
        dmask = offs_d < D
        m2 = mask_m[:, None] & dmask[None, :]
        x = tl.load(x_ptr + offs_m[:, None] * D + offs_d[None, :], mask=m2, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / D
        xc = tl.where(m2, x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / D
        rstd = 1.0 / tl.sqrt(var + EPS)

        mvals = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_d, mask=dmask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs_d, mask=dmask, other=0.0).to(tl.float32)
        x_norm = xc * rstd[:, None] * w[None, :] + b[None, :]
        out = x_norm * mvals[:, None]
        tl.store(
            out_ptr + offs_m[:, None] * D + offs_d[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=m2,
        )
    else:
        s = tl.zeros([BLOCK_M1], dtype=tl.float32)
        ss = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for d0 in range(0, D, BLOCK_K):
            offs_d = d0 + tl.arange(0, BLOCK_K)
            m2 = mask_m[:, None] & (offs_d[None, :] < D)
            x = tl.load(x_ptr + offs_m[:, None] * D + offs_d[None, :], mask=m2, other=0.0).to(tl.float32)
            s += tl.sum(x, axis=1)
            ss += tl.sum(x * x, axis=1)
        mean = s / D
        var = ss / D - mean * mean          # Σx²/D − mean² (same algebra as _stats_kernel)
        rstd = 1.0 / tl.sqrt(var + EPS)

        # Per-row mask
        mvals = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

        for d0 in range(0, D, BLOCK_K):
            offs_d = d0 + tl.arange(0, BLOCK_K)
            dmask = offs_d < D
            m2 = mask_m[:, None] & dmask[None, :]
            x = tl.load(x_ptr + offs_m[:, None] * D + offs_d[None, :], mask=m2, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs_d, mask=dmask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + offs_d, mask=dmask, other=0.0).to(tl.float32)
            x_norm = (x - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]
            out = x_norm * mvals[:, None]
            tl.store(
                out_ptr + offs_m[:, None] * D + offs_d[None, :],
                out.to(out_ptr.dtype.element_ty),
                mask=m2,
            )


def _fused_ln_mask_fake(x, weight, bias, mask, eps=1e-5):
    """(B, L, L, D): masked LN is elementwise, so the output matches x's shape and dtype."""
    return torch.empty_like(x)


@opaque(fake=_fused_ln_mask_fake, name="fused_ln_mask")
def fused_ln_mask(
    x: torch.Tensor,  # (B, L, L, D), bf16/fp16
    weight: torch.Tensor,  # (D,)
    bias: torch.Tensor,  # (D,)
    mask: torch.Tensor,  # (B, L, L) bool / float
    eps: float = 1e-5,
) -> torch.Tensor:
    """LN(x) over last axis then multiply by per-row mask. One HBM pass."""
    assert x.dim() == 4 and x.is_cuda
    B, L1, L2, D = x.shape
    M = B * L1 * L2
    x_flat = x.reshape(M, D)
    mask_flat = mask.reshape(M).to(x.dtype).contiguous()
    out = torch.empty_like(x_flat)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)
    _fused_ln_mask_kernel[grid](
        x_flat,
        mask_flat,
        weight,
        bias,
        out,
        M,
        D,
        EPS=eps,
        # `token_key`, because registry.csv declares this kernel level=token -- and token_key's
        # own docstring is "the key for a token/pair-level kernel". This used to call
        # `both_key(rows_of(...))`, the level=both function, which buckets the ROW count: a pair
        # of side L has L*L rows, so the cache recorded 16384/65536/147456/262144 where every
        # other level=token kernel records 128/256/384/512. No launch was served the wrong config
        # -- this wrapper is the only caller and it keyed reads the same way it keyed writes --
        # but `dev audit` compares the cache against the DECLARED bucket, which for level=token
        # is L, and reported all four missing on a cache that was complete.
        # `length_of` reads shape[-2], which is L for the (B, L, L, D) pair activation.
        shape_key=token_key(length_of(x.shape), D=D),
    )
    return out.view(B, L1, L2, D)

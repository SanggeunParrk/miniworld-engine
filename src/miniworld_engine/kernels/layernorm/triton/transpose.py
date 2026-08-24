"""``layer_norm_transpose`` — a cuequivariance-FREE drop-in for
``cuequivariance_ops_torch.fused_layer_norm_torch.layer_norm_transpose``.

Built on our own Triton LayerNorm, at the HBM bandwidth wall. Supports the two
layouts the trimul cute path uses:

    "nd->nd"   : x (M, D) -> LN over D -> (M, D)          (LN_in; our triton_layernorm)
    "dbn->bnd" : x (D, B, N) -> LN over D -> (B, N, D)    (LN_out; FUSED transpose-LN kernel)

The dbn->bnd case is a FUSED transpose+LN (one coalesced read of the channel-major
(D, M) input + one coalesced row-major write), NOT a materialised transpose then LN —
so it matches cuequiv's single-pass cost. Same channel-major-LN math as the trimul
back-half kernel (``_back_kernel``), just without the proj/gate.
"""

from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for

import torch

from miniworld_engine.kernels._compile import opaque
import triton
import triton.language as tl


from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm




# The channel axis was `tl.arange(0, D)` — the shape used directly as the tile extent, which also
# forced D to be a power of two and left no column mask. It is the REDUCE axis (mean/var over D),
# so it is a CSV tile; a row at or above the extent keeps the whole-row schedule
# is still reachable and the tuner picks it wherever the second pass is not worth the registers.
from miniworld_engine.autotune.buckets import bucket_mixed as _bucket
from miniworld_engine.autotune.shape_key import both_key


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)


@triton.autotune(configs=configs_for("layernorm_fwd_mmajor_triton"), key=['D', 'shape_key'])
@triton.jit
def _ln_transpose_dbn_kernel(
    x_ptr,   # (D, M) channel-major: x[k, m] at k*M + m
    y_ptr,   # (M, D) row-major:     y[m, k] at m*D + k
    w_ptr, b_ptr,  # (D,)
    M, eps,
    D: tl.constexpr, BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key,
):
    pid = tl.program_id(0).to(tl.int64)
    rm = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    mmask = rm < M
    # TWO-PASS: pass 1 accumulates Σx and Σx² over the D tiles (plain sums -> exact across tiles,
    # fp32); pass 2 re-reads x to normalize + transpose-store. LN re-uses the row after reducing
    # it, so with a tiled reduce axis the row is either re-read or carried in a Welford state —
    # the re-read is simpler and vanishes when the tuner picks BLOCK_K >= D.
    #
    # COVERING TILE (BLOCK_K >= D): both loops are single-trip, but the two tl.loads of x_ptr are
    # NOT CSE'd across the loop boundary (Triton cannot prove the raw x_ptr/y_ptr do not alias), so
    # the covering config read x twice instead of collapsing to the single-pass schedule this
    # module documents. D and BLOCK_K are both tl.constexpr, so the guard is resolved at TRACE time
    # and only ONE branch is emitted. The fast path uses the CENTRED variance Σ(x-mean)²/D
    # (numerically stabler, and x is already in registers); the uncentered Σx²/D - mean² stays in
    # the tiled branch, where it exists precisely to keep that branch one read per tile.
    if BLOCK_K >= D:
        rk = tl.arange(0, BLOCK_K)
        kmask = rk < D
        mask = mmask[:, None] & kmask[None, :]
        # channel-major load -> (BLOCK_M1 rows, BLOCK_K channels), coalesced along m per channel
        x = tl.load(x_ptr + rk[None, :] * M + rm[:, None], mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / D
        xc = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + rk, mask=kmask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + rk, mask=kmask, other=0.0).to(tl.float32)
        y = ((xc * rstd[:, None]) * w[None, :] + b[None, :]).to(y_ptr.dtype.element_ty)
        # row-major store -> (BLOCK_M1, BLOCK_K), coalesced along k within each row (the transpose)
        tl.store(y_ptr + rm[:, None] * D + rk[None, :], y, mask=mask)
    else:
        s = tl.zeros([BLOCK_M1], dtype=tl.float32)
        ss = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for k0 in range(0, D, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            mask = mmask[:, None] & (rk[None, :] < D)
            # channel-major load -> (BLOCK_M1 rows, BLOCK_K channels), coalesced along m per channel
            x = tl.load(x_ptr + rk[None, :] * M + rm[:, None], mask=mask, other=0.0).to(tl.float32)
            s += tl.sum(x, axis=1)
            ss += tl.sum(x * x, axis=1)
        mean = s / D
        var = ss / D - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)
        for k0 in range(0, D, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            kmask = rk < D
            mask = mmask[:, None] & kmask[None, :]
            x = tl.load(x_ptr + rk[None, :] * M + rm[:, None], mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + rk, mask=kmask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + rk, mask=kmask, other=0.0).to(tl.float32)
            y = (((x - mean[:, None]) * rstd[:, None]) * w[None, :] + b[None, :]).to(y_ptr.dtype.element_ty)
            # row-major store -> (BLOCK_M1, BLOCK_K), coalesced along k within each row (the transpose)
            tl.store(y_ptr + rm[:, None] * D + rk[None, :], y, mask=mask)


def _ln_transpose_dbn_bnd_fake(x, weight, bias, eps):
    """(B, N, D): the op transposes x's (D, B, N) layout as it normalizes over D."""
    return x.new_empty((x.shape[1], x.shape[2], x.shape[0]))


@opaque(fake=_ln_transpose_dbn_bnd_fake, name="layernorm_transpose_dbn_bnd")
def _ln_transpose_dbn_bnd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                          eps: float) -> torch.Tensor:
    """x: (D, B, N) -> LN over D -> (B, N, D), fused (no materialised transpose)."""
    d, b, n = x.shape
    M = b * n
    x_dm = x.reshape(d, M)  # (D, M) channel-major (contiguous view when x is contiguous)
    if not x_dm.is_contiguous():
        x_dm = x_dm.contiguous()
    y = torch.empty(M, d, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731
    _ln_transpose_dbn_kernel[grid](
        x_dm, y, weight.contiguous(), bias.contiguous(), M, float(eps), D=d,
        # M, the row count this launch iterates. It used to pass the token axis `n` instead,
        # because a length was what the key was made of; rows are, and M is right here.
        shape_key=both_key(M),
    )
    return y.view(b, n, d)


def layer_norm_transpose(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-5,
    layout: str = "nd->nd",
) -> torch.Tensor:
    """LayerNorm over the D axis with an optional layout transpose. Drop-in for the
    cuequiv op (same call signature); our Triton LN underneath, cuequiv-free."""
    if layout == "nd->nd":
        return triton_layernorm(x, weight, bias, eps)  # (M, D) -> (M, D)
    if layout == "dbn->bnd":
        return _ln_transpose_dbn_bnd(x, weight, bias, eps)  # (D, B, N) -> (B, N, D), fused
    msg = f"layer_norm_transpose: unsupported layout {layout!r} (have nd->nd, dbn->bnd)"
    raise ValueError(msg)

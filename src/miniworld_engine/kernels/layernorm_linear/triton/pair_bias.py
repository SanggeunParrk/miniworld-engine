"""Fused LayerNorm + projection Triton kernel for pair bias.

The normalized pair ``[..., d]`` is consumed only by a small projection to
``n_head``. This kernel reads the pair once, normalizes each row in fp32, and
projects to ``n_head`` in-register -- it never materializes the full normalized
``[..., d]`` tensor, so it beats ``torch.compile`` (which would emit a LayerNorm
kernel plus a separate GEMM). ``ln_weight``/``proj_weight`` are the affine scale
of an *unbiased* ``nn.LayerNorm`` and the weight of an *unbiased* ``Linear``.

Registered through ``kernels._compile.opaque`` like every other kernel launch, so it stays in the compiled graph
without a graph break -- surviving ``torch.compile`` is the entire point.
"""


from miniworld_engine.autotune.configs import configs_for
import torch
from miniworld_engine import settings
import triton
import triton.language as tl

from jaxtyping import Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.kernels._compile import opaque

# tl.dot needs every dim >= 16; below this the backward uses the scalar loop.
MIN_TL_DOT_DIM = 16

# BLOCK_K_D tiles the d / channel axis and BLOCK_K_NH the n_head / projection axis; the k-loops
# below make a tile narrower than the extent correct, and a row that sets either >= its extent
# keeps the whole-row-in-one-tile schedule. n_head is tiny, so a single NH tile normally wins.





# shape_key is keyed: the grid is cdiv(M, BLOCK_M1) and M is the pair row count (B*L^2), so one
# BLOCK_M1 was being reused from L=128 to L=1024+.
from miniworld_engine.autotune.buckets import bucket_mixed as _bucket
from miniworld_engine.autotune.shape_key import both_key, length_of, rows_of


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)


@triton.autotune(configs=configs_for("layernorm_linear_fwd_fp32_triton"),
                 key=['N', 'NH', 'shape_key'])
@triton.jit
def _layer_norm_linear_fwd(
    x_ptr,
    lnw_ptr,
    pw_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    stride_m,
    M,
    N: tl.constexpr,
    NH: tl.constexpr,
    eps,
    BLOCK_M1: tl.constexpr,
    BLOCK_K_D: tl.constexpr,
    BLOCK_K_NH: tl.constexpr,
    shape_key,
):
    # BLOCK_K_D tiles the channel axis (d) and BLOCK_K_NH the projection axis (n_head); when the
    # tuner picks tiles >= the extents, every loop below is a single iteration and this is the
    # original whole-row schedule.
    rows = (tl.program_id(0) * BLOCK_M1 + tl.arange(0, BLOCK_M1)).to(tl.int64)
    row_mask = rows < M

    if BLOCK_K_D >= N:
        # COVERING TILE. BLOCK_K_D and N are both tl.constexpr, so this test is resolved at COMPILE
        # time and only one branch is emitted. The whole row fits one tile, so read x ONCE, keep
        # the centered row (and the LN-scaled `y`) in registers, and reuse them for every
        # projection subtile -- the pre-tiling single-pass schedule. The `else` below is the
        # general N-tiled form; at BLOCK_K_D >= N its loops are single-trip and each expression
        # here matches it term for term, so the two branches are numerically identical.
        cols = tl.arange(0, BLOCK_K_D)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptr + rows[:, None] * stride_m + cols[None, :],
                    mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / N
        centered = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(centered * centered, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        lnw = tl.load(lnw_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * rstd[:, None] * lnw[None, :]
        for h0 in range(0, NH, BLOCK_K_NH):
            hcols = h0 + tl.arange(0, BLOCK_K_NH)
            h_mask = hcols < NH
            oacc = tl.zeros([BLOCK_M1, BLOCK_K_NH], dtype=tl.float32)
            pw = tl.load(pw_ptr + hcols[None, :] * N + cols[:, None],
                         mask=h_mask[None, :] & col_mask[:, None], other=0.0).to(tl.float32)
            oacc = tl.dot(y, pw, oacc, allow_tf32=False)
            tl.store(out_ptr + rows[:, None] * NH + hcols[None, :], oacc,
                     mask=row_mask[:, None] & h_mask[None, :])
        tl.store(mean_ptr + rows, mean, mask=row_mask)
        tl.store(rstd_ptr + rows, rstd, mask=row_mask)
    else:
        # --- row statistics over N-tiles. Two passes (mean, then centered variance) so the fp32
        # algebra is bit-for-bit the original one at BLOCK_K_D >= N; mean/rstd are SAVED and reused
        # by the backward, so this is the one place not to trade exactness for a pass. ---
        acc = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K_D):
            cols = n0 + tl.arange(0, BLOCK_K_D)
            col_mask = cols < N
            x = tl.load(x_ptr + rows[:, None] * stride_m + cols[None, :],
                        mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(x, axis=1)
        mean = acc / N
        acc = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K_D):
            cols = n0 + tl.arange(0, BLOCK_K_D)
            col_mask = cols < N
            x = tl.load(x_ptr + rows[:, None] * stride_m + cols[None, :],
                        mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
            centered = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
            acc += tl.sum(centered * centered, axis=1)
        var = acc / N
        rstd = 1.0 / tl.sqrt(var + eps)

        # --- projection: out[m, h] = sum_n y[m, n] * pw[h, n]. The (n_head, d) contraction is a
        # tl.dot over the BLOCK_K_D tile (fp32 operands, allow_tf32=False -> plain fp32 FMA, the
        # same arithmetic the old tl.sum(y * pw) reduction did). ---
        for h0 in range(0, NH, BLOCK_K_NH):
            hcols = h0 + tl.arange(0, BLOCK_K_NH)
            h_mask = hcols < NH
            oacc = tl.zeros([BLOCK_M1, BLOCK_K_NH], dtype=tl.float32)
            for n0 in range(0, N, BLOCK_K_D):
                cols = n0 + tl.arange(0, BLOCK_K_D)
                col_mask = cols < N
                mask = row_mask[:, None] & col_mask[None, :]
                x = tl.load(x_ptr + rows[:, None] * stride_m + cols[None, :],
                            mask=mask, other=0.0).to(tl.float32)
                centered = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
                lnw = tl.load(lnw_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
                y = centered * rstd[:, None] * lnw[None, :]
                pw = tl.load(pw_ptr + hcols[None, :] * N + cols[:, None],
                             mask=h_mask[None, :] & col_mask[:, None], other=0.0).to(tl.float32)
                oacc = tl.dot(y, pw, oacc, allow_tf32=False)
            tl.store(out_ptr + rows[:, None] * NH + hcols[None, :], oacc,
                     mask=row_mask[:, None] & h_mask[None, :])
        tl.store(mean_ptr + rows, mean, mask=row_mask)
        tl.store(rstd_ptr + rows, rstd, mask=row_mask)




# USE_DOT is deliberately NOT keyed even though it selects a code path (tl.dot tensor-core
# projection vs the scalar `tl.static_range(NH)` loop): the sole launcher passes `NH=nh` and
# `USE_DOT=nh >= MIN_TL_DOT_DIM`, so it is a pure function of NH -- already in the key.
@triton.autotune(configs=configs_for("layernorm_linear_bwd_fp32_triton"),
                 key=['N', 'NH', 'shape_key'],
                 reset_to_zero=['dlnw_ptr', 'dpw_ptr'])
@triton.jit
def _layer_norm_linear_bwd(
    dout_ptr,
    x_ptr,
    lnw_ptr,
    pw_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    dlnw_ptr,
    dpw_ptr,
    stride_m,
    M,
    N: tl.constexpr,
    NH: tl.constexpr,
    USE_DOT: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_K_D: tl.constexpr,
    BLOCK_K_NH: tl.constexpr,
    shape_key,
):
    # BLOCK_K_D tiles d, BLOCK_K_NH tiles n_head. dx needs the row sums c1/c2 over ALL of d, so the
    # channel axis is walked TWICE: pass A accumulates c1/c2 (and emits the dpw/dlnw atomics),
    # pass B forms dx. At BLOCK_K_D >= N each loop is one iteration = the original single-tile
    # schedule, with the second pass reading an L2-hot row.
    rows = (tl.program_id(0) * BLOCK_M1 + tl.arange(0, BLOCK_M1)).to(tl.int64)
    row_mask = rows < M
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0)
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)

    c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
    c2 = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for n0 in range(0, N, BLOCK_K_D):
        cols = n0 + tl.arange(0, BLOCK_K_D)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None] * stride_m + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        lnw = tl.load(lnw_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = xhat * lnw[None, :]
        dy = tl.zeros((BLOCK_M1, BLOCK_K_D), dtype=tl.float32)
        if USE_DOT:
            for h0 in range(0, NH, BLOCK_K_NH):
                hcols = h0 + tl.arange(0, BLOCK_K_NH)
                h_mask = hcols < NH
                dout = tl.load(
                    dout_ptr + rows[:, None] * NH + hcols[None, :],
                    mask=row_mask[:, None] & h_mask[None, :],
                    other=0.0,
                )
                pw = tl.load(
                    pw_ptr + hcols[:, None] * N + cols[None, :],
                    mask=h_mask[:, None] & col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                dy = tl.dot(dout, pw, dy, allow_tf32=False)
                tl.atomic_add(
                    dpw_ptr + hcols[:, None] * N + cols[None, :],
                    tl.dot(tl.trans(dout), y, allow_tf32=False),
                    mask=h_mask[:, None] & col_mask[None, :],
                )
        else:
            for j in tl.static_range(NH):
                dout_j = tl.load(dout_ptr + rows * NH + j, mask=row_mask, other=0.0)
                pw = tl.load(pw_ptr + j * N + cols, mask=col_mask, other=0.0).to(tl.float32)
                dy += dout_j[:, None] * pw[None, :]
                tl.atomic_add(
                    dpw_ptr + j * N + cols,
                    tl.sum(dout_j[:, None] * y, axis=0),
                    mask=col_mask,
                )
        dy = tl.where(mask, dy, 0.0)
        dxhat = dy * lnw[None, :]
        c1 += tl.sum(dxhat * xhat, axis=1)
        c2 += tl.sum(dxhat, axis=1)
        tl.atomic_add(dlnw_ptr + cols, tl.sum(dy * xhat, axis=0), mask=col_mask)
    c1 = c1 / N
    c2 = c2 / N

    for n0 in range(0, N, BLOCK_K_D):
        cols = n0 + tl.arange(0, BLOCK_K_D)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None] * stride_m + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        lnw = tl.load(lnw_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        dy = tl.zeros((BLOCK_M1, BLOCK_K_D), dtype=tl.float32)
        if USE_DOT:
            for h0 in range(0, NH, BLOCK_K_NH):
                hcols = h0 + tl.arange(0, BLOCK_K_NH)
                h_mask = hcols < NH
                dout = tl.load(
                    dout_ptr + rows[:, None] * NH + hcols[None, :],
                    mask=row_mask[:, None] & h_mask[None, :],
                    other=0.0,
                )
                pw = tl.load(
                    pw_ptr + hcols[:, None] * N + cols[None, :],
                    mask=h_mask[:, None] & col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                dy = tl.dot(dout, pw, dy, allow_tf32=False)
        else:
            for j in tl.static_range(NH):
                dout_j = tl.load(dout_ptr + rows * NH + j, mask=row_mask, other=0.0)
                pw = tl.load(pw_ptr + j * N + cols, mask=col_mask, other=0.0).to(tl.float32)
                dy += dout_j[:, None] * pw[None, :]
        dy = tl.where(mask, dy, 0.0)
        dxhat = dy * lnw[None, :]
        dx = (dxhat - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
        tl.store(dx_ptr + offs, dx, mask=mask)


def _fwd_op_fake(x, ln_weight, proj_weight, eps):
    """(..., n_head) projection, plus the flat (M,) fp32 LN mean and rstd the backward reuses."""
    rows = 1
    for size in x.shape[:-1]:
        rows *= size
    return (
        x.new_empty(*x.shape[:-1], proj_weight.shape[0]),
        x.new_empty(rows, dtype=torch.float32),
        x.new_empty(rows, dtype=torch.float32),
    )


@opaque(fake=_fwd_op_fake, name="layernorm_linear_pair_bias_fwd")
def _fwd_op(
    x: torch.Tensor, ln_weight: torch.Tensor, proj_weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    M, N = x2.shape
    nh = proj_weight.shape[0]
    out = torch.empty(M, nh, dtype=torch.float32, device=x.device)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)
    _layer_norm_linear_fwd[grid](
        x2,
        ln_weight,
        proj_weight,
        out,
        mean,
        rstd,
        x2.stride(0),
        M,
        N,
        nh,
        eps,
        # L = x.shape[-2] (the op takes the pre-flatten activation), never the row count M.
        shape_key=both_key(rows_of(x.shape)),
    )
    return out.reshape(*x.shape[:-1], nh).to(x.dtype), mean, rstd


def _bwd_op_fake(dout, x2, ln_weight, proj_weight, mean, rstd, shape_key=None):
    """(dx, dln_weight, dproj_weight), each shaped like the tensor it is the gradient of."""
    return (
        torch.empty_like(x2),
        torch.empty_like(ln_weight),
        torch.empty_like(proj_weight),
    )


@opaque(fake=_bwd_op_fake, name="layernorm_linear_pair_bias_bwd")
def _bwd_op(
    dout: torch.Tensor,
    x2: torch.Tensor,
    ln_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    # x2 arrives already flattened to (M, N), so L is gone by here: the key is computed by
    # `_backward` below from ctx.xshape (the forward's pre-flatten shape) and passed in.
    # None only for the coordinator-owned drivers_ln / checks_ln call sites.
    shape_key: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x2.shape
    nh = proj_weight.shape[0]
    dout2 = dout.reshape(M, nh).contiguous().to(torch.float32)
    dx = torch.empty_like(x2)
    dlnw = torch.zeros(N, dtype=torch.float32, device=x2.device)
    dpw = torch.zeros(nh, N, dtype=torch.float32, device=x2.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)
    _layer_norm_linear_bwd[grid](
        dout2,
        x2,
        ln_weight,
        proj_weight,
        mean,
        rstd,
        dx,
        dlnw,
        dpw,
        x2.stride(0),
        M,
        N,
        nh,
        USE_DOT=nh >= MIN_TL_DOT_DIM,
        shape_key=both_key(M) if shape_key is None else shape_key,
    )
    return dx, dlnw.to(ln_weight.dtype), dpw.to(proj_weight.dtype)


class _LayerNormLinearPairBias(torch.autograd.Function):
    """Autograd for the fused LN+proj, in this repo's one shape: the launches are ops and the
    Function owns ``ctx``.

    It used to be ``_fwd_op.register_autograd(...)`` instead, which worked -- everything the
    backward needs (mean, rstd) is a forward OUTPUT, which is all ``setup_context`` may save. But
    it made this the only file whose ops bypassed ``kernels._compile.opaque``, and so the only
    file ``compile_wrap`` did not reach. Two patterns for one job is worse than the marginal
    tidiness of the torch-native one.
    """

    @staticmethod
    def forward(ctx, x, ln_weight, proj_weight, eps):
        out, mean, rstd = _fwd_op(x, ln_weight, proj_weight, eps)
        ctx.save_for_backward(
            x.reshape(-1, x.shape[-1]).contiguous(), ln_weight, proj_weight, mean, rstd)
        ctx.xshape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x2, ln_weight, proj_weight, mean, rstd = ctx.saved_tensors
        dx, dlnw, dpw = _bwd_op(grad_out, x2, ln_weight, proj_weight, mean, rstd,
                                shape_key=both_key(rows_of(ctx.xshape)))
        return dx.reshape(ctx.xshape), dlnw, dpw, None   # eps takes no gradient


@typecheck
def triton_layer_norm_linear(
    x: Float[torch.Tensor, "*batch d"],
    ln_weight: Float[torch.Tensor, " d"],
    proj_weight: Float[torch.Tensor, "n_head d"],
    eps: float = 1e-5,
) -> Float[torch.Tensor, "*batch n_head"]:
    """Fused ``Linear(LayerNorm(x))`` over the last dim (both affine, no bias).

    ``ln_weight`` is the LayerNorm scale ``[d]``; ``proj_weight`` is the Linear
    weight ``[n_head, d]``. Returns ``[..., n_head]``. Compile-safe (custom_op).
    The projection out-dim ``n_head`` must be a power of two when it is >= 16
    (the tensor-core backward path indexes it with ``tl.arange``).
    """
    nh = proj_weight.shape[0]
    if nh >= MIN_TL_DOT_DIM and (nh & (nh - 1)):
        msg = f"n_head={nh} must be a power of two when >= {MIN_TL_DOT_DIM}"
        raise ValueError(msg)
    return _LayerNormLinearPairBias.apply(x, ln_weight, proj_weight, eps)

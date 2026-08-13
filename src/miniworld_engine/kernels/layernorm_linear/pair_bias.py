"""Fused LayerNorm + projection Triton kernel for pair bias.

The normalized pair ``[..., d]`` is consumed only by a small projection to
``n_head``. This kernel reads the pair once, normalizes each row in fp32, and
projects to ``n_head`` in-register -- it never materializes the full normalized
``[..., d]`` tensor, so it beats ``torch.compile`` (which would emit a LayerNorm
kernel plus a separate GEMM). ``ln_weight``/``proj_weight`` are the affine scale
of an *unbiased* ``nn.LayerNorm`` and the weight of an *unbiased* ``Linear``.

Registered as a ``torch.library.custom_op`` so it stays in the compiled graph
without a graph break -- surviving ``torch.compile`` is the entire point.
"""


import torch
from miniworld_engine import settings
import triton
import triton.language as tl
from jaxtyping import Float

from miniworld_engine.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of
from miniworld_engine._typecheck import typecheck

# tl.dot needs every dim >= 16; below this the backward uses the scalar loop.
MIN_TL_DOT_DIM = 16

AUTOTUNE = settings.current().autotunes("layer_norm_linear")
if AUTOTUNE:
    configs = [
        triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns)
        for bm in (16, 32, 64)
        for nw in (2, 4, 8)
        for ns in (2, 3)
    ]
else:
    configs = [
        triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=2),
    ]


_layernorm_linear_pair_bias_fwd_prune = make_cache_prune(
    "layernorm_linear_pair_bias_fwd", dtype_of=tensor_dtype_of("x_ptr"),
    bucket_of=key_bucket_of("N", "NH"),
)


@triton.autotune(configs=configs, key=["N", "NH"],
                 prune_configs_by={"early_config_prune": _layernorm_linear_pair_bias_fwd_prune})
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    row_mask, col_mask = rows < M, cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    offs = rows[:, None].to(tl.int64) * stride_m + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / N
    centered = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    lnw = tl.load(lnw_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = centered * rstd[:, None] * lnw[None, :]
    for j in tl.static_range(NH):
        pw = tl.load(pw_ptr + j * N + cols, mask=col_mask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + rows * NH + j, tl.sum(y * pw[None, :], axis=1), mask=row_mask
        )
    tl.store(mean_ptr + rows, mean, mask=row_mask)
    tl.store(rstd_ptr + rows, rstd, mask=row_mask)


_layernorm_linear_pair_bias_bwd_prune = make_cache_prune(
    "layernorm_linear_pair_bias_bwd", dtype_of=tensor_dtype_of("dout_ptr"),
    bucket_of=key_bucket_of("N", "NH"),
)


@triton.autotune(
    configs=configs, key=["N", "NH"], reset_to_zero=["dlnw_ptr", "dpw_ptr"],
    prune_configs_by={"early_config_prune": _layernorm_linear_pair_bias_bwd_prune},
)
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    row_mask, col_mask = rows < M, cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    offs = rows[:, None].to(tl.int64) * stride_m + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0)
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
    xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
    lnw = tl.load(lnw_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = xhat * lnw[None, :]
    if USE_DOT:
        hcols = tl.arange(0, NH)
        dout = tl.load(
            dout_ptr + rows[:, None] * NH + hcols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        pw = tl.load(
            pw_ptr + hcols[:, None] * N + cols[None, :],
            mask=col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        dy = tl.dot(dout, pw, allow_tf32=False)
        tl.atomic_add(
            dpw_ptr + hcols[:, None] * N + cols[None, :],
            tl.dot(tl.trans(dout), y, allow_tf32=False),
            mask=col_mask[None, :],
        )
    else:
        dy = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
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
    c1 = tl.sum(dxhat * xhat, axis=1) / N
    c2 = tl.sum(dxhat, axis=1) / N
    dx = (dxhat - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
    tl.store(dx_ptr + offs, dx, mask=mask)
    tl.atomic_add(dlnw_ptr + cols, tl.sum(dy * xhat, axis=0), mask=col_mask)


@torch.library.custom_op("miniworld_engine::layer_norm_linear_fwd", mutates_args=())
def _fwd_op(
    x: torch.Tensor, ln_weight: torch.Tensor, proj_weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    M, N = x2.shape
    nh = proj_weight.shape[0]
    out = torch.empty(M, nh, dtype=torch.float32, device=x.device)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
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
        BLOCK_N=triton.next_power_of_2(N),
    )
    return out.reshape(*x.shape[:-1], nh).to(x.dtype), mean, rstd


@_fwd_op.register_fake
def _(x, ln_weight, proj_weight, eps):
    rows = 1
    for s in x.shape[:-1]:
        rows *= s
    return (
        x.new_empty(*x.shape[:-1], proj_weight.shape[0]),
        x.new_empty(rows, dtype=torch.float32),
        x.new_empty(rows, dtype=torch.float32),
    )


@torch.library.custom_op("miniworld_engine::layer_norm_linear_bwd", mutates_args=())
def _bwd_op(
    dout: torch.Tensor,
    x2: torch.Tensor,
    ln_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x2.shape
    nh = proj_weight.shape[0]
    dout2 = dout.reshape(M, nh).contiguous().to(torch.float32)
    dx = torch.empty_like(x2)
    dlnw = torch.zeros(N, dtype=torch.float32, device=x2.device)
    dpw = torch.zeros(nh, N, dtype=torch.float32, device=x2.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
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
        BLOCK_N=triton.next_power_of_2(N),
    )
    return dx, dlnw.to(ln_weight.dtype), dpw.to(proj_weight.dtype)


@_bwd_op.register_fake
def _(dout, x2, ln_weight, proj_weight, mean, rstd):
    return (
        torch.empty_like(x2),
        torch.empty_like(ln_weight),
        torch.empty_like(proj_weight),
    )


def _setup_context(ctx, inputs, output):
    x, ln_weight, proj_weight, _eps = inputs
    _, mean, rstd = output
    ctx.save_for_backward(
        x.reshape(-1, x.shape[-1]).contiguous(), ln_weight, proj_weight, mean, rstd
    )
    ctx.xshape = x.shape


def _backward(ctx, grad_out, grad_mean, grad_rstd):
    x2, ln_weight, proj_weight, mean, rstd = ctx.saved_tensors
    dx, dlnw, dpw = _bwd_op(grad_out, x2, ln_weight, proj_weight, mean, rstd)
    return dx.reshape(ctx.xshape), dlnw, dpw, None


_fwd_op.register_autograd(_backward, setup_context=_setup_context)


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
    return _fwd_op(x, ln_weight, proj_weight, eps)[0]

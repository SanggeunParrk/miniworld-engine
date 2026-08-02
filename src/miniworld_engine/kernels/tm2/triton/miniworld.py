# vendored from team-gm origin/miniworld@7c3c67e : src/team_gm/modules/kernels/tm2.py
import os

import torch
import triton
import triton.language as tl
from einops import rearrange
from jaxtyping import Float

from miniworld_engine.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of
from miniworld_engine._typecheck import typecheck

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "tri_multi"

if AUTOTUNE:
    fwd_configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_K": k}, w, s)
        for m in [16, 32, 64]
        for k in [16, 32, 64]
        for w in [4, 8]
        for s in [1, 2, 3]
        if not m * k > 32 * 64
    ]
    bwd_configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_K": k}, w, s)
        for m in [16, 32, 64, 128]
        for k in [16, 32, 64, 128]
        for w in [4, 8]
        for s in [1, 2, 3]
        if not m * k > 32 * 64
    ]
else:
    fwd_configs = [
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, 4, 1),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, 4, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, 8, 2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, 4, 2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, 4, 1),
    ]
    bwd_configs = [
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, 4, 1),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, 4, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, 8, 2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, 4, 2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, 4, 1),
    ]


_tm2_miniworld_fwd_prune = make_cache_prune(
    "tm2_miniworld_fwd", dtype_of=tensor_dtype_of("x_gate_ptr"),
    bucket_of=key_bucket_of("GROUP_M", "d"),
)


@triton.autotune(configs=fwd_configs, key=["GROUP_M", "d"],
                 prune_configs_by={"early_config_prune": _tm2_miniworld_fwd_prune})
@triton.jit
def fused_sigmoid_gate2_fwd_kernel(
    x_gate_ptr,
    x_out_ptr,
    W_gate_ptr,
    W_out_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_d_out = tl.arange(0, N)

    A_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_gate_tile = tl.load(
            x_gate_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        x_out_tile = tl.load(
            x_out_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        W_gate_tile = tl.load(
            W_gate_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )
        W_out_tile = tl.load(
            W_out_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )
        A_tile += tl.dot(x_gate_tile, W_gate_tile, input_precision="ieee")
        B_tile += tl.dot(x_out_tile, W_out_tile, input_precision="ieee")

    g_tile = tl.sigmoid(A_tile)
    out_tile = g_tile * B_tile
    out_ptr_ = out_ptr + (offs_m[:, None] * N + offs_d_out[None, :])
    tl.store(out_ptr_, out_tile, mask=(offs_m[:, None] < M))


_tm2_miniworld_bwd_prune = make_cache_prune(
    "tm2_miniworld_bwd", dtype_of=tensor_dtype_of("x_gate_ptr"),
    bucket_of=key_bucket_of("GROUP_M", "d"),
)


@triton.autotune(configs=bwd_configs, key=["GROUP_M", "d"],
                 prune_configs_by={"early_config_prune": _tm2_miniworld_bwd_prune})
@triton.jit
def fused_sigmoid_gate2_bwd_kernel(
    x_gate_ptr,
    x_out_ptr,
    W_gate_ptr,
    W_out_ptr,
    grad_out_ptr,
    dA_ptr,
    dB_ptr,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_d_full = tl.arange(0, N)

    A_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_gate_tile = tl.load(
            x_gate_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        x_out_tile = tl.load(
            x_out_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        W_gate_tile = tl.load(
            W_gate_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        W_out_tile = tl.load(
            W_out_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        A_tile += tl.dot(x_gate_tile, W_gate_tile, input_precision="ieee")
        B_tile += tl.dot(x_out_tile, W_out_tile, input_precision="ieee")

    g_tile = tl.sigmoid(A_tile)
    grad_tile = tl.load(
        grad_out_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        mask=((offs_m[:, None] < M) & (offs_d_full[None, :] < N)),
        other=0.0,
    )
    dB_tile = grad_tile * g_tile
    dA_tile = dB_tile * (B_tile * (1.0 - g_tile))

    tl.store(
        dA_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dA_tile,
        mask=((offs_m[:, None] < M) & (offs_d_full[None, :] < N)),
    )
    tl.store(
        dB_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dB_tile,
        mask=((offs_m[:, None] < M) & (offs_d_full[None, :] < N)),
    )


def get_seq_group(length: int) -> int:
    """Get sequence group based on length."""
    GROUP_LENGTHS = [32 * 32, 64 * 64, 128 * 128, 256 * 256]
    for group_idx, group_length in enumerate(GROUP_LENGTHS):
        if length <= group_length:
            return group_idx
    return len(GROUP_LENGTHS) - 1


class TritonTM2Function(torch.autograd.Function):
    @typecheck
    @staticmethod
    def forward(
        ctx,
        x: Float[torch.Tensor, "* d"],
        x_out_normalized: Float[torch.Tensor, "* d"],
        gate_weight: Float[torch.Tensor, "d d"],
        out_weight: Float[torch.Tensor, "d d"],
    ) -> Float[torch.Tensor, "* d"]:
        original_shape = x.shape
        op_dtype = x.dtype
        x = rearrange(x, "... d -> (...) d").contiguous()
        y = rearrange(x_out_normalized, "... d -> (...) d").contiguous().contiguous()
        M, N = y.shape

        gate_weight = gate_weight.contiguous().to(op_dtype)
        out_weight = out_weight.contiguous().to(op_dtype)
        out = torch.empty_like(x)

        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        fused_sigmoid_gate2_fwd_kernel[grid](
            x,
            y,
            gate_weight,
            out_weight,
            out,
            M,
            N,
            GROUP_M=get_seq_group(M),
        )

        ctx.save_for_backward(
            x.to(torch.bfloat16),
            y.to(torch.bfloat16),
            gate_weight,
            out_weight,
        )
        ctx.original_shape = original_shape
        ctx.op_dtype = op_dtype

        return out.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # pyright: ignore[reportIncompatibleMethodOverride]
        x, y, gate_weight, out_weight = ctx.saved_tensors
        op_dtype = ctx.op_dtype
        x = x.to(op_dtype)
        y = y.to(op_dtype)
        gate_weight = gate_weight.to(op_dtype)
        out_weight = out_weight.to(op_dtype)
        original_shape = ctx.original_shape
        M, N = x.shape

        grad_out = rearrange(grad_out, "... d -> (...) d").contiguous().to(op_dtype)
        dA = torch.empty_like(x)
        dB = torch.empty_like(x)

        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        fused_sigmoid_gate2_bwd_kernel[grid](
            x,
            y,
            gate_weight,
            out_weight,
            grad_out,
            dA,
            dB,
            M,
            N,
            GROUP_M=get_seq_group(M),
        )
        dx = dA @ gate_weight.T
        dy = dB @ out_weight.T
        dW_gate = torch.matmul(x.T, dA)
        dW_out = torch.matmul(y.T, dB)

        dx = dx.reshape(original_shape)
        dy = dy.reshape(original_shape)
        return dx, dy, dW_gate, dW_out


triton_tm2 = TritonTM2Function.apply

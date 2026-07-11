# vendored from team-gm origin/miniworld@7c3c67e : src/team_gm/modules/kernels/transition.py
import os

import torch
import triton
import triton.language as tl
from einops import rearrange
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "transition"

if AUTOTUNE:
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_K": k}, w, s)
        for m in [16, 32, 64]
        for k in [16, 32, 64]
        for w in [4, 8, 16]
        for s in [1, 2, 3]
        if not m * k > 32 * 64
    ]
else:
    configs = [
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, 8, 2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, 8, 3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, 8, 3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, 8, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, 8, 3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, 16, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, 8, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, 16, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, 8, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, 16, 2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 16}, 4, 2),
    ]


@triton.autotune(configs=configs, key=["GROUP_M", "n", "N"])
@triton.jit
def transition_fwd_kernel(
    x_ptr,
    W1_ptr,
    W2_ptr,
    out_ptr,
    M,
    n: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, n * N)

    A_tile = tl.zeros((BLOCK_M, n * N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, n * N), dtype=tl.float32)

    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        W1_tile = tl.load(
            W1_ptr + (offs_n[None, :] * N + offs_k[:, None]),
        )
        W2_tile = tl.load(
            W2_ptr + (offs_n[None, :] * N + offs_k[:, None]),
        )

        A_tile += tl.dot(x_tile, W1_tile, input_precision="ieee")
        B_tile += tl.dot(x_tile, W2_tile, input_precision="ieee")

    swish_A = A_tile * tl.sigmoid(A_tile)
    swish_AB = swish_A * B_tile

    out_ptr_ = out_ptr + (offs_m[:, None] * n * N + offs_n[None, :])
    tl.store(out_ptr_, swish_AB, mask=(offs_m[:, None] < M))


@triton.autotune(configs=configs, key=["GROUP_M", "n", "N"])
@triton.jit
def transition_bwd_kernel(
    x_ptr,
    W1_ptr,
    W2_ptr,
    grad_expand_ptr,
    dA_ptr,
    dB_ptr,
    M,
    n: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, n * N)

    A_tile = tl.zeros((BLOCK_M, n * N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, n * N), dtype=tl.float32)

    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )

        W1_tile = tl.load(W1_ptr + (offs_n[None, :] * N + offs_k[:, None]))
        W2_tile = tl.load(W2_ptr + (offs_n[None, :] * N + offs_k[:, None]))

        A_tile += tl.dot(x_tile, W1_tile, input_precision="ieee")
        B_tile += tl.dot(x_tile, W2_tile, input_precision="ieee")

    grad_expand_tile = tl.load(
        grad_expand_ptr + (offs_m[:, None] * n * N + offs_n[None, :]),
        mask=(offs_m[:, None] < M),
        other=0.0,
    )

    sigmoid_A = tl.sigmoid(A_tile)
    swish_diff_A = sigmoid_A + A_tile * sigmoid_A * (1 - sigmoid_A)
    dA_tile = grad_expand_tile * B_tile * swish_diff_A
    dB_tile = A_tile * sigmoid_A * grad_expand_tile

    tl.store(
        dA_ptr + (offs_m[:, None] * n * N + offs_n[None, :]),
        dA_tile,
        mask=(offs_m[:, None] < M),
    )

    tl.store(
        dB_ptr + (offs_m[:, None] * n * N + offs_n[None, :]),
        dB_tile,
        mask=(offs_m[:, None] < M),
    )


def get_seq_group(length: int) -> int:
    """Get sequence group based on length."""
    GROUP_LENGTHS = [32 * 32, 64 * 64, 128 * 128, 256 * 256]
    for group_idx, group_length in enumerate(GROUP_LENGTHS):
        if length <= group_length:
            return group_idx
    return len(GROUP_LENGTHS) - 1


class TritonTransitionFunction(torch.autograd.Function):
    @typecheck
    @staticmethod
    def forward(
        ctx,
        x: Float[torch.Tensor, "... d"],
        expand_a_weight: Float[torch.Tensor, "nd d"],
        expand_b_weight: Float[torch.Tensor, "nd d"],
        squeeze_weight: Float[torch.Tensor, "d nd"],
        n: int,
    ) -> Float[torch.Tensor, "... d"]:
        orig_shape = x.shape
        op_dtype = x.dtype
        x = rearrange(x, "... d -> (...) d").contiguous()
        M, N = x.shape

        expand = torch.empty(M, n * N, dtype=op_dtype, device=x.device)
        expand_a_weight = expand_a_weight.contiguous().to(op_dtype)
        expand_b_weight = expand_b_weight.contiguous().to(op_dtype)
        squeeze_weight = squeeze_weight.contiguous().to(op_dtype)

        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        transition_fwd_kernel[grid](
            x,
            expand_a_weight,
            expand_b_weight,
            expand,
            M,
            n,
            N,
            GROUP_M=get_seq_group(M),
        )

        ctx.save_for_backward(
            x.to(torch.bfloat16),
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
        )
        ctx.n = n
        ctx.op_dtype = op_dtype

        output = torch.matmul(expand, squeeze_weight.T)
        return output.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyright: ignore[reportIncompatibleMethodOverride]
        (
            x,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
        ) = ctx.saved_tensors
        op_dtype = ctx.op_dtype
        x = x.to(op_dtype)
        grad_output = grad_output.to(op_dtype)
        expand_a_weight = expand_a_weight.to(op_dtype)
        expand_b_weight = expand_b_weight.to(op_dtype)
        squeeze_weight = squeeze_weight.to(op_dtype)
        n = ctx.n
        M, N = x.shape

        expand = torch.empty(M, n * N, dtype=op_dtype, device=x.device)

        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        transition_fwd_kernel[grid](
            x,
            expand_a_weight,
            expand_b_weight,
            expand,
            M,
            n,
            N,
            GROUP_M=get_seq_group(M),
        )

        orig_shape = grad_output.shape
        grad_output = rearrange(grad_output, "... d -> (...) d").contiguous()
        grad_expand = torch.matmul(grad_output, squeeze_weight)
        grad_squeeze_weight = torch.matmul(grad_output.T, expand)
        dA = torch.empty(M, n * N, dtype=op_dtype, device=x.device)
        dB = torch.empty(M, n * N, dtype=op_dtype, device=x.device)

        transition_bwd_kernel[grid](
            x,
            expand_a_weight,
            expand_b_weight,
            grad_expand,
            dA,
            dB,
            M,
            n,
            N,
            GROUP_M=get_seq_group(M),
        )

        grad_a_weight = torch.matmul(dA.T, x)
        grad_b_weight = torch.matmul(dB.T, x)
        dx = torch.matmul(dA, expand_a_weight) + torch.matmul(dB, expand_b_weight)
        dx = dx.reshape(orig_shape)

        return dx.float(), grad_a_weight.float(), grad_b_weight.float(), grad_squeeze_weight.float(), None


triton_transition = TritonTransitionFunction.apply

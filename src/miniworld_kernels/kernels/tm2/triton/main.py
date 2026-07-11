# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/tm2.py
import os

import torch
import triton
import triton.language as tl
from einops import rearrange
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "tri_multi"

fwd_configs = [
    triton.Config({"BLOCK_K": 32, "BLOCK_M": 64, "BLOCK_N": 128}, 8, 3),  # 128 at H100
    triton.Config(
        {"BLOCK_K": 32, "BLOCK_M": 64, "BLOCK_N": 128}, 4, 3
    ),  # 256, 384 at H100
]


bwd_configs = [
    triton.Config({"BLOCK_K": 32, "BLOCK_M": 64, "BLOCK_N": 64}, 4, 3),  # 128 at H100
    triton.Config({"BLOCK_K": 64, "BLOCK_M": 32, "BLOCK_N": 64}, 4, 2),  # 128 at H100
    triton.Config({"BLOCK_K": 32, "BLOCK_M": 64, "BLOCK_N": 64}, 8, 3),  # 128 at H100
]


@triton.autotune(configs=fwd_configs, key=["GROUP_M", "N"])
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
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # int64 M-index: offs_m*N (M=B*L*L) overflows int32 at large logical L.
    offs_m = row_start + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    A_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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
            W_gate_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )
        W_out_tile = tl.load(
            W_out_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )
        A_tile += tl.dot(x_gate_tile, W_gate_tile)
        B_tile += tl.dot(x_out_tile, W_out_tile)

    g_tile = tl.sigmoid(A_tile)
    out_tile = g_tile * B_tile
    tl.store(
        out_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        out_tile.to(out_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
    )


@triton.autotune(configs=bwd_configs, key=["GROUP_M", "N"])
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
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # int64 M-index: offs_m*N (M=B*L*L) overflows int32 at large logical L.
    offs_m = row_start + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    A_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
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
            W_gate_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )
        W_out_tile = tl.load(
            W_out_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )
        A_tile += tl.dot(x_gate_tile, W_gate_tile)
        B_tile += tl.dot(x_out_tile, W_out_tile)

    g_tile = tl.sigmoid(A_tile)
    grad_tile = tl.load(
        grad_out_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
        other=0.0,
    )
    dB_tile = grad_tile * g_tile
    dA_tile = dB_tile * (B_tile * (1.0 - g_tile))

    tl.store(
        dA_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        dA_tile.to(dA_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
    )
    tl.store(
        dB_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        dB_tile.to(dB_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
    )


def get_seq_group(length: int) -> int:
    """Get sequence group based on length."""
    GROUP_LENGTHS = [32 * 32, 64 * 64, 128 * 128, 256 * 256, 384 * 384]
    for group_idx, group_length in enumerate(GROUP_LENGTHS):
        if length <= group_length:
            return group_idx
    return len(GROUP_LENGTHS) - 1


class TritonTM2Function(torch.autograd.Function):
    @typecheck
    @staticmethod
    @torch.compiler.disable()
    def forward(
        ctx,
        x: Float[torch.Tensor, "* d"],
        x_out_normalized: Float[torch.Tensor, "* d"],
        gate_weight: Float[torch.Tensor, "d d"],
        out_weight: Float[torch.Tensor, "d d"],
    ) -> Float[torch.Tensor, "* d"]:
        original_shape = x.shape
        x = rearrange(x, "... d -> (...) d").contiguous()
        y = rearrange(x_out_normalized, "... d -> (...) d").contiguous()
        M, N = y.shape

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            x = x.to(dtype)
            y = y.to(dtype)
            gate_weight = gate_weight.to(dtype)
            out_weight = out_weight.to(dtype)

        gate_weight = gate_weight.contiguous()
        out_weight = out_weight.contiguous()
        out = torch.empty_like(x)

        grid = lambda META: [
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        ]
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

        ctx.save_for_backward(x, y, gate_weight, out_weight)
        ctx.original_shape = original_shape

        return out.reshape(original_shape)

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, grad_out: torch.Tensor):
        x, y, gate_weight, out_weight = ctx.saved_tensors
        original_shape = ctx.original_shape
        M, N = x.shape

        if grad_out.dtype != x.dtype:
            grad_out = grad_out.to(x.dtype)

        grad_out = rearrange(grad_out, "... d -> (...) d").contiguous()
        dA = torch.empty_like(x)
        dB = torch.empty_like(x)

        grid = lambda META: [
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        ]
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

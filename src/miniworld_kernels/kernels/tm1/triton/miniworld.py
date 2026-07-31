# vendored from team-gm origin/miniworld@7c3c67e : src/team_gm/modules/kernels/tm1.py
import os

import torch
import triton
import triton.language as tl
from einops import rearrange
from jaxtyping import Float

from miniworld_kernels.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of
from miniworld_kernels._typecheck import typecheck

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "tri_multi"

if AUTOTUNE:
    fwd_configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_K": k}, w, s)
        for m in [16, 32, 64]
        for k in [16, 32, 64]
        for w in [4, 8]
        for s in [2, 3]
        if not m * k > 32 * 64
    ]
    bwd_configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_K": k}, w, s)
        for m in [16, 32, 64, 128]
        for k in [16, 32, 64, 128]
        for w in [4, 8]
        for s in [2, 3]
        if not m * k > 32 * 64
    ]
else:
    fwd_configs = [
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, 4, 1),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, 4, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, 4, 1),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, 4, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, 4, 1),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 16}, 4, 1),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, 4, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, 4, 2),
    ]
    bwd_configs = [
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, 4, 1),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, 4, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, 4, 1),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, 4, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, 4, 2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, 4, 2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, 4, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, 4, 2),
    ]


_tm1_miniworld_fwd_prune = make_cache_prune(
    "tm1_miniworld_fwd", dtype_of=tensor_dtype_of("x_ptr"),
    bucket_of=key_bucket_of("GROUP_M", "N"),
)


@triton.autotune(configs=fwd_configs, key=["GROUP_M", "N"],
                 prune_configs_by={"early_config_prune": _tm1_miniworld_fwd_prune})
@triton.jit
def fused_sigmoid_gate_fwd_kernel(
    x_ptr,
    WLg_ptr,
    WL_ptr,
    WRg_ptr,
    WR_ptr,
    left_ptr,
    right_ptr,
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

    LA_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    LB_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    RA_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    RB_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )

        WLg_tile = tl.load(
            WLg_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )

        WL_tile = tl.load(
            WL_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )

        WRg_tile = tl.load(
            WRg_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )

        WR_tile = tl.load(
            WR_ptr + (offs_k[:, None] * N + offs_d_out[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_out[None, :] < N)),
            other=0.0,
        )

        LA_tile += tl.dot(x_tile, WLg_tile, input_precision="ieee")
        LB_tile += tl.dot(x_tile, WL_tile, input_precision="ieee")
        RA_tile += tl.dot(x_tile, WRg_tile, input_precision="ieee")
        RB_tile += tl.dot(x_tile, WR_tile, input_precision="ieee")

    Lg_tile = tl.sigmoid(LA_tile)
    left_tile = Lg_tile * LB_tile
    Rg_tile = tl.sigmoid(RA_tile)
    right_tile = Rg_tile * RB_tile

    tl.store(
        left_ptr + (offs_m[:, None] * N + offs_d_out[None, :]),
        left_tile,
        mask=(offs_m[:, None] < M),
    )
    tl.store(
        right_ptr + (offs_m[:, None] * N + offs_d_out[None, :]),
        right_tile,
        mask=(offs_m[:, None] < M),
    )


_tm1_miniworld_bwd_prune = make_cache_prune(
    "tm1_miniworld_bwd", dtype_of=tensor_dtype_of("x_ptr"),
    bucket_of=key_bucket_of("GROUP_M", "d"),
)


@triton.autotune(configs=bwd_configs, key=["GROUP_M", "d"],
                 prune_configs_by={"early_config_prune": _tm1_miniworld_bwd_prune})
@triton.jit
def fused_sigmoid_gate_bwd_kernel(
    x_ptr,
    WLg_ptr,
    WL_ptr,
    WRg_ptr,
    WR_ptr,
    grad_left_out_ptr,
    grad_right_out_ptr,
    dLA_ptr,
    dLB_ptr,
    dRA_ptr,
    dRB_ptr,
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

    LA_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    LB_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    RA_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    RB_tile = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        WLg_tile = tl.load(
            WLg_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        WL_tile = tl.load(
            WL_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        WRg_tile = tl.load(
            WRg_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        WR_tile = tl.load(
            WR_ptr + (offs_k[:, None] * N + offs_d_full[None, :]),
            mask=((offs_k[:, None] < N) & (offs_d_full[None, :] < N)),
            other=0.0,
        )
        LA_tile += tl.dot(x_tile, WLg_tile, input_precision="ieee")
        LB_tile += tl.dot(x_tile, WL_tile, input_precision="ieee")
        RA_tile += tl.dot(x_tile, WRg_tile, input_precision="ieee")
        RB_tile += tl.dot(x_tile, WR_tile, input_precision="ieee")

    Lg_tile = tl.sigmoid(LA_tile)
    Rg_tile = tl.sigmoid(RA_tile)
    grad_left_tile = tl.load(
        grad_left_out_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        mask=(offs_m[:, None] < M),
    )
    grad_right_tile = tl.load(
        grad_right_out_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        mask=(offs_m[:, None] < M),
    )
    dLB_tile = grad_left_tile * Lg_tile
    dRB_tile = grad_right_tile * Rg_tile
    dLA_tile = dLB_tile * LB_tile * (1.0 - Lg_tile)
    dRA_tile = dRB_tile * RB_tile * (1.0 - Rg_tile)

    tl.store(
        dLA_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dLA_tile,
        mask=(offs_m[:, None] < M),
    )
    tl.store(
        dLB_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dLB_tile,
        mask=(offs_m[:, None] < M),
    )
    tl.store(
        dRA_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dRA_tile,
        mask=(offs_m[:, None] < M),
    )
    tl.store(
        dRB_ptr + (offs_m[:, None] * N + offs_d_full[None, :]),
        dRB_tile,
        mask=(offs_m[:, None] < M),
    )


def get_seq_group(length: int) -> int:
    """Get sequence group based on length."""
    GROUP_LENGTHS = [32 * 32, 64 * 64, 128 * 128, 256 * 256]
    for group_idx, group_length in enumerate(GROUP_LENGTHS):
        if length <= group_length:
            return group_idx
    return len(GROUP_LENGTHS) - 1


class TritonTM1Function(torch.autograd.Function):
    @typecheck
    @staticmethod
    def forward(
        ctx,
        x: Float[torch.Tensor, "* d"],
        left_weight: Float[torch.Tensor, "d d"],
        left_gate_weight: Float[torch.Tensor, "d d"],
        right_weight: Float[torch.Tensor, "d d"],
        right_gate_weight: Float[torch.Tensor, "d d"],
    ) -> tuple[
        Float[torch.Tensor, "* d"],
        Float[torch.Tensor, "* d"],
    ]:
        original_shape = x.shape
        x = rearrange(x, "... d -> (...) d").contiguous()
        op_dtype = x.dtype
        M, N = x.shape

        left_weight = left_weight.contiguous().to(op_dtype)
        left_gate_weight = left_gate_weight.contiguous().to(op_dtype)
        right_weight = right_weight.contiguous().to(op_dtype)
        right_gate_weight = right_gate_weight.contiguous().to(op_dtype)
        left = torch.empty_like(x)
        right = torch.empty_like(x)

        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        fused_sigmoid_gate_fwd_kernel[grid](
            x,
            left_gate_weight,
            left_weight,
            right_gate_weight,
            right_weight,
            left,
            right,
            M,
            N,
            GROUP_M=get_seq_group(M),
        )

        ctx.save_for_backward(
            x.to(torch.bfloat16),
            left_weight,
            left_gate_weight,
            right_weight,
            right_gate_weight,
        )
        ctx.original_shape = original_shape
        ctx.op_dtype = op_dtype

        left = left.reshape(original_shape)
        right = right.reshape(original_shape)
        return left, right

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx,
        grad_out_left: torch.Tensor,
        grad_out_right: torch.Tensor,
    ):
        x, left_weight, left_gate_weight, right_weight, right_gate_weight = (
            ctx.saved_tensors
        )
        op_dtype = ctx.op_dtype
        x = x.to(op_dtype)
        left_weight = left_weight.to(op_dtype)
        left_gate_weight = left_gate_weight.to(op_dtype)
        right_weight = right_weight.to(op_dtype)
        right_gate_weight = right_gate_weight.to(op_dtype)
        original_shape = ctx.original_shape
        M, N = x.shape

        grad_out_left = rearrange(grad_out_left, "... d -> (...) d").contiguous().to(op_dtype)
        grad_out_right = rearrange(grad_out_right, "... d -> (...) d").contiguous().to(op_dtype)

        dLA = torch.empty_like(x)
        dLB = torch.empty_like(x)
        dRA = torch.empty_like(x)
        dRB = torch.empty_like(x)

        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        fused_sigmoid_gate_bwd_kernel[grid](
            x,
            left_gate_weight,
            left_weight,
            right_gate_weight,
            right_weight,
            grad_out_left,
            grad_out_right,
            dLA,
            dLB,
            dRA,
            dRB,
            M,
            N,
            GROUP_M=get_seq_group(M),
        )

        dx = dLA @ left_gate_weight.T + dLB @ left_weight.T
        dx += dRA @ right_gate_weight.T + dRB @ right_weight.T
        dx = dx.reshape(original_shape)

        x_t = x.T
        dWLg = torch.matmul(x_t, dLA)
        dWL = torch.matmul(x_t, dLB)
        dWRg = torch.matmul(x_t, dRA)
        dWR = torch.matmul(x_t, dRB)
        return dx.float(), dWL.float(), dWLg.float(), dWR.float(), dWRg.float()


triton_tm1 = TritonTM1Function.apply

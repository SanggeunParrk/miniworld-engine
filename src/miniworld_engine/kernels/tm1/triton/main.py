
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/tm1.py

from miniworld_engine.autotune.configs import configs_for
import torch
import triton
import triton.language as tl

from einops import rearrange
from jaxtyping import Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.autotune import key_bucket_of, tensor_dtype_of
from miniworld_engine.autotune.shape_key import length_of, token_key


def get_seq_group(length) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_squared
    return bucket_squared(length)


# Real cross-product tile search (was: 2 fwd / 1 bwd pinned configs, only num_warps varied).
# BLOCK_M1 (grid M-tile), BLOCK_N (grid N-output tile) and BLOCK_K (contraction-loop tile) are
# all genuine tunable tiles of the `for k_start in range(0, N, BLOCK_K)` GEMM. The smem prune
# below drops configs whose pipelined tiles overflow device shared memory before compile.







@triton.autotune(configs=configs_for("trimul_gemm_gate_triton"), key=['shape_key', 'N'])
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
    BLOCK_M1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    shape_key,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    row_start = pid_m * BLOCK_M1
    col_start = pid_n * BLOCK_N

    # int64 M-index: offs_m*N (M=B*L*L) overflows int32 at large logical L; weight
    # offsets (offs_k*N, offs_n) stay int32 (bounded by K/N).
    offs_m = row_start + tl.arange(0, BLOCK_M1).to(tl.int64)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    LA_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    LB_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    RA_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    RB_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )

        WLg_tile = tl.load(
            WLg_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )

        WL_tile = tl.load(
            WL_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )

        WRg_tile = tl.load(
            WRg_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )

        WR_tile = tl.load(
            WR_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )

        LA_tile += tl.dot(x_tile, WLg_tile)
        LB_tile += tl.dot(x_tile, WL_tile)
        RA_tile += tl.dot(x_tile, WRg_tile)
        RB_tile += tl.dot(x_tile, WR_tile)

    Lg_tile = tl.sigmoid(LA_tile)
    left_tile = Lg_tile * LB_tile
    Rg_tile = tl.sigmoid(RA_tile)
    right_tile = Rg_tile * RB_tile

    tl.store(
        left_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        left_tile.to(left_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
    )
    tl.store(
        right_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        right_tile.to(right_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
    )




@triton.autotune(configs=configs_for("trimul_bwd_gate_recompute_triton"), key=['shape_key', 'N'])
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
    BLOCK_M1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    shape_key,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    row_start = pid_m * BLOCK_M1
    col_start = pid_n * BLOCK_N

    # int64 M-index: offs_m*N (M=B*L*L) overflows int32 at large logical L; weight
    # offsets (offs_k*N, offs_n) stay int32 (bounded by K/N).
    offs_m = row_start + tl.arange(0, BLOCK_M1).to(tl.int64)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    LA_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    LB_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    RA_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    RB_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        WLg_tile = tl.load(
            WLg_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )
        WL_tile = tl.load(
            WL_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )
        WRg_tile = tl.load(
            WRg_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )
        WR_tile = tl.load(
            WR_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=((offs_k[:, None] < N) & (offs_n[None, :] < N)),
            other=0.0,
        )
        LA_tile += tl.dot(x_tile, WLg_tile)
        LB_tile += tl.dot(x_tile, WL_tile)
        RA_tile += tl.dot(x_tile, WRg_tile)
        RB_tile += tl.dot(x_tile, WR_tile)

    Lg_tile = tl.sigmoid(LA_tile)
    Rg_tile = tl.sigmoid(RA_tile)
    grad_left_tile = tl.load(
        grad_left_out_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
        other=0.0,
    )
    grad_right_tile = tl.load(
        grad_right_out_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
        other=0.0,
    )
    dLB_tile = grad_left_tile * Lg_tile
    dRB_tile = grad_right_tile * Rg_tile
    dLA_tile = dLB_tile * LB_tile * (1.0 - Lg_tile)
    dRA_tile = dRB_tile * RB_tile * (1.0 - Rg_tile)

    tl.store(
        dLA_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        dLA_tile.to(dLA_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
    )
    tl.store(
        dLB_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        dLB_tile.to(dLB_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
    )
    tl.store(
        dRA_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        dRA_tile.to(dRA_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
    )
    tl.store(
        dRB_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        dRB_tile.to(dRB_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < N)),
    )


class TritonTM1Function(torch.autograd.Function):
    @typecheck
    @staticmethod
    @opaque()
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
        M, N = x.shape

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            x = x.to(dtype)
            left_weight = left_weight.to(dtype)
            left_gate_weight = left_gate_weight.to(dtype)
            right_weight = right_weight.to(dtype)
            right_gate_weight = right_gate_weight.to(dtype)

        left_weight = left_weight.contiguous()
        left_gate_weight = left_gate_weight.contiguous()
        right_weight = right_weight.contiguous()
        right_gate_weight = right_gate_weight.contiguous()
        left = torch.empty_like(x)
        right = torch.empty_like(x)

        grid = lambda META: [
            triton.cdiv(M, META["BLOCK_M1"]) * triton.cdiv(N, META["BLOCK_N"]),
        ]
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
            shape_key=token_key(length_of(original_shape)),
        )

        ctx.save_for_backward(
            x,
            left_weight,
            left_gate_weight,
            right_weight,
            right_gate_weight,
        )
        ctx.original_shape = original_shape

        left = left.reshape(original_shape)
        right = right.reshape(original_shape)
        return left, right

    @staticmethod
    @opaque()
    def backward(
        ctx,
        grad_out_left: torch.Tensor,
        grad_out_right: torch.Tensor,
    ):
        x, left_weight, left_gate_weight, right_weight, right_gate_weight = (
            ctx.saved_tensors
        )
        original_shape = ctx.original_shape
        M, N = x.shape

        if grad_out_left.dtype != x.dtype:
            grad_out_left = grad_out_left.to(x.dtype)
        if grad_out_right.dtype != x.dtype:
            grad_out_right = grad_out_right.to(x.dtype)

        grad_out_left = rearrange(grad_out_left, "... d -> (...) d").contiguous()
        grad_out_right = rearrange(grad_out_right, "... d -> (...) d").contiguous()

        dLA = torch.empty_like(x)
        dLB = torch.empty_like(x)
        dRA = torch.empty_like(x)
        dRB = torch.empty_like(x)

        grid = lambda META: [
            triton.cdiv(M, META["BLOCK_M1"]) * triton.cdiv(N, META["BLOCK_N"]),
        ]
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
            shape_key=token_key(length_of(original_shape)),
        )

        dx = dLA @ left_gate_weight.T + dLB @ left_weight.T
        dx += dRA @ right_gate_weight.T + dRB @ right_weight.T
        dx = dx.reshape(original_shape)

        x_t = x.T
        dWLg = torch.matmul(x_t, dLA)
        dWL = torch.matmul(x_t, dLB)
        dWRg = torch.matmul(x_t, dRA)
        dWR = torch.matmul(x_t, dRB)
        return dx, dWL, dWLg, dWR, dWRg


triton_tm1 = TritonTM1Function.apply

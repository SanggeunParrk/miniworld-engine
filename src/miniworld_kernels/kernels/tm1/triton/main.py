# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/tm1.py
import os

import torch
import triton
import triton.language as tl
from einops import rearrange
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck
from miniworld_kernels.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "tri_multi"


def get_seq_group(length: int) -> int:
    """Get sequence group based on length."""
    GROUP_LENGTHS = [32 * 32, 64 * 64, 128 * 128, 256 * 256, 384 * 384]
    for group_idx, group_length in enumerate(GROUP_LENGTHS):
        if length <= group_length:
            return group_idx
    return len(GROUP_LENGTHS) - 1


# Real cross-product tile search (was: 2 fwd / 1 bwd pinned configs, only num_warps varied).
# BLOCK_M (grid M-tile), BLOCK_N (grid N-output tile) and BLOCK_K (contraction-loop tile) are
# all genuine tunable tiles of the `for k_start in range(0, N, BLOCK_K)` GEMM. The smem prune
# below drops configs whose pipelined tiles overflow device shared memory before compile.
fwd_configs = [
    triton.Config(
        {"BLOCK_M": block_m, "BLOCK_K": block_k, "BLOCK_N": block_n},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for block_m in [64, 128]
    for block_n in [64, 128, 256]
    for block_k in [32, 64]
    for num_warps in [4, 8]
    for num_stages in [2, 3, 4]
]

bwd_configs = list(fwd_configs)


def _tm1_smem_early_prune(configs, named_args, **kwargs):  # noqa: ARG001
    """Drop configs whose shared-memory footprint exceeds the device limit BEFORE compile.

    Both the fwd and bwd fused sigmoid-gate GEMMs pipeline, per k-step, ONE x tile
    [BLOCK_M, BLOCK_K] and FOUR weight tiles [BLOCK_K, BLOCK_N] -- WLg, WL, WRg, WR (W = 4),
    across ``num_stages`` (bf16 = 2 B). The largest cross-product tiles (e.g. BLOCK_M=128,
    BLOCK_K=64, BLOCK_N=256, num_stages=4 ~= 576 KB) overflow A100's ~163 KB smem, so prune up
    front: Triton's bench-time OOM pruning is unsafe under CUDA-graph capture (fires mid-capture
    and poisons the stream). Device-aware via ``max_shared_mem``; sm_90/sm_100 keep more configs.
    """
    import triton as _triton

    try:
        limit = _triton.runtime.driver.active.utils.get_device_properties(
            torch.cuda.current_device(),
        )["max_shared_mem"]
    except Exception:  # noqa: BLE001 -- conservative sm100 budget
        limit = 227 * 1024

    def _smem(c):
        bm = c.kwargs["BLOCK_M"]
        bk = c.kwargs["BLOCK_K"]
        bn = c.kwargs["BLOCK_N"]
        return c.num_stages * (bm * bk + 4 * bk * bn) * 2

    kept = [c for c in configs if _smem(c) <= limit]
    return kept or [min(configs, key=_smem)]


_tm1_fwd_prune = make_cache_prune(
    "tm1_fwd", dtype_of=tensor_dtype_of("x_ptr"), bucket_of=key_bucket_of("GROUP_M", "N"),
    base_prune=_tm1_smem_early_prune,
)


@triton.autotune(
    configs=fwd_configs, key=["GROUP_M", "N"],
    prune_configs_by={"early_config_prune": _tm1_fwd_prune},
)
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
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # int64 M-index: offs_m*N (M=B*L*L) overflows int32 at large logical L; weight
    # offsets (offs_k*N, offs_n) stay int32 (bounded by K/N).
    offs_m = row_start + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    LA_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    LB_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    RA_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    RB_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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


_tm1_bwd_prune = make_cache_prune(
    "tm1_bwd", dtype_of=tensor_dtype_of("x_ptr"), bucket_of=key_bucket_of("GROUP_M", "N"),
    base_prune=_tm1_smem_early_prune,
)


@triton.autotune(
    configs=bwd_configs, key=["GROUP_M", "N"],
    prune_configs_by={"early_config_prune": _tm1_bwd_prune},
)
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
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # int64 M-index: offs_m*N (M=B*L*L) overflows int32 at large logical L; weight
    # offsets (offs_k*N, offs_n) stay int32 (bounded by K/N).
    offs_m = row_start + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    LA_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    LB_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    RA_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    RB_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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
    @torch.compiler.disable()
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
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
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
            GROUP_M=get_seq_group(M),
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
    @torch.compiler.disable()
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
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
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
        return dx, dWL, dWLg, dWR, dWRg


triton_tm1 = TritonTM1Function.apply

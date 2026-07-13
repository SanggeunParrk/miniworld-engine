# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/transition.py
import os

import torch
import triton
import triton.language as tl
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "transition"


def get_seq_group(length: int) -> int:
    """Get sequence group based on length."""
    GROUP_LENGTHS = [
        32 * 32,
        64 * 64,
        128 * 128,
        256 * 256,
        384 * 384,
        48 * 128,
        48 * 256,
        48 * 384,
        48 * 512,
        48 * 1024,
        48 * 2048,
        48 * 3072,
        48 * 4096,
    ]
    GROUP_LENGTHS = sorted(GROUP_LENGTHS)
    for group_idx, group_length in enumerate(GROUP_LENGTHS):
        if length <= group_length:
            return group_idx
    return len(GROUP_LENGTHS) - 1


if AUTOTUNE or True:
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_K": k, "BLOCK_N": n}, w, s)
        for m in [32, 64, 128, 256]
        for k in [16, 32, 64]
        for n in [128]
        for w in [4, 8]
        for s in [2, 3, 4, 5]
        # if not m * k > 32 * 64
    ]
else:
    configs = [
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32, "BLOCK_N": 128}, 8, 2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32, "BLOCK_N": 128}, 8, 3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16, "BLOCK_N": 128}, 8, 3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_N": 128}, 8, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_N": 128}, 8, 3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_N": 256}, 16, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_N": 128}, 8, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_N": 128}, 16, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16, "BLOCK_N": 64}, 8, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16, "BLOCK_N": 64}, 16, 2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 16, "BLOCK_N": 64}, 4, 2),
    ]


def _smem_early_prune(configs, named_args, **kwargs):  # noqa: ARG001
    """Drop configs whose shared-memory footprint exceeds the device limit BEFORE compile.

    The split transition GEMM pipelines, per k-step, an x tile [BLOCK_M, BLOCK_K] and two
    weight tiles [BLOCK_K, BLOCK_N] (W1, W2) across ``num_stages`` (bf16 = 2 B). The largest
    configs (e.g. BLOCK_M=256, BLOCK_K=64, num_stages=5 ~= 320 KB) overflow A100's 163 KB.
    Triton's bench-time OOM pruning is unsafe under CUDA-graph capture (it fires mid-capture
    and poisons the stream), and this is now A100's DEFAULT large-d route (module routes
    pre-Hopper d>=256 here), so prune up front. Device-aware: sm_90/sm_100 keep their configs.
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
        return c.num_stages * (bm * bk + 2 * bk * bn) * 2

    kept = [c for c in configs if _smem(c) <= limit]
    return kept or [min(configs, key=_smem)]


@triton.autotune(
    configs=configs, key=["GROUP_M", "n", "N"],
    prune_configs_by={"early_config_prune": _smem_early_prune},
)
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
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pid_n = tl.cdiv(n * N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    A_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )
        W1_tile = tl.load(
            W1_ptr + (offs_n[None, :] * N + offs_k[:, None]),
            mask=((offs_n[None, :] < n * N) & (offs_k[:, None] < N)),
            other=0.0,
        )
        W2_tile = tl.load(
            W2_ptr + (offs_n[None, :] * N + offs_k[:, None]),
            mask=((offs_n[None, :] < n * N) & (offs_k[:, None] < N)),
            other=0.0,
        )

        A_tile += tl.dot(x_tile, W1_tile)
        B_tile += tl.dot(x_tile, W2_tile)

    swish_A = A_tile * tl.sigmoid(A_tile)
    swish_AB = swish_A * B_tile

    out_ptr_ = out_ptr + (offs_m[:, None] * n * N + offs_n[None, :])
    tl.store(
        out_ptr_,
        swish_AB.to(out_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < n * N)),
    )


@triton.autotune(
    configs=configs, key=["GROUP_M", "n", "N"],
    prune_configs_by={"early_config_prune": _smem_early_prune},
)
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
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pid_n = tl.cdiv(n * N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    A_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * N + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < N)),
            other=0.0,
        )

        W1_tile = tl.load(
            W1_ptr + (offs_n[None, :] * N + offs_k[:, None]),
            mask=((offs_n[None, :] < n * N) & (offs_k[:, None] < N)),
            other=0.0,
        )
        W2_tile = tl.load(
            W2_ptr + (offs_n[None, :] * N + offs_k[:, None]),
            mask=((offs_n[None, :] < n * N) & (offs_k[:, None] < N)),
            other=0.0,
        )

        A_tile += tl.dot(x_tile, W1_tile)
        B_tile += tl.dot(x_tile, W2_tile)

    grad_expand_tile = tl.load(
        grad_expand_ptr + (offs_m[:, None] * n * N + offs_n[None, :]),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < n * N)),
        other=0.0,
    )

    sigmoid_A = tl.sigmoid(A_tile)
    swish_diff_A = sigmoid_A + A_tile * sigmoid_A * (1 - sigmoid_A)
    dA_tile = grad_expand_tile * B_tile * swish_diff_A
    dB_tile = A_tile * sigmoid_A * grad_expand_tile

    tl.store(
        dA_ptr + (offs_m[:, None] * n * N + offs_n[None, :]),
        dA_tile.to(dA_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < n * N)),
    )

    tl.store(
        dB_ptr + (offs_m[:, None] * n * N + offs_n[None, :]),
        dB_tile.to(dB_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < n * N)),
    )


class TritonTransitionFunction(torch.autograd.Function):
    @typecheck
    @staticmethod
    @torch.compiler.disable()
    def forward(
        ctx,
        x: Float[torch.Tensor, "... d"],
        expand_a_weight: Float[torch.Tensor, "nd d"],
        expand_b_weight: Float[torch.Tensor, "nd d"],
        squeeze_weight: Float[torch.Tensor, "d nd"],
        n: int,
    ) -> Float[torch.Tensor, "... d"]:
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1]).contiguous()
        M, N = x.shape

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            x = x.to(dtype)
            expand_a_weight = expand_a_weight.to(dtype)
            expand_b_weight = expand_b_weight.to(dtype)
            squeeze_weight = squeeze_weight.to(dtype)

        expand = torch.empty(M, n * N, dtype=x.dtype, device=x.device)

        grid = lambda META: [
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(n * N, META["BLOCK_N"]),
        ]
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
            x,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
        )
        ctx.n = n

        output = torch.matmul(expand, squeeze_weight.T)
        return output.reshape(orig_shape)

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, grad_output: torch.Tensor):
        (
            x,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
        ) = ctx.saved_tensors
        n = ctx.n
        M, N = x.shape

        if grad_output.dtype != x.dtype:
            grad_output = grad_output.to(x.dtype)

        orig_shape = grad_output.shape
        grad_output = grad_output.view(-1, orig_shape[-1]).contiguous()
        expand = torch.empty(M, n * N, dtype=x.dtype, device=x.device)

        grid = lambda META: [
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(n * N, META["BLOCK_N"]),
        ]
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

        grad_expand = torch.matmul(grad_output, squeeze_weight)
        grad_squeeze_weight = torch.matmul(grad_output.T, expand)
        dA = torch.empty(M, n * N, dtype=x.dtype, device=x.device)
        dB = torch.empty(M, n * N, dtype=x.dtype, device=x.device)

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

        return dx, grad_a_weight, grad_b_weight, grad_squeeze_weight, None


triton_transition = TritonTransitionFunction.apply

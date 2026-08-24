
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/tm2.py

from miniworld_engine.autotune.configs import configs_for
import torch
import triton
import triton.language as tl

from einops import rearrange
from jaxtyping import Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.autotune.shape_key import length_of, token_key

# Real cross-product tile search (was: 2 fwd pinned / 3 bwd configs with BLOCK_N pinned to 64).
# BLOCK_M1 (grid M-tile), BLOCK_N (grid N-output tile) and BLOCK_K (contraction-loop tile) are
# all genuine tunable tiles of the `for k_start in range(0, N, BLOCK_K)` GEMM. The smem prune
# below drops configs whose pipelined tiles overflow device shared memory before compile.








@triton.autotune(configs=configs_for("trimul_outproj_gemm_gate_triton"), key=['shape_key', 'N'])
@triton.jit
def fused_sigmoid_gate2_fwd_kernel(
    x_gate_ptr,
    x_out_ptr,
    W_gate_ptr,
    W_out_ptr,
    out_ptr,
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

    # int64 M-index: offs_m*N (M=B*L*L) overflows int32 at large logical L.
    offs_m = row_start + tl.arange(0, BLOCK_M1).to(tl.int64)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    A_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)

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




@triton.autotune(configs=configs_for("trimul_outproj_bwd_gate_recompute_triton"), key=['shape_key', 'N'])
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

    # int64 M-index: offs_m*N (M=B*L*L) overflows int32 at large logical L.
    offs_m = row_start + tl.arange(0, BLOCK_M1).to(tl.int64)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    A_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
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


def get_seq_group(length) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_squared
    return bucket_squared(length)


def _tm2_fwd_fake(x, y, gate_weight, out_weight, shape_key):
    """`out`, shaped and typed like the flat `x`."""
    return torch.empty_like(x)


@opaque(fake=_tm2_fwd_fake, name="tm2_fwd")
def _tm2_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    gate_weight: torch.Tensor,
    out_weight: torch.Tensor,
    shape_key: int,
) -> torch.Tensor:
    """The fused gate+out projection -> ``out``, flat.

    Split out of ``TritonTM2Function.forward`` so the rearranges, the autocast casts and
    ``save_for_backward`` stay traceable -- see ``kernels._compile``.
    """
    M, N = y.shape
    out = torch.empty_like(x)
    grid = lambda META: [
        triton.cdiv(M, META["BLOCK_M1"]) * triton.cdiv(N, META["BLOCK_N"]),
    ]
    fused_sigmoid_gate2_fwd_kernel[grid](
        x,
        y,
        gate_weight,
        out_weight,
        out,
        M,
        N,
        shape_key=shape_key,
    )
    return out


def _tm2_bwd_fake(x, y, gate_weight, out_weight, grad_out, shape_key):
    """(dA, dB), both shaped and typed like the flat `x`."""
    return torch.empty_like(x), torch.empty_like(x)


@opaque(fake=_tm2_bwd_fake, name="tm2_bwd")
def _tm2_bwd(
    x: torch.Tensor,
    y: torch.Tensor,
    gate_weight: torch.Tensor,
    out_weight: torch.Tensor,
    grad_out: torch.Tensor,
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate backward -> ``(dA, dB)``; the four GEMMs that consume them stay in the caller."""
    M, N = x.shape
    dA = torch.empty_like(x)
    dB = torch.empty_like(x)
    grid = lambda META: [
        triton.cdiv(M, META["BLOCK_M1"]) * triton.cdiv(N, META["BLOCK_N"]),
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
        shape_key=shape_key,
    )
    return dA, dB


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
        x = rearrange(x, "... d -> (...) d").contiguous()
        y = rearrange(x_out_normalized, "... d -> (...) d").contiguous()

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            x = x.to(dtype)
            y = y.to(dtype)
            gate_weight = gate_weight.to(dtype)
            out_weight = out_weight.to(dtype)

        gate_weight = gate_weight.contiguous()
        out_weight = out_weight.contiguous()

        out = _tm2_fwd(x, y, gate_weight, out_weight, token_key(length_of(original_shape)))

        ctx.save_for_backward(x, y, gate_weight, out_weight)
        ctx.original_shape = original_shape

        return out.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, y, gate_weight, out_weight = ctx.saved_tensors
        original_shape = ctx.original_shape

        if grad_out.dtype != x.dtype:
            grad_out = grad_out.to(x.dtype)

        grad_out = rearrange(grad_out, "... d -> (...) d").contiguous()
        dA, dB = _tm2_bwd(x, y, gate_weight, out_weight, grad_out,
                          token_key(length_of(original_shape)))

        dx = dA @ gate_weight.T
        dy = dB @ out_weight.T
        dW_gate = torch.matmul(x.T, dA)
        dW_out = torch.matmul(y.T, dB)

        dx = dx.reshape(original_shape)
        dy = dy.reshape(original_shape)
        return dx, dy, dW_gate, dW_out


triton_tm2 = TritonTM2Function.apply

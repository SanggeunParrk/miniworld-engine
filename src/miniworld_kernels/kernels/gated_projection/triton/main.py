# vendored from team-gm origin/miniworld@7c3c67e : src/team_gm/modules/kernels/gated_projection.py
import os

import torch
import triton
import triton.language as tl
from einops import rearrange
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0") == "tri_attention"

if AUTOTUNE:
    configs = [
        triton.Config({"BLOCK_M": m}, w, s)
        for m in [16, 32, 64, 128, 256]
        for w in [4, 8]
        for s in [1, 2, 3, 4]
    ]
else:
    configs = [
        triton.Config({"BLOCK_M": 64}, 8, 4),
        triton.Config({"BLOCK_M": 64}, 4, 4),
        triton.Config({"BLOCK_M": 32}, 8, 3),
        triton.Config({"BLOCK_M": 32}, 8, 1),
        triton.Config({"BLOCK_M": 16}, 4, 4),
        triton.Config({"BLOCK_M": 16}, 8, 3),
    ]


@triton.autotune(configs=configs, key=["GROUP_M", "n_elements"])
@triton.jit
def sigmoid_gate_fwd_kernel(
    gate_ptr,
    rep_ptr,
    stride_gate,
    stride_rep,
    out_ptr,
    n_elements,
    R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    row = tl.program_id(0)
    offset_row = tl.arange(0, BLOCK_M) + row * BLOCK_M
    offset_col = tl.arange(0, BLOCK_N)
    row_mask = offset_row < n_elements
    col_mask = offset_col < R
    offset = offset_row[:, None] * stride_rep + offset_col[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    gate = tl.load(gate_ptr + offset, mask=mask).to(tl.float32)
    rep = tl.load(rep_ptr + offset, mask=mask)

    s = 1.0 + tl.math.exp2(-1.44269504 * gate)
    out_val = rep / s

    tl.store(out_ptr + offset, out_val, mask=mask)


@triton.autotune(configs=configs, key=["GROUP_M", "n_elements"])
@triton.jit
def sigmoid_gate_bwd_kernel(
    gate_ptr,
    rep_ptr,
    grad_out_ptr,
    dgate_ptr,
    drep_ptr,
    stride_gate,
    stride_rep,
    n_elements,
    R,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    row = tl.program_id(0)
    offset_row = tl.arange(0, BLOCK_M) + row * BLOCK_M
    offset_col = tl.arange(0, BLOCK_N)
    row_mask = offset_row < n_elements
    col_mask = offset_col < R
    offset = offset_row[:, None] * stride_rep + offset_col[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    gate = tl.load(gate_ptr + offset, mask=mask).to(tl.float32)
    rep = tl.load(rep_ptr + offset, mask=mask)
    grad_out = tl.load(grad_out_ptr + offset, mask=mask)

    s = 1.0 / (1.0 + tl.math.exp2(-1.44269504 * gate))

    dgate_val = grad_out * (rep * s * (1 - s))
    drep_val = grad_out * s

    tl.store(dgate_ptr + offset, dgate_val, mask=mask)
    tl.store(drep_ptr + offset, drep_val, mask=mask)


def get_seq_group(length: int) -> int:
    """Get sequence group based on length."""
    GROUP_LENGTHS = [32, 64, 128, 256, 512]
    for group_idx, group_length in enumerate(GROUP_LENGTHS):
        if length <= group_length:
            return group_idx
    return len(GROUP_LENGTHS) - 1


class TritonGatedProjectionFunction(torch.autograd.Function):
    @typecheck
    @staticmethod
    def forward(
        ctx,
        gate: Float[torch.Tensor, "* hd"],
        x: Float[torch.Tensor, "* hd"],
        out_weight: Float[torch.Tensor, "hd d"],
    ) -> Float[torch.Tensor, "* d"]:
        original_shape = x.shape
        gate = rearrange(gate, "... d -> (...) d").contiguous()
        x = rearrange(x, "... d -> (...) d").contiguous()
        op_dtype = x.dtype
        gate = gate.to(op_dtype)
        M, N = x.shape

        out = torch.empty_like(x)

        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        sigmoid_gate_fwd_kernel[grid](
            gate,
            x,
            gate.stride(0),
            x.stride(0),
            out,
            M,
            N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )

        ctx.save_for_backward(
            gate.to(torch.bfloat16),
            x.to(torch.bfloat16),
            out_weight,
        )
        ctx.original_shape = original_shape
        ctx.op_dtype = op_dtype

        out = torch.matmul(out, out_weight)
        return out.reshape(*original_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # pyright: ignore[reportIncompatibleMethodOverride]
        gate, x, out_weight = ctx.saved_tensors
        op_dtype = ctx.op_dtype
        gate = gate.to(op_dtype)
        x = x.to(op_dtype)
        grad_out = grad_out.to(op_dtype)
        out_weight = out_weight.to(op_dtype)
        original_shape = ctx.original_shape
        M, N = x.shape

        out = torch.empty_like(x)

        grid = lambda META: [triton.cdiv(M, META["BLOCK_M"])]
        sigmoid_gate_fwd_kernel[grid](
            gate,
            x,
            gate.stride(0),
            x.stride(0),
            out,
            M,
            N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )

        grad_out = rearrange(grad_out, "... W -> (...) W").contiguous()
        dW_out = torch.matmul(out.T, grad_out)
        grad_out = torch.matmul(grad_out, out_weight.T)

        dgate = torch.empty_like(gate)
        dx = torch.empty_like(x)

        sigmoid_gate_bwd_kernel[grid](
            gate,
            x,
            grad_out,
            dgate,
            dx,
            gate.stride(0),
            x.stride(0),
            M,
            N,
            BLOCK_N=triton.next_power_of_2(N),
            GROUP_M=get_seq_group(M),
        )

        dgate = dgate.reshape(original_shape)
        dx = dx.reshape(original_shape)
        return dgate.float(), dx.float(), dW_out.float()


triton_gated_projection = TritonGatedProjectionFunction.apply

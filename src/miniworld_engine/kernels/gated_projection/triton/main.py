# vendored from team-gm origin/miniworld@7c3c67e : src/team_gm/modules/kernels/gated_projection.py
from miniworld_engine.autotune.configs import configs_for
import os

import torch
from miniworld_engine import settings
import triton
import triton.language as tl

from einops import rearrange
from jaxtyping import Float

from miniworld_engine.autotune.shape_key import both_key, length_of
from miniworld_engine._typecheck import typecheck

AUTOTUNE = settings.current().autotunes("tri_attention")
# BOTH tile axes are searched. BLOCK_N used to be pinned at the launch site to
# next_power_of_2(R) — a whole-row register tile decided by the shape, not by measurement, and
# one that grows without bound as the hidden width grows. The R axis now loops in BLOCK_N tiles.




# AUTOTUNE KEY: ['shape_key', 'R'].
# `n_elements` is a MISNOMER: both launch sites pass the flattened ROW count M into it (it is only
# ever read as `offset_row < n_elements`), so keying it added the raw M back beside its own bucket
# and minted a full config sweep per distinct M. shape_key is that axis, bucketed -- and it is
# bucketed from L (both_key(length_of(original_shape))), NOT from M: M alone cannot say whether
# it came from L or L*L, which is what autotune/shape_key.py exists to fix.
# `R` (the launcher's N = hidden/projection width) IS a real config axis -- it is the extent of the
# BLOCK_K column loop -- and was absent from the key, so a new width recompiled (it is constexpr
# here) but silently reused the config tuned for a different width. It is a searched axis now.
@triton.autotune(configs=configs_for("gated_projection_gate_triton"),
                 key=['shape_key', 'R'])
@triton.jit
def sigmoid_gate_fwd_kernel(
    gate_ptr,
    rep_ptr,
    stride_gate,
    stride_rep,
    out_ptr,
    n_elements,
    R: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    shape_key,
):
    row = tl.program_id(0).to(tl.int64)
    offset_row = row * BLOCK_M1 + tl.arange(0, BLOCK_M1).to(tl.int64)
    row_mask = offset_row < n_elements

    for c0 in range(0, R, BLOCK_K):
        offset_col = c0 + tl.arange(0, BLOCK_K)
        col_mask = offset_col < R
        offset = offset_row[:, None] * stride_rep + offset_col[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        gate = tl.load(gate_ptr + offset, mask=mask).to(tl.float32)
        rep = tl.load(rep_ptr + offset, mask=mask)

        s = 1.0 + tl.math.exp2(-1.44269504 * gate)
        out_val = rep / s

        tl.store(out_ptr + offset, out_val, mask=mask)




# AUTOTUNE KEY: ['shape_key', 'R'] -- same reasoning as the forward: `n_elements` receives the raw
# row count M (shape_key is its bucket), and `R` (the column-loop extent, a plain runtime arg here)
# is the second real axis.
@triton.autotune(configs=configs_for("gated_projection_bwd_gate_triton"),
                 key=['shape_key', 'R'])
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
    BLOCK_M1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    shape_key,
):
    row = tl.program_id(0).to(tl.int64)
    offset_row = row * BLOCK_M1 + tl.arange(0, BLOCK_M1).to(tl.int64)
    row_mask = offset_row < n_elements

    for c0 in range(0, R, BLOCK_K):
        offset_col = c0 + tl.arange(0, BLOCK_K)
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


def get_seq_group(length) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_mixed
    return bucket_mixed(length)


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

        grid = lambda META: [triton.cdiv(M, META["BLOCK_M1"])]
        sigmoid_gate_fwd_kernel[grid](
            gate,
            x,
            gate.stride(0),
            x.stride(0),
            out,
            M,
            N,
            # L = original_shape[-2], captured BEFORE the rearrange to (M, hd) -- one rule
            # for pair (B, L, L, D) and token/atom (B, L, D). Never the row count M.
            shape_key=both_key(length_of(original_shape)),
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
    # `Function.backward(ctx, *grad_outputs)` in torch's stubs; this op has exactly one
    # output, so the concrete signature is narrower. Covered by `invalid-method-override`
    # being off in `[tool.ty.rules]`.
    def backward(ctx, grad_out: torch.Tensor):
        gate, x, out_weight = ctx.saved_tensors
        op_dtype = ctx.op_dtype
        gate = gate.to(op_dtype)
        x = x.to(op_dtype)
        grad_out = grad_out.to(op_dtype)
        out_weight = out_weight.to(op_dtype)
        original_shape = ctx.original_shape
        M, N = x.shape

        out = torch.empty_like(x)

        grid = lambda META: [triton.cdiv(M, META["BLOCK_M1"])]
        sigmoid_gate_fwd_kernel[grid](
            gate,
            x,
            gate.stride(0),
            x.stride(0),
            out,
            M,
            N,
            shape_key=both_key(length_of(original_shape)),
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
            shape_key=both_key(length_of(original_shape)),
        )

        dgate = dgate.reshape(original_shape)
        dx = dx.reshape(original_shape)
        return dgate.float(), dx.float(), dW_out.float()


triton_gated_projection = TritonGatedProjectionFunction.apply


# ── flat (1-D) form of the two kernels above ────────────────────────────────────────────
# Moved here from bias_only_attention/triton/gate_out.py. conditioned_transition/triton/
# training.py carried a bitwise-equal copy of each (.bench/direct.out); both files import
# these now. The tiled kernels above take (M, N, strides); these take one element count and
# assume every operand is contiguous.
@triton.autotune(configs=configs_for("gated_projection_gate_flat_triton"), key=['shape_key'])
@triton.jit
def _sigmul_fwd(g_ptr, o_ptr, a_ptr, n, BLOCK_E: tl.constexpr, shape_key):
    off = tl.program_id(0).to(tl.int64) * BLOCK_E + tl.arange(0, BLOCK_E)
    m = off < n
    g = tl.sigmoid(tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32))
    o = tl.load(o_ptr + off, mask=m, other=0.0).to(tl.float32)
    tl.store(a_ptr + off, (g * o).to(a_ptr.dtype.element_ty), mask=m)


@triton.autotune(configs=configs_for("gated_projection_bwd_gate_flat_triton"), key=['shape_key'])
@triton.jit
def _sigmul_bwd(da_ptr, g_ptr, o_ptr, dg_ptr, do_ptr, n, BLOCK_E: tl.constexpr, shape_key):
    off = tl.program_id(0).to(tl.int64) * BLOCK_E + tl.arange(0, BLOCK_E)
    m = off < n
    da = tl.load(da_ptr + off, mask=m, other=0.0).to(tl.float32)
    s = tl.sigmoid(tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32))
    o = tl.load(o_ptr + off, mask=m, other=0.0).to(tl.float32)
    tl.store(do_ptr + off, (da * s).to(do_ptr.dtype.element_ty), mask=m)
    tl.store(dg_ptr + off, (da * o * s * (1.0 - s)).to(dg_ptr.dtype.element_ty), mask=m)

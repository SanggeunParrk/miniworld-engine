
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/transition.py

from miniworld_engine.autotune.configs import configs_for
import torch
import triton
import triton.language as tl

from miniworld_engine.kernels._tiles import tile_grid, tile_order

from jaxtyping import Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.autotune.shape_key import both_key, length_of, pack, rows_of

def get_seq_group(length) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_mixed
    return bucket_mixed(length)






# Cache-narrowing prunes composed OVER the smem safety prune (see autotune package). Bucket on
# the autotune key (shape_key, ND, K); dtype from x (defaults bf16). Separate op ids for fwd/bwd
# since their best tiles differ. Miss/stale -> warn once + full grid.


@triton.autotune(configs=configs_for("transition_expand_swiglu_triton"), key=['shape_key'])
@triton.jit
def transition_fwd_kernel(
    x_ptr,
    W1_ptr,
    W2_ptr,
    out_ptr,
    M,
    K: tl.constexpr,
    ND: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    shape_key,
):
    # Visit order, tuned: see kernels/_tiles.py. This walked the COLUMNS first, which is
    # GROUP_M = 1 on that axis -- the opposite end from the kernels that launch a 2-D grid,
    # and just as fixed. Which end suits the shape is what the tuner now decides.
    pid_m, pid_n = tile_order(tl.program_id(0).to(tl.int64),
                              tl.cdiv(M, BLOCK_M1), tl.cdiv(ND, BLOCK_N), GROUP_M)

    row_start = pid_m * BLOCK_M1
    col_start = pid_n * BLOCK_N

    offs_m = row_start + tl.arange(0, BLOCK_M1)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    A_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * K + offs_k[None, :]),
            mask=((offs_m[:, None] < M) & (offs_k[None, :] < K)),
            other=0.0,
        )
        W1_tile = tl.load(
            W1_ptr + (offs_n[None, :] * K + offs_k[:, None]),
            mask=((offs_n[None, :] < ND) & (offs_k[:, None] < K)),
            other=0.0,
        )
        W2_tile = tl.load(
            W2_ptr + (offs_n[None, :] * K + offs_k[:, None]),
            mask=((offs_n[None, :] < ND) & (offs_k[:, None] < K)),
            other=0.0,
        )

        A_tile += tl.dot(x_tile, W1_tile)
        B_tile += tl.dot(x_tile, W2_tile)

    swish_A = A_tile * tl.sigmoid(A_tile)
    swish_AB = swish_A * B_tile

    out_ptr_ = out_ptr + (offs_m[:, None] * ND + offs_n[None, :])
    tl.store(
        out_ptr_,
        swish_AB.to(out_ptr.dtype.element_ty),
        mask=((offs_m[:, None] < M) & (offs_n[None, :] < ND)),
    )


def _expand_swiglu_fake(x, expand_a_weight, expand_b_weight, shape_key):
    """The expand activation, (M, ND) in ``x``'s dtype: the two SwiGLU halves are already
    multiplied together in the epilogue, so the width is ``ND`` and not ``2*ND``."""
    return x.new_empty((x.shape[0], expand_a_weight.shape[0]))


@opaque(fake=_expand_swiglu_fake, name="transition_expand_swiglu")
def _expand_swiglu(
    x: torch.Tensor,
    expand_a_weight: torch.Tensor,
    expand_b_weight: torch.Tensor,
    shape_key: int,
) -> torch.Tensor:
    """The launch, and only the launch: ``SwiGLU(x @ Wa^T, x @ Wb^T)`` -> ``(M, n*N)``.

    Split out of ``TritonTransitionFunction.forward`` so the flatten, the autocast casts, the
    squeeze GEMM and ``save_for_backward`` stay traceable and only this stays opaque -- see
    ``kernels._compile``. ``x`` arrives already 2-D and contiguous; ``shape_key`` is computed from
    the PRE-flatten shape by the caller, which is the only place that still knows it.
    """
    M, N = x.shape
    # ND comes from the WEIGHT, not from an expansion ratio. The kernel only ever needed the
    # expanded width; taking it as ``n * N`` made a caller state twice something the weight
    # already fixes, so an `n` that disagreed with the weight silently read the wrong extent --
    # and it barred any expansion the ratio cannot spell, which is exactly what an FFN whose
    # hidden size is rounded to a multiple of 256 has. `conditioned_transition`'s own
    # `_expand_swiglu` already derives it this way.
    nd = expand_a_weight.shape[0]
    expand = torch.empty(M, nd, dtype=x.dtype, device=x.device)
    grid = lambda META: tile_grid(M, nd, META["BLOCK_M1"], META["BLOCK_N"])  # noqa: E731
    transition_fwd_kernel[grid](
        x,
        expand_a_weight,
        expand_b_weight,
        expand,
        M,
        N,          # K -- the input width
        nd,         # ND -- the expanded width
        shape_key=pack(shape_key, K=N, ND=nd),
    )
    return expand


def swiglu_squeeze_backward(
    x: torch.Tensor, expand_a_weight: torch.Tensor, expand_b_weight: torch.Tensor,
    squeeze_weight: torch.Tensor, grad_output: torch.Tensor,
    orig_shape: torch.Size, nd: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(dx, dWa, dWb, dWs)`` for ``squeeze(SwiGLU(x @ Wa^T, x @ Wb^T))``. ``x`` is 2-D.

    Shared by the split forward here and by the fully fused no-LayerNorm forward in
    ``fused.py``: the two differ only in how the FORWARD is scheduled -- whether the (M, ND)
    intermediate reaches HBM -- and not at all in the gradient, which recomputes it either way.
    Written once so they cannot drift.

    Via the STACKED save-xn kernel: one launch emits h (the SwiGLU output, for dWs) and
    dAB=[dA|dB], so the squeeze input is not recomputed by a second forward kernel and
    dWa/dWb/dx collapse to single stacked GEMMs. ~9% faster than recompute-expand plus
    transition_bwd_kernel at d>=256 (measured, A100), and bit-for-bit the same math.

    ``x`` is whatever the forward normalized: the post-LN activation for `Transition`, and the
    module input itself where there is no LayerNorm. Either way the LN boundary is the caller's.
    """
    from miniworld_engine.kernels.transition.triton.fused import (
        _transition_expand_gatebwd_savedxn_stacked,
    )

    grad_expand = torch.matmul(grad_output, squeeze_weight)        # dh  [M, ND]
    h, dAB = _transition_expand_gatebwd_savedxn_stacked(
        x, expand_a_weight, expand_b_weight, grad_expand,
        shape_key=both_key(rows_of(orig_shape)),   # grad_output's pre-flatten shape
    )
    grad_squeeze_weight = torch.matmul(grad_output.T, h)           # dWs  [D, ND]
    grad_ab = torch.matmul(dAB.T, x)                               # [2*ND, K] = [dWa; dWb]
    w_ab = torch.cat((expand_a_weight, expand_b_weight), dim=0)     # [2*ND, K]
    dx = torch.matmul(dAB, w_ab).reshape(orig_shape)               # [M, K] single GEMM
    return dx, grad_ab[:nd], grad_ab[nd:], grad_squeeze_weight


class TritonTransitionFunction(torch.autograd.Function):
    @typecheck
    @staticmethod
    def forward(
        ctx,
        x: Float[torch.Tensor, "... d"],
        expand_a_weight: Float[torch.Tensor, "nd d"],
        expand_b_weight: Float[torch.Tensor, "nd d"],
        squeeze_weight: Float[torch.Tensor, "d nd"],
        n: int | None = None,
    ) -> Float[torch.Tensor, "... d"]:
        """``squeeze(SwiGLU(x @ Wa^T, x @ Wb^T))`` -- no normalization, no residual.

        ``n`` is the expansion RATIO and is now optional and advisory: the expanded width is
        ``expand_a_weight.shape[0]``, which is the only place it was ever really written down.
        Passed, it is checked; an FFN sized to a multiple of 256 rather than to a multiple of
        ``d`` simply leaves it out.
        """
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1]).contiguous()

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            x = x.to(dtype)
            expand_a_weight = expand_a_weight.to(dtype)
            expand_b_weight = expand_b_weight.to(dtype)
            squeeze_weight = squeeze_weight.to(dtype)

        nd = expand_a_weight.shape[0]
        if n is not None and n * orig_shape[-1] != nd:
            msg = (f"expansion ratio n={n} says the expand width is {n * orig_shape[-1]}, but "
                   f"expand_a_weight is {tuple(expand_a_weight.shape)} -- width {nd}. The "
                   f"weight decides; drop n or make it agree.")
            raise ValueError(msg)
        expand = _expand_swiglu(
            x,
            expand_a_weight,
            expand_b_weight,
            # L = shape[-2] of x BEFORE the view(-1, d) above -- one rule for pair
            # (B, L, L, D) and token/atom (B, L, D). Never the row count M.
            both_key(rows_of(orig_shape)),
        )

        ctx.save_for_backward(
            x,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
        )
        ctx.nd = nd

        output = torch.matmul(expand, squeeze_weight.T)
        return output.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
        ) = ctx.saved_tensors
        nd = ctx.nd
        M, N = x.shape

        if grad_output.dtype != x.dtype:
            grad_output = grad_output.to(x.dtype)

        orig_shape = grad_output.shape
        grad_output = grad_output.view(-1, orig_shape[-1]).contiguous()
        # Backward via the fast STACKED save-xn kernel: one kernel emits h (the SwiGLU output,
        # for dWs) + dAB=[dA|dB], so the squeeze input is NOT recomputed (no second
        # transition_fwd_kernel) and dWa/dWb/dxn collapse to single stacked GEMMs. ~9% faster
        # than the old recompute-expand + transition_bwd_kernel at d>=256 (measured, A100),
        # bit-for-bit the same math: it consumes the SAVED post-LN xn (`x`) directly, so there
        # is no LN-boundary or xn-recompute mismatch. (LN backward stays in the module's ln_in.)
        dx, grad_a_weight, grad_b_weight, grad_squeeze_weight = swiglu_squeeze_backward(
            x, expand_a_weight, expand_b_weight, squeeze_weight, grad_output, orig_shape, nd,
        )
        return dx, grad_a_weight, grad_b_weight, grad_squeeze_weight, None


triton_transition = TritonTransitionFunction.apply

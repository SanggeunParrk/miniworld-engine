
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/transition.py

from miniworld_engine.autotune.configs import configs_for
import torch

# FP32 projections follow torch's TF32 switch (kernels/_dot_precision.py): IEEE dot precision unless torch allows TF32
# (implicit TF32 changes DiT gradients, so it is never taken by default). BF16 tensor-core arithmetic is unchanged.
import triton
import triton.language as tl

from miniworld_engine.kernels._dot_precision import dot_precision
from miniworld_engine.kernels._tiles import tile_grid, tile_order
from miniworld_engine.kernels.transition.triton._tma import num_sms, tma_ok

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


@triton.autotune(configs=configs_for("transition_expand_swiglu_triton"), key=['shape_key', 'STORE_AB', 'USE_TMA'])
@triton.jit
def transition_fwd_kernel(
    x_ptr,
    Wab_ptr,
    out_ptr,
    a_ptr,
    b_ptr,
    M,
    K: tl.constexpr,
    ND: tl.constexpr,
    stride_wn, stride_wk,
    BLOCK_M1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    shape_key,
    STORE_AB: tl.constexpr = False,
    USE_TMA: tl.constexpr = False,
    NUM_SMS: tl.constexpr = 1,
    PREC: tl.constexpr = "ieee",
):
    # ``Wab`` is the two expand weights INTERLEAVED row by row (row 2j = Wa[j], row 2j+1 = Wb[j]), so one
    # dot over a (BLOCK_K, 2*BLOCK_N) weight tile produces the a and b halves of BLOCK_N hidden columns
    # side by side. One wgmma of N = 2*BLOCK_N reads the activation tile once and keeps the tensor cores
    # busier than two dots of N = BLOCK_N (measured on H100, bf16: the 128 x 128 dual tile ran at ~520 TFLOPS,
    # the same tile as one 128 x 256 dot at the squeeze kernel's ~640). The epilogue splits the accumulator.
    #
    # STORE_AB: a training forward also stores the two pre-activation halves (rounded to the activation
    # dtype, as the PyTorch statements hold them), so the backward is one GEMM with a gate epilogue
    # (kernels/transition/triton/training.py) instead of a recompute of both expand GEMMs.
    #
    # USE_TMA (Hopper, aligned operands -- see _tma.py): tensor-map loads and stores in a persistent grid.
    # Visit order, tuned: see kernels/_tiles.py.
    n_m = tl.cdiv(M, BLOCK_M1)
    n_n = tl.cdiv(ND, BLOCK_N)
    if USE_TMA:
        x_desc = tl.make_tensor_descriptor(x_ptr, [M, K], [K, 1], [BLOCK_M1, BLOCK_K])
        w_desc = tl.make_tensor_descriptor(Wab_ptr, [2 * ND, K], [stride_wn, 1], [2 * BLOCK_N, BLOCK_K])
        out_desc = tl.make_tensor_descriptor(out_ptr, [M, ND], [ND, 1], [BLOCK_M1, BLOCK_N])
        a_desc = tl.make_tensor_descriptor(a_ptr, [M, ND], [ND, 1], [BLOCK_M1, BLOCK_N])
        b_desc = tl.make_tensor_descriptor(b_ptr, [M, ND], [ND, 1], [BLOCK_M1, BLOCK_N])
        for tile in tl.range(tl.program_id(0), n_m * n_n, NUM_SMS, flatten=True):
            pid_m, pid_n = tile_order(tile, n_m, n_n, GROUP_M)
            acc = tl.zeros((BLOCK_M1, 2 * BLOCK_N), dtype=tl.float32)
            for k in range(0, K, BLOCK_K):
                x_tile = x_desc.load([pid_m * BLOCK_M1, k])
                w_tile = w_desc.load([pid_n * (2 * BLOCK_N), k])
                acc = tl.dot(x_tile, w_tile.T, acc, input_precision=PREC)
            A_tile, B_tile = tl.split(tl.reshape(acc, (BLOCK_M1, BLOCK_N, 2)))
            swish_AB = (A_tile * tl.sigmoid(A_tile)) * B_tile
            out_desc.store([pid_m * BLOCK_M1, pid_n * BLOCK_N], swish_AB.to(out_ptr.dtype.element_ty))
            if STORE_AB:
                a_desc.store([pid_m * BLOCK_M1, pid_n * BLOCK_N], A_tile.to(a_ptr.dtype.element_ty))
                b_desc.store([pid_m * BLOCK_M1, pid_n * BLOCK_N], B_tile.to(b_ptr.dtype.element_ty))
    else:
        pid_m, pid_n = tile_order(tl.program_id(0).to(tl.int64), n_m, n_n, GROUP_M)

        offs_m = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_w = pid_n * (2 * BLOCK_N) + tl.arange(0, 2 * BLOCK_N)    # rows of Wab

        acc = tl.zeros((BLOCK_M1, 2 * BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            x_tile = tl.load(
                x_ptr + (offs_m[:, None] * K + offs_k[None, :]),
                mask=((offs_m[:, None] < M) & (offs_k[None, :] < K)),
                other=0.0,
            )
            w_tile = tl.load(
                Wab_ptr + (offs_w[None, :] * stride_wn + offs_k[:, None] * stride_wk),
                mask=((offs_w[None, :] < 2 * ND) & (offs_k[:, None] < K)),
                other=0.0,
            )
            acc = tl.dot(x_tile, w_tile, acc, input_precision=PREC)

        A_tile, B_tile = tl.split(tl.reshape(acc, (BLOCK_M1, BLOCK_N, 2)))
        swish_AB = (A_tile * tl.sigmoid(A_tile)) * B_tile

        out_ptr_ = out_ptr + (offs_m[:, None] * ND + offs_n[None, :])
        store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < ND)
        tl.store(out_ptr_, swish_AB.to(out_ptr.dtype.element_ty), mask=store_mask)
        if STORE_AB:
            tl.store(a_ptr + (offs_m[:, None] * ND + offs_n[None, :]), A_tile.to(a_ptr.dtype.element_ty), mask=store_mask)
            tl.store(b_ptr + (offs_m[:, None] * ND + offs_n[None, :]), B_tile.to(b_ptr.dtype.element_ty), mask=store_mask)


def _expand_swiglu_fake(x, expand_a_weight, expand_b_weight, shape_key):
    """The expand activation, (M, ND) in ``x``'s dtype: the two SwiGLU halves are already
    multiplied together in the epilogue, so the width is ``ND`` and not ``2*ND``."""
    return x.new_empty((x.shape[0], expand_a_weight.shape[0]))


def _launch_expand(x, expand_a_weight, expand_b_weight, shape_key, store_ab):
    M, N = x.shape
    # ND comes from the WEIGHT, not from an expansion ratio. The kernel only ever needed the
    # expanded width; taking it as ``n * N`` made a caller state twice something the weight
    # already fixes, so an `n` that disagreed with the weight silently read the wrong extent --
    # and it barred any expansion the ratio cannot spell, which is exactly what an FFN whose
    # hidden size is rounded to a multiple of 256 has. `conditioned_transition`'s own
    # `_expand_swiglu` already derives it this way.
    nd = expand_a_weight.shape[0]
    expand = torch.empty(M, nd, dtype=x.dtype, device=x.device)
    # the halves are kept in fp32: the backward gate then sees the same pre-activations the PyTorch-free CUDA
    # kernels recompute (bf16 halves moved dx by ~4e-3 against them, an order past the parity bands)
    a, b = (torch.empty(M, nd, dtype=torch.float32, device=x.device) for _ in range(2)) if store_ab else (expand, expand)
    # both weights as one operand, row-interleaved (2*ND, K): ~10 MB at the widest, a few microseconds
    w_ab = torch.stack((expand_a_weight, expand_b_weight), dim=1).reshape(2 * nd, N)
    use_tma = tma_ok(x, w_ab, expand, a, b)
    if use_tma:
        sms = num_sms(x.device)
        grid = lambda META: (min(sms, tile_grid(M, nd, META["BLOCK_M1"], META["BLOCK_N"])[0]),)  # noqa: E731
    else:
        sms = 1
        grid = lambda META: tile_grid(M, nd, META["BLOCK_M1"], META["BLOCK_N"])  # noqa: E731
    transition_fwd_kernel[grid](
        x,
        w_ab,
        expand,
        a,
        b,
        M,
        N,          # K -- the input width
        nd,         # ND -- the expanded width
        w_ab.stride(0), w_ab.stride(1),
        shape_key=pack(shape_key, K=N, ND=nd),
        STORE_AB=store_ab,
        USE_TMA=use_tma,
        NUM_SMS=sms,
        PREC=dot_precision(x.dtype),   # fp32: TF32 when torch allows it (kernels/_dot_precision.py), else IEEE
    )
    return expand, a, b


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
    return _launch_expand(x, expand_a_weight, expand_b_weight, shape_key, False)[0]


def _expand_swiglu_saveact_fake(x, expand_a_weight, expand_b_weight, shape_key):
    """``h (M, ND)`` in ``x``'s dtype and the halves ``a``, ``b`` ``(M, ND)`` in fp32."""
    out = x.new_empty((x.shape[0], expand_a_weight.shape[0]))
    return out, torch.empty(out.shape, dtype=torch.float32, device=out.device), torch.empty(out.shape, dtype=torch.float32, device=out.device)


@opaque(fake=_expand_swiglu_saveact_fake, name="transition_expand_swiglu_saveact")
def _expand_swiglu_saveact(
    x: torch.Tensor,
    expand_a_weight: torch.Tensor,
    expand_b_weight: torch.Tensor,
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The training forward launch: ``(h, a, b)`` with ``h = SwiGLU(a, b)`` and ``a, b`` the two
    expand halves, ``h`` in ``x``'s dtype, ``a`` and ``b`` ``(M, n*N)`` in fp32. The backward consumes ``a`` and ``b``."""
    return _launch_expand(x, expand_a_weight, expand_b_weight, shape_key, True)


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
        x = x.reshape(-1, orig_shape[-1]).contiguous()

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
        # L = shape[-2] of x BEFORE the view(-1, d) above -- one rule for pair
        # (B, L, L, D) and token/atom (B, L, D). Never the row count M.
        key = both_key(rows_of(orig_shape))
        expand = _expand_swiglu(x, expand_a_weight, expand_b_weight, key)
        ctx.save_for_backward(x, expand_a_weight, expand_b_weight, squeeze_weight)
        ctx.key = key
        ctx.nd = nd

        output = torch.matmul(expand, squeeze_weight.T)
        return output.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, expand_a_weight, expand_b_weight, squeeze_weight = ctx.saved_tensors
        if grad_output.dtype != x.dtype:
            grad_output = grad_output.to(x.dtype)

        orig_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, orig_shape[-1]).contiguous()
        from miniworld_engine.kernels.transition.triton.training import swiglu_squeeze_backward_saved

        dx, grad_a_weight, grad_b_weight, grad_squeeze_weight = swiglu_squeeze_backward_saved(
            x, expand_a_weight, expand_b_weight, squeeze_weight, grad_output, ctx.key,
        )
        dx = dx.reshape(orig_shape)
        return dx, grad_a_weight, grad_b_weight, grad_squeeze_weight, None


triton_transition = TritonTransitionFunction.apply

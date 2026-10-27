
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/transition.py

from miniworld_engine.autotune.configs import configs_for
import torch

# FP32 projections follow torch's TF32 switch (kernels/_dot_precision.py): IEEE dot precision unless torch allows TF32
# (implicit TF32 changes DiT gradients, so it is never taken by default). BF16 tensor-core arithmetic is unchanged.
import triton
import triton.language as tl

from miniworld_engine.kernels._tf32 import tf32_cached, tf32_dot, tf32_grid_many, tf32_grid_tile

from miniworld_engine.kernels._dot_precision import dot_precision
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

def _whole_k_tiles(configs, named_args, **meta):
    """With ``ROUND_X`` the kernel rounds the activation tile it loads, which is only exact when no K tile is ragged: keep the configs whose
    ``BLOCK_K`` divides ``K``."""
    a = {**named_args, **meta}
    if not a.get("ROUND_X"):
        return list(configs)
    return [c for c in configs if int(a["K"]) % c.kwargs["BLOCK_K"] == 0] or list(configs)

@triton.autotune(configs=configs_for("transition_expand_swiglu_triton"), key=["shape_key", "STORE_AB", "HALF"],
                 prune_configs_by={"early_config_prune": _whole_k_tiles})
@triton.jit
def transition_fwd_kernel(
    x_ptr,
    Wab_ptr,
    out_ptr,
    a_ptr,
    b_ptr,
    scale_ptr,
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
    HALF: tl.constexpr = False,
    PREC: tl.constexpr = "ieee",
    PRE: tl.constexpr = False,
    ROUND_X: tl.constexpr = False,
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
    # Visit order, tuned: see kernels/_tiles.py.
    n_m = tl.cdiv(M, BLOCK_M1)
    n_n = tl.cdiv(ND, BLOCK_N)
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
        acc = tf32_dot(x_tile, w_tile, acc, input_precision=PREC, pre_a=PRE and not ROUND_X, pre_b=PRE)

    A_tile, B_tile = tl.split(tl.reshape(acc, (BLOCK_M1, BLOCK_N, 2)))
    swish_AB = (A_tile * tl.sigmoid(A_tile)) * B_tile

    if PRE:
        swish_AB = tf32_grid_tile(swish_AB)
    out_ptr_ = out_ptr + (offs_m[:, None] * ND + offs_n[None, :])
    store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < ND)
    tl.store(out_ptr_, swish_AB.to(out_ptr.dtype.element_ty), mask=store_mask)
    if STORE_AB:
        if HALF:
            A_tile = A_tile * (1.0 / tl.load(scale_ptr))
            B_tile = B_tile * (1.0 / tl.load(scale_ptr))
        tl.store(a_ptr + (offs_m[:, None] * ND + offs_n[None, :]), A_tile.to(a_ptr.dtype.element_ty), mask=store_mask)
        tl.store(b_ptr + (offs_m[:, None] * ND + offs_n[None, :]), B_tile.to(b_ptr.dtype.element_ty), mask=store_mask)


def _expand_swiglu_fake(x, expand_a_weight, expand_b_weight, shape_key):
    """The expand activation, (M, ND) in ``x``'s dtype: the two SwiGLU halves are already
    multiplied together in the epilogue, so the width is ``ND`` and not ``2*ND``."""
    return x.new_empty((x.shape[0], expand_a_weight.shape[0]))

#: A/B switch for the tests and records: True keeps the saved halves in fp32 under fp32 activations (the pre-fp16 behaviour).
_SAVED_HALVES_FP32 = False

def _launch_expand(x, expand_a_weight, expand_b_weight, shape_key, store_ab, half_scale=None):
    M, N = x.shape
    # ND comes from the WEIGHT, not from an expansion ratio. The kernel only ever needed the
    # expanded width; taking it as ``n * N`` made a caller state twice something the weight
    # already fixes, so an `n` that disagreed with the weight silently read the wrong extent --
    # and it barred any expansion the ratio cannot spell, which is exactly what an FFN whose
    # hidden size is rounded to a multiple of 256 has. `conditioned_transition`'s own
    # `_expand_swiglu` already derives it this way.
    nd = expand_a_weight.shape[0]
    expand = torch.empty(M, nd, dtype=x.dtype, device=x.device)
    # fp32 activations (TF32 tensor cores, a 10-bit mantissa): when the caller passes `half_scale` the saved halves are fp16 -- the same
    # mantissa, half the bytes -- stored divided by that device scalar, which the caller derives from a bound on |a|, |b| (see
    # residual.half_scale) so that nothing can overflow. No host sync, nothing data dependent to fall back on.
    half = store_ab and x.dtype == torch.float32 and half_scale is not None and _SAVED_HALVES_FP32 is False
    hd = torch.float16 if half else x.dtype
    a, b = (torch.empty(M, nd, dtype=hd, device=x.device) for _ in range(2)) if store_ab else (expand, expand)
    scale = half_scale if half else expand
    # both weights as one operand, row-interleaved (2*ND, K): ~10 MB at the widest, a few microseconds
    prec = dot_precision(x.dtype)
    pre = prec == "tf32" and x.dtype is torch.float32     # operands rounded to nearest here, once, and h at its store (kernels/_tf32.py)
    stack = lambda: torch.stack((expand_a_weight, expand_b_weight), dim=1).reshape(2 * nd, N)    # noqa: E731
    w_ab = tf32_cached((expand_a_weight, expand_b_weight), lambda: tf32_grid_many(stack()))[0] if pre else stack()
    round_x = pre and N % 64 == 0      # the activation tile is rounded inside the kernel (no extra pass over x); a ragged K tile is why the other shapes round it here, in a launcher pass
    x_op = x if round_x or not pre else tf32_grid_many(x)[0]
    grid = lambda META: tile_grid(M, nd, META["BLOCK_M1"], META["BLOCK_N"])  # noqa: E731
    transition_fwd_kernel[grid](
        x_op,
        w_ab,
        expand,
        a,
        b,
        scale,
        M,
        N,          # K -- the input width
        nd,         # ND -- the expanded width
        w_ab.stride(0), w_ab.stride(1),
        shape_key=pack(shape_key, K=N, ND=nd),
        STORE_AB=store_ab,
        HALF=half,
        PREC=prec,   # fp32: TF32 when torch allows it (kernels/_dot_precision.py), else IEEE
        PRE=pre,
        ROUND_X=round_x,
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

def _expand_swiglu_saveact_fake(x, expand_a_weight, expand_b_weight, shape_key, half_scale=None):
    """``(h, a, b)``: each ``(M, ND)``, ``h`` in ``x``'s dtype, ``a`` / ``b`` fp16 when ``half_scale`` is given (fp32 ``x``)."""
    out = x.new_empty((x.shape[0], expand_a_weight.shape[0]))
    half = x.dtype == torch.float32 and half_scale is not None and _SAVED_HALVES_FP32 is False
    halves = torch.empty(out.shape, dtype=torch.float16 if half else out.dtype, device=out.device)
    return out, halves, torch.empty_like(halves)

@opaque(fake=_expand_swiglu_saveact_fake, name="transition_expand_swiglu_saveact")
def _expand_swiglu_saveact(
    x: torch.Tensor,
    expand_a_weight: torch.Tensor,
    expand_b_weight: torch.Tensor,
    shape_key: int,
    half_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The training forward launch: ``(h, a, b)`` with ``h = SwiGLU(a, b)`` and ``a, b`` the two expand halves, each ``(M, n*N)``
    (``x``'s dtype; under fp32 ``x`` with a ``half_scale`` (one fp32 on the device): fp16, stored divided by it). The backward consumes them."""
    return _launch_expand(x, expand_a_weight, expand_b_weight, shape_key, True, half_scale)

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
        save_act: bool = True,
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
        if save_act:
            # training: keep the expand halves (rounded to the activation dtype, as the PyTorch statements hold them), so the
            # backward is ONE GEMM with the gate epilogue instead of a recompute of both expand GEMMs
            expand, a, b = _expand_swiglu_saveact(x, expand_a_weight, expand_b_weight, key)
            ctx.save_for_backward(x, expand_a_weight, expand_b_weight, squeeze_weight, expand, a, b)
        else:
            expand = _expand_swiglu(x, expand_a_weight, expand_b_weight, key)
            ctx.save_for_backward()
        ctx.key = key
        ctx.nd = nd

        output = torch.matmul(expand, squeeze_weight.T)
        return output.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, expand_a_weight, expand_b_weight, squeeze_weight, h, a, b = ctx.saved_tensors
        if grad_output.dtype != x.dtype:
            grad_output = grad_output.to(x.dtype)

        orig_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, orig_shape[-1]).contiguous()
        from miniworld_engine.kernels.transition.triton.training import swiglu_squeeze_backward_saved

        dx, grad_a_weight, grad_b_weight, grad_squeeze_weight = swiglu_squeeze_backward_saved(
            x, expand_a_weight, expand_b_weight, squeeze_weight, grad_output, ctx.key, saved=(h, a, b),
        )
        dx = dx.reshape(orig_shape)
        return dx, grad_a_weight, grad_b_weight, grad_squeeze_weight, None, None

def triton_transition(x, expand_a_weight, expand_b_weight, squeeze_weight, n=None):
    """``squeeze(SwiGLU(x @ Wa^T, x @ Wb^T))`` (:class:`TritonTransitionFunction`). The forward keeps the expand halves only when a
    gradient is wanted: grad mode on and an operand requiring one (``ctx.needs_input_grad`` is true for parameters even under
    ``torch.no_grad()``, so the flag is decided here)."""
    save = torch.is_grad_enabled() and any(t.requires_grad for t in (x, expand_a_weight, expand_b_weight, squeeze_weight))
    return TritonTransitionFunction.apply(x, expand_a_weight, expand_b_weight, squeeze_weight, n, save)

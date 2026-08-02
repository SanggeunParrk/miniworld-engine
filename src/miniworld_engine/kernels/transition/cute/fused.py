"""Full Transition fwd+bwd with the GEMM-bearing kernels on quack SM90 WGMMA.

Mirrors ``triton/fused.py`` (``TritonTransitionFusedFunction``) EXACTLY in structure — same
dataflow, same separate-backward design — but swaps the two GEMM-bearing triton kernels for
cute WGMMA:

  * FORWARD: ``transition_expand_swiglu_cute`` (LN-folded gated dual-GEMM) + torch.matmul
    squeeze  (replaces the triton fused expand + squeeze).
  * BACKWARD: ``grad_expand = go @ Ws`` (cuBLAS) -> ONE cute ``transition_expand_gatebwd_cute``
    (dual-accumulator WGMMA recompute + SwiGLU-backward epilogue -> h, dA, dB) -> dWs/dWa/dWb/
    d_xn (cuBLAS) -> the EXISTING triton ``_transition_ln_bwd`` (NOT ported — it is a
    memory-bound M-reduction; cute would not help and keeping it preserves the structure).

The cute backward consumes the *pre-normalized* xn (Version-B style). xn is materialized
once with a plain ``layer_norm`` (memory-bound, cheap) and reused by both the gatebwd GEMM
operand and the wgrad GEMMs — the LN-bwd kernel still gets the raw x2 + saved stats.
"""

import torch
import triton
import triton.language as tl
from jaxtyping import Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
from miniworld_engine.kernels.transition.triton.fused import (
    _transition_expand_gatebwd_savedxn,
    _transition_ln_bwd,
)


# fmt: off
@triton.jit
def _xn_recompute_kernel(
    x_ptr, rstd_ptr, c1_ptr, g_ptr, b_ptr, o_ptr, M, K, sm, sk,
    BM: tl.constexpr, BK: tl.constexpr,
):
    # xn = (x*rstd - c1)*gamma + beta from saved stats. One bandwidth-bound pass (read x,
    # write xn). Replaces the eager fp32 layer_norm recompute, which materialized several
    # (M,K) fp32 temporaries and was ~10x slower.
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BM + tl.arange(0, BM)
    k = tl.arange(0, BK)
    rm = rows < M
    km = k < K
    mask = rm[:, None] & km[None, :]
    off = rows[:, None] * sm + k[None, :] * sk
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=rm, other=0.0)
    c1 = tl.load(c1_ptr + rows, mask=rm, other=0.0)
    g = tl.load(g_ptr + k, mask=km, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + k, mask=km, other=0.0).to(tl.float32)
    xn = (x * rstd[:, None] - c1[:, None]) * g[None, :] + b[None, :]
    tl.store(o_ptr + off, xn.to(o_ptr.dtype.element_ty), mask=mask)
# fmt: on


def _xn_recompute(x2, rstd, c1, gamma, beta):
    M, K = x2.shape
    xn = torch.empty_like(x2)
    _xn_recompute_kernel[(triton.cdiv(M, 64),)](
        x2, rstd, c1, gamma.contiguous(), beta.contiguous(), xn,
        M, K, x2.stride(0), x2.stride(1),
        BM=64, BK=triton.next_power_of_2(K),
    )
    return xn
from miniworld_engine.kernels.transition.cute.gemm_transition_swiglu import (
    transition_expand_swiglu_cute,
    fold_swiglu,
)
from miniworld_engine.kernels.transition.cute.backward_gatebwd import (
    transition_expand_gatebwd_cute,
)


class CuteTransitionFusedFunction(torch.autograd.Function):
    """Forward: cute LN+expand+SwiGLU + torch squeeze. Backward: cute gate-bwd + cuBLAS + triton LN-bwd."""

    @typecheck
    @staticmethod
    @torch.compiler.disable()
    def forward(
        ctx,
        x: Float[torch.Tensor, "... d"],
        ln_weight: Float[torch.Tensor, "d"],
        ln_bias: Float[torch.Tensor, "d"],
        expand_a_weight: Float[torch.Tensor, "nd d"],
        expand_b_weight: Float[torch.Tensor, "nd d"],
        squeeze_weight: Float[torch.Tensor, "d nd"],
        n: int,
        eps: float,
        backward_backend: str = "triton",
    ) -> Float[torch.Tensor, "... d"]:
        if backward_backend not in {"triton", "cute"}:
            msg = f"backward_backend must be 'triton' or 'cute', got {backward_backend!r}"
            raise ValueError(msg)

        orig_shape = x.shape
        K = orig_shape[-1]
        x2 = x.reshape(-1, K)

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            x2 = x2.to(dtype)
            ln_weight = ln_weight.to(dtype)
            ln_bias = ln_bias.to(dtype)
            expand_a_weight = expand_a_weight.to(dtype)
            expand_b_weight = expand_b_weight.to(dtype)
            squeeze_weight = squeeze_weight.to(dtype)
        x2 = x2.contiguous()

        # LN stats computed once and saved for the separate backward (no recompute there).
        rstd, c1 = stats_triton(x2, eps)

        # cute fused LN + SwiGLU expand (LN folded into the gated dual-GEMM), then cuBLAS squeeze.
        expand = transition_expand_swiglu_cute(
            x2,
            ln_weight,
            ln_bias,
            expand_a_weight,
            expand_b_weight,
            eps,
            stats=(rstd, c1),
        )
        out = torch.matmul(expand, squeeze_weight.T)

        # Recompute-all training policy: do NOT save the (M, ND) expand/h activation — it is
        # re-derived in the backward (store_h=True) to keep the training memory footprint minimal.
        ctx.save_for_backward(
            x2, rstd, c1, ln_weight, ln_bias,
            expand_a_weight, expand_b_weight, squeeze_weight,
        )
        ctx.n = n
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.backward_backend = backward_backend
        return out.reshape(orig_shape)

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, grad_output: torch.Tensor):
        (
            x2, rstd, c1, ln_weight, ln_bias,
            expand_a_weight, expand_b_weight, squeeze_weight,
        ) = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        dt = x2.dtype
        K = x2.shape[-1]
        D = squeeze_weight.shape[0]
        eps = ctx.eps

        go = grad_output.reshape(-1, D)
        if go.dtype != dt:
            go = go.to(dt)

        N = expand_a_weight.shape[0]

        # Re-materialize xn (one bandwidth-bound triton pass) from raw x + saved stats:
        # reused by the cute gatebwd GEMM operand AND the stacked wgrad GEMM.
        xn = _xn_recompute(x2, rstd, c1, ln_weight, ln_bias)

        grad_expand = go @ squeeze_weight                 # dh  [M, ND]
        if ctx.backward_backend == "cute":
            # ONE cute WGMMA: recompute a,b once -> h (for dWs) + interleaved dAB=[dA|dB].
            h, dAB, Bw = transition_expand_gatebwd_cute(
                xn, grad_expand, expand_a_weight, expand_b_weight
            )
            dWs = go.t() @ h
            d_xn = dAB @ Bw
            dW_stack = dAB.t() @ xn
            dWa = dW_stack[0::2].contiguous()
            dWb = dW_stack[1::2].contiguous()
        else:
            # Avoid the cute path's huge duplicated C operand (M, 2N); this costs one extra
            # cuBLAS GEMM but removes a bandwidth-heavy materialization at wide d.
            # Recompute h here (store_h=True) instead of saving it in the forward.
            h, dA, dB = _transition_expand_gatebwd_savedxn(
                xn, expand_a_weight, expand_b_weight, grad_expand, store_h=True
            )
            dWs = go.t() @ h
            dWa = dA.t() @ xn
            dWb = dB.t() @ xn
            d_xn = dA @ expand_a_weight + dB @ expand_b_weight

        # LayerNorm backward (EXISTING triton kernel from saved stats: dx, dgamma, dbeta).
        dx, dgamma, dbeta = _transition_ln_bwd(d_xn, x2, rstd, c1, ln_weight)

        return (
            dx.reshape(orig_shape),
            dgamma.to(ln_weight.dtype),
            dbeta.to(ln_bias.dtype),
            dWa, dWb, dWs, None, None, None,
        )


def cute_transition_fused(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    expand_a_weight: torch.Tensor,
    expand_b_weight: torch.Tensor,
    squeeze_weight: torch.Tensor,
    n: int,
    eps: float = 1e-5,
    backward_backend: str = "triton",
) -> torch.Tensor:
    """Fully fused Transition fwd+bwd with the GEMMs on quack SM90 WGMMA."""
    return CuteTransitionFusedFunction.apply(
        x,
        ln_weight,
        ln_bias,
        expand_a_weight,
        expand_b_weight,
        squeeze_weight,
        n,
        eps,
        backward_backend,
    )

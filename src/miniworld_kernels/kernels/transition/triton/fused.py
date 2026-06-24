"""Fused LayerNorm + SwiGLU-expand forward for the Transition module.

The Transition op is

    x   = LayerNorm(x)                       # ln_in (affine, eps)
    a   = x @ Wa^T,  b = x @ Wb^T            # expand_a / expand_b  (N, d)->(N, n*d)
    h   = a * sigmoid(a) * b                 # SwiGLU gate -> (M, n*d)
    out = h @ Ws^T                           # squeeze  (n*d -> d)

This module fuses the **front half** (LN + both expand GEMMs + SwiGLU gate) into a
single Triton kernel and leaves the ``squeeze`` as a plain ``torch.matmul`` (a clean,
well-tuned (M, n*d) x (n*d, d) GEMM). Fusing squeeze would force the expand kernel to
hold a full ``n*d``-wide row per block (no N-tiling) and blow up the accumulator register
budget, so it is deliberately left out.

Following the M1 LayerNormLinear design, the LN row statistics are computed in a
SEPARATE pass (``stats_triton`` -> rstd[m], c1[m]=mean*rstd) so the expand kernel does no
reduction and is free to tile BLOCK_K < d. The fused kernel then loads x once per
M-block, normalizes on-chip (``x*rstd - c1`` then affine), and reuses the one normalized
tile for BOTH the A and B projections across the N-loop.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck
from miniworld_kernels.kernels.layernorm_linear.triton.stats import stats_triton

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "transition"

if AUTOTUNE:
    _configs = [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
        for bm in (16, 32, 64, 128)
        for bn in (64, 128, 256, 512)
        for nw in (4, 8)
        for ns in (2, 3, 4)
    ]
else:
    _configs = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=8, num_stages=2),
    ]


# fmt: off
@triton.autotune(configs=_configs, key=["ND", "K"])
@triton.jit
def _transition_expand_gate_kernel(
    x_ptr, rstd_ptr, c1_ptr, g_ptr, beta_ptr,
    wa_ptr, wb_ptr, out_ptr,
    M, ND, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,   # Wa, Wb share layout: (ND, K) row-major -> stride_wn=K, stride_wk=1
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program owns BLOCK_M rows and ALL of ND: LayerNorm is applied ONCE per row and
    # the normalized tile is reused for both the A and B projections across the N-loop.
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, BLOCK_K)
    row_mask = rows < M
    k_mask = k < K

    # --- normalize once (stats precomputed): xn = (x*rstd - c1) * g + beta ---
    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
        mask=row_mask[:, None] & k_mask[None, :], other=0.0,
    ).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
    c1 = tl.load(c1_ptr + rows, mask=row_mask, other=0.0)
    g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
    xhat = x * rstd[:, None] - c1[:, None]
    xn = (xhat * g[None, :] + beta[None, :]).to(x_ptr.dtype.element_ty)  # (BM, BK)

    # --- loop the two projections over N-tiles, reusing the one normalized X tile ---
    for n0 in range(0, ND, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < ND
        wa = tl.load(  # (BLOCK_K, BLOCK_N): w[k, n] = W[cols[n], k]
            wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        wb = tl.load(
            wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        a = tl.dot(xn, wa, out_dtype=tl.float32)
        b = tl.dot(xn, wb, out_dtype=tl.float32)
        gate = a * tl.sigmoid(a) * b
        tl.store(
            out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,
            gate.to(out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )
# fmt: on


def transition_expand_gate(
    x2: torch.Tensor,         # (M, K) contiguous
    ln_weight: torch.Tensor,  # (K,)
    ln_bias: torch.Tensor,    # (K,)
    wa: torch.Tensor,         # (ND, K) = expand_a.weight
    wb: torch.Tensor,         # (ND, K) = expand_b.weight
    eps: float,
) -> torch.Tensor:
    """LayerNorm(x) then SwiGLU(expand_a, expand_b) -> expand (M, ND). Stats fused-out."""
    M, K = x2.shape
    ND = wa.shape[0]
    assert wa.shape[1] == K and wb.shape == wa.shape
    assert K <= 1024, "fused expand assumes K fits one BLOCK_K (next_pow2(K) <= 1024)"

    rstd, c1 = stats_triton(x2, eps)
    expand = torch.empty(M, ND, device=x2.device, dtype=x2.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731 (N looped in-kernel)
    _transition_expand_gate_kernel[grid](
        x2, rstd, c1, ln_weight.contiguous(), ln_bias.contiguous(),
        wa.contiguous(), wb.contiguous(), expand,
        M, ND, K,
        x2.stride(0), x2.stride(1),
        wa.stride(0), wa.stride(1),
        expand.stride(0), expand.stride(1),
        BLOCK_K=triton.next_power_of_2(K),
    )
    return expand


# Single baked winner (BLOCK_M=64, BLOCK_N=64) for the d=128 b2b path. NOT env-gated:
# multi-config autotune was timing-UNSTABLE here (cached bad configs -> 0.49-0.64ms runs);
# the single baked config is stable at ~0.31ms. (Unlike the expand kernel, which autotunes.)
_b2b_configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
]


# fmt: off
@triton.autotune(configs=_b2b_configs, key=["ND", "K", "D"])
@triton.jit
def _transition_b2b_kernel(
    x_ptr, rstd_ptr, c1_ptr, g_ptr, beta_ptr,
    wa_ptr, wb_ptr, ws_ptr, out_ptr,
    M, ND, K: tl.constexpr, D: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,    # Wa, Wb: (ND, K) row-major
    stride_sd, stride_sn,    # Ws: (D, ND) row-major
    stride_om, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Back-to-back: one program owns BLOCK_M rows and ALL of ND. It builds the gated h
    # tile-by-tile and ACCUMULATES the squeeze out[BM, D] += h_chunk @ Ws[:, chunk]^T, so
    # the (M, ND) intermediate h never touches HBM. Only valid when K fits one BLOCK_K.
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, BLOCK_K)
    row_mask = rows < M
    k_mask = k < K

    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
        mask=row_mask[:, None] & k_mask[None, :], other=0.0,
    ).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
    c1 = tl.load(c1_ptr + rows, mask=row_mask, other=0.0)
    g = tl.load(g_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
    xn = ((x * rstd[:, None] - c1[:, None]) * g[None, :] + beta[None, :]).to(x_ptr.dtype.element_ty)

    dcols = tl.arange(0, D)
    out_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < ND
        wa = tl.load(
            wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        wb = tl.load(
            wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        a = tl.dot(xn, wa, out_dtype=tl.float32)
        b = tl.dot(xn, wb, out_dtype=tl.float32)
        h = (a * tl.sigmoid(a) * b).to(x_ptr.dtype.element_ty)  # (BM, BN)
        ws_t = tl.load(  # (BN, D): Ws[d, cols]^T
            ws_ptr + cols[:, None] * stride_sn + dcols[None, :] * stride_sd,
            mask=col_mask[:, None], other=0.0,
        )
        out_acc += tl.dot(h, ws_t, out_dtype=tl.float32)
    tl.store(
        out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od,
        out_acc.to(out_ptr.dtype.element_ty), mask=row_mask[:, None],
    )
# fmt: on


def transition_b2b(
    x2: torch.Tensor,         # (M, K) contiguous
    ln_weight: torch.Tensor,  # (K,)
    ln_bias: torch.Tensor,    # (K,)
    wa: torch.Tensor,         # (ND, K)
    wb: torch.Tensor,         # (ND, K)
    ws: torch.Tensor,         # (D, ND) = squeeze.weight
    eps: float,
) -> torch.Tensor:
    """Fully fused LN + SwiGLU expand + squeeze -> out (M, D). h never hits HBM.

    Requires K to fit one BLOCK_K (K = next_pow2(K) <= 1024) AND the (x row + weight tiles)
    working set to fit smem — practical only for small K (d <= 128). Caller falls back to
    ``transition_expand_gate`` + ``torch.matmul`` for larger K.
    """
    M, K = x2.shape
    ND = wa.shape[0]
    D = ws.shape[0]
    rstd, c1 = stats_triton(x2, eps)
    out = torch.empty(M, D, device=x2.device, dtype=x2.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _transition_b2b_kernel[grid](
        x2, rstd, c1, ln_weight.contiguous(), ln_bias.contiguous(),
        wa.contiguous(), wb.contiguous(), ws.contiguous(), out,
        M, ND, K, D,
        x2.stride(0), x2.stride(1),
        wa.stride(0), wa.stride(1),
        ws.stride(0), ws.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=triton.next_power_of_2(K),
    )
    return out


# Back-to-back squeeze fusion fits smem only for small K (the model's d=128). Above this,
# the full-K-row load + weight tiles overflow shared memory, so fall back to the two-step
# path (expand kernel writes h, then a cuBLAS squeeze).
_B2B_MAX_K = 128


class TritonTransitionFusedFunction(torch.autograd.Function):
    """Forward: fused (stats + LN + expand + SwiGLU) + torch squeeze.

    Backward recomputes the whole op in torch autograd from the saved raw inputs — correct
    and simple (the current bench target is forward; the ``full`` mode still trains
    correctly). Optimizing backward is a deliberate follow-up.
    """

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
    ) -> Float[torch.Tensor, "... d"]:
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

        if K <= _B2B_MAX_K:
            # Back-to-back fused: squeeze folded in, h never materialized in HBM.
            out = transition_b2b(
                x2, ln_weight, ln_bias,
                expand_a_weight, expand_b_weight, squeeze_weight, eps,
            )
        else:
            expand = transition_expand_gate(
                x2, ln_weight, ln_bias, expand_a_weight, expand_b_weight, eps
            )
            out = torch.matmul(expand, squeeze_weight.T)

        ctx.save_for_backward(
            x2, ln_weight, ln_bias,
            expand_a_weight, expand_b_weight, squeeze_weight,
        )
        ctx.n = n
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        return out.reshape(orig_shape)

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, grad_output: torch.Tensor):
        (
            x2, ln_weight, ln_bias,
            expand_a_weight, expand_b_weight, squeeze_weight,
        ) = ctx.saved_tensors
        eps = ctx.eps
        orig_shape = ctx.orig_shape
        K = x2.shape[-1]

        go = grad_output.reshape(-1, K)
        if go.dtype != x2.dtype:
            go = go.to(x2.dtype)

        with torch.enable_grad():
            xin = x2.detach().requires_grad_(True)
            lw = ln_weight.detach().requires_grad_(True)
            lb = ln_bias.detach().requires_grad_(True)
            wa = expand_a_weight.detach().requires_grad_(True)
            wb = expand_b_weight.detach().requires_grad_(True)
            ws = squeeze_weight.detach().requires_grad_(True)

            xn = torch.nn.functional.layer_norm(
                xin.float(), (K,), lw.float(), lb.float(), eps
            ).to(xin.dtype)
            a = xn @ wa.T
            b = xn @ wb.T
            h = a * torch.sigmoid(a) * b
            y = h @ ws.T

            dx, dwa, dwb, dws, dlw, dlb = torch.autograd.grad(
                y, [xin, wa, wb, ws, lw, lb], grad_outputs=go
            )

        return (
            dx.reshape(orig_shape), dlw, dlb, dwa, dwb, dws, None, None,
        )


def triton_transition_fused(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    expand_a_weight: torch.Tensor,
    expand_b_weight: torch.Tensor,
    squeeze_weight: torch.Tensor,
    n: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fully fused Transition forward (LN folded in)."""
    return TritonTransitionFusedFunction.apply(
        x, ln_weight, ln_bias, expand_a_weight, expand_b_weight, squeeze_weight, n, eps
    )

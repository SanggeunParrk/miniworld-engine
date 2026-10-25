"""The triangle-attention variants (RoseTTAFold3's gate / out-projection biases, Chai-1's untransposed output and dual out-projection,
AlphaFold 3's untransposed ending-node bias) on the portable Triton core, forward and backward.

Only the epilogue and the orientation differ from the module's standard statements; the attention itself (flash-style softmax over the
keys with the pair bias, its own backward) is the family's Triton kernel, so no [B, L, H, L, L] logits tensor is ever materialised (the
PyTorch statements, ``TriangleAttention._biased_reference``, build it in fp32). Around the core:

    x = LN(pair); xt = x (starting) or x^T (ending)             the frame the queries / keys / values live in
    bias = Wb (x if starting or not transposed_bias else xt)    [B, H, L, L]; masked keys -> bf16 min (the module's fill)
    q, k, v = Wq xt, Wk xt, Wv xt                               [B, H, L, L, D]
    o = Triton core(q, k, v, bias)                              a row with every key masked -> exactly 0 (the core's l_i guard)
    u = sigmoid(Wg xt + b_g) * o
    out = Wo u + b_o;  dual: other = Wo_t u + b_t, starting: out + other^T, ending: out^T + other;  else ending: out^T (or out untransposed)

Every projection is a cuBLAS GEMM with its bias in the epilogue; gradients flow through autograd (the core and the gated product carry
their own backward kernels). bf16 activations (the core's contract), bf16 or fp32 master parameters (cast inside); the caller checks :func:`refusal` first.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from miniworld_engine.kernels.triangle_attention.triton.main import (
    triton_triangle_attention_pair_bias,
)


def refusal(pair: torch.Tensor, weights: tuple[torch.Tensor, ...], n_head: int, d_hidden: int, *, use_qk_norm: bool = False) -> str | None:
    """None if the Triton variant path can run this call, else why it cannot. Never raises."""
    try:
        if use_qk_norm:
            return "QK-norm is not part of the Triton variant path"
        if not pair.is_cuda:
            return "the input is not on a CUDA device"
        if pair.ndim != 4 or pair.shape[1] != pair.shape[2]:
            return f"the pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
        if pair.dtype != torch.bfloat16:
            return "the Triton variant path is bf16 (activations)"
        if any(w.dtype not in (torch.bfloat16, torch.float32) for w in weights if w is not None):
            return "the Triton variant path takes bf16 or fp32 (master) parameters"
        if d_hidden % n_head:
            return f"d_hidden ({d_hidden}) must be divisible by n_head ({n_head})"
        head = d_hidden // n_head
        if head < 16 or head > 128:
            return f"the head width must be 16..128 for the Triton core, got {head}"
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def triton_triangle_attention_variant(
    pair: torch.Tensor, mask: torch.Tensor | None, ln_weight: torch.Tensor, ln_bias: torch.Tensor, w_query: torch.Tensor,
    w_key: torch.Tensor, w_value: torch.Tensor, w_bias: torch.Tensor, w_gate: torch.Tensor, b_gate: torch.Tensor | None,
    w_out: torch.Tensor, b_out: torch.Tensor | None, w_out_t: torch.Tensor | None = None, b_out_t: torch.Tensor | None = None, *,
    n_head: int, starting: bool = True, transposed_bias: bool = True, untransposed_output: bool = False, dual_output: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    """The triangle-attention update (no residual, no dropout), differentiable. pair [B, L, L, C] bf16, mask [B, L] bool over keys."""
    B, L, _, _ = pair.shape
    from miniworld_engine import kernels
    from miniworld_engine.kernels.bias_only_attention import dispatch as bo_dispatch

    # fp32 master parameters under bf16 activations (the mixed-precision layout): the products run in the activation dtype, the casts are
    # autograd ops, so every gradient comes back in the master dtype (bf16 parameters pass through unchanged). The norm affine stays fp32.
    cast = lambda w: None if w is None else w.to(pair.dtype)   # noqa: E731
    w_query, w_key, w_value, w_bias, w_gate, b_gate, w_out, b_out, w_out_t, b_out_t = (
        cast(w) for w in (w_query, w_key, w_value, w_bias, w_gate, b_gate, w_out, b_out, w_out_t, b_out_t))

    x = kernels.layernorm_kernel(pair, ln_weight, ln_bias, eps) if bo_dispatch.use_kernels(L) else F.layer_norm(
        pair.float(), (pair.shape[-1],), ln_weight, ln_bias, eps).to(pair.dtype)     # the affine is fp32; short sequences: torch's
    xt = x if starting else x.transpose(1, 2).contiguous()
    bias = rearrange(F.linear(x if (starting or not transposed_bias) else xt, w_bias), "B L L2 H -> B H L L2")
    if mask is not None:
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
    q, k, v = (rearrange(F.linear(xt, w), "B L L2 (H D) -> B H L L2 D", H=n_head) for w in (w_query, w_key, w_value))
    o = rearrange(triton_triangle_attention_pair_bias(q, k, v, bias), "B H L L2 D -> B L L2 (H D)")
    u = kernels.sigmoid_gate_fused(F.linear(xt, w_gate, b_gate), o)
    out = F.linear(u, w_out, b_out)
    if dual_output:
        other = F.linear(u, w_out_t, b_out_t)
        return out + other.transpose(1, 2) if starting else out.transpose(1, 2) + other
    return out if starting or untransposed_output else out.transpose(1, 2)

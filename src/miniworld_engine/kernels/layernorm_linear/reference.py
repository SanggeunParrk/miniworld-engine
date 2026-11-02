"""PyTorch reference for the fused LayerNorm + Linear kernel (`layernorm_linear`).

Mirrors NVIDIA Transformer Engine's `te.LayerNormLinear`: a LayerNorm over the
last dim immediately followed by a `Linear` (GEMM + bias). This is the math our
Triton kernels must match; it is also the `torch.compile` baseline the
kernels are measured against.

Both a functional form (`layernorm_linear_pytorch`) and an `nn.Module` form
(`LayerNormLinearRef`, convenient for fwd+bwd benches and weight copying) are
provided; they share the same math.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def layernorm_linear_pytorch(
    x: torch.Tensor,          # (..., d_in)
    ln_weight: torch.Tensor,  # (d_in,)
    ln_bias: torch.Tensor | None,  # (d_in,)  — None for a LayerNorm without beta
    weight: torch.Tensor,     # (d_out, d_in)   — torch.nn.Linear layout
    bias: torch.Tensor | None,  # (d_out,)
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute ``Linear(LayerNorm(x))`` over the last dimension.

    LayerNorm statistics are taken in fp32 (matching TE and `nn.LayerNorm`),
    then the normalized activations feed a plain affine GEMM.
    """
    normed = F.layer_norm(x, (x.shape[-1],), ln_weight, ln_bias, eps)
    return F.linear(normed, weight, bias)


def fold_weights(
    weight: torch.Tensor,     # (d_out, d_in)  — nn.Linear layout
    ln_weight: torch.Tensor,  # (d_in,)  gamma
    ln_bias: torch.Tensor,    # (d_in,)  beta
    bias: torch.Tensor | None,  # (d_out,)
    *,
    w2_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute the folded GEMM operands (prologue).

    The GEMM contracts over K=d_in, so it wants W with shape (K, N) = weight.T.
    Returns ``W2 (K, N), S (N,), B2 (N,)`` matching the kernel's math:

        W2[k,n] = gamma[k] * W[k,n]          (stored in the GEMM operand dtype)
        S[n]    = sum_k W2[k,n]              (FP32, reduced from the *stored* W2!)
        B2[n]   = sum_k beta[k] * W[k,n] + bias[n]   (FP32)

    Crucial: ``S`` is the column-sum of the *rounded* ``W2`` (the values the GEMM
    actually multiplies), not of ``gamma*W`` in FP32 — otherwise ``acc - mean*S``
    leaves a residual common-mode term.
    """
    W = weight.t().contiguous()                       # (K, N)
    W2 = (ln_weight[:, None].float() * W.float()).to(w2_dtype)   # (K, N)
    S = W2.float().sum(dim=0)                          # (N,)  from the stored W2
    B2 = ln_bias.float() @ W.float()                   # (N,)
    if bias is not None:
        B2 = B2 + bias.float()
    return W2, S, B2


class LayerNormLinearRef(nn.Module):
    """`nn.LayerNorm` -> `nn.Linear`, the eager module handed to `torch.compile`.

    Attribute names mirror `te.LayerNormLinear` so weights can be copied 1:1:
    `layer_norm_weight`, `layer_norm_bias`, `weight`, `bias`.
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(in_features, eps=eps)
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.ln(x))

    # TE-compatible aliases ---------------------------------------------------
    @property
    def layer_norm_weight(self) -> torch.Tensor:
        return self.ln.weight

    @property
    def layer_norm_bias(self) -> torch.Tensor:
        return self.ln.bias

    @property
    def weight(self) -> torch.Tensor:
        return self.fc.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.fc.bias

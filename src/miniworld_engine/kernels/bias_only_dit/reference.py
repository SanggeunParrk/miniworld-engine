"""PyTorch reference for the bias-only token DiT kernels (``kernels/bias_only_dit/cuda``).

The two operations the CUDA path adds to the token DiT's, in the layouts the kernels take:

    attention_weights(bias, mask)   P = softmax over the keys of the pair bias [R, L] (R = blocks x heads x queries), masked
                                    keys at the largest negative finite logit -- a fully masked row is uniform, not NaN
    pv_gate(vg, P, S)               a[s l, h d] = sigmoid(g[s l, h d]) * sum_j P[h l, j] v[s j, h d]

``vg`` [S L, 2 D] is the v|g GEMM output (v the first D columns, g the rest), ``P`` one block's [H L, L] weights; H heads of
D / H channels (16 x 48, 24 x 32, 12 x 64 or 16 x 64 on the CUDA path). The whole block is
``modules.bias_only_dit.BiasOnlyDiTBlock`` with ``implementation=PYTORCH``.
"""

from __future__ import annotations

import torch


def attention_weights(bias: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """bias [R, L] -> P [R, L] in bias's dtype; the softmax runs in fp32."""
    logits = bias.float()
    if mask is not None:
        logits = logits.masked_fill(~mask.reshape(1, -1), torch.finfo(torch.float32).min)
    return torch.softmax(logits, dim=-1).to(bias.dtype)


def pv_gate(vg: torch.Tensor, P: torch.Tensor, S: int) -> torch.Tensor:
    """-> a [S L, D] in vg's dtype (P [H L, L]: H heads of D / H channels); the product accumulates in fp32 over the operands
    as given."""
    M, D2 = vg.shape
    D, L = D2 // 2, M // S
    H = P.shape[0] // L
    DH = D // H
    v = vg[:, :D].float().view(S, L, H, DH)
    g = vg[:, D:].float()
    o = torch.einsum("hij,sjhd->sihd", P.float().view(H, L, L), v).reshape(M, D)
    return (torch.sigmoid(g) * o).to(vg.dtype)


__all__ = ["attention_weights", "pv_gate"]

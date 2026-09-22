"""PyTorch reference for the fused augmented-attention pair-bias kernel.

Mirrors ``triton_augmented_attention_pair_bias``
(``miniworld_engine.kernels.augmented_attention.__init__``, which dispatches to
either ``triton/main.py`` or ``triton/memory_efficient.py``). The two backends are
numerically equivalent, so one reference covers both and the ``compute_efficient``
switch has no analogue here.

Layout is token-major, exactly as the kernel entry point takes it::

    query, key, value : (A, B, L, H, D)      A = augmentation, H = heads
    bias              : (B, L, L, H)         shared across the A axis
    mask              : (A, B, L) bool       key-side; ``None`` means all-valid
    out               : (A, B, L, H, D)

Formula, with ``i`` the query token, ``j`` the key token and ``s = D ** -0.5``::

    logits[a,b,h,i,j] = s * <query[a,b,i,h,:], key[a,b,j,h,:]> + bias[b,i,j,h]

The pair bias is added straight to the *scaled* dot product -- it is not itself
multiplied by ``s`` (the kernel divides it by ``sm_scale`` before the common scale
is applied, which is the same thing). Masking is applied to the logits, on the key
axis only, before the softmax::

    logits[a,b,h,i,j] = -inf   where mask[a,b,j] is False
    p = softmax_j(logits)
    out[a,b,i,h,:] = sum_j p[a,b,h,i,j] * value[a,b,j,h,:]

A row whose keys are all masked would be ``0/0``; the kernel floors the softmax
denominator so such a row yields all-zero probabilities and a finite zero output,
and this reference does the same.

Provided as an ``nn.Module`` (:class:`AugmentedAttentionPairBiasReference`) so a
kernel can be checked on both the forward output and the backward gradients, plus
a plain functional form (:func:`augmented_attention_pair_bias_pytorch`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The kernel floors the running row max before subtracting it, and floors the softmax
# denominator, so a fully masked row stays finite instead of producing 0/0 = NaN.
_ROW_MAX_FLOOR = -1e38
_DENOM_FLOOR = 1e-30


def _accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    """Softmax/matmul accumulation dtype: fp32, unless the caller is already wider."""
    return dtype if dtype == torch.float64 else torch.float32


def augmented_attention_pair_bias_pytorch(
    query: torch.Tensor,  # (A, B, L, H, D)
    key: torch.Tensor,  # (A, B, L, H, D)
    value: torch.Tensor,  # (A, B, L, H, D)
    bias: torch.Tensor,  # (B, L, L, H)
    mask: torch.Tensor | None = None,  # (A, B, L) bool, key side
) -> torch.Tensor:
    """Softmax attention with an additive pair bias and a key-side mask.

    Returns ``(A, B, L, H, D)`` in the dtype of ``query``.
    """
    compute_dtype = query.dtype
    acc_dtype = _accumulator_dtype(compute_dtype)
    scale = query.shape[-1] ** -0.5

    # Both matmuls take operands at ``compute_dtype`` and accumulate in ``acc_dtype``,
    # matching the kernel's ``tl.dot`` (bf16 operands, fp32 accumulator). Upcasting the
    # operands is lossless: they are already rounded to ``compute_dtype``.
    q_acc, k_acc = query.to(acc_dtype), key.to(acc_dtype)
    logits = torch.einsum("abihd,abjhd->abhij", q_acc, k_acc) * scale
    logits = logits + bias.permute(0, 3, 1, 2).to(acc_dtype).unsqueeze(0)

    if mask is not None:
        logits = logits.masked_fill(~mask[:, :, None, None, :], float("-inf"))

    # ``detach`` on the row max: it cancels analytically in the softmax gradient, and
    # the kernel likewise treats its running max as a constant.
    row_max = logits.amax(dim=-1, keepdim=True).clamp_min(_ROW_MAX_FLOOR).detach()
    weights = torch.exp(logits - row_max)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(_DENOM_FLOOR)

    # Round the probabilities back to ``compute_dtype`` before the PV product: the
    # kernel stores them at the value dtype for its second ``tl.dot``.
    out = torch.einsum(
        "abhij,abjhd->abihd",
        weights.to(compute_dtype).to(acc_dtype),
        value.to(acc_dtype),
    )
    return out.to(compute_dtype)


class AugmentedAttentionPairBiasReference(nn.Module):
    """nn.Module reference for augmented attention with pair bias.

    Holds no parameters. The kernel entry point takes ``query``/``key``/``value``/
    ``bias`` directly and owns no weights, so a backward comparison differentiates
    with respect to those four input tensors rather than module state; the module
    exists only to give tests the same shape of object as the weight-owning
    references, e.g.::

        ref = AugmentedAttentionPairBiasReference()
        o = ref(q, k, v, bias, mask)                                  # reference
        ok = triton_augmented_attention_pair_bias(q, k, v, bias, mask)  # kernel
        o.sum().backward()                                    # -> q.grad, bias.grad
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass; differentiable in ``query``, ``key``, ``value``, ``bias``."""
        return augmented_attention_pair_bias_pytorch(query, key, value, bias, mask)

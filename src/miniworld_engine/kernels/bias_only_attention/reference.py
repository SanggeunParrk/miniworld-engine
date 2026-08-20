"""PyTorch reference for the bias-only triangle-attention kernel.

Mirrors ``triton_bias_only_attention``
(``miniworld_engine.kernels.bias_only_attention.triton.main``), the degenerate
triangle attention whose logits come entirely from the projected pair bias: there
is no query and no key, so the attention weights depend only on ``bias``.

Layout is head-major, exactly as the kernel entry point takes it::

    value : (B, H, L, L, D)      dims are (batch, head, t, key, channel)
    bias  : (B, H, L, L)         dims are (batch, head, query, key)
    out   : (B, H, L, L, D)      dims are (batch, head, t, query, channel)

``bias`` carries no ``t`` axis: one set of attention weights is broadcast over the
``t`` slices of ``value``, which is why the backward sums ``dbias`` over ``t``.

Formula, with ``t`` the broadcast axis, ``m`` the query token and ``n`` the key
token::

    p[b,h,m,n]     = softmax_n(bias[b,h,m,:])[n]
    out[b,h,t,m,:] = sum_n p[b,h,m,n] * value[b,h,t,n,:]

The pair bias *is* the logit -- it enters unscaled and undivided, with no ``1/sqrt(D)``
factor, because there is no dot product to scale. The kernel takes no mask argument:
masking is expected to be baked into ``bias`` as ``-inf`` (or a large negative fill)
at the positions to drop, which the softmax then sends to zero probability. A row of
``bias`` that is entirely ``-inf`` is ``0/0`` in the kernel, and likewise NaN here --
the kernel applies no denominator floor, so neither does this reference.

Provided as an ``nn.Module`` (:class:`BiasOnlyAttentionReference`) so a kernel can be
checked on both the forward output and the backward gradients, plus a plain
functional form (:func:`bias_only_attention_pytorch`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    """Softmax/matmul accumulation dtype: fp32, unless the caller is already wider."""
    return dtype if dtype == torch.float64 else torch.float32


def bias_only_attention_pytorch(
    value: torch.Tensor,  # (B, H, L, L, D)
    bias: torch.Tensor,  # (B, H, L, L)
) -> torch.Tensor:
    """Attention whose logits are the pair bias alone.

    Returns ``(B, H, L, L, D)`` in the dtype of ``value``.
    """
    compute_dtype = value.dtype
    acc_dtype = _accumulator_dtype(compute_dtype)

    # fp32 softmax accumulation, matching the kernel's fp32 running max / denominator.
    weights = torch.softmax(bias.to(acc_dtype), dim=-1)

    # Round the probabilities back to ``compute_dtype`` before the PV product, then
    # accumulate in ``acc_dtype``: the kernel's ``tl.dot`` takes bf16 operands into an
    # fp32 accumulator. Upcasting the operands here is lossless.
    out = torch.einsum(
        "bhmn,bhtnd->bhtmd",
        weights.to(compute_dtype).to(acc_dtype),
        value.to(acc_dtype),
    )
    return out.to(compute_dtype)


class BiasOnlyAttentionReference(nn.Module):
    """nn.Module reference for bias-only triangle attention.

    Holds no parameters. The kernel entry point takes ``value`` and ``bias`` directly
    and owns no weights, so a backward comparison differentiates with respect to those
    two input tensors rather than module state; the module exists only to give tests
    the same shape of object as the weight-owning references, e.g.::

        ref = BiasOnlyAttentionReference()
        o = ref(v, bias)                          # reference
        ok = triton_bias_only_attention(v, bias)  # kernel
        o.sum().backward()                        # -> v.grad, bias.grad
    """

    def forward(self, value: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Forward pass; differentiable in ``value`` and ``bias``."""
        return bias_only_attention_pytorch(value, bias)

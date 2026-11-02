"""Whole-op wrapper for augmented (pair-biased) attention.

Exposes :func:`augmented_attention_pair_bias` — the pair-biased multi-head attention
core (``softmax(qkᵀ·scale + bias)·v``) as a single autograd-transparent call, with the
fused Triton kernel inside and the layout conversion handled here.

Boundary note: the q/k/v/gate/out projections and the adaptive conditioning are model
concerns (they carry EDM2 magnitude-preserving research variants that normalize weights
on use, so they must own the projection); this op takes ALREADY-PROJECTED ``q/k/v`` plus
the precomputed pair ``bias`` — the same boundary the model already used for the kernel.
The step-invariant bias itself is produced by :func:`ops.layer_norm_linear`.
"""

from __future__ import annotations

from typing import Literal

import torch


def augmented_attention_pair_bias(
    query: torch.Tensor,                  # (A, B, H, L, D)  — model (head-major) layout
    key: torch.Tensor,                    # (A, B, H, L, D)
    value: torch.Tensor,                  # (A, B, H, L, D)
    bias: torch.Tensor,                   # (B, H, L, L)
    mask: torch.Tensor | None = None,     # (A, B, L) key mask
    *,
    kernel_type: Literal["compute_efficient", "memory_efficient"] = "compute_efficient",
) -> torch.Tensor:
    """Pair-biased attention — whole-op call. Input/output in the model's head-major
    ``(A,B,H,L,D)`` / ``(B,H,L,L)`` layout; the internal token-major kernel layout
    (``(A,B,L,H,D)`` / ``(B,L,L,H)``) translation is absorbed here.

    Autograd-transparent. Preserve the caller's precision: native BF16 stays BF16,
    while an FP32 caller does not silently run a quantized attention core.
    """
    if kernel_type not in ("compute_efficient", "memory_efficient"):
        msg = f"unknown kernel_type {kernel_type!r}"
        raise ValueError(msg)
    from miniworld_engine.kernels.augmented_attention.interface import (
        triton_augmented_attention_pair_bias as _fn,
    )
    from miniworld_engine.modules.dispatch import portable_blocked, restricted_card

    in_dtype = query.dtype
    if query.is_cuda and restricted_card(query.device):
        # H100 / B200 launch no Triton. bf16, B == 1, 16 x 48 or 4 x 32 heads, L a multiple of 128: the hand-CUDA sm_90 core (the one
        # modules.AugmentedAttention takes); every other call: the PyTorch reference (fp32 accumulation, the caller's dtype back).
        q_, k_, v_, b_ = query.transpose(2, 3), key.transpose(2, 3), value.transpose(2, 3), bias.permute(0, 2, 3, 1)
        if in_dtype is torch.bfloat16:
            from miniworld_engine.kernels.augmented_attention import cuda as cuda_sm90

            if cuda_sm90.available(q_, b_):
                return cuda_sm90.augmented_attention_bf16_sm90(q_, k_, v_, b_, mask).transpose(2, 3).to(in_dtype)
        if query.device.type == "cuda" and not torch.compiler.is_compiling() and query.dim() == 5:
            # any other dtype / head layout / length: the memory-efficient SDPA kernel with the pair bias (fp32 accumulation, no L^2 scores)
            return augmented_attention_sdpa(query, key, value, bias, mask).to(in_dtype)
        from miniworld_engine.kernels.augmented_attention.reference import (
            augmented_attention_pair_bias_pytorch,
        )

        return augmented_attention_pair_bias_pytorch(q_, k_, v_, b_, mask).transpose(2, 3).to(in_dtype)
    if query.is_cuda and portable_blocked("augmented_attention_pair_bias", query.device, query, key, value, bias):
        # a card the engine holds no record for: the PyTorch reference (one UnrecordedCardWarning)
        from miniworld_engine.kernels.augmented_attention.reference import (
            augmented_attention_pair_bias_pytorch,
        )

        out = augmented_attention_pair_bias_pytorch(
            query.transpose(2, 3), key.transpose(2, 3), value.transpose(2, 3), bias.permute(0, 2, 3, 1), mask,
        )
        return out.transpose(2, 3).to(in_dtype)
    # (A,B,H,L,D) -> (A,B,L,H,D); (B,H,L,L) -> (B,L,L,H)
    q = query.transpose(2, 3)
    k = key.transpose(2, 3)
    v = value.transpose(2, 3)
    b = bias.permute(0, 2, 3, 1)
    out = _fn(q, k, v, b, mask, compute_efficient=kernel_type == "compute_efficient")           # (A,B,L,H,D)
    return out.transpose(2, 3).to(in_dtype)  # -> (A,B,H,L,D)


def augmented_attention_sdpa(query, key, value, bias, mask):
    """Pair-biased attention for the calls the sm_90 core does not take (any dtype, head width and length): ``(A, B, H, L, D)`` q / k / v,
    bias ``(B, H, L, L)``, key mask ``(A, B, L)``, through ``F.scaled_dot_product_attention`` (memory-efficient kernel with an additive
    bias; fp32 accumulation, the scores are never materialised). Masked keys carry the dtype's minimum (a fully masked row averages uniformly
    where the reference returns zeros: no row of the model's masks is fully masked)."""
    import torch.nn.functional as F

    from miniworld_engine.kernels._device import on_h100

    A, B, H, L, D = query.shape
    add = bias.unsqueeze(0).expand(A, B, H, L, L)
    if mask is not None:
        add = torch.where(mask[:, :, None, None, :], add, torch.finfo(add.dtype).min)
    q, k, v = (t.reshape(A * B, H, L, D) for t in (query, key, value))
    add = add.reshape(A * B, H, L, L).to(query.dtype)
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value, bias))
    if not grad and query.dtype is torch.bfloat16 and on_h100(query):
        # no gradient: the cuDNN attention (1.3 - 1.6x the memory-efficient kernel at bf16; it has no gradient for an additive bias), the
        # memory-efficient kernel when cuDNN refuses the call
        from torch.nn.attention import SDPBackend, sdpa_kernel

        try:
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=add).reshape(A, B, H, L, D)
        except RuntimeError:
            pass
    return F.scaled_dot_product_attention(q, k, v, attn_mask=add).reshape(A, B, H, L, D)

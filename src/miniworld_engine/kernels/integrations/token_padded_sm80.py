"""A100 Token450/700 inference only; generic shared AA remains unchanged.

The dispatcher reads physical metadata only.  The original DiT fused/train
routes run first.  This path owns a separate registered attention copy,
cache and original seven source Tensor owners; the transition instance and
all physical parameters are passed unchanged.  Unsupported paths keep the
literal original DiT fallback.  This source makes no runtime qualification.
"""

import torch


def _typed_eligible(module, single, cond, pair, mask, compute_dtype):
    from miniworld_engine.modules.augmented_attention.module import (
        AugmentedAttentionPairBias,
    )
    from miniworld_engine.modules.conditioned_transition.module import (
        ConditionedTransition,
    )
    from miniworld_engine.modules.dispatch import KernelBackend
    from miniworld_engine.modules.dit.module import DiTBlock

    if (
        type(module) is not DiTBlock
        or type(module.attention) is not AugmentedAttentionPairBias
        or type(module.transition) is not ConditionedTransition
    ):
        return False
    if torch.is_grad_enabled():
        return False
    if (
        module.attention._backend != KernelBackend.TRITON
        or module.transition._backend != KernelBackend.TRITON
        or module.attention.use_qk_norm
        or module.attention.n_head != 16
    ):
        return False
    if compute_dtype not in (None, torch.bfloat16):
        return False
    if any(
        not isinstance(v, torch.Tensor)
        or not v.is_cuda
        or v.dtype is not torch.bfloat16
        for v in (single, cond, pair)
    ):
        return False
    if (
        single.ndim != 4
        or single.shape[0] != 5
        or single.shape[1] != 1
        or single.shape[2] not in (128, 256, 384, 512, 640, 768, 200, 450, 700)
        or single.shape[3] != 768
    ):
        return False
    length = single.shape[2]
    if (
        tuple(cond.shape) != (5, 1, length, 384)
        or cond.stride(0) != 0
        or tuple(pair.shape) != (1, length, length, 128)
    ):
        return False
    if cond.device != single.device or pair.device != single.device:
        return False
    if mask is not None:
        if (
            not isinstance(mask, torch.Tensor)
            or not mask.is_cuda
            or mask.dtype is not torch.bool
            or mask.device != single.device
        ):
            return False
        if tuple(mask.shape) != (1, length):
            return False
    return not (
        tuple(module.transition.expand_a.weight.shape) != (1536, 768)
        or tuple(module.transition.expand_b.weight.shape) != (1536, 768)
        or tuple(module.transition.squeeze.weight.shape) != (768, 1536)
    )


def serves(module, single, cond, pair, mask, compute_dtype=None):
    if not _typed_eligible(module, single, cond, pair, mask, compute_dtype):
        return False
    if torch.cuda.get_device_capability(single.device) != (8, 0):
        return False
    from miniworld_engine.integrations import token_padded_augattn_sm80 as attention

    return attention.serves(
        module.attention, single, pair, mask, compute_dtype, cond=cond
    )


def block(module, single, cond, pair, mask):
    assert _typed_eligible(module, single, cond, pair, mask, None)
    from miniworld_engine.integrations import token_padded_augattn_sm80 as attention

    single = attention.forward(module.attention, single, cond, pair, mask)
    return module.transition(single, cond)

"""Automatic dispatch to the B200 (sm_100a) TriangleAttention, d_pair 128, 4 heads x 32 (inference and training, with the
module's broadcast dropout)."""

from __future__ import annotations

import torch
from einops import rearrange

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.modules.dispatch import KernelBackend


def serves(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    """Check the mathematical and resource contract before touching the CUDA builder."""
    # The module resolves `miniworld` (and an explicit `triton`) to its TRITON backend, which also hosts the H100 CUDA paths;
    # `_b200_cuda = False` keeps the Triton kernels (the benchmark baseline), as the `_fuse_*` switches do on H100.
    if (
        getattr(module, "_backend", None) != KernelBackend.TRITON
        or not getattr(module, "_b200_cuda", True)
        or settings.current().engine_backend == "triton"
    ):
        return False
    if not module.use_self_attention or module.use_qk_norm or module.n_head != 4:
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16:
        return False
    if pair.ndim != 4 or pair.shape[1] != pair.shape[2] or pair.shape[-1] != 128:
        return False
    if module.to_query.weight.shape != (128, 128):
        return False
    if mask is not None and (mask.dtype != torch.bool or mask.shape != pair.shape[:2]):
        return False
    from miniworld_engine.kernels.triangle_attention.cuda.b200_triattn import supports

    if not supports(pair.shape[1]):
        return False
    return torch.cuda.get_device_capability(pair.device) == (10, 0)


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """pair + TriangleAttention(pair). The casts stay in autograd so the module's parameters receive gradients."""
    from miniworld_engine.kernels.triangle_attention.cuda.b200_triattn import (
        triangle_attention,
    )

    drop = None
    if module.training and module.p_drop > 0.0:
        # the module's own draw on the original layout ([B, 1, L, C] starting / [B, L, 1, C] ending): in the kernels' starting
        # layout both index the row's last position, so either is [B, L, C] as it is
        B, L = pair.shape[:2]
        drop = module._make_drop_scale(pair, module.p_drop).reshape(B, L, pair.shape[-1])
    if not module.starting:
        pair = rearrange(pair, "B I J D -> B J I D")
    # parameters go in as they are (bf16 or fp32 master weights): the kernels stage them and return gradients in their dtype
    weights = (module.to_query.weight, module.to_key.weight, module.to_value.weight, module.to_gate.weight,
               module.to_bias.weight, module.to_out.weight)
    out = triangle_attention(pair.contiguous(), module.ln_pair.weight, module.ln_pair.bias, *weights, mask, module.ln_pair.eps, drop)
    if not module.starting:
        out = rearrange(out, "B J I D -> B I J D").contiguous()
    return out


def serves_wide(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    """The other widths (d_pair 64 .. 512, heads of 16 / 32 channels), inference and training (with the module's dropout)."""
    if (
        getattr(module, "_backend", None) != KernelBackend.TRITON
        or not getattr(module, "_b200_cuda", True)
        or settings.current().engine_backend == "triton"
    ):
        return False
    if not module.use_self_attention or module.use_qk_norm:
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or pair.ndim != 4 or pair.shape[1] != pair.shape[2]:
        return False
    if mask is not None and (mask.dtype != torch.bool or mask.shape != pair.shape[:2]):
        return False
    from miniworld_engine.kernels.triangle_attention.cuda.b200_triattn import (
        wide_supports,
    )

    d_hidden, d_pair = module.to_query.weight.shape
    if pair.shape[-1] != d_pair or not wide_supports(d_pair, d_hidden, module.n_head, pair.shape[1]):
        return False
    return torch.cuda.get_device_capability(pair.device) == (10, 0)


def forward_wide(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """pair + TriangleAttention(pair) on the wide path (inference, or training with the module's dropout); the packed bf16
    weights are cached until a parameter changes (data pointer or version: an optimizer step re-packs)."""
    from miniworld_engine.kernels.triangle_attention.cuda.b200_triattn import (
        pack_wide_weights,
        wide_inference,
        wide_train,
    )

    weights = (module.to_query.weight, module.to_key.weight, module.to_value.weight, module.to_gate.weight,
               module.to_bias.weight, module.to_out.weight)
    # scoped to the CUDA-graph capture (kernels._capture): a pack made outside a capture is not reused inside one, whose
    # replays would otherwise run on the weights of capture time after every optimizer step
    key = _capture.scoped(tuple((w.data_ptr(), w._version) for w in weights))
    cache = getattr(module, "_b200_wide_pack", None)
    if key is None or cache is None or cache[0] != key:
        with torch.no_grad():
            cache = (key, pack_wide_weights(*weights))
        module._b200_wide_pack = cache
    head_dim = module.to_query.weight.shape[0] // module.n_head
    drop = None
    if module.training and module.p_drop > 0.0:
        # the module's own draw on the original layout: in the kernels' starting layout it indexes the row's last position
        B, L = pair.shape[:2]
        drop = module._make_drop_scale(pair, module.p_drop).reshape(B, L, pair.shape[-1])
    if not module.starting:
        pair = rearrange(pair, "B I J D -> B J I D")
    grad = torch.is_grad_enabled() and (pair.requires_grad or any(w.requires_grad for w in weights))
    if grad or drop is not None:
        out = wide_train(pair.contiguous(), module.ln_pair.weight, module.ln_pair.bias, *weights, cache[1], module.n_head, head_dim,
                         mask, module.ln_pair.eps, drop)
    else:
        out = wide_inference(pair.contiguous(), module.ln_pair.weight, module.ln_pair.bias, cache[1], module.n_head, head_dim,
                             mask, module.ln_pair.eps)
    if not module.starting:
        out = rearrange(out, "B J I D -> B I J D").contiguous()
    return out

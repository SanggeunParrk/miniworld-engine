"""Automatic dispatch of the A100 (sm_80) hand-CUDA triangle attention (``kernels.triangle_attention.cuda``): the whole module in inference (``serves_module`` / ``forward``) and in
training (``serves_train`` / ``forward_train``: the same forward with what the backward needs, the module's broadcast dropout, and an autograd function), or only the attention core
of the module's Triton backend (``serves`` / ``attention``) when the rest of the module is outside the contract.

Two whole-module paths, picked by the module's geometry:

  * ``fused``: d_pair 128, 4 heads x 32 (``sm80.py``): the front, the core and the back are three persistent kernels with the weights resident in shared memory;
  * ``wide``: every other width the model code uses (``sm80_wide.py``): d_pair 64 .. 512 (a multiple of 64), 1 .. 16 heads of 16 or 32 channels: hand-CUDA row kernels around cuBLAS GEMMs
    and the same core.  ``MINIWORLD_TRIATTN_SM80_WIDE=1`` makes d_pair 128 take this path too (tests, A/B).
"""

from __future__ import annotations

import os

import torch

from miniworld_engine import settings
from miniworld_engine.modules.dispatch import KernelBackend


def _contract(module) -> bool:
    """The module's contract: the Triton backend that ``miniworld`` and an explicit ``triton`` resolve to, a non-strict engine backend, the module's
    ``_sm80_cuda`` switch."""
    return not (
        getattr(module, "_backend", None) != KernelBackend.TRITON
        or not getattr(module, "_sm80_cuda", True)
        or settings.current().engine_backend == "triton"
    )


def serves(module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, bias: torch.Tensor) -> bool:
    """The attention core only (inference): the module's contract, no autograd (the backward of the core alone is not written), then the kernel's gate."""
    if not _contract(module) or torch.is_grad_enabled():
        return False
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    return sm80.available(query, key, value, bias)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``[B, H, L, L2, D]`` out, the layout the module's Triton attention returns; a view of a fresh token-major tensor."""
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    return sm80.attention(query, key, value, bias)[0]


def _weights(module) -> tuple[torch.Tensor, ...]:
    return (module.to_query.weight, module.to_key.weight, module.to_value.weight, module.to_gate.weight, module.to_bias.weight)


def _kind(module, pair: torch.Tensor, mask: torch.Tensor | None):
    """``("fused", None)`` for d_pair 128 / 4 x 32, ``("wide", cfg)`` for the other geometries ``sm80_wide`` serves, else None (before the kernels' own gates)."""
    if not module.use_self_attention or module.use_qk_norm:
        return None
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or pair.ndim != 4:
        return None
    if mask is not None and not (mask.dtype == torch.bool and mask.shape == pair.shape[:2]):
        return None
    d_hidden, d_pair = module.to_query.weight.shape
    if pair.shape[-1] != d_pair:
        return None
    if (d_pair, d_hidden, module.n_head) == (128, 128, 4) and os.environ.get("MINIWORLD_TRIATTN_SM80_WIDE", "0") != "1":
        return ("fused", None)
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

    cfg = sm80_wide.cfg_for(d_pair, d_hidden, module.n_head)
    return None if cfg is None else ("wide", cfg)


def _serves_kind(module, pair: torch.Tensor, mask: torch.Tensor | None, *, train: bool) -> bool:
    kind = _kind(module, pair, mask)
    if kind is None:
        return False
    from miniworld_engine.kernels.triangle_attention.cuda import sm80, sm80_wide

    ln = module.ln_pair
    wo = module.to_out.weight
    if kind[0] == "fused":
        if train:
            return sm80.supports_train(pair, _weights(module), ln.weight, ln.bias, wo, mask) and sm80._built(pair)
        return sm80.available_module(pair, _weights(module), ln.weight, ln.bias, wo, mask)
    return sm80_wide.supports(pair, kind[1], _weights(module), wo, ln.weight, ln.bias, mask, train=train) and sm80_wide.loads(kind[1])


def serves_module(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    """The whole module in inference, ``pair + TriangleAttention(pair)``: the module's contract, no autograd, a self-attention module without q/k norm
    (dropout off), then the gates of the path its geometry selects."""
    if not _contract(module) or torch.is_grad_enabled() or (module.training and module.p_drop > 0.0):
        return False
    return _serves_kind(module, pair, mask, train=False)


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """``pair + TriangleAttention(pair)``.  ``fused``: front (LayerNorm, q | k | v | g, bias), attention (with device-side key compaction on large masked calls), back (gate, output projection,
    residual).  ``wide``: ``sm80_wide.forward`` (row kernels around two GEMMs and the core).  The ending node reads and writes its transposed positions inside the kernels
    (no transposing copies)."""
    kind = _kind(module, pair, mask)
    ln = module.ln_pair
    ending = not module.starting
    if kind is not None and kind[0] == "wide":
        from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

        return sm80_wide.forward(pair, kind[1], _weights(module), module.to_out.weight, ln.weight, ln.bias, ln.eps, mask, transposed=ending)
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    qkvg, bias, _ = sm80.front(pair, _weights(module), ln.weight, ln.bias, ln.eps, mask, transposed=ending)
    q, k, v, _ = sm80.qkv_views(qkvg)
    out, _ = sm80.attention(q, k, v, bias, compact_key_mask=mask)
    b, length = pair.shape[:2]
    o = out.permute(0, 2, 3, 1, 4).reshape(b, length, length, 128)         # "B H L L2 D -> B L L2 (H D)": a view of the core's token-major result
    return sm80.back(o, qkvg, module.to_out.weight, pair, transposed=ending)


def serves_train(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    """The whole module with autograd (training or an eval-mode module under grad): the module's contract, the shapes of the inference path, gradients
    wanted by the pair tensor or a parameter, then the gates of the path its geometry selects."""
    if not _contract(module) or not torch.is_grad_enabled():
        return False
    wo = module.to_out.weight
    if not (pair.requires_grad or wo.requires_grad or any(w.requires_grad for w in _weights(module))):
        return False
    return _serves_kind(module, pair, mask, train=True)


def forward_train(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """``pair + drop(TriangleAttention(pair))`` with autograd.  The dropout scale is the module's own draw on the original layout (``[B, 1, L, C]`` starting /
    ``[B, L, 1, C]`` ending); in the kernels' starting frame both index the token's second index, so either is ``[B, L, C]`` as it is."""
    kind = _kind(module, pair, mask)
    ds = None
    if module.training and module.p_drop > 0.0:
        b, length = pair.shape[:2]
        ds = module._make_drop_scale(pair, module.p_drop).reshape(b, length, pair.shape[-1])
    ln = module.ln_pair
    if kind is not None and kind[0] == "wide":
        from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide

        return sm80_wide.trainable(pair, kind[1], _weights(module), module.to_out.weight, ln.weight, ln.bias, ln.eps, mask, transposed=not module.starting, ds=ds)
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    return sm80.trainable(pair, _weights(module), ln.weight, ln.bias, module.to_out.weight, ln.eps, mask, transposed=not module.starting, ds=ds)

"""Automatic dispatch to the B200 (sm_100a) bidirectional D128 TriMul (inference and training)."""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType


def serves(module, pair: torch.Tensor) -> bool:
    """Check the mathematical and resource contract before touching the CUDA builder."""
    if (
        module.implementation != ImplementationType.MINIWORLD
        or settings.current().engine_backend == "triton"
    ):
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if pair.ndim != 4 or pair.shape[0] != 1 or pair.shape[1] != pair.shape[2]:
        return False
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_bidir import supports

    if pair.shape[-1] != 128 or module.d_hidden != 128 or not supports(pair.shape[1]):
        return False
    if module.ln_pair.eps != 1e-5 or module.ln_out.eps != 1e-5:
        return False
    # Qualified on the full 148-SM B200 only: B1r / B7r are cooperative grids sized by SM count.
    if torch.cuda.get_device_capability(pair.device) != (10, 0):
        return False
    return torch.cuda.get_device_properties(pair.device).multi_processor_count == 148


def update(module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None) -> torch.Tensor:
    """Keep the casts in autograd so the module's original parameters receive gradients."""
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_bidir import (
        bidirectional_trimul,
    )

    n = pair.shape[1]
    if mask is None:
        pair_mask = pair.new_ones((n, n), dtype=torch.float32)
    else:
        m = mask.reshape(n)
        pair_mask = (m.unsqueeze(-1) & m.unsqueeze(-2)).float().contiguous()
    scale = None if dropscale is None else dropscale.reshape(n, 128).contiguous()
    bf = pair.dtype
    # pack_w1 reads the four front weights with their strides; no contiguous copies here.
    return bidirectional_trimul(
        pair,
        module.to_left.weight.to(bf),
        module.to_left_gate.weight.to(bf),
        module.to_right.weight.to(bf),
        module.to_right_gate.weight.to(bf),
        module.to_gate.weight.to(bf).contiguous(),
        module.to_out.weight.to(bf).contiguous(),
        module.ln_pair.weight.float().contiguous(),
        module.ln_pair.bias.float().contiguous(),
        module.ln_out.weight.float().contiguous(),
        module.ln_out.bias.float().contiguous(),
        pair_mask,
        scale,
    )


def serves_inference(module, pair: torch.Tensor, *, bidirectional: bool, dropscale: torch.Tensor | None = None) -> bool:
    """Inference (no autograd) at every width D64-D512, either direction (``b200_infer``)."""
    if torch.is_grad_enabled():
        return False
    if (
        module.implementation != ImplementationType.MINIWORLD
        or settings.current().engine_backend == "triton"
    ):
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if pair.ndim != 4 or pair.shape[0] != 1 or pair.shape[1] != pair.shape[2]:
        return False
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import supports

    width = pair.shape[-1]
    if module.d_hidden != width or not supports(width, pair.shape[1], dropscale is not None):
        return False
    if module.ln_pair.eps != 1e-5 or module.ln_out.eps != 1e-5:
        return False
    return torch.cuda.get_device_capability(pair.device) == (10, 0)


def update_inference(module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None,
                     *, bidirectional: bool) -> torch.Tensor:
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import inference

    n, d = pair.shape[1], pair.shape[-1]
    token_mask = None if mask is None else mask.reshape(n).to(torch.bool).contiguous()
    bf = pair.dtype
    leaves = [
        pair,
        module.to_left.weight.to(bf),
        module.to_left_gate.weight.to(bf),
        module.to_right.weight.to(bf),
        module.to_right_gate.weight.to(bf),
        module.to_gate.weight.to(bf).contiguous(),
        module.to_out.weight.to(bf).contiguous(),
        module.ln_pair.weight.float().contiguous(),
        module.ln_pair.bias.float().contiguous(),
        module.ln_out.weight.float().contiguous(),
        module.ln_out.bias.float().contiguous(),
    ]
    scale = None if dropscale is None else dropscale.reshape(n, d).contiguous()
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    return inference(leaves, token_mask, scale, direction)


def serves_train(module, pair: torch.Tensor, *, bidirectional: bool) -> bool:
    """Training (autograd, ``b200_train``): D64 / D256 / D384 / D512 either direction, D128 one direction (D128 bidirectional:
    ``serves``)."""
    if not torch.is_grad_enabled():
        return False
    if (
        module.implementation != ImplementationType.MINIWORLD
        or settings.current().engine_backend == "triton"
    ):
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if pair.ndim != 4 or pair.shape[0] != 1 or pair.shape[1] != pair.shape[2]:
        return False
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_train import supports

    width = pair.shape[-1]
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    if module.d_hidden != width or not supports(width, pair.shape[1], direction):
        return False
    if module.ln_pair.eps != 1e-5 or module.ln_out.eps != 1e-5:
        return False
    return torch.cuda.get_device_capability(pair.device) == (10, 0)


def update_train(module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None,
                 *, bidirectional: bool) -> torch.Tensor:
    """Keep the casts in autograd so the module's original parameters receive gradients."""
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_train import trimul_train

    n, d = pair.shape[1], pair.shape[-1]
    token_mask = None if mask is None else mask.reshape(n).to(torch.bool).contiguous()
    bf = pair.dtype
    leaves = [
        pair,
        module.to_left.weight.to(bf),
        module.to_left_gate.weight.to(bf),
        module.to_right.weight.to(bf),
        module.to_right_gate.weight.to(bf),
        module.to_gate.weight.to(bf).contiguous(),
        module.to_out.weight.to(bf).contiguous(),
        module.ln_pair.weight.float().contiguous(),
        module.ln_pair.bias.float().contiguous(),
        module.ln_out.weight.float().contiguous(),
        module.ln_out.bias.float().contiguous(),
    ]
    scale = None if dropscale is None else dropscale.reshape(n, d).to(bf).contiguous()
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    return trimul_train(leaves, token_mask, scale, direction)

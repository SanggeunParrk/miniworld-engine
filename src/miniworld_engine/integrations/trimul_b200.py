"""Automatic dispatch to the B200 (sm_100a) TriMul: inference (``b200_infer``) and training (``b200_train``), D64-D512,
both modules; the hidden width equals the pair width, or is twice it in one direction at D64 / D128 (``b200_infer.hidden_ok``:
AF3 / Protenix template blocks, pair 64, hidden 128)."""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType

#: Largest batch the native D64 path takes: the j-class assignment of k3g / b1s needs (B L / 128) CTAs at most, and the plane bmm batch is d B.
#: Public: a caller that folds samples into the batch (MiniWorld's template embedder) asks it; an engine without batched samples has no such name.
MAX_BATCH = 8


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
    if pair.ndim != 4 or pair.shape[1] != pair.shape[2]:
        return False
    if pair.shape[0] != 1 and (pair.shape[-1] != 64 or pair.shape[0] > MAX_BATCH):
        return False                  # samples are native (b-major tokens) for D64 only; the others take one sample
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import supports

    width = pair.shape[-1]
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    if not supports(width, pair.shape[1], dropscale is not None, module.d_hidden, direction):
        return False
    if module.ln_pair.eps != 1e-5 or module.ln_out.eps != 1e-5:
        return False
    return torch.cuda.get_device_capability(pair.device) == (10, 0)


def update_inference(module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None,
                     *, bidirectional: bool) -> torch.Tensor:
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import inference

    bsz, n, d = pair.shape[0], pair.shape[1], pair.shape[-1]
    token_mask = None if mask is None else mask.reshape(bsz * n).to(torch.bool).contiguous()
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
    scale = None if dropscale is None else dropscale.reshape(bsz * n, d).contiguous()
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    return inference(leaves, token_mask, scale, direction)


def serves_train(module, pair: torch.Tensor, *, bidirectional: bool) -> bool:
    """Training (autograd, ``b200_train``) at every width D64-D512, either direction."""
    if not torch.is_grad_enabled():
        return False
    if (
        module.implementation != ImplementationType.MINIWORLD
        or settings.current().engine_backend == "triton"
    ):
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if pair.ndim != 4 or pair.shape[1] != pair.shape[2]:
        return False
    if pair.shape[0] != 1 and (pair.shape[-1] != 64 or pair.shape[0] > MAX_BATCH):
        return False                  # samples are native (b-major tokens) for D64 only; the others take one sample
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_train import supports

    width = pair.shape[-1]
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    if not supports(width, pair.shape[1], direction, module.d_hidden):
        return False
    if module.ln_pair.eps != 1e-5 or module.ln_out.eps != 1e-5:
        return False
    return torch.cuda.get_device_capability(pair.device) == (10, 0)


def update_train(module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None,
                 *, bidirectional: bool) -> torch.Tensor:
    """Keep the casts in autograd so the module's original parameters receive gradients."""
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_train import trimul_train

    bsz, n, d = pair.shape[0], pair.shape[1], pair.shape[-1]
    token_mask = None if mask is None else mask.reshape(bsz * n).to(torch.bool).contiguous()
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
    scale = None if dropscale is None else dropscale.reshape(bsz * n, d).to(bf).contiguous()
    direction = 0 if bidirectional else (1 if module.outgoing else 2)
    return trimul_train(leaves, token_mask, scale, direction)

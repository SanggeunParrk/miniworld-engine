"""Fused H100 token DiT inference wired to DiTBlock's existing parameter contract.

Weights are packed on every forward, so CUDA replay observes live parameters.
Only launch configurations are cached. Calls with per-sample conditioning,
QK-norm, autograd or different model dimensions keep the general module path.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.modules.exceptions import ImplementationType

WEIGHTS = (
    "attention.to_query.weight",
    "attention.to_query.bias",
    "attention.to_key.weight",
    "attention.to_value.weight",
    "attention.to_gate.weight",
    "attention.to_out.weight",
    "attention.ada_ln_in.ln_cond.weight",
    "attention.ada_ln_in.to_scale.weight",
    "attention.ada_ln_in.to_scale.bias",
    "attention.ada_ln_in.to_bias.weight",
    "attention.to_scale.weight",
    "attention.to_scale.bias",
    "attention.ln_pair.weight",
    "attention.to_bias.weight",
    "transition.ada_ln_in.ln_cond.weight",
    "transition.ada_ln_in.to_scale.weight",
    "transition.ada_ln_in.to_scale.bias",
    "transition.ada_ln_in.to_bias.weight",
    "transition.to_scale.weight",
    "transition.to_scale.bias",
    "transition.expand_a.weight",
    "transition.expand_b.weight",
    "transition.squeeze.weight",
)
_TUNING: dict = {}


def serves(module, single, cond, pair, compute_dtype=None):
    if torch.is_grad_enabled() or settings.current().engine_backend == "triton":
        return False
    a = module.attention
    if a.implementation != ImplementationType.MINIWORLD or a.use_qk_norm:
        return False
    if not single.is_cuda or single.dtype not in (torch.bfloat16, torch.float32):
        return False
    if compute_dtype is not None and compute_dtype != single.dtype:
        return False
    if (
        single.ndim != 4
        or single.shape[1] != 1
        or single.shape[-1] != 768
        or single.shape[2] % 128
    ):
        return False
    if (
        a.n_head,
        cond.shape[-1],
        pair.shape[-1],
        module.transition.expand_a.weight.shape[0],
    ) != (16, 384, 128, 1536):
        return False
    # The stack algorithm shares conditioning across samples; never silently take cond[0] otherwise.
    if cond.shape[0] != 1 and cond.stride(0) != 0:
        return False
    norms = (
        a.ada_ln_in.ln_in,
        a.ada_ln_in.ln_cond,
        a.ln_pair,
        module.transition.ada_ln_in.ln_in,
        module.transition.ada_ln_in.ln_cond,
    )
    if any(norm.eps != 1e-5 for norm in norms):
        return False
    return torch.cuda.get_device_capability(single.device) == (9, 0)


def _fake(single, cond, pair, mask, weights):
    return torch.empty_like(single)


@opaque(fake=_fake, name="token_dit_h100_infer")
def _infer(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor,
    weights: list[torch.Tensor],
) -> torch.Tensor:
    from miniworld_engine.kernels.conditioned_transition.triton.token_dit_runner import (
        FusedTokenDiT,
    )

    block = SimpleNamespace()
    for path, weight in zip(WEIGHTS, weights, strict=True):
        node = block
        names = path.split(".")
        for name in names[:-1]:
            if not hasattr(node, name):
                setattr(node, name, SimpleNamespace())
            node = getattr(node, name)
        setattr(node, names[-1], weight)
    block.attention.use_qk_norm = False
    block.attention.n_head = 16
    with torch.cuda.device(single.device):
        runner = FusedTokenDiT([block], dtype=single.dtype)
        key = (single.device, single.dtype)
        caches = _TUNING.setdefault(key, ({}, {}))
        runner._mm_cfg, runner._gated_cfg = caches
        bias = runner.hoist(pair, mask)
        return runner.step(single, cond, bias)


def update(module, single, cond, pair, mask):
    mask = (
        mask
        if mask is not None
        else torch.ones((1, single.shape[2]), device=single.device, dtype=torch.bool)
    )
    return _infer(
        single.contiguous(), cond, pair.contiguous(), mask, [module.get_parameter(name) for name in WEIGHTS]
    )

"""Automatic dispatch to the packaged, selected bidirectional H100 training path."""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType


def serves(module, pair: torch.Tensor) -> bool:
    """Check the mathematical and resource contract before touching CUDA builders."""
    policy = settings.current()
    if (
        module.implementation != ImplementationType.MINIWORLD
        or policy.engine_backend == "triton"
    ):
        return False
    if pair.shape[-1] not in policy.trimul_h100_training_widths:
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if (
        pair.ndim != 4
        or pair.shape[0] != 1
        or pair.shape[1] != pair.shape[2]
        or pair.shape[1] not in (384, 768)
    ):
        return False
    if (
        module.d_hidden != pair.shape[-1]
        or module.ln_pair.eps != 1e-5
        or module.ln_out.eps != 1e-5
    ):
        return False
    if torch.cuda.get_device_capability(pair.device) != (9, 0):
        return False
    # Cooperative B1/B7 grids were qualified on full 132-SM H100s, not MIG slices.
    if (
        pair.shape[-1] == 128
        and torch.cuda.get_device_properties(pair.device).multi_processor_count != 132
    ):
        return False
    return all(
        p.dtype == torch.float32
        for p in (
            module.ln_pair.weight,
            module.ln_pair.bias,
            module.ln_out.weight,
            module.ln_out.bias,
        )
    )


def update(
    module,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    dropscale: torch.Tensor | None,
    *, bidirectional: bool = True,
) -> torch.Tensor:
    """Keep casts in autograd so the module's original parameters receive gradients."""
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_training import (
        bidirectional_trimul,
    )

    n, d = pair.shape[1], pair.shape[-1]
    if mask is None:
        pair_mask = torch.ones((n, n), device=pair.device, dtype=torch.bfloat16)
    else:
        pair_mask = (
            (mask.unsqueeze(-1) & mask.unsqueeze(-2)) if mask.ndim == 2 else mask
        )
        pair_mask = pair_mask.to(torch.bfloat16).contiguous()
    scale = (
        pair.new_ones((n, d))
        if dropscale is None
        else dropscale.reshape(n, d).contiguous()
    )
    weights = (
        module.to_left.weight,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
    )
    if bidirectional:
        apply = bidirectional_trimul
    else:
        from miniworld_engine.kernels.trimul_inproj.cuda.h100_single import single_trimul
        def apply(*args):
            return single_trimul(module.outgoing, *args)
    return apply(
        pair,
        *(w.to(pair.dtype) if bidirectional and d == 128 and i < 4 else w.to(pair.dtype).contiguous()
          for i, w in enumerate(weights)),
        module.ln_pair.weight,
        module.ln_pair.bias,
        module.ln_out.weight,
        module.ln_out.bias,
        pair_mask,
        scale,
    )


def serves_inference(
    module, pair: torch.Tensor, *, bidirectional: bool, dropscale=None
) -> bool:
    if (
        module.implementation != ImplementationType.MINIWORLD
        or settings.current().engine_backend == "triton"
    ):
        return False
    if torch.is_grad_enabled() or dropscale is not None:
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or not pair.is_contiguous():
        return False
    if (
        pair.ndim != 4
        or pair.shape[0] != 1
        or pair.shape[1] != pair.shape[2]
        or pair.shape[1] <= 0
    ):
        return False
    if module.ln_pair.eps != module.ln_out.eps:
        return False
    if torch.cuda.get_device_capability(pair.device) != (9, 0):
        return False
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_infer_kernel as table

    hidden = module.d_hidden * (2 if bidirectional else 1)
    entry = table.TILE_TABLE.get(("sm_90a", pair.shape[-1], hidden, "b"))
    return bool(entry and entry.get("k1") and entry.get("k3"))


def update_inference(module, pair, mask, *, bidirectional):
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_inference import inference

    n = pair.shape[1]
    pm = (
        pair.new_ones((n, n))
        if mask is None
        else (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float().contiguous()
    )
    weights = (
        module.to_left.weight,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
    )
    # K1 packs the first four matrices with stride-aware torch operations.
    # Preserve native column-major parameters instead of copying them first.
    weights = [w.to(pair.dtype) if i < 4 else w.to(pair.dtype).contiguous()
               for i, w in enumerate(weights)] + [
        module.ln_pair.weight.float().contiguous(),
        module.ln_pair.bias.float().contiguous(),
        module.ln_out.weight.float().contiguous(),
        module.ln_out.bias.float().contiguous(),
    ]
    direction = 0 if bidirectional else 1 if module.outgoing else 2
    return inference(pair, weights, pm, direction, module.ln_pair.eps)


def serves_single(module, pair: torch.Tensor) -> bool:
    """Qualified single-direction native training contract; inference is separate."""
    return pair.shape[-1] == 128 and serves(module, pair)

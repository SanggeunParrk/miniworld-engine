"""One-time optimizer checkpoint migration for native parameter layouts."""

from __future__ import annotations

import torch


@torch.no_grad()
def align_optimizer_state_layout_(optimizer: torch.optim.Optimizer) -> int:
    """Match dense per-parameter state strides after ``load_state_dict``.

    Fused AdamW requires moments and gradients to have the parameter's layout.
    Old checkpoints can carry row-major moments for now column-major TriMul
    parameters. Copy only mismatched, same-shape dense state tensors; preserve
    their values, dtype/device and all scalar/factored state. Call once after
    loading an optimizer checkpoint, before capturing a training CUDA graph.
    Returns the number of state tensors converted. New optimizers need no call.
    """
    changed = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.layout != torch.strided:
                continue
            for name, value in optimizer.state.get(parameter, {}).items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.layout == torch.strided
                    and value.shape == parameter.shape
                    and value.stride() != parameter.stride()
                ):
                    aligned = torch.empty_strided(
                        value.shape, parameter.stride(),
                        dtype=value.dtype, device=value.device,
                    )
                    aligned.copy_(value)
                    optimizer.state[parameter][name] = aligned
                    changed += 1
    return changed

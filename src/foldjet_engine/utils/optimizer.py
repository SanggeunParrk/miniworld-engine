"""One-time optimizer checkpoint migration for parameter memory layouts.

Engine 2.4.x stored some parameters in a non-default memory layout (the 2.4.4 ``BidirectionalTriangleMultiplication`` D128 path kept its
four front projection matrices column-major); 2.5 parameters are row-major (contiguous). An optimizer checkpoint saved under one layout
and loaded under the other carries moments (Adam ``exp_avg`` / ``exp_avg_sq``) whose strides differ from the parameter's. Fused AdamW
requires the moments and the gradients to have the parameter's layout, so a resumed run calls :func:`align_optimizer_state_layout_`
once after ``optimizer.load_state_dict``. MiniWorld's training resume (``miniworld.training.engine_backend``) imports it from here.
"""

from __future__ import annotations

import torch

__all__ = ["align_optimizer_state_layout_"]


@torch.no_grad()
def align_optimizer_state_layout_(optimizer: torch.optim.Optimizer) -> int:
    """Re-lay every dense per-parameter state tensor whose strides differ from its parameter's, in place.

    Only same-shape, strided tensor state is touched (Adam moments); values, dtype and device are preserved, scalar state (``step``)
    and factored / differently shaped state are left alone. Call once after loading an optimizer checkpoint, before capturing a
    training CUDA graph. Returns the number of state tensors converted (0 when the layouts already match; new optimizers need no call).
    """
    changed = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.layout != torch.strided:
                continue
            state = optimizer.state.get(parameter, {})
            for name, value in list(state.items()):
                if (
                    isinstance(value, torch.Tensor)
                    and value.layout == torch.strided
                    and value.shape == parameter.shape
                    and value.stride() != parameter.stride()
                ):
                    aligned = torch.empty_strided(value.shape, parameter.stride(), dtype=value.dtype, device=value.device)
                    aligned.copy_(value)
                    state[name] = aligned
                    changed += 1
    return changed

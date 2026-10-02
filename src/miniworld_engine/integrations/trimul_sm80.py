"""Automatic dispatch to the A100 (sm_80) hand-CUDA TriMul (``kernels.trimul_inproj.cuda.sm80``), inference and training."""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType


def serves(module, pair: torch.Tensor, mask: torch.Tensor | None) -> bool:
    """The module's contract (implementation, backend policy, matching LayerNorm eps), then the kernels' own gate."""
    if module.implementation != ImplementationType.MINIWORLD or settings.current().engine_backend == "triton":
        return False
    if module.ln_pair.eps != module.ln_out.eps:
        return False
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80

    return sm80.available(pair, module.d_hidden, mask)


def update(module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None, *, bidirectional: bool,
           ) -> torch.Tensor:
    """``pair + drop_row(trimul(pair))``.  The weight casts stay in autograd so the module's own parameters receive gradients."""
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80

    n = pair.shape[1]
    direction = sm80.BIDIR if bidirectional else sm80.OUTGOING if module.outgoing else sm80.INCOMING
    ds = (pair.new_empty((0,)) if dropscale is None
          else dropscale.reshape(n, pair.shape[-1]).to(pair.dtype).contiguous())
    weights = [w.to(pair.dtype) for w in (module.to_left.weight, module.to_left_gate.weight, module.to_right.weight,
                                          module.to_right_gate.weight, module.to_gate.weight, module.to_out.weight)]
    return sm80.trimul(pair, *weights, module.ln_pair.weight, module.ln_pair.bias, module.ln_out.weight, module.ln_out.bias,
                       sm80.token_mask(mask, n, pair.device), ds, direction, module.ln_pair.eps, module.ln_out.eps)

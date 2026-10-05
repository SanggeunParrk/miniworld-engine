"""Automatic dispatch to the A100 (sm_80) hand-CUDA TriMul, inference and training.

Two kernel sets, one gate (``serves``):

* ``kernels.trimul_inproj.cuda.sm80`` -- the fused D128 path (bf16, one direction or bidirectional): literal tile shapes, the on-chip backward;
* ``kernels.trimul_inproj.cuda.sm80_wide`` -- every other width D = 64 / 256 / 384 (and the dtypes the fused path does not take): the LayerNorm-folded
  GEMM kernels ``k1w`` / ``k3w`` with the cuBLAS contraction, and the backward as cuBLAS GEMMs between row kernels.

``MINIWORLD_TRIMUL_SM80=0`` turns both off (the Triton path serves); ``MINIWORLD_TRIMUL_SM80_WIDE=0`` turns the wide set off alone.
"""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType


def _wide_hs(module) -> int:
    """Hidden channels of the plane pair: the leading size of the front matrices (D one direction, 2 D bidirectional)."""
    return module.to_left.weight.shape[0]


def serves(module, pair: torch.Tensor, mask: torch.Tensor | None) -> bool:
    """The module's contract (implementation, backend policy, matching LayerNorm eps), then the kernels' own gate."""
    if module.implementation != ImplementationType.MINIWORLD or settings.current().engine_backend == "triton":
        return False
    if module.ln_pair.eps != module.ln_out.eps:
        return False
    # the input LayerNorm's gradients are written by the kernel, in the parameters' dtype (bf16 or fp32, both alike)
    if module.ln_pair.weight.dtype != module.ln_pair.bias.dtype or module.ln_pair.weight.dtype not in (torch.bfloat16, torch.float32):
        return False
    # the six weight matrices share one dtype (bf16, or fp32 masters over bf16 activations); the kernels cast them inside the op and write the
    # weight gradients in that dtype
    wd = module.to_left.weight.dtype
    if wd not in (torch.bfloat16, torch.float32) or any(
            w.weight.dtype != wd for w in (module.to_left_gate, module.to_right, module.to_right_gate, module.to_gate, module.to_out)):
        return False
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80, sm80_wide

    if sm80.available(pair, module.d_hidden, mask):
        return True
    return sm80_wide.available(pair, module.d_hidden, mask, hs=_wide_hs(module))


def update(module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None, *, bidirectional: bool,
           ) -> torch.Tensor:
    """``pair + drop_row(trimul(pair))``.  The parameters go to the op as leaves; it casts the weights to the kernels' dtype outside autograd and returns
    each gradient in its parameter's own dtype (fp32 masters get the kernels' fp32 accumulators unrounded)."""
    if pair.shape[0] > 1:                                  # the kernels take one square plane: a batch runs plane by plane (autograd sums the weight gradients)
        return torch.cat([update(module, pair[i:i + 1], None if mask is None else mask[i:i + 1],
                                 None if dropscale is None else (dropscale if dropscale.shape[0] == 1 else dropscale[i:i + 1]),
                                 bidirectional=bidirectional) for i in range(pair.shape[0])], dim=0)
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80, sm80_wide

    n = pair.shape[1]
    direction = sm80.BIDIR if bidirectional else sm80.OUTGOING if module.outgoing else sm80.INCOMING
    ds = (pair.new_empty((0,)) if dropscale is None
          else dropscale.reshape(n, pair.shape[-1]).to(pair.dtype).contiguous())
    kernels = sm80 if sm80.supports(pair, module.d_hidden, mask) else sm80_wide
    return kernels.trimul(pair, module.to_left.weight, module.to_left_gate.weight, module.to_right.weight,
                          module.to_right_gate.weight, module.to_gate.weight, module.to_out.weight,
                          module.ln_pair.weight, module.ln_pair.bias, module.ln_out.weight, module.ln_out.bias,
                          sm80.token_mask(mask, n, pair.device), ds, direction, module.ln_pair.eps, module.ln_out.eps)

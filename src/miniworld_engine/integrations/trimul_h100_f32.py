"""H100 dispatch of the fp32 (TF32 tensor core) bidirectional TriMul: ``kernels/trimul_inproj/cuda/h100_f32.py``.

Served when the module is the bidirectional D = hidden = 128 one, ``implementation="miniworld"`` on a Hopper card with fp32 parameters and
activations (no autocast) and TF32 allowed (``torch.set_float32_matmul_precision`` not ``"highest"`` or ``allow_tf32``: the switch cuBLAS
follows).  Inference and training; B samples run sample by sample (the planes are ``[channel][token]`` of one sample).  A call outside this
contract runs the next path (the PyTorch statements on H100).
"""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType

WIDTH = 128


def _tf32_allowed() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32) or torch.get_float32_matmul_precision() != "highest"


def serves(module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None) -> bool:
    policy = settings.current()
    if (module.implementation != ImplementationType.MINIWORLD or policy.force_portable_kernels
            or hasattr(module, "outgoing")):
        return False
    if (not pair.is_cuda or pair.dtype != torch.float32 or pair.ndim != 4 or pair.shape[-1] != WIDTH or pair.shape[1] != pair.shape[2]
            or pair.shape[1] < 16 or not pair.is_contiguous() or module.d_hidden != WIDTH or module.d_pair != WIDTH):
        return False
    if torch.is_autocast_enabled() or not _tf32_allowed():
        return False
    if torch.cuda.get_device_capability(pair.device) != (9, 0):
        return False
    length = pair.shape[1]
    if torch.is_grad_enabled() and length % 4:
        return False
    if mask is not None and (mask.dtype != torch.bool or mask.ndim != 2 or mask.shape != (pair.shape[0], length)):
        return False
    if dropscale is not None and (dropscale.ndim != 4 or dropscale.shape[1] != 1 or dropscale.shape[2:] != (length, WIDTH)
                                  or dropscale.shape[0] not in (1, pair.shape[0])):
        return False
    params = (module.to_left.weight, module.to_left_gate.weight, module.to_right.weight, module.to_right_gate.weight,
              module.to_gate.weight, module.to_out.weight, module.ln_pair.weight, module.ln_pair.bias, module.ln_out.weight, module.ln_out.bias)
    return all(p.dtype == torch.float32 and p.device == pair.device and p.is_contiguous() or _column_major_ok(p) for p in params)


def _column_major_ok(p: torch.Tensor) -> bool:
    return p.dtype == torch.float32 and p.ndim == 2 and p.stride() == (1, p.shape[0])        # the D128 module's front matrices


#: Bytes the training forward keeps per pair token (fp32: x_n 128, LN stats 2 + 2, planes 512, tri 256, on 256, gate logit 128, projection 128).
SAVED_BYTES_PER_TOKEN = (128 + 2 + 512 + 256 + 256 + 2 + 128 + 128) * 4
#: A batch whose saved state would pass this fraction of the device's TOTAL memory (never of the live free memory, so the plan is a function of
#: the shapes and the card: torch.compile and CUDA-graph capture see the same one every time) runs every sample checkpointed.
CHECKPOINT_FRACTION = 0.5


def _checkpointed(pair: torch.Tensor) -> bool:
    if not torch.is_grad_enabled() or pair.shape[0] == 1:
        return False
    saved = pair.shape[0] * pair.shape[1] * pair.shape[2] * SAVED_BYTES_PER_TOKEN
    return saved > CHECKPOINT_FRACTION * torch.cuda.get_device_properties(pair.device).total_memory


def update(module, pair: torch.Tensor, mask: torch.Tensor | None, dropscale: torch.Tensor | None) -> torch.Tensor:
    """The B samples run one after the other (the planes are ``[channel][token]`` of one sample); when their saved state would not fit
    next to the caller's tensors every sample is checkpointed (forward without saves, again with grad in the backward)."""
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_f32 import trimul

    ckpt = _checkpointed(pair)
    outs = []
    for b in range(pair.shape[0]):
        ds = None if dropscale is None else dropscale[b if dropscale.shape[0] > 1 else 0, 0].contiguous()
        args = (pair[b:b + 1], module.to_left_gate.weight, module.to_left.weight, module.to_right_gate.weight, module.to_right.weight,
                module.to_out.weight, module.to_gate.weight, module.ln_pair.weight, module.ln_pair.bias, module.ln_out.weight,
                module.ln_out.bias, module.ln_pair.eps, module.ln_out.eps, None if mask is None else mask[b].contiguous(), ds)
        if ckpt:
            from torch.utils.checkpoint import checkpoint

            outs.append(checkpoint(trimul, *args, use_reentrant=False))
        else:
            outs.append(trimul(*args))
    return outs[0] if len(outs) == 1 else torch.cat(outs, 0)

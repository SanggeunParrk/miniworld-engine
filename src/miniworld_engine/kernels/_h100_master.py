"""H100 bf16 casts of live fp32 master weights (the part of the old ``utils.h100_master`` the kernels use)."""

import torch

from miniworld_engine.kernels._compile import device_constant, opaque


@device_constant
def is_h100(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


def _pack_fake(weights):
    return [torch.empty_like(w, dtype=torch.bfloat16, memory_format=torch.contiguous_format) for w in weights]


@opaque(fake=_pack_fake, name="h100_master_projection_pack")
def _pack(weights: list[torch.Tensor]) -> list[torch.Tensor]:
    # One multi-tensor PyTorch copy (a cast): not worth a kernel of its own.
    outputs = _pack_fake(weights)
    torch._foreach_copy_(outputs, weights)
    return outputs


def pack(weights):
    """Fresh bf16 casts on every call / replay; the projection owns the raw fp32 parameter gradient."""
    with torch.no_grad():
        return _pack(list(weights))

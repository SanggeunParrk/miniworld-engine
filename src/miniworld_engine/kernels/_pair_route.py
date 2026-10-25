"""Where a call of the pair-embedding ops (``pair_embed_geom``, ``pair_embed_adaln``, ``token_pair_init``) runs on the card in front of us.

One rule for the three families (the card rule of ``docs/dispatch.md``):

* a card with a hand-written CUDA kernel for the op (H100 for the two La-Proteina pair builders, H100 and B200 for the token-pair initialisation): CUDA;
* any other card the engine holds a record for (every portable kernel the call launches has an ``ok`` row in that card's manifest): the portable kernels;
* any other card: one ``UnrecordedCardWarning`` and the PyTorch reference;
* H100 / B200 never run the portable kernels unless the internal A/B switch is on (``_forced``), and a call no CUDA kernel serves there is the PyTorch reference;
* a tensor that is not on a CUDA device is the PyTorch reference.
"""
from __future__ import annotations

import torch

CUDA = "cuda"
PORTABLE = "portable"
REFERENCE = "reference"


def _forced() -> bool:
    """The internal A/B switch: the portable kernels on every card, recorded or not, H100 and B200 included."""
    from miniworld_engine import settings

    return settings.current().force_portable_kernels


def _capability(device) -> tuple[int, int]:
    index = device.index if getattr(device, "type", None) == "cuda" else None
    return tuple(torch.cuda.get_device_capability(index))


def cuda_served(device, cuda_caps) -> bool:
    """Does a hand-written CUDA kernel serve this card (and the A/B switch is off)?"""
    return not _forced() and _capability(device) in cuda_caps


def restricted_reason(device, cuda_caps) -> str | None:
    """Why a restricted card (H100 / B200 without a CUDA kernel of the op) is not served; None on a card that is, or that routes by its record."""
    from miniworld_engine.modules import dispatch

    if _forced() or _capability(device) in cuda_caps or not dispatch.restricted_card(device):
        return None
    cap = _capability(device)
    return f"no CUDA kernel of this op on sm_{cap[0]}{cap[1]}: H100 and B200 run the PyTorch reference where no CUDA kernel serves the call"


def route(op: str, device, *tensors: torch.Tensor, cuda_caps=((9, 0),)) -> str:
    """``CUDA``, ``PORTABLE`` or ``REFERENCE`` for a call of ``op`` (the tensors decide inference / training: a tensor that needs a gradient)."""
    from miniworld_engine.modules import dispatch

    if getattr(device, "type", "cuda") != "cuda" or not torch.cuda.is_available():
        return REFERENCE                                  # a CPU tensor: no kernel to run, the PyTorch reference is the op
    if _forced():
        return PORTABLE
    if _capability(device) in cuda_caps:
        return CUDA
    if dispatch.restricted_card(device):
        return REFERENCE
    return REFERENCE if dispatch.portable_blocked(op, device, *tensors) else PORTABLE

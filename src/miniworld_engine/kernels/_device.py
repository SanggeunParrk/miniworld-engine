"""Device facts the kernel launchers ask for.

``on_h100`` and ``no_triton`` are the H100 preconditions the routed entry points ask.

``num_sms(device)``: the streaming-multiprocessor count of a CUDA device (``device`` an index, a ``torch.device``, a string or ``None`` for the current
device). Persistent-grid kernels size their grid by it; the answer is fixed per device, so it is read once, and it is a ``device_constant`` to a
compiled graph (Dynamo cannot trace the device-index lookup inside it).
"""

from __future__ import annotations

import functools

import torch

from miniworld_engine.kernels._compile import device_constant


@functools.lru_cache(maxsize=32)
def _sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


@device_constant
def num_sms(device: torch.device | int | str | None = None) -> int:
    """SM count of ``device`` (``None``: the current device)."""
    return _sms(torch.cuda._get_device_index(device, optional=True))  # pyright: ignore[reportPrivateImportUsage]  -- torch internal, on purpose


#: CUDA device index -> compute capability (fixed per device: read once). ``torch.cuda.get_device_capability`` costs ~0.85 us a call
#: (device-index normalisation, lazy-init checks, the properties object); the per-call backend gates ask it several times a block call --
#: ~70 000 times a FoldForge AF3 forward of 200 steps (~60 ms on the RTX 5090, where a small target's forward is host-bound).
_CAPABILITY: dict[int, tuple[int, int]] = {}


def capability(device: torch.device | int | str | None = None) -> tuple[int, int]:
    """``torch.cuda.get_device_capability(device)``, cached per device index (the same value; ~0.1 us a hit). A ``torch.device`` with an
    explicit CUDA index (what ``tensor.device`` is), an int index or ``None`` (the current device) take the cache; anything else (a string,
    an index-less or non-CUDA device) and every call under ``torch.compile`` tracing go to torch itself, as before."""
    if torch.compiler.is_compiling():
        return torch.cuda.get_device_capability(device)
    kind = type(device)
    if kind is torch.device:
        index = device.index if device.type == "cuda" else None  # pyright: ignore[reportAttributeAccessIssue]  -- a torch.device here
    elif kind is int:
        index = device
    elif device is None:
        index = torch.cuda.current_device()
    else:
        index = None
    if index is None:
        return torch.cuda.get_device_capability(device)
    cap = _CAPABILITY.get(index)  # pyright: ignore[reportArgumentType]  -- an int here
    if cap is None:
        cap = _CAPABILITY[index] = torch.cuda.get_device_capability(index)  # pyright: ignore[reportArgumentType]
    return cap


def on_h100(*tensors: torch.Tensor) -> bool:
    """Real CUDA tensors on an sm_90 device, outside ``torch.compile`` tracing and fake-tensor mode (the raw extension calls of the H100 CUDA wrappers
    are not traceable): the precondition of every one of them."""
    t = tensors[0]
    if t.device.type != "cuda" or torch.compiler.is_compiling():
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if any(isinstance(x, FakeTensor) for x in tensors):
        return False
    index = t.device.index if t.device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index) == (9, 0)


def no_triton(x: torch.Tensor) -> bool:
    """H100 / B200 (and no A/B switch): a routed entry point must not launch a Triton kernel on this tensor
    (the cards that run hand-written CUDA, no portable fallback)."""
    return x.is_cuda and torch.cuda.get_device_capability(x.device.index)[0] in (9, 10)

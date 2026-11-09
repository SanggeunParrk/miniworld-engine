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
    return _sms(torch.cuda._get_device_index(device, optional=True))


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

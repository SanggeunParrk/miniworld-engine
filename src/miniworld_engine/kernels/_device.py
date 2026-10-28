"""Device facts the kernel launchers ask for.

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

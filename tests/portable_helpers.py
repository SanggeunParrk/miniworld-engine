"""The one place the module-parity tests spell the internal A/B switch that puts the portable kernels on every card.

``settings.force_portable_kernels`` is the switch; if it is renamed again, this is the only line that changes. A test uses
``with portable_on():`` (or the ``portable`` fixture in the test module) and never names the setting itself.
"""
from __future__ import annotations

import contextlib

import torch

from miniworld_engine import settings

_SWITCH = "force_portable_kernels"


@contextlib.contextmanager
def portable_on(on: bool = True):
    previous = settings.configure(**{_SWITCH: on})
    try:
        yield
    finally:
        settings.configure(**vars(previous))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error in fp64."""
    a, b = a.detach().double(), b.detach().double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()

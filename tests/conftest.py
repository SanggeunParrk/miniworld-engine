"""Shared pytest setup: GPU / card markers skip cleanly where the hardware is absent.

``gpu``  needs a CUDA device; ``h100`` / ``a100`` additionally need that card (sm_90 / sm_80). A test without a GPU marker must run on a
CPU-only node.
"""

from __future__ import annotations

import functools

import pytest

#: marker -> the CUDA capability it needs
CARDS = {"h100": (9, 0), "a100": (8, 0)}


@functools.cache
def _capability() -> tuple[int, int] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return (major, minor)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        needs_card = [m for m in CARDS if item.get_closest_marker(m)]
        if not (item.get_closest_marker("gpu") or needs_card):
            continue
        cap = _capability()
        if cap is None:
            item.add_marker(pytest.mark.skip(reason="needs a CUDA device"))
            continue
        for m in needs_card:
            if cap != CARDS[m]:
                item.add_marker(pytest.mark.skip(reason=f"needs an {m.upper()} (sm_{CARDS[m][0]}{CARDS[m][1]}), this is sm_{cap[0]}{cap[1]}"))

"""Shared pytest setup: GPU / card markers skip cleanly where the hardware is absent.

``gpu``  needs a CUDA device; ``h100`` / ``a100`` / ``rtx5090`` additionally need that card (sm_90 / sm_80 / sm_120). A test without a GPU marker must run on a
CPU-only node.
"""

from __future__ import annotations

import functools

import pytest

#: marker -> the CUDA capability it needs
CARDS = {"h100": (9, 0), "a100": (8, 0), "rtx5090": (12, 0)}


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
                item.add_marker(
                    pytest.mark.skip(reason=f"needs an {m.upper()} (sm_{CARDS[m][0]}{CARDS[m][1]}), this is sm_{cap[0]}{cap[1]}")
                )


@pytest.fixture(autouse=True)
def _restore_tf32_flags():
    """The measure harness (``speed.make_step`` / ``accuracy``) sets ``torch.backends.{cuda.matmul,cudnn}.allow_tf32`` process-wide; a test that builds a
    step through it would leak that into every later test of the process (the B200 sampler glue test compares against fp32 PyTorch and fails under TF32).
    Restore the flags after each test (n14 integration: found when the atom DiT FP8 test file ran before ``test_misc_b200.py`` in one process)."""
    import sys

    torch = sys.modules.get("torch")
    if torch is None or not hasattr(torch, "backends"):
        yield
        return
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags

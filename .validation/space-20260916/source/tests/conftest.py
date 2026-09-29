"""A test that needs a card must SKIP without one, not fail.

`-m "not gpu"` is how CI runs the CPU suite, so the marker was never exercised anywhere else --
and a plain `pytest tests/` on a login node produced eleven red failures whose whole content was
"Found no NVIDIA driver on your system". A suite that is red by default on the machine most people
type into is a suite whose red stops meaning anything, which is the same failure mode the
collection guard exists to prevent from the other direction.

Collection is untouched: the items are still collected and still counted, so
`--collect-only -m gpu` reports the same list it always did.
"""
from __future__ import annotations

# `tests/paths.py` holds the repo paths and the two file readers that 27 test modules used to
# spell out for themselves. pytest puts each test's OWN directory on sys.path (there are no
# __init__.py files here), not this one, so it goes on explicitly.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest


def pytest_collection_modifyitems(config, items) -> None:
    import torch

    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device on this machine")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)

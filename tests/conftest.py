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


def _default_mathdx_home() -> None:
    """The three CUDA transition extensions include cuBLASDx (kernels/_nvcc.py ``mathdx_home``). When no variable names a mathdx root
    (``MINIWORLD_MATHDX_HOME`` / ``MATHDX_HOME`` / ``NVIDIA_MATHDX_HOME``),
    point ``MATHDX_HOME`` at the unpacked wheel the cluster keeps in ``~/mathdx_dl`` -- the same role
    ``CUTLASS_PATH`` plays for the CUTLASS kernels: a test run needs no export. Only a root that really holds the headers is used."""
    import os

    if any(os.environ.get(v) for v in ("MINIWORLD_MATHDX_HOME", "MATHDX_HOME", "NVIDIA_MATHDX_HOME")):
        return
    root = Path.home() / "mathdx_dl" / "extracted" / "nvidia" / "mathdx"
    if (root / "include" / "cublasdx.hpp").is_file():
        os.environ["MATHDX_HOME"] = str(root)


_default_mathdx_home()


def pytest_collection_modifyitems(config, items) -> None:
    import torch

    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device on this machine")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def force_portable():
    """The internal A/B switch ``settings.force_portable_kernels`` for one test: the engine's Triton kernel family on every card, every CUDA stage off."""
    from miniworld_engine import settings

    previous = settings.configure(force_portable_kernels=True)
    yield
    settings.configure(**vars(previous))

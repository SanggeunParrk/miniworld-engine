"""The anthropic backends select the rows the model kits run in-model (their fast rows), not a step-aside, wherever the stack loads them.

CPU only: the selection logic (no kernel runs). The kits run their sealed CUDA-13.0 cubins (trimul_native, esm_v61, the ef2_t16 transition
cubin) under a CUDA-13 driver; below it those rows are excluded by name (a launch dumps core), at or above it they are served."""

import pytest

from miniworld_engine.modules import _anthropic
from miniworld_engine.modules.transition import Transition
from miniworld_engine.modules.triangle import TriangleMultiplication


@pytest.fixture
def driver(monkeypatch):
    def set_version(v: int) -> None:
        monkeypatch.setattr(_anthropic, "driver_version", lambda: v)

    return set_version


def test_cuda13_cubins_follow_the_driver(driver):
    driver(12090)
    assert not _anthropic.cuda13_cubins_load()
    driver(13000)
    assert _anthropic.cuda13_cubins_load()
    driver(0)  # no libcuda: nothing CUDA-13 is assumed loadable
    assert not _anthropic.cuda13_cubins_load()


def test_trimul_excludes_the_sealed_rows_only_below_cuda13(driver, monkeypatch):
    """TriMul asks the provider with ``exclude`` = its CUDA-13 rows on a 12.x driver and with no exclusion on a CUDA-13 driver (the kits'
    ``native`` row then serves, as in their in-model runs)."""
    import torch

    seen = {}

    class Prov:
        @staticmethod
        def select(*args, **kwargs):
            seen["exclude"] = kwargs.get("exclude")
            return "selection"

        class Refusal(Exception):
            pass

    monkeypatch.setattr(_anthropic, "load", lambda *a, **k: Prov)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    m = TriangleMultiplication(d_pair=128)
    pair = torch.zeros(1, 4, 4, 128, dtype=torch.bfloat16)
    driver(12090)
    m._anthropic_select(pair)
    assert seen["exclude"] == TriangleMultiplication._ANTHROPIC_EXCLUDED_ROWS
    assert "native" in seen["exclude"]
    driver(13000)
    m._anthropic_select(pair)
    assert seen["exclude"] is None


def test_transition_pair_folds_the_residual(monkeypatch):
    """The pair transition asks the provider for the residual-folded cell (the kits' block statement ``z += T(z)``: row esm_t16 at
    c 256 x 4 on H100); single / MSA-row transitions keep the update-only cell."""
    import torch

    seen = []

    class Prov:
        @staticmethod
        def select(word, **kwargs):
            seen.append((kwargs["family"], kwargs["residual"]))
            return "selection"

        class Refusal(Exception):
            pass

    monkeypatch.setattr(_anthropic, "load", lambda *a, **k: Prov)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    m = Transition(d_hidden=256, n=4)
    m._anthropic_select(torch.zeros(1, 8, 8, 256))  # pair [B, L, L, c]
    m._anthropic_select(torch.zeros(1, 8, 256))  # single [B, L, c]
    m._anthropic_select(torch.zeros(1, 3, 8, 256))  # MSA rows [B, S, L, c]
    assert seen == [("pair", True), ("single", False), ("rows", False)]

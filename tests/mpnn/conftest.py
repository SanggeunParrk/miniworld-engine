"""The MPNN kernel tests exercise the Triton and PyTorch paths.

On an A100 the four edge families (tail, MLP, LayerNorm backward, dropout mask) run their hand-written CUDA kernels by default (``integrations/mpnn_edge_sm80.py``), and those
have their own tests (``tests/integrations/test_a100_mpnn_edge_gpu.py``): here the switch is off, so every test in this directory keeps testing the kernel it names on any card.
"""

import pytest


@pytest.fixture(autouse=True)
def _triton_edge_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIWORLD_MPNN_EDGE_SM80", "0")

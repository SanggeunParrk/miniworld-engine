"""The MSA pair's portable Triton path: who takes it, and what a call it cannot serve is told.

Before these families existed an `OuterProductMean` / `MSAPairWeightedAveraging` built with `implementation="triton"` ran the
PyTorch statements without a word -- the label named a backend that did not exist for either module. Now the explicit request
either runs the Triton kernels or refuses with the reason, and `miniworld` (auto) falls through to the statements silently.
CPU only: every refusal below is decided before a kernel could launch.
"""
from __future__ import annotations

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels.outer_product_mean.triton.main import (
    refusal as opm_refusal,
)
from miniworld_engine.kernels.pair_weighted_averaging.triton.main import (
    refusal as pwa_refusal,
)
from miniworld_engine.modules import dispatch
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)
from miniworld_engine.modules.outer_product import OuterProductMean


@pytest.fixture(autouse=True)
def restore_policy():
    previous = settings.current()
    yield
    settings.configure(**vars(previous))


def _msa(d=64, dtype=torch.bfloat16):
    return torch.randn(1, 8, 16, d, dtype=dtype)


def test_auto_resolves_the_msa_pair_to_triton():
    assert dispatch.resolve("outer_product_mean", I.MINIWORLD) == dispatch.KernelBackend.TRITON
    assert dispatch.resolve("msa_pair_weighted_averaging", I.MINIWORLD) == dispatch.KernelBackend.TRITON
    settings.configure(engine_backend="triton")
    assert dispatch.resolve("outer_product_mean", I.MINIWORLD) == dispatch.KernelBackend.TRITON


def test_refusals_name_the_reason():
    assert "CUDA" in str(opm_refusal(_msa(), 32, 128, interchain=False))
    assert "interchain" in str(opm_refusal(_msa(), 32, 128, interchain=True))
    assert "CUDA" in str(pwa_refusal(_msa(), torch.randn(1, 16, 16, 128, dtype=torch.bfloat16), 8, 32))


def test_explicit_triton_refuses_rather_than_running_the_statements():
    torch.manual_seed(0)
    opm = OuterProductMean(64, 128, 32, implementation=I.TRITON).bfloat16()
    with pytest.raises(NotImplementedError, match="not on a CUDA device"):
        opm(_msa())
    pwa = MSAPairWeightedAveraging(64, 128, 8, 32, implementation=I.TRITON).bfloat16()
    with pytest.raises(NotImplementedError, match="not on a CUDA device"):
        pwa(_msa(), torch.randn(1, 16, 16, 128, dtype=torch.bfloat16))


def test_auto_falls_through_to_the_statements_where_triton_cannot_serve():
    torch.manual_seed(0)
    auto = OuterProductMean(64, 128, 32, implementation=I.MINIWORLD)
    ref = OuterProductMean(64, 128, 32, implementation=I.PYTORCH)
    ref.load_state_dict(auto.state_dict())
    x = _msa(dtype=torch.float32)
    torch.testing.assert_close(auto(x), ref(x), rtol=0, atol=0)

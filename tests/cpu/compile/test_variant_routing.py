"""The model variants with a Triton leg (OuterProductMean ``groups`` / ``projection_bias``, TriangleAttention gate / out-projection
biases and Chai-1's output forms) follow the same per-card rule as the standard ops: a recorded card routes to the Triton kernels, a call
they refuse warns once (``KernelRefusedWarning``) and runs the PyTorch statements, an unrecorded card runs them too (TriangleAttention
resolves when the module is built; OuterProductMean per call from the tensor's device, which a CPU-tensor test cannot fake). A fake card and a fake manifest (``tests/fake_card.py``); the tensors are CPU ones that read as CUDA."""
from __future__ import annotations

import dataclasses
import warnings
from functools import partial

import pytest
import torch
from fake_card import fake_card, write_manifest

from miniworld_engine import settings
from miniworld_engine.autotune import records
from miniworld_engine.modules import dispatch
from miniworld_engine.modules.dispatch import KernelRefusedWarning
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.outer_product import OuterProductMean
from miniworld_engine.modules.triangle_attention import TriangleAttention

RECORDED = "NVIDIA RTX A6000"      # sm86
OTHER = "NVIDIA A40"                # same capability, never recorded


@pytest.fixture(autouse=True)
def _clean():
    previous = settings.current()
    settings.configure(force_portable_kernels=False, portable_on_hopper_blackwell=False, b200_portable_kernels=False, trimul_impl=None,
                       capture=False)
    for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):
        cache.clear()
    yield
    settings.configure(**dataclasses.asdict(previous))
    for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):
        cache.clear()


@pytest.fixture
def manifests(tmp_path):
    with records.use_manifest_dir(tmp_path):
        yield tmp_path


@pytest.fixture
def cuda_tensors(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))


def _recorded(kind, *calls):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = [call() for call in calls]
    return results, [w for w in caught if issubclass(w.category, kind)]


OPM_VARIANTS = [{"d_pair": 128, "d_hidden": 32, "projection_bias": True}, {"d_pair": 128, "d_hidden": 64, "groups": 8}]


def _opm(options, implementation):
    options = dict(options)
    return OuterProductMean(64, options.pop("d_pair"), options.pop("d_hidden"), implementation=implementation, **options)


@pytest.mark.parametrize("options", OPM_VARIANTS)
def test_an_opm_variant_the_kernels_refuse_warns_once_and_runs_pytorch(monkeypatch, manifests, cuda_tensors, options):
    fake_card(monkeypatch, RECORDED, (8, 6))
    write_manifest(manifests, RECORDED, (8, 6), ["outer_product_mean"])
    mod, ref = _opm(options, ImplementationType.MINIWORLD), _opm(options, ImplementationType.PYTORCH)
    ref.load_state_dict(mod.state_dict())
    x = torch.randn(1, 8, 16, 64)                                 # fp32: the Triton path is bf16
    (y, _), caught = _recorded(KernelRefusedWarning, partial(mod, x), partial(mod, x))
    assert len(caught) == 1
    assert "bf16" in str(caught[0].message)
    torch.testing.assert_close(y, ref(x))


def test_an_opm_variant_on_a_cpu_tensor_is_silent(monkeypatch, manifests):
    fake_card(monkeypatch, RECORDED, (8, 6))
    write_manifest(manifests, RECORDED, (8, 6), ["outer_product_mean"])
    mod = _opm(OPM_VARIANTS[1], ImplementationType.MINIWORLD)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mod(torch.randn(1, 8, 16, 64))


TRI_VARIANTS = [{"gate_bias": True, "out_bias": True}, {"dual_output": True}, {"untransposed_output": True, "starting": False, "transposed_bias": False}]


@pytest.mark.parametrize("options", TRI_VARIANTS)
def test_a_triangle_attention_variant_in_fp32_warns_once_and_runs_pytorch(monkeypatch, manifests, cuda_tensors, options):
    fake_card(monkeypatch, RECORDED, (8, 6))
    write_manifest(manifests, RECORDED, (8, 6), ["triangle_attention"])
    mod = TriangleAttention(64, 4, implementation=ImplementationType.MINIWORLD, p_drop=0.0, **options)
    ref = TriangleAttention(64, 4, implementation=ImplementationType.PYTORCH, p_drop=0.0, **options)
    ref.load_state_dict(mod.state_dict())
    x = torch.randn(1, 12, 12, 64)
    (y, _), caught = _recorded(KernelRefusedWarning, partial(mod, x), partial(mod, x))
    assert len(caught) == 1
    assert "bf16" in str(caught[0].message)
    torch.testing.assert_close(y, ref(x))


@pytest.mark.parametrize("options", TRI_VARIANTS)
def test_a_triangle_attention_variant_on_an_unrecorded_card_runs_pytorch(monkeypatch, manifests, cuda_tensors, options):
    fake_card(monkeypatch, OTHER, (8, 6))
    mod = TriangleAttention(64, 4, implementation=ImplementationType.MINIWORLD, p_drop=0.0, **options)
    ref = TriangleAttention(64, 4, implementation=ImplementationType.PYTORCH, p_drop=0.0, **options)
    ref.load_state_dict(mod.state_dict())
    x = torch.randn(1, 12, 12, 64)
    torch.testing.assert_close(mod(x), ref(x))

"""CPU module calls must stay on CPU even when CUDA and flash are installed."""
import pytest
import torch

from miniworld_engine.modules.swa_atom_attention import module as swa


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_non_cuda_tensor_does_not_query_cuda_capability(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(swa, "_FA2_SPEC", True)
    monkeypatch.setattr(swa, "_FA4_SPEC", True)

    def unexpected_query(_device):
        pytest.fail("A non-CUDA tensor must not query a CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_query)
    assert swa._flash_backend(torch.device(device)) is None


@pytest.mark.parametrize(("major", "expected"), [(7, None), (8, "fa2"), (9, "fa4")])
def test_cuda_selection_keeps_architecture_policy(monkeypatch, major, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (major, 0))
    monkeypatch.setattr(swa, "_FA2_SPEC", True)
    monkeypatch.setattr(swa, "_FA4_SPEC", True)
    assert swa._flash_backend(torch.device("cuda:0")) == expected

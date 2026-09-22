"""Shape-only dispatch must never build a CUDA extension or wait on its lock."""
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.parametrize("wide", [False, True])
def test_supported_fake_transition_never_builds(monkeypatch, wide):
    from miniworld_engine.kernels.transition.cuda import fused_sm90a, fused_wide_sm90a

    backend = fused_wide_sm90a if wide else fused_sm90a
    monkeypatch.setattr(backend, "supported", lambda *args: True)
    monkeypatch.setattr(backend, "_BUILD_FAILED", set() if wide else False)

    def forbidden(*args):
        pytest.fail("FakeTensor availability invoked the CUDA extension builder")

    monkeypatch.setattr(backend, "_ext_for", forbidden)
    with FakeTensorMode():
        x = torch.empty(128, 256 if wide else 128)
        assert backend.available(x, x, x)

"""The A100 norm kernels' gates and fallbacks that need no GPU: a CPU tensor is never served (nothing initialises CUDA), the switch and the engine backend are read at call time, the
dispatching entry points fall back to the PyTorch / Triton paths, and ``rms_norm_modulation`` is the public op."""

import pytest
import torch

from miniworld_engine import ops, settings
from miniworld_engine.kernels.layernorm.cuda import sm80 as ln
from miniworld_engine.kernels.rmsnorm.cuda import sm80 as rms
from miniworld_engine.kernels.rope.cuda import sm80 as rope


@pytest.fixture(autouse=True)
def default_settings():
    previous = settings.current()
    yield
    settings.configure(engine_backend=previous.engine_backend)


def test_cpu_tensors_are_never_served():
    x = torch.randn(8, 128).to(torch.bfloat16)
    w = torch.ones(128)
    assert not ln.supports(x, w, w)
    assert not ln.supports(x, None, None)
    assert not rms.supports(x, w)
    q = torch.randn(2, 8, 4, 32).to(torch.bfloat16)
    cos = torch.randn(2, 8, 16)
    assert not rope.supports_qk(q, q, cos, cos)
    assert not rope.supports_rope(q, cos, cos)
    c = torch.randn(8, 128).to(torch.bfloat16)
    ws = [torch.randn(128, 128).to(torch.bfloat16) for _ in range(3)]
    assert not rms.supports_adamod(x, c, *ws)


def test_the_switch_and_the_backend_are_read_at_call_time(monkeypatch):
    """``enabled`` needs a CUDA device only for the capability: the two switches answer first."""
    monkeypatch.setenv("MINIWORLD_NORMS_SM80", "0")
    assert not ln.enabled(torch.device("cuda", 0))
    monkeypatch.delenv("MINIWORLD_NORMS_SM80")
    settings.configure(engine_backend="triton")
    assert not ln.enabled(torch.device("cuda", 0))


def test_layernorm_kernel_on_cpu_is_the_reference():
    from miniworld_engine.kernels.layernorm.interface import layernorm_kernel

    x = torch.randn(4, 7, 96)
    w, b = torch.randn(96), torch.randn(96)
    y = layernorm_kernel(x, w, b, 1e-5)
    assert torch.allclose(y, torch.nn.functional.layer_norm(x, (96,), w, b, 1e-5), atol=1e-6)


def test_rms_norm_modulation_is_the_dispatching_op():
    """``ops.rms_norm_modulation`` resolves to the family's dispatching entry (the A100 kernels where they serve the call, Triton elsewhere), not to the Triton function itself."""
    from miniworld_engine.kernels.rmsnorm.interface import rms_norm_modulation

    assert ops.rms_norm_modulation is rms_norm_modulation


def test_the_public_triton_entries_keep_their_names():
    from miniworld_engine import kernels

    for name in ("triton_layernorm", "triton_rmsnorm", "triton_rmsnorm_adamod", "triton_rope_3d"):
        assert callable(getattr(kernels, name))

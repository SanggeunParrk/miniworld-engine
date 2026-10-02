"""LayerNorm / RMSNorm on a B200 run as PyTorch ops that ``torch.compile`` fuses; every other card keeps the engine kernel.

``settings.b200_engine_triton`` brings the engine's Triton kernels back on B200 (the A/B switch), and a strict
``engine_backend="triton"`` process still wins. No GPU is needed: the card is faked.

Run: ``pixi run python -m pytest tests/compile/test_b200_norm_dispatch.py -q``
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules import dispatch
from miniworld_engine.modules.dispatch import KernelBackend

#: (op, the engine kernel every card but a B200 keeps)
_NORMS = [("layernorm", KernelBackend.CUDA), ("rmsnorm", KernelBackend.TRITON)]


@pytest.fixture(autouse=True)
def _restore_settings():
    previous = settings.current()
    settings.configure(engine_backend="auto", b200_engine_triton=False)
    yield
    settings.configure(**dataclasses.asdict(previous))


def _card(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)


@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_b200_runs_the_norms_as_pytorch_ops(monkeypatch, op, engine):
    _card(monkeypatch, (10, 0))
    assert dispatch.resolve(op, "miniworld") is KernelBackend.PYTORCH
    assert dispatch.resolve(op, "miniworld", torch.device("cuda", 0)) is KernelBackend.PYTORCH


@pytest.mark.parametrize("capability", [(9, 0), (8, 0), (8, 6)])
@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_other_cards_keep_the_engine_kernel(monkeypatch, op, engine, capability):
    _card(monkeypatch, capability)
    assert dispatch.resolve(op, "miniworld") is engine


@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_without_a_gpu_the_engine_kernel_stays(monkeypatch, op, engine):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert dispatch.resolve(op, "miniworld") is engine


@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_b200_engine_triton_brings_the_engine_kernel_back(monkeypatch, op, engine):
    _card(monkeypatch, (10, 0))
    settings.configure(b200_engine_triton=True)
    assert dispatch.resolve(op, "miniworld") is engine


@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_a_strict_triton_process_still_wins_on_b200(monkeypatch, op, engine):
    _card(monkeypatch, (10, 0))
    settings.configure(engine_backend="triton")
    assert dispatch.resolve(op, "miniworld") is KernelBackend.TRITON


@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_an_explicit_backend_is_never_rewritten(monkeypatch, op, engine):
    _card(monkeypatch, (10, 0))
    assert dispatch.resolve(op, "pytorch") is KernelBackend.PYTORCH
    assert dispatch.resolve(op, "triton") is KernelBackend.TRITON


def test_the_pytorch_stand_in_for_rmsnorm_returns_the_input_dtype(monkeypatch):
    """Torch's rms_norm runs in fp32 under autocast and returns fp32 (torch 2.13); the Triton kernel the B200 policy replaces
    returns the input dtype. A qk-norm feeds the attention kernels, which need q and k in the same dtype."""
    from miniworld_engine.modules import RMSNorm

    def autocast_like(self, x):  # what ``nn.RMSNorm.forward`` returns under CUDA autocast
        return torch.nn.functional.rms_norm(x.float(), self.normalized_shape, self.weight, self.eps)

    _card(monkeypatch, (10, 0))
    monkeypatch.setattr(torch.nn.RMSNorm, "forward", autocast_like)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    stand_in = RMSNorm(16, implementation="miniworld")
    assert stand_in._backend is KernelBackend.PYTORCH
    assert stand_in(x).dtype is torch.bfloat16
    assert RMSNorm(16, implementation="pytorch")(x).dtype is torch.float32  # an explicit request keeps torch's behaviour

"""AdaptiveLayerNorm / ConditionedTransition (no CUDA path yet) on an H100 or B200 run as PyTorch ops that ``torch.compile`` fuses -- the
engine has no Triton fallback on those cards; every other card keeps the engine kernel. LayerNorm and RMSNorm resolve to CUDA on H100 /
B200 (the general kernels, ``kernels/layernorm/cuda/general`` and ``kernels/rmsnorm/cuda``); elsewhere LayerNorm stays CUDA and RMSNorm
Triton.

``settings.portable_on_hopper_blackwell`` (and, on B200 only, the older ``b200_portable_kernels``) bring the engine's Triton kernels back
(the A/B switch), and a strict ``force_portable_kernels=True`` process still wins. No GPU is needed: the card is faked.

Run: ``pixi run python -m pytest tests/cpu/compile/test_b200_norm_dispatch.py -q``
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from fake_card import fake_card

from miniworld_engine import settings
from miniworld_engine.modules import dispatch
from miniworld_engine.modules.dispatch import KernelBackend

#: (op, the engine kernel every card but a B200 keeps)
_NORMS = [("layernorm", KernelBackend.CUDA), ("rmsnorm", KernelBackend.PORTABLE)]


@pytest.fixture(autouse=True)
def _restore_settings():
    previous = settings.current()
    settings.configure(force_portable_kernels=False, b200_portable_kernels=False)
    yield
    settings.configure(**dataclasses.asdict(previous))


_NAMES = {(8, 0): "NVIDIA A100 80GB PCIe", (8, 6): "NVIDIA RTX A6000", (9, 0): "NVIDIA H100 80GB HBM3", (10, 0): "NVIDIA B200"}


def _card(monkeypatch, capability):
    fake_card(monkeypatch, _NAMES[capability], capability)


@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
@pytest.mark.parametrize(("op", "engine"), [("adaptive_layernorm", KernelBackend.PORTABLE), ("conditioned_transition", KernelBackend.PORTABLE)])
def test_h100_runs_adaln_and_conditioned_transition_on_cuda_and_b200_as_pytorch_ops(monkeypatch, op, engine, capability):
    """H100 has hand-written sm_90a kernels for these two (kernels/adaln/cuda, kernels/conditioned_transition/cuda); B200 still runs the PyTorch reference."""
    _card(monkeypatch, capability)
    expected = KernelBackend.CUDA if capability == (9, 0) else KernelBackend.PYTORCH
    assert dispatch.resolve(op, "miniworld") is expected
    assert dispatch.resolve(op, "miniworld", torch.device("cuda", 0)) is expected


@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
@pytest.mark.parametrize("op", ["layernorm", "rmsnorm"])
def test_h100_and_b200_run_the_norms_on_cuda(monkeypatch, op, capability):
    _card(monkeypatch, capability)
    assert dispatch.resolve(op, "miniworld") is KernelBackend.CUDA
    assert dispatch.resolve(op, "miniworld", torch.device("cuda", 0)) is KernelBackend.CUDA


@pytest.mark.parametrize("capability", [(8, 0), (8, 6)])
@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_other_cards_keep_the_engine_kernel_where_they_are_recorded(monkeypatch, op, engine, capability):
    """A100 holds a record for the RMSNorm kernels, the A6000 does not (autotune/manifests): there ``miniworld`` warns and runs PyTorch."""
    _card(monkeypatch, capability)
    if op == "rmsnorm" and capability == (8, 6):
        with pytest.warns(UserWarning, match="not recorded"):
            assert dispatch.resolve(op, "miniworld") is KernelBackend.PYTORCH
    else:
        assert dispatch.resolve(op, "miniworld") is engine


@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_without_a_gpu_the_engine_kernel_stays(monkeypatch, op, engine):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert dispatch.resolve(op, "miniworld") is engine


@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_b200_portable_kernels_brings_the_engine_kernel_back(monkeypatch, op, engine):
    _card(monkeypatch, (10, 0))
    settings.configure(b200_portable_kernels=True)
    assert dispatch.resolve(op, "miniworld") is engine


@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_portable_on_hopper_blackwell_brings_the_engine_kernel_back(monkeypatch, op, engine):
    for capability in ((9, 0), (10, 0)):
        _card(monkeypatch, capability)
        settings.configure(portable_on_hopper_blackwell=True)
        assert dispatch.resolve(op, "miniworld") is engine
        settings.configure(portable_on_hopper_blackwell=False)


@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_a_strict_triton_process_still_wins_on_h100_and_b200(monkeypatch, op, engine, capability):
    _card(monkeypatch, capability)
    settings.configure(force_portable_kernels=True)
    assert dispatch.resolve(op, "miniworld") is KernelBackend.PORTABLE


@pytest.mark.parametrize(("op", "engine"), _NORMS)
def test_an_explicit_backend_is_never_rewritten(monkeypatch, op, engine):
    _card(monkeypatch, (10, 0))
    assert dispatch.resolve(op, "pytorch") is KernelBackend.PYTORCH
    with pytest.raises(ValueError, match="miniworld"):     # "triton" is no value of implementation any more
        dispatch.resolve(op, "triton")


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
    assert stand_in._backend is KernelBackend.CUDA  # a CPU tensor runs the PyTorch statements
    assert stand_in(x).dtype is torch.bfloat16
    assert RMSNorm(16, implementation="pytorch")(x).dtype is torch.float32  # an explicit request keeps torch's behaviour

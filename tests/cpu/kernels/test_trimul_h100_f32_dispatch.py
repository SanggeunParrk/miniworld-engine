"""``integrations.trimul_h100_f32.serves``: the fp32 (TF32) H100 bidirectional TriMul takes only its own contract, and says no everywhere else
without touching CUDA (a CPU tensor, a bf16 pair, a single-direction module, the opt-out switch)."""

import types

import torch

from miniworld_engine.integrations import trimul_h100_f32
from miniworld_engine.modules.exceptions import ImplementationType


def _module(implementation=ImplementationType.MINIWORLD, d=128, hidden=128, outgoing=None):
    m = types.SimpleNamespace(implementation=implementation, d_pair=d, d_hidden=hidden)
    if outgoing is not None:
        m.outgoing = outgoing
    return m


def test_a_cpu_tensor_is_not_served():
    pair = torch.zeros(1, 32, 32, 128)
    assert not trimul_h100_f32.serves(_module(), pair, None, None)


def test_bf16_and_other_widths_are_not_served():
    assert not trimul_h100_f32.serves(_module(), torch.zeros(1, 32, 32, 128, dtype=torch.bfloat16), None, None)
    assert not trimul_h100_f32.serves(_module(d=64, hidden=64), torch.zeros(1, 32, 32, 64), None, None)


def test_a_single_direction_module_is_not_served():
    assert not trimul_h100_f32.serves(_module(outgoing=True), torch.zeros(1, 32, 32, 128), None, None)


def test_other_implementations_are_not_served():
    for impl in (ImplementationType.PYTORCH, ImplementationType.CUEQUIVARIANCE):
        assert not trimul_h100_f32.serves(_module(implementation=impl), torch.zeros(1, 32, 32, 128), None, None)


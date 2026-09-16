"""ESMFold2 block semantics, including active conditioning and both residuals."""

import copy

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.modules.swa_atom_attention import build_attention_params
from miniworld_engine.modules.swa_dit import SWADiTBlock


def inputs(dtype):
    x = torch.randn(2, 7, 32, dtype=dtype, requires_grad=True)
    cond = torch.randn(2, 7, 24, dtype=dtype, requires_grad=True)
    angles = torch.randn(1, 7, 4)
    valid = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    params = build_attention_params(angles.cos(), angles.sin(), valid, 2)
    return x, cond, params


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_init_is_identity_and_modulation_can_learn(dtype):
    torch.manual_seed(19)
    block = SWADiTBlock(32, 24, 4, half_window=2).to(dtype)
    x, cond, params = inputs(dtype)
    out = block(x, cond, params)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    grad = torch.randn_like(out)
    out.backward(grad)
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)
    torch.testing.assert_close(cond.grad, torch.zeros_like(cond), rtol=0, atol=0)
    modulation_grad = block.adaln_modulation[1].weight.grad
    assert isinstance(modulation_grad, torch.Tensor)
    assert torch.isfinite(modulation_grad).all()
    assert modulation_grad[2 * 32:3 * 32].abs().sum() > 0
    assert modulation_grad[5 * 32:6 * 32].abs().sum() > 0


def reference(block, x, cond, params):
    # MiniWorld SWAAtomBlock equations, without calling engine block/FFN forward.
    values = F.linear(F.silu(cond), block.adaln_modulation[1].weight).chunk(6, -1)
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = values
    attn_in = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_a) + shift_a
    x = x + gate_a * block.attn(attn_in, params)
    ffn_in = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_f) + shift_f
    up_a, up_b = F.linear(ffn_in, block.ffn.w_up.weight).chunk(2, -1)
    return x + gate_f * F.linear(F.silu(up_a) * up_b, block.ffn.w_down.weight)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_active_modulation_matches_reference_outputs_and_gradients(dtype):
    torch.manual_seed(23)
    block = SWADiTBlock(32, 24, 4, half_window=2).to(dtype)
    # Zero gates would hide differences in attention and FFN.
    modulation = block.adaln_modulation[1]
    assert isinstance(modulation, torch.nn.Linear)
    with torch.no_grad():
        modulation.weight.normal_(std=0.1)
    expected_block = copy.deepcopy(block)
    x, cond, params = inputs(dtype)
    rx = x.detach().clone().requires_grad_()
    rc = cond.detach().clone().requires_grad_()
    actual = block(x, cond, params)
    expected = reference(expected_block, rx, rc, params)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, rx.grad)
    torch.testing.assert_close(cond.grad, rc.grad)
    for (name, p), (rname, rp) in zip(
        block.named_parameters(), expected_block.named_parameters(), strict=True,
    ):
        assert name == rname
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, rp.grad)


@pytest.mark.parametrize(("width", "expansion", "hidden"), [(128, 2, 256), (128, 4, 512), (384, 2, 512)])
def test_ffn_width_matches_miniworld(width, expansion, hidden):
    block = SWADiTBlock(width, n=expansion)
    assert block.ffn.w_up.weight.shape == (2 * hidden, width)
    assert block.ffn.w_down.weight.shape == (width, hidden)

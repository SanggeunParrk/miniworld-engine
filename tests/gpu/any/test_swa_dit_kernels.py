"""Dispatch and CUDA parity for the complete ESMFold2 SWA DiT kernel path."""
import copy

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine import ops
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.swa_atom_attention import build_attention_params
from miniworld_engine.modules.swa_dit import SWADiTBlock


@pytest.mark.parametrize("implementation", [ImplementationType.MINIWORLD, ImplementationType.TRITON])
def test_all_block_ops_are_reached(monkeypatch, implementation):
    calls = []

    def modulation(x, c, scale, shift, gate, weight=None, eps=1e-5):
        calls.append("modulation")
        assert eps == torch.finfo(torch.float32).eps
        return F.rms_norm(x, (x.shape[-1],), eps=eps) * (1 + F.linear(c, scale)) + F.linear(c, shift), F.linear(c, gate)

    def ffn(x, wa, wb, ws):
        calls.append("ffn")
        return F.linear(F.silu(F.linear(x, wa)) * F.linear(x, wb), ws)

    def residual(x, gate, branch):
        calls.append("residual")
        return x + gate * branch

    monkeypatch.setattr(ops, "rms_norm_modulation", modulation)
    monkeypatch.setattr(ops, "swiglu_ffn", ffn)
    monkeypatch.setattr(ops, "gated_residual", residual)
    model = SWADiTBlock(32, 24, 4, half_window=2, implementation=implementation)
    projection = model.adaln_modulation[1]
    assert isinstance(projection, torch.nn.Linear)
    with torch.no_grad():
        projection.weight.normal_(std=0.1)
    baseline = copy.deepcopy(model)
    baseline.implementation = ImplementationType.PYTORCH
    baseline.ffn.implementation = ImplementationType.PYTORCH
    baseline.attn.implementation = ImplementationType.PYTORCH
    x = torch.randn(2, 7, 32, requires_grad=True)
    c = torch.randn(2, 7, 24, requires_grad=True)
    rx, rc = x.detach().clone().requires_grad_(), c.detach().clone().requires_grad_()
    angle = torch.randn(1, 7, 4)
    ap = build_attention_params(angle.cos(), angle.sin(), torch.ones(2, 7, dtype=torch.bool), 2)
    y, ry = model(x, c, ap), baseline(rx, rc, ap)
    assert calls == ["modulation", "residual", "modulation", "ffn", "residual"]
    torch.testing.assert_close(y, ry)
    dy = torch.randn_like(y)
    y.backward(dy)
    ry.backward(dy)
    for a, b in [(x.grad, rx.grad), (c.grad, rc.grad)]:
        torch.testing.assert_close(a, b)
    for p, rp in zip(model.parameters(), baseline.parameters(), strict=True):
        torch.testing.assert_close(p.grad, rp.grad)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_gated_residual_cuda(dtype, compiled):
    from miniworld_engine.kernels.gated_projection.triton.residual import gated_residual

    tensors = [torch.randn(2, 17, 9, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_() for _ in range(3)]
    refs = [x.detach().clone().requires_grad_() for x in tensors]
    fn = torch.compile(gated_residual, fullgraph=True) if compiled else gated_residual
    y = fn(*tensors)
    ry = refs[0] + refs[1] * refs[2]
    torch.testing.assert_close(y, ry, rtol=0, atol=0)
    dy = torch.randn_like(y)
    y.backward(dy)
    ry.backward(dy)
    for x, rx in zip(tensors, refs, strict=True):
        torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_swa_dit_cuda_forward_backward(dtype, active, compiled):
    torch.manual_seed(51)
    model = SWADiTBlock(128, 128, 4, half_window=4, implementation=ImplementationType.MINIWORLD).cuda().to(dtype)
    projection = model.adaln_modulation[1]
    assert isinstance(projection, torch.nn.Linear)
    if active:
        with torch.no_grad():
            projection.weight.normal_(std=0.01)
    baseline = copy.deepcopy(model)
    baseline.implementation = ImplementationType.PYTORCH
    baseline.ffn.implementation = ImplementationType.PYTORCH
    baseline.attn.implementation = ImplementationType.PYTORCH
    x = torch.randn(2, 17, 128, device="cuda", dtype=dtype, requires_grad=True)
    c = torch.randn_like(x, requires_grad=True)
    rx, rc = x.detach().clone().requires_grad_(), c.detach().clone().requires_grad_()
    angle = torch.randn(1, 17, 16, device="cuda")
    valid = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    valid[1, 13:] = False
    ap = build_attention_params(angle.cos(), angle.sin(), valid, 2)
    run = torch.compile(model, fullgraph=True) if compiled else model
    y, ry = run(x, c, ap), baseline(rx, rc, ap)
    tol = {"atol": 0.04, "rtol": 0.04} if dtype == torch.bfloat16 else {"atol": 0.005, "rtol": 0.01}
    torch.testing.assert_close(y, ry, atol=tol["atol"], rtol=tol["rtol"])
    if not active:
        torch.testing.assert_close(y, x, rtol=0, atol=0)
    dy = torch.randn_like(y)
    y.backward(dy)
    ry.backward(dy)
    pairs = [("x", x.grad, rx.grad), ("cond", c.grad, rc.grad)]
    pairs += [(name, p.grad, dict(baseline.named_parameters())[name].grad) for name, p in model.named_parameters()]
    for name, actual, expected in pairs:
        assert actual is not None, name
        assert torch.isfinite(actual).all(), name
        assert expected is not None, name
        if name in {"x", "cond"}:
            torch.testing.assert_close(actual, expected, atol=tol["atol"], rtol=tol["rtol"], msg=name)
        else:
            # Weight gradients sum across rows. Cancellation makes per-element
            # relative errors near zero misleading; bound BOTH norm and peak error.
            # Fusion retains fp32 intermediates that eager BF16 rounds separately.
            delta = actual.float() - expected.float()
            rel_l2 = delta.norm() / expected.float().norm().clamp_min(1e-12)
            rel_max = delta.abs().max() / expected.float().abs().max().clamp_min(1e-12)
            limit = 0.02 if dtype == torch.bfloat16 else 0.005
            assert rel_l2 < limit, (name, "relative_l2", rel_l2.item())
            assert rel_max < limit, (name, "relative_max", rel_max.item())

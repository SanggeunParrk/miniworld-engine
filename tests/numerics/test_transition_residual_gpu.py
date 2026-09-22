"""Residual fusion keeps the old rounding and contributes only to the input gradient."""
import pytest
import torch
import triton

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("tile", [(32, 32, 16, 1), (64, 64, 32, 4), (128, 128, 64, 4)])
@pytest.mark.parametrize("shape", [(35, 97, 211), (128, 128, 512)])
def test_squeeze_residual_strides_tails_rounding(tile, shape, monkeypatch):
    from miniworld_engine.autotune.shape_key import both_key
    from miniworld_engine.kernels.transition.triton import residual

    bm, bn, bk, group = tile
    cfg = triton.Config({"BLOCK_M1": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": group},
                        num_warps=4, num_stages=3)
    monkeypatch.setattr(residual._squeeze_residual_kernel, "configs", [cfg])
    monkeypatch.setattr(residual._squeeze_residual_kernel, "cache", {})
    m, d, k = shape
    torch.manual_seed(51)
    h = torch.randn(m, k * 2, device="cuda", dtype=torch.bfloat16)[:, ::2] / k**.5
    w = torch.randn(k, d, device="cuda", dtype=torch.bfloat16).T
    r = torch.randn(d, m, device="cuda", dtype=torch.bfloat16).T
    actual = residual.squeeze_residual(h, w, r, both_key(m))
    expected = ((h.float() @ w.float().T).bfloat16().float() + r.float()).bfloat16()
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative < 1e-4


@pytest.mark.parametrize("shape", [(2, 17, 96), (1, 128, 128), (1, 128, 384)])
@pytest.mark.parametrize("n", [2, 4])
def test_transition_residual_all_gradients(shape, n, monkeypatch):
    from miniworld_engine import settings
    from miniworld_engine.modules import Transition

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    settings.configure(engine_backend="triton", transition_residual_fusion=True)
    torch.manual_seed(72)
    module = Transition(shape[-1], n=n, implementation=ImplementationType.TRITON).cuda().bfloat16()
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 2:
                param.normal_(std=shape[-1]**-.5)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    params = [x, *module.parameters()]
    reference = module._old_triton_forward(x) + x
    expected_grads = torch.autograd.grad(reference, params, dy)
    actual = module(x)
    actual_grads = torch.autograd.grad(actual, params, dy)
    for a, e in zip((actual, *actual_grads), (reference, *expected_grads), strict=True):
        relative = (a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-12)
        assert relative < 1e-4

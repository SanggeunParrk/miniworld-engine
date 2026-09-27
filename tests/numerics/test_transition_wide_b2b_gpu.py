"""Explicit experimental b2b: tails, affine zero, residual and all six gradients."""
import pytest
import torch


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from miniworld_engine import settings
    monkeypatch.setattr(settings, '_ACTIVE', settings.current())
    settings.configure(engine_backend='triton', autotune_miss_cap=3)
    torch.manual_seed(513)


@pytest.mark.parametrize('width', [384, 512])
@pytest.mark.parametrize('fused_ln', [False, True])
@pytest.mark.parametrize('bk', [64, 128, 512])
def test_reference_gradients_and_zero_residual(width, fused_ln, bk):
    from miniworld_engine.kernels.transition.triton.wide_b2b import transition_wide_b2b, transition_wide_b2b_fused_ln
    from miniworld_engine.kernels.transition.triton.residual import transition_residual
    x = torch.randn(1, 129, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    gamma = torch.rand(width, device='cuda', requires_grad=True)
    beta = torch.randn(width, device='cuda', requires_grad=True)
    wa = (torch.randn(4*width, width, device='cuda', dtype=torch.bfloat16)/width**.5).requires_grad_()
    wb = torch.randn_like(wa).mul_(width**-.5).requires_grad_()
    ws = (torch.randn(width, 4*width, device='cuda', dtype=torch.bfloat16)/(4*width)**.5).requires_grad_()
    with torch.no_grad():
        gamma[0] = 0
    leaves = (x, gamma, beta, wa, wb, ws)
    dy = torch.randn_like(x)
    config = (dict(BM=64, BN=128, BK=64, num_warps=8, num_stages=3) if bk == 64
              else dict(BM=16, BN=32, BK=bk, num_warps=4, num_stages=1))
    fn = transition_wide_b2b_fused_ln if fused_ln else transition_wide_b2b
    ref = transition_residual(*leaves, 1e-5)
    expected = torch.autograd.grad(ref, leaves, dy)
    actual = fn(*leaves, 1e-5, config=config)
    observed = torch.autograd.grad(actual, leaves, dy)
    for a, e in zip((actual, *observed), (ref, *expected)):
        assert torch.isfinite(a).all()
        rel = (a.float()-e.float()).norm()/e.float().norm().clamp_min(1e-12)
        assert rel < .02, rel
    with torch.no_grad():
        ws.zero_()
    actual = fn(*leaves, 1e-5, config=config)
    dx, dg, db = torch.autograd.grad(actual, (x, gamma, beta), dy)
    torch.testing.assert_close(actual, x, rtol=0, atol=0)
    torch.testing.assert_close(dx, dy, rtol=0, atol=0)
    assert torch.count_nonzero(dg) == 0 and torch.count_nonzero(db) == 0


@pytest.mark.parametrize('width', [384, 512])
@pytest.mark.parametrize('fused_ln', [False, True])
def test_fullgraph_forward_backward(width, fused_ln):
    from miniworld_engine.kernels.transition.triton.wide_b2b import transition_wide_b2b, transition_wide_b2b_fused_ln
    from miniworld_engine.modules import Transition
    model = Transition(width).cuda().bfloat16()
    with torch.no_grad():
        model.squeeze.weight.normal_(std=(4*width)**-.5)
    x = torch.randn(1, 256, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    fn = transition_wide_b2b_fused_ln if fused_ln else transition_wide_b2b
    config = dict(BM=32, BN=64, BK=128, num_warps=4, num_stages=2)
    def call(x):
        return fn(x, model.ln_in.weight, model.ln_in.bias, model.expand_a.weight,
                  model.expand_b.weight, model.squeeze.weight, model.ln_in.eps, config=config)
    compiled = torch.compile(call, fullgraph=True, dynamic=False)
    expected = call(x)
    actual = compiled(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x, *model.parameters()))

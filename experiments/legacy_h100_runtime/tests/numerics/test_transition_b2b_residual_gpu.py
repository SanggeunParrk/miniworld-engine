"""Unified forward: six gradients, affine edge, mode dispatch and fullgraph."""
import pytest
import torch

@pytest.fixture(autouse=True)
def setup(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('SM90 required')
    from miniworld_engine import settings
    monkeypatch.setattr(settings, '_ACTIVE', settings.current())
    settings.configure(engine_backend='triton', autotune_miss_cap=3)
    torch.manual_seed(512)

@pytest.mark.parametrize('d', [128, 256])
@pytest.mark.parametrize('affine', [torch.float32, torch.bfloat16])
def test_gradients_and_residual(d, affine):
    from miniworld_engine.kernels.transition.triton.b2b_residual import transition_b2b_residual
    from miniworld_engine.kernels.transition.triton.residual import transition_residual
    x=torch.randn(1,129,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    g=torch.rand(d,device='cuda',dtype=affine,requires_grad=True)
    b=torch.randn_like(g,requires_grad=True)
    wa=(torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5).requires_grad_()
    wb=(torch.randn_like(wa)*d**-.5).requires_grad_()
    ws=(torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5).requires_grad_()
    with torch.no_grad():g[0]=0
    leaves=(x,g,b,wa,wb,ws);dy=torch.randn_like(x)
    expected=transition_residual(*leaves,1e-5)
    grads=torch.autograd.grad(expected,leaves,dy)
    y=transition_b2b_residual(*leaves,1e-5)
    got=torch.autograd.grad(y,leaves,dy)
    for actual,reference in zip((y,*got),(expected,*grads)):
        assert torch.isfinite(actual).all()
        assert (actual.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12)<.02
    with torch.no_grad():
        inference=transition_b2b_residual(*leaves,1e-5)
        torch.testing.assert_close(inference,y,rtol=0,atol=0)
        ws.zero_()
    y=transition_b2b_residual(*leaves,1e-5)
    dx,dg,db=torch.autograd.grad(y,(x,g,b),dy)
    torch.testing.assert_close(y,x,rtol=0,atol=0)
    torch.testing.assert_close(dx,dy,rtol=0,atol=0)
    assert torch.count_nonzero(dg)==torch.count_nonzero(db)==0

@pytest.mark.parametrize('d', [128, 256])
def test_module_ops_fullgraph(d):
    from miniworld_engine import settings,ops
    from miniworld_engine.modules import Transition
    from miniworld_engine.kernels.transition.triton.b2b_residual import enabled
    model=Transition(d, implementation="triton").cuda().bfloat16()
    with torch.no_grad():model.squeeze.weight.normal_(std=(4*d)**-.5)
    x=torch.randn(1,128,128,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    assert enabled(x,model.expand_a.weight,model.expand_b.weight,model.squeeze.weight)
    def whole(x):
        return ops.transition(x,ln_in_weight=model.ln_in.weight,ln_in_bias=model.ln_in.bias,
            expand_a_weight=model.expand_a.weight,expand_b_weight=model.expand_b.weight,
            squeeze_weight=model.squeeze.weight,n=4,eps=model.ln_in.eps)
    expected=model(x)
    torch.testing.assert_close(whole(x),expected,rtol=0,atol=0)
    compiled=torch.compile(whole,fullgraph=True,dynamic=False)
    actual=compiled(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (x,*model.parameters()))
    settings.configure(transition_force_split=True)
    assert not enabled(x,model.expand_a.weight,model.expand_b.weight,model.squeeze.weight)
    settings.configure(transition_force_split=False,transition_triton_b2b=False)
    assert not enabled(x,model.expand_a.weight,model.expand_b.weight,model.squeeze.weight)

@pytest.mark.parametrize('d', [384, 512])
@pytest.mark.parametrize('variant', [1, 2, 3])
def test_wide_layouts(d, variant):
    from miniworld_engine.kernels.transition.triton.wide_b2b import transition_wide_b2b
    from miniworld_engine.kernels.transition.triton.residual import transition_residual
    x=torch.randn(1,129,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    g=torch.rand(d,device='cuda',requires_grad=True);b=torch.randn_like(g,requires_grad=True)
    wa=(torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5).requires_grad_()
    wb=(torch.randn_like(wa)*d**-.5).requires_grad_()
    ws=(torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5).requires_grad_()
    leaves=(x,g,b,wa,wb,ws);dy=torch.randn_like(x)
    config=dict(BM=64,BN=128,BK=64,num_warps=8,num_stages=3)
    config.update(BO=128) if variant==3 else config.update(PACKED=variant)
    expected=transition_residual(*leaves,1e-5)
    grads=torch.autograd.grad(expected,leaves,dy)
    actual=transition_wide_b2b(*leaves,1e-5,config=config)
    got=torch.autograd.grad(actual,leaves,dy)
    for a,e in zip((actual,*got),(expected,*grads)):
        assert torch.isfinite(a).all()
        assert (a.float()-e.float()).norm()/e.float().norm().clamp_min(1e-12)<.02

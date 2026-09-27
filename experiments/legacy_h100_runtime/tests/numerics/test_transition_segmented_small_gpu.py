"""Small-D segmented b2b: shared forward modes, boundaries and gradient parity."""
import pytest
import torch

@pytest.fixture(autouse=True)
def setup(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9,0):
        pytest.skip('SM90 required')
    from miniworld_engine import settings
    monkeypatch.setattr(settings,'_ACTIVE',settings.current())
    settings.configure(engine_backend='triton',autotune_miss_cap=3)
    torch.manual_seed(817)

@pytest.mark.parametrize('d',[128,256])
@pytest.mark.parametrize('parts',[2,4])
@pytest.mark.parametrize('fused',[False,True])
@pytest.mark.parametrize('full_k',[False,True])
def test_segmented_gradients(d,parts,fused,full_k):
    from miniworld_engine.kernels.transition.triton.wide_b2b import transition_wide_b2b,transition_wide_b2b_fused_ln
    from miniworld_engine.kernels.transition.triton.residual import transition_residual
    x=torch.randn(1,129,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    g=torch.rand(d,device='cuda',requires_grad=True);b=torch.randn_like(g,requires_grad=True)
    with torch.no_grad():g[0]=0
    wa=(torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5).requires_grad_()
    wb=(torch.randn_like(wa)*d**-.5).requires_grad_()
    ws=(torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5).requires_grad_()
    leaves=(x,g,b,wa,wb,ws);dy=torch.randn_like(x)
    config=dict(BM=64,BN=32,BK=d if full_k else 64,BO=d//parts,num_warps=8,num_stages=2)
    fn=transition_wide_b2b_fused_ln if fused else transition_wide_b2b
    expected=transition_residual(*leaves,1e-5);ref_grads=torch.autograd.grad(expected,leaves,dy)
    actual=fn(*leaves,1e-5,config=config);grads=torch.autograd.grad(actual,leaves,dy)
    for a,e in zip((actual,*grads),(expected,*ref_grads)):
        assert torch.isfinite(a).all()
        assert (a.float()-e.float()).norm()/e.float().norm().clamp_min(1e-12)<.02
    with torch.no_grad():
        inference=fn(*leaves,1e-5,config=config)
        torch.testing.assert_close(inference,actual,rtol=0,atol=0)
        ws.zero_()
    actual=fn(*leaves,1e-5,config=config);dx,dg,db=torch.autograd.grad(actual,(x,g,b),dy)
    torch.testing.assert_close(actual,x,rtol=0,atol=0)
    torch.testing.assert_close(dx,dy,rtol=0,atol=0)
    assert torch.count_nonzero(dg)==torch.count_nonzero(db)==0

@pytest.mark.parametrize('d',[128,256])
@pytest.mark.parametrize('fused',[False,True])
def test_segmented_fullgraph(d,fused):
    from miniworld_engine.kernels.transition.triton.wide_b2b import transition_wide_b2b,transition_wide_b2b_fused_ln
    from miniworld_engine.modules import Transition
    model=Transition(d,implementation='triton').cuda()
    with torch.no_grad():model.squeeze.weight.normal_(std=(4*d)**-.5)
    x=torch.randn(1,257,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    config=dict(BM=64,BN=32,BK=d,BO=d//2,num_warps=8,num_stages=2)
    fn=transition_wide_b2b_fused_ln if fused else transition_wide_b2b
    def call(x):
        return fn(x,model.ln_in.weight,model.ln_in.bias,model.expand_a.weight,model.expand_b.weight,model.squeeze.weight,model.ln_in.eps,config=config)
    expected=call(x);actual=torch.compile(call,dynamic=False,fullgraph=True)(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x,*model.parameters()))

@pytest.mark.parametrize('d',[128,256])
@pytest.mark.parametrize('fused',[False,True])
def test_autotuned_wrapper(d,fused):
    from miniworld_engine.kernels.transition.triton.segmented_residual import transition_segmented
    from miniworld_engine.kernels.transition.triton.residual import transition_residual
    from miniworld_engine.modules import Transition
    model=Transition(d,implementation='triton').cuda()
    with torch.no_grad():model.squeeze.weight.normal_(std=(4*d)**-.5)
    x=torch.randn(1,129,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    params=(x,model.ln_in.weight,model.ln_in.bias,model.expand_a.weight,model.expand_b.weight,model.squeeze.weight)
    dy=torch.randn_like(x)
    expected=transition_residual(*params,model.ln_in.eps)
    ref_grads=torch.autograd.grad(expected,params,dy)
    def call(x):
        return transition_segmented(x,*params[1:],model.ln_in.eps,fused_ln=fused)
    actual=torch.compile(call,fullgraph=True,dynamic=False)(x)
    grads=torch.autograd.grad(actual,params,dy)
    for a,e in zip((actual,*grads),(expected,*ref_grads)):
        assert torch.isfinite(a).all()
        assert (a.float()-e.float()).norm()/e.float().norm().clamp_min(1e-12)<.02

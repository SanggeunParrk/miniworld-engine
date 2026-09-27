"""Both native schedules: FP32 affine, residual, six gradients and graph capture."""
import pytest
import torch


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9,0):
        pytest.skip('SM90 required')
    from miniworld_engine import settings
    monkeypatch.setattr(settings,'_ACTIVE',settings.current())
    settings.configure(engine_backend='triton',autotune_miss_cap=3)
    torch.manual_seed(1918)


def fixture(d,variant):
    from miniworld_engine.kernels.transition.cuda.variants import transition
    x=torch.randn(1,129,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    g=torch.rand(d,device='cuda',requires_grad=True);b=torch.randn_like(g,requires_grad=True)
    with torch.no_grad():g[0]=0
    wa=(torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5).requires_grad_()
    wb=(torch.randn_like(wa)*d**-.5).requires_grad_()
    ws=(torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5).requires_grad_()
    leaves=(x,g,b,wa,wb,ws)
    c=dict(bk=(1<<(d-1).bit_length()) if variant=='full_k' else 64,bn=32,bo=64,mgroups=1,ngroups=2 if d>=384 else 1,stages=2,min_blocks=1)
    def call(x):return transition(x,*leaves[1:],variant=variant,forward_config=c,backward_config=c)
    return leaves,call


def assert_relative(actual,expected,tolerance=.02):
    assert torch.isfinite(actual).all()
    error=(actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)
    assert error<tolerance,error


@pytest.mark.parametrize('variant',['streamed_k','full_k'])
@pytest.mark.parametrize('d',[128,256,384,512])
def test_gradients_and_identity(d,variant):
    from miniworld_engine.kernels.transition.triton.residual import transition_residual
    leaves,call=fixture(d,variant);x,g,b,wa,wb,ws=leaves;dy=torch.randn_like(x)
    ref=transition_residual(*leaves,1e-5);rg=torch.autograd.grad(ref,leaves,dy)
    y=call(x);gg=torch.autograd.grad(y,leaves,dy)
    for a,e in zip((y,*gg),(ref,*rg)):assert_relative(a,e)
    # Independent PyTorch formulation; preserve the forward rounding boundaries.
    xn=torch.nn.functional.layer_norm(x.float(),(d,),g,b,1e-5).bfloat16()
    aa=xn.float()@wa.float().T;bb=xn.float()@wb.float().T
    h=(torch.nn.functional.silu(aa)*bb).bfloat16()
    ref=(h.float()@ws.float().T).bfloat16()+x
    rg=torch.autograd.grad(ref,leaves,dy)
    for a,e in zip((y,*gg),(ref,*rg)):assert_relative(a,e)
    with torch.no_grad():
        inference=call(x);torch.testing.assert_close(inference,y,rtol=0,atol=0)
        ws.zero_()
    y=call(x);dx,dg,db=torch.autograd.grad(y,(x,g,b),dy)
    torch.testing.assert_close(y,x,rtol=0,atol=0)
    torch.testing.assert_close(dx,dy,rtol=0,atol=0)
    assert torch.count_nonzero(dg)==torch.count_nonzero(db)==0


@pytest.mark.parametrize('variant',['streamed_k','full_k'])
@pytest.mark.parametrize('d',[128,256,384,512])
def test_compile_and_cudagraph_training(d,variant):
    # Compile, build the autograd graph and capture on the SAME non-default
    # stream. Autograd routes backward to each forward node's original stream.
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        leaves,call=fixture(d,variant);x=leaves[0];dy=torch.randn_like(x)
        compiled=torch.compile(call,dynamic=False,fullgraph=True)
        y=compiled(x);expected=torch.autograd.grad(y,leaves,dy)
        for _ in range(2):
            out=compiled(x);torch.autograd.grad(out,leaves,dy)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            out=compiled(x);grads=torch.autograd.grad(out,leaves,dy)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(out,y,rtol=0,atol=0)
    for a,e in zip(grads,expected):assert_relative(a,e,.001)


@pytest.mark.parametrize('variant',['streamed_k','full_k'])
def test_module_gradients(variant):
    from miniworld_engine import settings
    from miniworld_engine.modules.transition.module import Transition
    from miniworld_engine.kernels.transition.triton.residual import transition_residual
    d=256
    settings.configure(engine_backend='auto')
    c=dict(bk=d if variant=='full_k' else 64,bn=32,bo=64,mgroups=1,ngroups=1,stages=2,min_blocks=1)
    model=Transition(d,implementation='cuda',cuda_variant=variant,
                     cuda_forward_config=c,cuda_backward_config=c).cuda().bfloat16()
    assert model.ln_in.weight.dtype==torch.float32
    with torch.no_grad():model.squeeze.weight.normal_(std=(4*d)**-.5)
    x=torch.randn(1,129,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    leaves=(x,model.ln_in.weight,model.ln_in.bias,model.expand_a.weight,model.expand_b.weight,model.squeeze.weight)
    dy=torch.randn_like(x)
    ref=transition_residual(*leaves,model.ln_in.eps)
    expected=torch.autograd.grad(ref,leaves,dy)
    compiled=torch.compile(model,dynamic=False,fullgraph=True)
    y=compiled(x);actual=torch.autograd.grad(y,leaves,dy)
    for a,e in zip((y,*actual),(ref,*expected)):assert_relative(a,e)
    model.eval()
    with torch.no_grad():torch.testing.assert_close(model(x),y,rtol=0,atol=0)

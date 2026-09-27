"""Anthropic replacements must preserve the existing training save/fusion policy."""
import torch
import pytest
from miniworld_engine import settings
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import bidirectional_trimul_triton
from miniworld_engine.kernels.trimul_inproj.triton.unidirectional import trimul_triton

@pytest.fixture(autouse=True)
def sm90(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(9,0):pytest.skip('SM90 required')
    monkeypatch.setattr(settings,'_ACTIVE',settings.current())
    settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=24)
    torch.backends.cuda.matmul.allow_tf32=False


def rel(a,b):return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()

def setup(n,kind):
    torch.manual_seed(812);c=128;h=c*(2 if kind=='bidir' else 1)
    x=torch.randn(1,n,n,c,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=[(torch.randn(h,c,device='cuda',dtype=torch.bfloat16)/c**.5).requires_grad_() for _ in range(4)]
    w += [(torch.randn(c,c,device='cuda',dtype=torch.bfloat16)/c**.5).requires_grad_(),(torch.randn(c,h,device='cuda',dtype=torch.bfloat16)/h**.5).requires_grad_()]
    norms=[torch.rand(c,device='cuda',requires_grad=True),torch.randn(c,device='cuda',requires_grad=True),torch.rand(h,device='cuda',requires_grad=True),torch.randn(h,device='cuda',requires_grad=True)]
    with torch.no_grad():norms[0][0]=0;norms[2][0]=0
    leaves=[x,*w,*norms]
    mask=torch.rand(1,n,n,device='cuda')>.2
    ds=(torch.rand(1,1,n,c,device='cuda')>.25).bfloat16()/.75
    def call(backend):
        args=(*leaves,1e-5,1e-5,c)
        if kind=='bidir':return bidirectional_trimul_triton(*args,mask=mask,dropscale=ds,output_backend=backend)
        return trimul_triton(*args,outgoing=kind=='outgoing',mask=mask,dropscale=ds,output_backend=backend)
    return leaves,call,mask,ds


@pytest.mark.parametrize('kind',['bidir','outgoing','incoming'])
@pytest.mark.parametrize('n',[64,72])
def test_forward_backward_and_identical_saves(kind,n):
    leaves,call,m,ds=setup(n,kind);dy=torch.randn_like(leaves[0]);saved={};results={}
    for backend in ('triton','anthropic_saved'):
        sizes=[]
        def pack(t):sizes.append((tuple(t.shape),t.dtype));return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            y=call(backend)
        saved[backend]=sizes
        results[backend]=(y,*torch.autograd.grad(y,leaves,dy))
    assert saved['triton']==saved['anthropic_saved']
    for a,b in zip(results['anthropic_saved'],results['triton']):
        assert torch.isfinite(a).all();assert rel(a,b)<.003


@pytest.mark.parametrize('zero',['dropout','mask','projection'])
def test_identity_zero_gradients(zero):
    leaves,call,m,ds=setup(64,'bidir');dy=torch.randn_like(leaves[0])
    with torch.no_grad():
        if zero=='dropout':ds.zero_()
        elif zero=='mask':m.zero_();leaves[-1].zero_()
        else:leaves[6].zero_()
    y=call('anthropic_saved');g=torch.autograd.grad(y,leaves,dy)
    torch.testing.assert_close(y,leaves[0],rtol=0,atol=0)
    torch.testing.assert_close(g[0],dy,rtol=0,atol=0)
    if zero=='dropout':assert all(torch.count_nonzero(a)==0 for a in g[1:])
    else:assert all(torch.count_nonzero(a)==0 for a in g[1:6])


def test_compile_graph_and_weight_update():
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        leaves,call,m,ds=setup(64,'bidir');dy=torch.randn_like(leaves[0])
        fn=torch.compile(lambda:call('anthropic_saved'),dynamic=False,fullgraph=True)
        for _ in range(3):
            y=fn();expected=torch.autograd.grad(y,leaves,dy)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            y=fn();g=torch.autograd.grad(y,leaves,dy)
        with torch.no_grad():leaves[1].mul_(.7);leaves[6].add_(.01);ds[:,:,::2].zero_()
        graph.replay()
        r=call('triton');rg=torch.autograd.grad(r,leaves,dy)
        for a,b in zip((y,*g),(r,*rg)):assert rel(a,b)<.003
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()


@pytest.mark.parametrize('kind',['bidir','outgoing','incoming'])
def test_module_entry_optimizer_and_eval(kind):
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication,BidirectionalTriangleMultiplication
    # The explicit Anthropic row executes CUDA; don't force global Triton-only dispatch.
    settings.configure(engine_backend='auto')
    cls=BidirectionalTriangleMultiplication if kind=='bidir' else TriangleMultiplication
    kw={} if kind=='bidir' else {'outgoing':kind=='outgoing'}
    model=cls(128,implementation='anthropic',anthropic_row='training_saved',p_drop=.25,**kw).cuda().bfloat16().train()
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim==2:p.normal_(std=p.shape[1]**-.5)
    x=torch.randn(1,64,64,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    m=torch.rand(1,64,device='cuda')>.2
    y=model(x,m);y.float().square().mean().backward()
    assert model.anthropic_selection['save_policy']=='unchanged'
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before=model.to_out.weight.detach().clone()
    torch.optim.SGD(model.parameters(),lr=.1).step()
    assert not torch.equal(before,model.to_out.weight)
    model.eval()
    y=model(x,m)
    with torch.no_grad():r=model(x,m)
    torch.testing.assert_close(y,r,rtol=0,atol=0)


def test_multiple_pending_forwards():
    leaves,call,m,ds=setup(64,'bidir');dy=torch.randn_like(leaves[0])
    y=call('anthropic_saved')+call('anthropic_saved');g=torch.autograd.grad(y,leaves,dy)
    r=call('triton')+call('triton');rg=torch.autograd.grad(r,leaves,dy)
    for a,b in zip((y,*g),(r,*rg)):assert rel(a,b)<.003

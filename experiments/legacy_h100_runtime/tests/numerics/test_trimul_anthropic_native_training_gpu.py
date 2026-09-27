"""Original native forward, reference backward: training baseline contract."""
import os
import pytest
import torch
from miniworld_engine.integrations.anthropic_training import (
    WEIGHT_KEYS, native_ops, triangle_multiplication_training,
)

@pytest.fixture(autouse=True)
def gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('SM90 required')
    if not os.environ.get('TRIMUL_NATIVE_BUILD_DIR'):
        pytest.skip('Original native rebuilt payload required')
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def setup(n, direction, weight_dtype=torch.bfloat16):
    torch.manual_seed(439)
    c, h = 128, 256 if direction == 'bidirectional' else 128
    x = torch.randn(1,n,n,c,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    shapes = [(c,), (c,), (h,c), (h,c), (h,c), (h,c), (h,), (h,), (c,h), (c,c)]
    weights = {}
    for key, shape in zip(WEIGHT_KEYS, shapes):
        v = torch.randn(shape,device='cuda',dtype=weight_dtype if len(shape)==2 else torch.float32)
        v = v / shape[1]**.5 if len(shape)==2 else v*.2 + (1 if key.endswith('_w') else 0)
        weights[key] = v.requires_grad_()
    mask = torch.rand(1,n,n,device='cuda')>.2
    ds = (torch.rand(1,1,n,c,device='cuda')>.25).bfloat16()/.75
    return x, weights, mask, ds


def reference(x, w, mask, direction):
    # Independent row-major mathematical statement; native FP32 GEMM accumulators
    # round only after gating. In/out input norms are separate native operations.
    def norm(t, gamma, beta):
        return torch.nn.functional.layer_norm(t.float(), (t.shape[-1],),
            gamma.float(), beta.float(), 1e-5).bfloat16()
    def mm(t, name):
        return t.float() @ w[name].bfloat16().float().T
    xn = norm(x, w['ln_in_w'], w['ln_in_b'])
    a = (torch.sigmoid(mm(xn,'w_ag'))*mm(xn,'w_ap')*mask[...,None]).bfloat16()
    b = (torch.sigmoid(mm(xn,'w_bg'))*mm(xn,'w_bp')*mask[...,None]).bfloat16()
    if direction == 'bidirectional':
        h = a.shape[-1]//2
        t = torch.cat((torch.einsum('bikd,bjkd->bijd',a[...,:h],b[...,:h]),
                       torch.einsum('bkid,bkjd->bijd',a[...,h:],b[...,h:])), -1)
    else:
        eq = 'bikd,bjkd->bijd' if direction == 'outgoing' else 'bkid,bkjd->bijd'
        t = torch.einsum(eq, a, b)
    xn_gate = norm(x, w['ln_in_w'], w['ln_in_b'])
    on = norm(t, w['ln_out_w'], w['ln_out_b'])
    return (mm(on,'w_o') * torch.sigmoid(mm(xn_gate,'w_og'))).bfloat16()


def relative(a,b):
    return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()


@pytest.mark.parametrize('direction',['outgoing','incoming','bidirectional'])
@pytest.mark.parametrize('n',[64,65])
def test_original_forward_and_all_gradients(direction,n):
    x,w,m,ds = setup(n,direction)
    y = triangle_multiplication_training(x,m,weights=w,direction=direction)
    if direction != 'bidirectional':
        with torch.no_grad():
            orig = native_ops().trimul(x,m,direction=direction,weights=w,residual=False,cache={})
        torch.testing.assert_close(y,orig,rtol=0,atol=0)
    dy = torch.randn_like(x)
    out = x+y*ds
    g = torch.autograd.grad(out,(x,*w.values()),dy)
    ref = x+reference(x,w,m,direction)*ds
    rg = torch.autograd.grad(ref,(x,*w.values()),dy)
    for a,b in zip((out,*g),(ref,*rg)):
        assert torch.isfinite(a).all()
        assert relative(a,b)<.01


@pytest.mark.parametrize('direction',['outgoing','incoming','bidirectional'])
def test_live_weights_and_multiple_pending_forwards(direction):
    x,w,m,ds = setup(64,direction,torch.float32)
    # Two retained forwards must not alias a cached native activation workspace.
    y1 = triangle_multiplication_training(x,m,weights=w,direction=direction)
    y2 = triangle_multiplication_training(x*.8,m,weights=w,direction=direction)
    loss = ((y1+y2)*ds).float().square().mean()
    loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (x,*w.values()))
    ref = (reference(x,w,m,direction)+reference(x*.8,w,m,direction))*ds
    rg = torch.autograd.grad(ref.float().square().mean(),(x,*w.values()))
    for a,b in zip((x,*w.values()),rg): assert relative(a.grad,b)<.015
    before = y1.detach().clone()
    optimizer = torch.optim.SGD(w.values(),lr=.05)
    optimizer.step()
    after = triangle_multiplication_training(x,m,weights=w,direction=direction)
    assert not torch.equal(before,after)
    assert relative(after,reference(x,w,m,direction))<.005


@pytest.mark.parametrize('kind',['outgoing','incoming','bidirectional'])
def test_module_dropout_residual_and_optimizer(kind):
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication,BidirectionalTriangleMultiplication
    cls = BidirectionalTriangleMultiplication if kind=='bidirectional' else TriangleMultiplication
    kwargs = {} if kind=='bidirectional' else {'outgoing':kind=='outgoing'}
    model = cls(128,implementation='anthropic',anthropic_row='native_rebuilt',p_drop=.25,**kwargs).cuda().bfloat16().train()
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim==2:p.normal_(std=p.shape[1]**-.5)
    x = torch.randn(1,64,64,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    mask = torch.rand(1,64,device='cuda')>.2
    y = model(x,mask)
    assert 'original K1' in model.anthropic_selection['forward']
    y.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.isfinite(x.grad).all()
    # Dropout must have an effect in training, then be disabled in eval.
    torch.manual_seed(29);a=model(x,mask)
    torch.manual_seed(38);b=model(x,mask)
    assert not torch.equal(a,b)
    model.eval()
    torch.manual_seed(29);a=model(x,mask)
    torch.manual_seed(38);b=model(x,mask)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    with torch.no_grad(): c=model(x,mask)
    torch.testing.assert_close(a,c,rtol=0,atol=0)


def test_zero_dropout_and_mask():
    x,w,m,ds=setup(64,'bidirectional')
    ds.zero_()
    dy=torch.randn_like(x)
    y=x+triangle_multiplication_training(x,m,weights=w,direction='bidirectional')*ds
    g=torch.autograd.grad(y,(x,*w.values()),dy)
    torch.testing.assert_close(y,x,rtol=0,atol=0)
    torch.testing.assert_close(g[0],dy,rtol=0,atol=0)
    assert all(torch.count_nonzero(v)==0 for v in g[1:])
    m.zero_()
    with torch.no_grad():w['ln_out_b'].zero_()
    y=triangle_multiplication_training(x,m,weights=w,direction='bidirectional')
    assert torch.count_nonzero(y)==0


def test_cuda_graph_replay_changed_weights():
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x,w,m,ds=setup(64,'bidirectional');dy=torch.randn_like(x)
        def call():
            y=x+triangle_multiplication_training(x,m,weights=w,direction='bidirectional')*ds
            return y,torch.autograd.grad(y,(x,*w.values()),dy)
        for _ in range(3):call()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):y,g=call()
        graph.replay()
        yr,gr=call()
        torch.testing.assert_close(y,yr,rtol=0,atol=0)
        for a,b in zip(g,gr):torch.testing.assert_close(a,b,rtol=0,atol=0)
        with torch.no_grad():w['w_o'].mul_(.7);ds[:,:,::2].zero_()
        graph.replay();yr,gr=call()
        torch.testing.assert_close(y,yr,rtol=0,atol=0)
        for a,b in zip(g,gr):torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

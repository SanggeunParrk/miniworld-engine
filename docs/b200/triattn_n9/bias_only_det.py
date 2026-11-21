import sys, torch
from miniworld_engine.modules import triangle as T
L=int(sys.argv[1]); torch.backends.cuda.matmul.allow_tf32=True
for name, mk in (("bidir", lambda: T.BidirectionalTriangleAttention(d_pair=128, backend="cuda")), ("ta_bo", lambda: T.TriangleAttention(d_pair=128, n_head=4, starting=True, use_self_attention=False, p_drop=0.0, backend="cuda"))):
    torch.manual_seed(0)
    m = mk().cuda().float().train()
    with torch.no_grad(): [p.normal_(0,0.1) for p in m.parameters()]
    x = torch.randn(1,L,L,128,device="cuda",dtype=torch.bfloat16,requires_grad=True); mask=torch.ones(1,L,dtype=torch.bool,device="cuda")
    up = torch.randn(1,L,L,128,device="cuda",dtype=torch.bfloat16)
    ps=list(m.parameters())+[x]
    def step():
        for t in ps: t.grad=None
        with torch.autocast("cuda",torch.bfloat16): y=m(x,mask)
        torch.autograd.backward([y],[up]); return [y.detach().clone()]+[t.grad.detach().clone() for t in ps]
    step(); a=step(); b=step()
    eq = all(torch.equal(u,v) for u,v in zip(a,b))
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): step()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for t in ps: t.grad=None
        with torch.autocast("cuda",torch.bfloat16): y=m(x,mask)
        torch.autograd.backward([y],[up])
    g.replay(); torch.cuda.synchronize()
    c=[y.detach()]+[t.grad.detach() for t in ps]
    eqg = all(torch.equal(u,v) for u,v in zip(a,c))
    print(name, "L",L,"eager==eager", eq, "graph==eager", eqg, flush=True)

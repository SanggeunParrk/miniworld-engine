import torch
from miniworld_engine.modules import triangle as T
for cls, kw, L in (("TriangleAttention", dict(d_pair=128, n_head=4, p_drop=0.0, starting=True, use_self_attention=False), 128),
                   ("TriangleAttention", dict(d_pair=128, n_head=4, p_drop=0.0, starting=False, use_self_attention=False), 256),
                   ("TriangleAttention", dict(d_pair=128, n_head=4, p_drop=0.0, starting=True, use_self_attention=False), 384),
                   ("BidirectionalTriangleAttention", dict(d_pair=128, n_head=4), 256)):
    torch.manual_seed(1)
    mod = getattr(T, cls)(**kw, backend="cuda").cuda().float().train()
    for p in mod.parameters(): torch.nn.init.normal_(p, std=0.1) if p.dim() > 1 else torch.nn.init.uniform_(p, 0.5, 1.5)
    x = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
    def step():
        for p in mod.parameters(): p.grad = None
        x.grad = None
        with torch.autocast("cuda", torch.bfloat16): y = mod(x)
        y.backward(dy)
        return [y.detach().clone(), x.grad.clone()] + [p.grad.clone() for p in mod.parameters() if p.grad is not None]
    e = step(); torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): step()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); outs = []
    with torch.cuda.graph(g): outs = step()
    g.replay(); torch.cuda.synchronize()
    ok = all(torch.equal(a, b) for a, b in zip(e, outs))
    print(f"{cls} {kw.get('starting')} L{L}: eager step == graph replay bitwise (output, dx, {len(e)-2} param grads): {ok}", flush=True)

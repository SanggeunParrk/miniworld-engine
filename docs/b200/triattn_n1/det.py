import sys, json, torch
from foldjet_engine.modules import triangle as T
for cls, kw, L in (("TriangleAttention", dict(d_pair=128, n_head=4, p_drop=0.25, starting=True, use_self_attention=False), 384),
                   ("TriangleAttention", dict(d_pair=128, n_head=4, p_drop=0.25, starting=False, use_self_attention=False), 512),
                   ("BidirectionalTriangleAttention", dict(d_pair=128, n_head=4), 384)):
    torch.manual_seed(1)
    mod = getattr(T, cls)(**kw, backend="cuda").cuda().float().train()
    for p in mod.parameters(): torch.nn.init.normal_(p, std=0.1) if p.dim() > 1 else torch.nn.init.uniform_(p, 0.5, 1.5)
    x = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
    outs = []
    for _ in range(3):
        for p in mod.parameters(): p.grad = None
        x.grad = None
        torch.manual_seed(5)
        with torch.autocast("cuda", torch.bfloat16): y = mod(x)
        y.backward(dy); torch.cuda.synchronize()
        outs.append([y.detach().clone(), x.grad.clone()] + [p.grad.clone() for p in mod.parameters() if p.grad is not None])
    same = all(all(torch.equal(a, b) for a, b in zip(outs[0], o)) for o in outs[1:])
    nz = min(float(t.float().abs().max()) for t in outs[0][2:])
    print(f"{cls} {kw} L{L}: 3 runs bitwise identical (output, dx, {len(outs[0])-2} param grads): {same}; min |grad|max {nz:.2e} (non-zero weights N(0, 0.1))", flush=True)

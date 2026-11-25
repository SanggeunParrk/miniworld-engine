import sys, json, torch
import numpy as np
from foldjet_engine.modules import triangle as T
tag = sys.argv[1]
res = {}
for kw, L in [({"d_pair": 64, "n_head": 4, "p_drop": 0.0, "starting": True}, 128), ({"d_pair": 64, "n_head": 2, "p_drop": 0.0, "starting": False}, 256), ({"d_pair": 256, "n_head": 8, "p_drop": 0.0, "starting": True}, 128), ({"d_pair": 64, "n_head": 4, "p_drop": 0.0, "starting": True}, 384)]:
    torch.manual_seed(0); np.random.seed(0)
    mod = T.TriangleAttention(**kw, backend="cuda").cuda().float().train()
    g_ = torch.Generator(device="cpu").manual_seed(7)
    for p in mod.parameters(): p.data.copy_(torch.randn(p.shape, generator=g_) * 0.1 + (1.0 if p.dim() == 1 else 0.0))
    x = torch.randn(1, L, L, kw["d_pair"], device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
    with torch.autocast("cuda", torch.bfloat16): y = mod(x)
    y.backward(dy)
    res[f"{kw['d_pair']}x{kw['n_head']}_{L}_{kw['starting']}"] = [x.grad.clone().cpu(), *[p.grad.clone().cpu() for p in mod.parameters()]]
torch.save(res, f"/tmp/s90_dump_{tag}.pt")
if len(sys.argv) > 2:
    a = torch.load(f"/tmp/s90_dump_{sys.argv[2]}.pt")
    for k in res: print(k, "bitwise equal to", sys.argv[2], ":", all(torch.equal(u, v) for u, v in zip(a[k], res[k])))

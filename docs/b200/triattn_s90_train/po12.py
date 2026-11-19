import sys, json, torch
from miniworld_engine.modules import triangle as T
kw = json.loads(sys.argv[1]); L = int(sys.argv[2])
torch.manual_seed(1)
x0 = torch.randn(1, L, L, kw["d_pair"], device="cuda", dtype=torch.bfloat16); dy0 = torch.randn_like(x0)
def run():
    torch.manual_seed(0)
    m = T.TriangleAttention(**kw, backend="pytorch").cuda().double().train()
    chk = [float(p.double().sum()) for p in m.parameters()]
    x = x0.double().clone().requires_grad_(True)
    y = m(x); y.backward(dy0.double())
    return chk, {n: p.grad.clone() for n, p in m.named_parameters()}, x.grad.clone(), y.detach().clone()
c1, g1, dx1, y1 = run(); c2, g2, dx2, y2 = run()
print("param checksums equal:", c1 == c2, "| y equal:", torch.equal(y1, y2), "| dx equal:", torch.equal(dx1, dx2))
for n in g1: print(n, "grad equal:", torch.equal(g1[n], g2[n]), "norm", float(g1[n].norm()), float(g2[n].norm()))

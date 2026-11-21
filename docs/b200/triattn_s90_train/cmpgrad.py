# cmpgrad.py: training step under CUDA-graph capture with the kv / q graph branch off vs on: output, dx and every parameter gradient bitwise equal; 3 replays of the branched graph bitwise equal
import torch
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.modules import triangle as T
cases = [({"d_pair": 64, "n_head": 4, "p_drop": 0.0, "starting": s}, L) for s in (True, False) for L in (128, 256)] + [
    ({"d_pair": 64, "n_head": 2, "p_drop": 0.0, "starting": True}, 128), ({"d_pair": 128, "n_head": 4, "p_drop": 0.0, "starting": True}, 128),
    ({"d_pair": 128, "n_head": 4, "p_drop": 0.0, "starting": False}, 256), ({"d_pair": 256, "n_head": 8, "p_drop": 0.0, "starting": True}, 256),
    ({"d_pair": 384, "n_head": 12, "p_drop": 0.0, "starting": True}, 128), ({"d_pair": 64, "n_head": 4, "p_drop": 0.0, "starting": True}, 384)]
def run(kw, L, branch):
    K.KVQ_BRANCH = branch
    torch.manual_seed(0)
    mod = T.TriangleAttention(**kw, backend="cuda").cuda().float().train()
    x = torch.randn(1, L, L, kw["d_pair"], device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
    def step():
        for p in mod.parameters(): p.grad = None
        x.grad = None
        torch.manual_seed(1234)
        with torch.autocast("cuda", torch.bfloat16): y = mod(x)
        y.backward(dy)
        return [y.detach().clone(), x.grad.clone(), *[p.grad.clone() for p in mod.parameters()]]
    for _ in range(3): step()
    torch.cuda.synchronize()
    outs = []
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): res = step()
    for _ in range(3):
        g.replay(); torch.cuda.synchronize(); outs.append([t.clone() for t in res])
    return outs
for kw, L in cases[:0]:
    a = run(kw, L, False); b = run(kw, L, True)
    same = all(torch.equal(u, v) for u, v in zip(a[0], b[0]))
    det = all(torch.equal(u, v) for r in b[1:] for u, v in zip(b[0], r))
    print(f"{kw} L{L}: branch on == off bitwise {same}; 3 replays of the branched graph equal {det}; tensors {len(a[0])}", flush=True)
kw, L = cases[0]
a = run(kw, L, False); b = run(kw, L, True)
print("per tensor equal:", [bool(torch.equal(u, v)) for u, v in zip(a[0], b[0])], "max abs diff:", [f"{float((u.float()-v.float()).abs().max()):.2e}" for u, v in zip(a[0], b[0])])
a2 = run(kw, L, False)
print("off vs off (a second module / graph):", [bool(torch.equal(u, v)) for u, v in zip(a[0], a2[0])])
torch.manual_seed(0)
mod = T.TriangleAttention(**kw, backend="cuda").cuda().float().train()
print([n for n, _ in mod.named_parameters()])

# poison.py [graph]: the training step with the allocator's free memory pre-filled with different byte patterns; any output / gradient difference = an uninitialised read
import sys, torch
import numpy as np
from miniworld_engine.modules import triangle as T
graph = len(sys.argv) > 1 and sys.argv[1] == "graph"
RAND = len(sys.argv) > 2 and sys.argv[2] == "rand"
cases = [({"d_pair": 64, "n_head": 4, "p_drop": 0.0, "starting": True}, 128), ({"d_pair": 64, "n_head": 4, "p_drop": 0.25, "starting": True}, 128),
         ({"d_pair": 128, "n_head": 4, "p_drop": 0.0, "starting": True}, 128), ({"d_pair": 256, "n_head": 8, "p_drop": 0.0, "starting": True}, 128),
         ({"d_pair": 64, "n_head": 4, "p_drop": 0.0, "starting": False}, 256),
         ({"d_pair": 128, "n_head": 4, "p_drop": 0.0, "starting": True, "use_self_attention": False}, 128)]
def poison(b):
    big = torch.empty(40 << 30, dtype=torch.uint8, device="cuda"); big.fill_(b); del big
    small = [torch.empty(1 << 16, dtype=torch.uint8, device="cuda").fill_(b) for _ in range(3000)]; del small
def run(kw, L, b):
    poison(b)
    torch.manual_seed(0); np.random.seed(0)
    mod = T.TriangleAttention(**kw, backend="cuda").cuda().float().train()
    if RAND:
        g_ = torch.Generator(device="cpu").manual_seed(7)
        for p in mod.parameters(): p.data.copy_(torch.randn(p.shape, generator=g_) * (0.1 if p.dim() == 2 else 0.2) + (1.0 if p.dim() == 1 and p is mod.ln_pair.weight else 0.0))
    x = torch.randn(1, L, L, kw["d_pair"], device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
    def step():
        for p in mod.parameters(): p.grad = None
        x.grad = None
        torch.manual_seed(1234)
        with torch.autocast("cuda", torch.bfloat16): y = mod(x)
        y.backward(dy)
        return [y.detach().clone(), x.grad.clone(), *[p.grad.clone() for p in mod.parameters()]]
    if not graph:
        step(); torch.cuda.synchronize(); poison(b); return step()
    for _ in range(3): step()
    torch.cuda.synchronize(); poison(b)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): res = step()
    g.replay(); torch.cuda.synchronize()
    return [t.clone() for t in res]
names = ["y", "dx", "ln_w", "ln_b", "to_query", "to_key", "to_value", "to_bias", "to_gate", "to_out"]
for kw, L in cases:
    outs = {b: run(kw, L, b) for b in (0x00, 0xFF, 0x7F)}
    ref = outs[0x00]
    bad = {}
    for b in (0xFF, 0x7F):
        for n, u, v in zip(names, ref, outs[b]):
            if v.shape != u.shape: continue
            if not torch.equal(u, v): bad.setdefault(n, []).append(f"{b:#x}: max abs diff {float((u.float()-v.float()).abs().max()):.3e} nan {int(torch.isnan(v).sum())}")
    print(("graph " if graph else "eager ") + f"{kw} L{L}: " + ("all tensors bitwise equal under 0x00 / 0xFF / 0x7F poison" if not bad else f"DIFFERENT: {bad}"), flush=True)

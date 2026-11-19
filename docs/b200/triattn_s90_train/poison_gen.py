# poison_gen.py: the training step of TriangleMultiplication / Transition (cuda) with the allocator's free memory pre-filled with 0x00 / 0xFF / 0x7F, eager and graph
import sys, torch
import numpy as np
from miniworld_engine.modules import triangle as T
from miniworld_engine.modules.transition import Transition
def poison(b):
    big = torch.empty(40 << 30, dtype=torch.uint8, device="cuda"); big.fill_(b); del big
    small = [torch.empty(1 << 16, dtype=torch.uint8, device="cuda").fill_(b) for _ in range(3000)]; del small
cases = [("TriMul d128 out L128", lambda: T.TriangleMultiplication(128, outgoing=True, p_drop=0.0, backend="cuda"), (1, 128, 128, 128)),
         ("TriMul d128 in L256", lambda: T.TriangleMultiplication(128, outgoing=False, p_drop=0.0, backend="cuda"), (1, 256, 256, 128)),
         ("TriMul d256 out L128", lambda: T.TriangleMultiplication(256, outgoing=True, p_drop=0.0, backend="cuda"), (1, 128, 128, 256)),
         ("Transition 128 L128", lambda: Transition(128, 4, backend="cuda"), (1, 128, 128, 128)),
         ("Transition 128 L256", lambda: Transition(128, 4, backend="cuda"), (1, 256, 256, 128))]
def run(mk, shape, b, graph):
    poison(b); torch.manual_seed(0); np.random.seed(0)
    mod = mk().cuda().float().train()
    g_ = torch.Generator(device="cpu").manual_seed(7)
    for p in mod.parameters(): p.data.copy_(torch.randn(p.shape, generator=g_) * 0.1 + (1.0 if p.dim() == 1 else 0.0))
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
    def step():
        for p in mod.parameters(): p.grad = None
        x.grad = None
        with torch.autocast("cuda", torch.bfloat16): y = mod(x)
        y.backward(dy)
        return [y.detach().clone(), x.grad.clone(), *[p.grad.clone() for p in mod.parameters() if p.grad is not None]]
    if not graph:
        step(); torch.cuda.synchronize(); poison(b); return step()
    for _ in range(3): step()
    torch.cuda.synchronize(); poison(b)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): res = step()
    g.replay(); torch.cuda.synchronize()
    return [t.clone() for t in res]
for name, mk, shape in cases:
    for graph in (False, True):
        try:
            ref = run(mk, shape, 0x00, graph); bad = []
            for b in (0xFF, 0x7F):
                got = run(mk, shape, b, graph)
                bad += [i for i, (u, v) in enumerate(zip(ref, got)) if not torch.equal(u, v)]
            print(f"{name} {'graph' if graph else 'eager'}: " + ("bitwise equal under poison" if not bad else f"DIFFERENT tensors {sorted(set(bad))}"), flush=True)
        except Exception as e:
            print(f"{name} {'graph' if graph else 'eager'}: error {type(e).__name__}: {str(e)[:120]}", flush=True)

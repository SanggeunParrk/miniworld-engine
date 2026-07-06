import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/snu_hwle/psk/mw-bidir/src")
import torch, triton
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication as TMB)
from miniworld_kernels.modules.triangle_multiplication import TriangleMultiplication as TM
from miniworld_kernels.modules import ImplementationType
from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training_b200 import BidirTriMulB200Train

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
d = 128; dev = "cuda"
Ls = [int(x) for x in sys.argv[1:]] or [384, 768, 1024]


def rnd(m):
    for n, p in m.named_parameters():
        if n.endswith("ln_pair.weight") or n.endswith("ln_out.weight"): p.data.normal_(1.0, 0.05)
        elif n.endswith(".bias"): p.data.normal_(0.0, 0.02)
        else: p.data.normal_(0.0, 0.05)
    return m


def bench(fn):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=30, rep=100, return_mode="median")


def make_step(mod, pair, cot):
    def step():
        for p in mod.parameters():
            p.grad = None
        y = mod(pair)
        (y * cot).sum().backward()
        return y
    return step


def graph_step(mod, pair, cot):
    # capture fwd+bwd; validate graph grads == eager grads
    params = [p for p in mod.parameters()]
    step = make_step(mod, pair, cot)
    for _ in range(3): step()
    torch.cuda.synchronize()
    eager_g = [p.grad.detach().clone() for p in params]
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): step()
    torch.cuda.current_stream().wait_stream(s)
    try:
        g = torch.cuda.CUDAGraph()
        for p in params: p.grad = None
        with torch.cuda.graph(g):
            y = mod(pair)
            (y * cot).sum().backward()
    except Exception as e:
        return None, f"CAPFAIL:{type(e).__name__}:{str(e)[:50]}"
    g.replay(); torch.cuda.synchronize()
    worst = 1.0
    for p, ge in zip(params, eager_g):
        if p.grad is None: worst = 0; continue
        worst = min(worst, torch.nn.functional.cosine_similarity(
            p.grad.float().flatten(), ge.float().flatten(), dim=0).item())
    return bench(lambda: g.replay()), worst


res = {}
for L in Ls:
    pair = torch.randn(1, L, L, d, device=dev, dtype=torch.bfloat16)
    cot = torch.randn(1, L, L, d, device=dev, dtype=torch.bfloat16)

    base = rnd(TMB(d_pair=d, d_hidden=d, implementation=ImplementationType("pytorch")).to(dev))
    cute_m = BidirTriMulB200Train(base).to(dev).to(torch.bfloat16)
    pc = pair.clone().requires_grad_(True)
    tc, wc = graph_step(cute_m, pc, cot)

    om = rnd(TM(d, outgoing=True, implementation=ImplementationType("cuequivariance")).to(dev)).to(torch.bfloat16)
    im = rnd(TM(d, outgoing=False, implementation=ImplementationType("cuequivariance")).to(dev)).to(torch.bfloat16)
    pq = pair.clone().requires_grad_(True)
    def cueq_step():
        for m in (om, im):
            for p in m.parameters(): p.grad = None
        if pq.grad is not None: pq.grad = None
        y = om(pq) + im(pq)
        (y * cot).sum().backward()
        return y
    # graph for cueq
    for _ in range(3): cueq_step()
    torch.cuda.synchronize()
    params_q = [p for m in (om, im) for p in m.parameters()]
    eg = [p.grad.detach().clone() for p in params_q]
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): cueq_step()
    torch.cuda.current_stream().wait_stream(s)
    try:
        gq = torch.cuda.CUDAGraph()
        for p in params_q: p.grad = None
        with torch.cuda.graph(gq):
            yq = om(pq) + im(pq); (yq * cot).sum().backward()
        gq.replay(); torch.cuda.synchronize()
        wq = min(torch.nn.functional.cosine_similarity(p.grad.float().flatten(), e.float().flatten(), dim=0).item()
                 for p, e in zip(params_q, eg) if p.grad is not None)
        tq = bench(lambda: gq.replay())
    except Exception as e:
        tq, wq = None, f"CAPFAIL:{str(e)[:40]}"

    res[L] = (tc, tq)
    sp = f"{tq/tc:.3f}x" if isinstance(tc, float) and isinstance(tq, float) else "n/a"
    print(f"L={L:5d}  cute-train={tc if not isinstance(tc,float) else round(tc,4)} ms (gradcos={wc if isinstance(wc,str) else round(wc,4)})  "
          f"cueq-train(2call)={tq if not isinstance(tq,float) else round(tq,4)} ms (gradcos={wq if isinstance(wq,str) else round(wq,4)})  speedup={sp}", flush=True)

print("=== TRAIN summary (fwd+bwd graph time, grad-cos-validated) ===")
print(f"{'L':>6} | {'cute-train':>12} | {'cueq-train':>12} | {'speedup':>8}")
for L in Ls:
    tc, tq = res[L]
    sp = f"{tq/tc:.3f}x" if isinstance(tc, float) and isinstance(tq, float) else "n/a"
    print(f"{L:>6} | {str(round(tc,4) if isinstance(tc,float) else tc):>12} | {str(round(tq,4) if isinstance(tq,float) else tq):>12} | {sp:>8}")

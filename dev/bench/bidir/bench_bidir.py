import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/snu_hwle/psk/mw-bidir/src")
import torch, triton
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication as TMB)
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication as TM
from miniworld_engine.modules import ImplementationType

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
    return triton.testing.do_bench(fn, warmup=30, rep=200, return_mode="median")


def graph_time(fn):
    with torch.no_grad():
        eager = fn().clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(8): fn()
        torch.cuda.current_stream().wait_stream(s)
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): gout = fn()
        except Exception as e:
            return None, f"CAPFAIL:{type(e).__name__}:{str(e)[:60]}"
        g.replay(); torch.cuda.synchronize()
        rcos = torch.nn.functional.cosine_similarity(
            gout.float().flatten(), eager.float().flatten(), dim=0).item()
        return bench(lambda: g.replay()), rcos


res = {}
for L in Ls:
    pair = torch.randn(1, L, L, d, device=dev, dtype=torch.bfloat16)

    cute_m = rnd(TMB(d_pair=d, d_hidden=d, implementation=ImplementationType("cute")).to(dev))
    cute_m = cute_m.to(torch.bfloat16).eval()

    out_m = rnd(TM(d, outgoing=True, implementation=ImplementationType("cuequivariance")).to(dev)).to(torch.bfloat16).eval()
    in_m = rnd(TM(d, outgoing=False, implementation=ImplementationType("cuequivariance")).to(dev)).to(torch.bfloat16).eval()

    with torch.no_grad():
        tc, cc = graph_time(lambda: cute_m(pair))
        tq, cq = graph_time(lambda: out_m(pair) + in_m(pair))
    res[L] = (tc, tq)
    sp = f"{tq/tc:.3f}x" if isinstance(tc, float) and isinstance(tq, float) else "n/a"
    print(f"L={L:5d}  cute-bidir={tc if not isinstance(tc,float) else round(tc,4)} ms (rcos={cc if isinstance(cc,str) else round(cc,4)})  "
          f"cueq-bidir(2call)={tq if not isinstance(tq,float) else round(tq,4)} ms (rcos={cq if isinstance(cq,str) else round(cq,4)})  speedup={sp}", flush=True)

print("=== summary (graph time, replay-cos-validated) ===")
print(f"{'L':>6} | {'cute-bidir':>12} | {'cueq-bidir':>12} | {'speedup':>8}")
for L in Ls:
    tc, tq = res[L]
    sp = f"{tq/tc:.3f}x" if isinstance(tc, float) and isinstance(tq, float) else "n/a"
    tcs = f"{tc:.4f}" if isinstance(tc, float) else str(tc)
    tqs = f"{tq:.4f}" if isinstance(tq, float) else str(tq)
    print(f"{L:>6} | {tcs:>12} | {tqs:>12} | {sp:>8}")

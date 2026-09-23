"""What a kernel boundary costs. The same four GEMMs, timed alone (20 identical launches back to back, so their tails
and prologues overlap) and cycled in the block's dependency order, which is how the step runs them."""
import statistics
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
from quack.gemm_act import gemm_act  # noqa: E402

L, S = int(sys.argv[1]) if len(sys.argv) > 1 else 768, 5
M, D, dev, bf = S * (int(sys.argv[1]) if len(sys.argv) > 1 else 768), 768, "cuda", torch.bfloat16


def time_us(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn()
    out = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / reps)
    return statistics.median(out)


xa = torch.randn(M, D, device=dev, dtype=bf) * D ** -0.5
wq = (torch.randn(4 * D, D, device=dev) * D ** -0.5).to(bf); bq = torch.randn(4 * D, device=dev, dtype=bf)
wo = (torch.randn(D, D, device=dev) * D ** -0.5).to(bf)
wab = (torch.randn(4 * D, D, device=dev) * D ** -0.5).to(bf)
ws = (torch.randn(D, 2 * D, device=dev) * (2 * D) ** -0.5).to(bf)
qkvg = torch.empty(M, 4 * D, device=dev, dtype=bf)
y = torch.empty(M, D, device=dev, dtype=bf)
h = torch.empty(M, 2 * D, device=dev, dtype=bf)

ops = {
    "qkvg  (cuBLAS addmm)": lambda: torch.addmm(bq, xa, wq.t(), out=qkvg),
    "Wo    (cuBLAS mm)": lambda: torch.mm(qkvg[:, :D], wo.t(), out=y),
    "expand+swiglu (quack)": lambda: gemm_act(xa[None], wab[None], None, None, h[None], None, "swiglu", 128, 192, 2, 1, pingpong=True),
    "squeeze (cuBLAS mm)": lambda: torch.mm(h, ws.t(), out=y),
}
alone = {}
for nm, fn in ops.items():
    alone[nm] = time_us(fn, 20)
    print(f"  {nm:<24s} alone {alone[nm]:6.2f} us", flush=True)


def block():
    for fn in ops.values():
        fn()


cyc = time_us(block, 8)
print(f"L={L} M={M}: sum alone {sum(alone.values()):6.2f} us   cycled in block order {cyc:6.2f} us   "
      f"boundary cost {cyc - sum(alone.values()):5.2f} us over 4 kernels", flush=True)

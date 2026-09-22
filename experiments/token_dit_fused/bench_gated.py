"""expand + SwiGLU at the token DiT shape: cuBLAS mm + swiglu_rows (today) vs quack gemm_gated_out (SwiGLU in the epilogue)."""
import statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit import kernels as K
from quack.gemm_act import gemm_act

def time_us(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(5):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps): g.replay()
        en.record(); torch.cuda.synchronize(); out.append(st.elapsed_time(en) * 1000 / reps)
    return statistics.median(out)

D, ND, dev, bf = 768, 1536, "cuda", torch.bfloat16
torch.manual_seed(0)
wa, wb = (torch.randn(ND, D, device=dev) * D ** -0.5 for _ in range(2))
wab = torch.cat([wa, wb], 0).to(bf).contiguous()                       # [2ND, D], nn.Linear layout
Bi = torch.stack([wa, wb], 1).reshape(2 * ND, D).to(bf).contiguous()[None]   # (1, 2ND, D): rows a0, b0, a1, b1, ...
CFGS = [(tm, tn, cm, pp) for tm in (128,) for tn in (128, 192, 256) for cm in (1, 2) for pp in (False, True)]
for L in (384, 768):
    M = 5 * L
    x = torch.randn(M, D, device=dev, dtype=bf)
    ref = (torch.nn.functional.silu(x.float() @ wa.t()) * (x.float() @ wb.t()))
    ab = torch.empty(M, 2 * ND, device=dev, dtype=bf); h1 = torch.empty(M, ND, device=dev, dtype=bf)
    h2 = torch.empty(M, ND, device=dev, dtype=bf)
    def today():
        torch.mm(x, wab.t(), out=ab); K.swiglu_rows(ab, h1)
    def mk(tm, tn, cm, pp):
        return lambda: gemm_act(x[None], Bi, None, None, h2[None], None, "swiglu", tm, tn, cm, 1, pingpong=pp)
    today(); torch.cuda.synchronize()
    r = lambda h: float((h.float() - ref).norm() / ref.norm())
    print(f"L={L} M={M}: cuBLAS mm + swiglu_rows {time_us(today):6.1f} us (rel {r(h1):.2e})", flush=True)
    for c in CFGS:
        try:
            fn = mk(*c); fn(); torch.cuda.synchronize()
            print(f"   quack gated tile {c[0]}x{c[1]} cluster {c[2]} pingpong {c[3]!s:5s} {time_us(fn):6.1f} us (rel {r(h2):.2e})", flush=True)
        except Exception as e:
            print(f"   quack gated {c}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)

"""expand + SwiGLU at the token DiT shape: cuBLAS mm + swiglu_rows (today) vs quack gemm_gated_out (SwiGLU in the epilogue)."""
import statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit import kernels as K
from quack.gemm_interface import gemm_gated_out, _concat_interleave

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
B = _concat_interleave(wab.t().contiguous())                            # [D, 2ND]: gate/up columns alternate
for L in (384, 768):
    M = 5 * L
    x = torch.randn(M, D, device=dev, dtype=bf)
    ref = (torch.nn.functional.silu(x.float() @ wa.t()) * (x.float() @ wb.t()))
    ab = torch.empty(M, 2 * ND, device=dev, dtype=bf); h1 = torch.empty(M, ND, device=dev, dtype=bf)
    h2 = torch.empty(M, ND, device=dev, dtype=bf)
    def today():
        torch.mm(x, wab.t(), out=ab); K.swiglu_rows(ab, h1)
    def quack():
        gemm_gated_out(x, B, None, h2, activation="swiglu")
    today(); quack(); torch.cuda.synchronize()
    r = lambda h: float((h.float() - ref).norm() / ref.norm())
    print(f"L={L} M={M}: cuBLAS+swiglu_rows {time_us(today):6.1f} us (rel {r(h1):.2e}) | quack gemm_gated {time_us(quack):6.1f} us (rel {r(h2):.2e})", flush=True)

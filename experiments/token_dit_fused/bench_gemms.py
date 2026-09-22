"""The block's three plain GEMMs at the token DiT shape: cuBLAS (torch) vs quack gemm (sm90, cluster / pingpong sweep)."""
import statistics
import torch
from quack.gemm import gemm as qgemm

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

dev, bf = "cuda", torch.bfloat16
CFGS = [(128, tn, cm, pp) for tn in (128, 192, 256) for cm in (1, 2) for pp in (False, True) if not (pp and tn > 208)]
CFGS += [(64, tn, cm, False) for tn in (128, 256) for cm in (1, 2)]
for L in (384, 768):
    M = 5 * L
    for name, K, N, has_bias in (("qkvg 768->3072 +bias", 768, 3072, True), ("Wo 768->768", 768, 768, False),
                                 ("squeeze 1536->768", 1536, 768, False)):
        x = torch.randn(M, K, device=dev, dtype=bf)
        w = (torch.randn(N, K, device=dev) * K ** -0.5).to(bf).contiguous()
        b = (torch.randn(N, device=dev) * 0.1).to(bf) if has_bias else None
        out = torch.empty(M, N, device=dev, dtype=bf)
        ref = (x.float() @ w.float().t() + (b.float() if has_bias else 0))
        tb = time_us(lambda: torch.addmm(b, x, w.t(), out=out) if has_bias else torch.mm(x, w.t(), out=out))
        best = None
        for c in CFGS:
            try:
                fn = lambda c=c: qgemm(x[None], w[None], out[None], None, None, c[0], c[1], c[2], 1, pingpong=c[3],
                                       rowvec_bias=b[None] if has_bias else None)
                fn(); torch.cuda.synchronize()
                t = time_us(fn)
                if best is None or t < best[0]:
                    best = (t, c, float((out.float() - ref).norm() / ref.norm()))
            except Exception as e:  # noqa: BLE001
                pass
        print(f"L={L} M={M} {name:<22s} cuBLAS {tb:6.1f} us | quack best {best[0]:6.1f} us {best[1]} rel {best[2]:.1e}", flush=True)

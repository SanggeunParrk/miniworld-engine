# cmpb.py DEFS [L...]: core backward, base build vs build with -D DEFS: bitwise equality of dq dk dv dbias, device time (graph), 2 runs determinism
import sys, torch
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels._nvcc import load_extension
dv = [d for d in sys.argv[1].split(",") if d]; LS = [int(x) for x in (sys.argv[2:] or ["128", "256", "512", "768"])]
H = 4
def build(defs, r):
    tag = "_".join(d.replace("=", "") for d in defs) or "base"
    return load_extension(name=f"s90cmp_r{r}_{tag}", sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), f"-DTA_R={r}", *(f"-D{d}" for d in defs)])
def t_of(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(n): g.replay()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n * 1e3
for L in LS:
    torch.manual_seed(0); sh = (1, L, L, H, 32)
    q, k, v, do = (torch.randn(sh, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    bias = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16); E = q.new_empty(0)
    r = K._rows_per_task(L, H); out = {}
    for nm, d in (("base", []), ("var", dv)):
        e = build(d, r)
        o, lse, _ = e.triattn_fwd(q, k, v, bias, 32 ** -0.5, True, E, E); delta = e.triattn_delta(do, o)
        bT = e.triattn_biasT(bias, L) if hasattr(e, "triattn_biasT") else E.float()
        f = lambda: e.triattn_bwd(q, k, v, bias, do, lse, delta, 32 ** -0.5, E.float(), E, E, E, E.float())
        res = [x.clone() for x in f()[:4]]; res2 = [x.clone() for x in f()[:4]]
        out[nm] = (res, t_of(f), all(torch.equal(a, b) for a, b in zip(res, res2)))
    same = [torch.equal(a, b) for a, b in zip(out["base"][0], out["var"][0])]
    mx = [float((a.float() - b.float()).abs().max()) for a, b in zip(out["base"][0], out["var"][0])]
    print(f"L{L}: base {out['base'][1]:.1f} us  var {out['var'][1]:.1f} us ({out['var'][1]/out['base'][1]:.3f}) | bitwise equal dq dk dv db: {same} maxdiff {['%.2e' % m for m in mx]} | det base/var {out['base'][2]}/{out['var'][2]}", flush=True)

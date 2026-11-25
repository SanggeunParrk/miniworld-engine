# kq.py L...: core backward which=0 (two kernels) vs which=3 (one grid): bitwise equality, determinism, graph time, 3+ interleaved rounds
import sys, torch
from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn as K
from foldjet_engine.kernels._nvcc import load_extension
LS = [int(x) for x in (sys.argv[1:] or ["128"])]; H = 4
def build(r):
    return load_extension(name=f"n1kq_r{r}", sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), f"-DTA_R={r}"])
def graph(f):
    for _ in range(3): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); return g
def timeit(g, flush, n=30):
    ts = []
    for _ in range(n):
        flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for L in LS:
    torch.manual_seed(0); sh = (1, L, L, H, 32)
    q, k, v, do = (torch.randn(sh, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    bias = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16); E = q.new_empty(0)
    r = K._rows_per_task(L, H); e = build(r)
    o, lse, _ = e.triattn_fwd(q, k, v, bias, 32 ** -0.5, True, E, E); delta = e.triattn_delta(do, o)
    mk = lambda w, pdl: (lambda: e.triattn_bwd(q, k, v, bias, do, lse, delta, 32 ** -0.5, E.float(), E, E, E, E.float(), False, pdl, w, E))
    out = {}; gs = {}
    for nm, w, pdl in (("two", 0, L == 128), ("one", 3, L == 128)):
        f = mk(w, pdl); r1 = [x.clone() for x in f()[:4]]; r2 = [x.clone() for x in f()[:4]]; torch.cuda.synchronize()
        out[nm] = (r1, all(torch.equal(a, b) for a, b in zip(r1, r2))); gs[nm] = graph(f)
    same = [torch.equal(a, b) for a, b in zip(out["two"][0], out["one"][0])]
    mx = [float((a.float() - b.float()).abs().max()) for a, b in zip(out["two"][0], out["one"][0])]
    tt = {"two": [], "one": []}
    for rd in range(4):
        for nm in ("two", "one", "two", "one"): tt[nm].append(timeit(gs[nm], flush, 15))
    m = {nm: sorted(v)[len(v) // 2] for nm, v in tt.items()}
    print(f"L{L}: two {m['two']:.1f} us  one {m['one']:.1f} us ({m['one']/m['two']:.3f}) | bitwise dq dk dv db: {same} maxdiff {['%.1e' % x for x in mx]} | det two/one {out['two'][1]}/{out['one'][1]}", flush=True)

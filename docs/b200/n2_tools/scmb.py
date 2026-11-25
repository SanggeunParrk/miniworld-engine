# scmb.py M : swa_embed_scatter vs scatter8 alone (graph, CUPTI kernel us, L2 flushed) + bitwise check on contiguous, ragged-run and masked layouts
import statistics, sys, torch
from torch.profiler import profile, ProfilerActivity
from foldjet_engine.kernels.swa_dit.cuda import sm100
M = int(sys.argv[1]); T = 384; dev = torch.device("cuda"); torch.manual_seed(0)
K = sm100.embed_glue(0); ns = sm100.num_sms()
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
def bench(fn, name, key):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(20):
        flush.zero_(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr: g.replay(); torch.cuda.synchronize()
        ts += [e.time_range.end - e.time_range.start for e in pr.events() if key in e.name]
    print(f"{name:30s} {statistics.median(ts):6.1f} us", flush=True)
def layouts():
    base = torch.arange(M, device=dev) // 8
    yield "contig8", torch.ones(M, device=dev, dtype=torch.uint8), base
    runs = torch.cumsum(torch.randint(1, 20, (M,), device=dev), 0); yield "ragged runs", (torch.rand(M, device=dev) > 0.2).to(torch.uint8), (runs % 50000)
    yield "long runs", torch.ones(M, device=dev, dtype=torch.uint8), torch.arange(M, device=dev) // 37
ok = True
for nm, mask, idx in layouts():
    mask = mask.unsqueeze(0).contiguous(); idx = idx.unsqueeze(0).contiguous(); ntok = int(idx.max().item()) + 1
    for bf in (0, 1):
        X = torch.randn(1, M, T, device=dev, dtype=torch.bfloat16 if bf else torch.float32)
        a1 = torch.zeros(1, ntok, T + 1, device=dev); a2 = torch.zeros_like(a1)
        K.scatter((min((M * (T // 8) + 255) // 256, 16 * ns), 1, 1), (256, 1, 1), X, mask, idx, a1, 1, M, ntok, T, 1, bf)
        K.scatter8((min(((M // 8) * (T // 8) + 255) // 256, 16 * ns), 1, 1), (256, 1, 1), X, mask, idx, a2, 1, M, ntok, T, 1, bf)
        torch.cuda.synchronize(); same = torch.equal(a1, a2); ok &= same
        print(f"{nm:12s} bf{bf}: scatter8 bitwise == scatter: {same}", flush=True)
print("PASS" if ok else "FAIL")
mask = torch.ones(1, M, device=dev, dtype=torch.uint8); idx = (torch.arange(M, device=dev) // 8).unsqueeze(0).contiguous(); ntok = (M + 7) // 8
for bf in (0, 1):
    X = torch.randn(1, M, T, device=dev, dtype=torch.bfloat16 if bf else torch.float32); acc = torch.zeros(1, ntok, T + 1, device=dev)
    bench(lambda: K.scatter((min((M * (T // 8) + 255) // 256, 16 * ns), 1, 1), (256, 1, 1), X, mask, idx, acc, 1, M, ntok, T, 1, bf), f"scatter bf{bf}", "scatter_sm100")
    bench(lambda: K.scatter8((min(((M // 8) * (T // 8) + 255) // 256, 16 * ns), 1, 1), (256, 1, 1), X, mask, idx, acc, 1, M, ntok, T, 1, bf), f"scatter8 bf{bf}", "scatter8")

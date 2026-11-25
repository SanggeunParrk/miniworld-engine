import sys, os, torch
from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn as K
from foldjet_engine.kernels._nvcc import load_extension
H = int(os.environ.get("H", "4")); B = int(os.environ.get("B", "1"))
DEFS = [["N1_DRAIN_EARLY"], []]
for L in [int(x) for x in sys.argv[1:]]:
    r = K._rows_per_task(L, H * B)
    es = [load_extension(name=f"n1kvb_r{r}_" + "_".join(d), sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), f"-DTA_R={r}", *("-D" + x for x in d)]) for d in DEFS]
    torch.manual_seed(0); sh = (B, L, L, H, 32)
    q, k, v, do = (torch.randn(sh, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    bias = torch.randn(B, H, L, L, device="cuda", dtype=torch.bfloat16); E = q.new_empty(0)
    o, lse, _ = es[0].triattn_fwd(q, k, v, bias, 32 ** -0.5, True, E, E); delta = es[0].triattn_delta(do, o)
    bT = (bias.float().transpose(-1, -2) * 1.4426950408889634).contiguous()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    gs = []; res = []
    for e in es:
        f = lambda: e.triattn_bwd(q, k, v, bias, do, lse, delta, 32 ** -0.5, E.float(), E, E, E, bT, False, L == 128, 1, E)
        r1 = [x.clone() for x in f()[1:3]]; r2 = [x.clone() for x in f()[1:3]]; torch.cuda.synchronize()
        res.append((r1, all(torch.equal(a, b) for a, b in zip(r1, r2))))
        torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): f()
        g.replay(); torch.cuda.synchronize(); gs.append(g)
    tt = [[], []]
    for rd in range(4):
        for i in (0, 1, 0, 1):
            for _ in range(15):
                flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); gs[i].replay(); b.record(); torch.cuda.synchronize(); tt[i].append(a.elapsed_time(b) * 1e3)
    m = [sorted(x)[len(x) // 2] for x in tt]
    same = [torch.equal(a, b) for a, b in zip(res[0][0], res[1][0])]
    print(f"L{L} H{H} B{B} kv kernel (graph, L2 flushed): old order {m[0]:.1f}  drain after next task's first stage {m[1]:.1f} ({m[1]/m[0]:.3f}) | dk dv bitwise {same} det {res[0][1]}/{res[1][1]}", flush=True)

# abl.py : the head-dim-32 core backward (bwd_kv, bwd_q) with compile-time ablations; per-kernel device time, graph replay (L512 / L768, 4 heads)
import sys, torch
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels._nvcc import load_extension
from torch.profiler import profile, ProfilerActivity
VARS = [v.split("+") if v != "base" else [] for v in (sys.argv[1] if len(sys.argv) > 1 else "base").split(",")]
LS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "512,768").split(",")]
H = 4
for L in LS:
    torch.manual_seed(0)
    sh = (1, L, L, H, 32)
    q, k, v, do = (torch.randn(sh, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    bias = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16)
    r = K._rows_per_task(L, H)
    for dv in VARS:
        tag = "_".join(d.replace("=", "") for d in dv) or "base"
        e = load_extension(name=f"s90abl_r{r}_{tag}", sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), f"-DTA_R={r}", *(f"-D{d}" for d in dv)])
        E = q.new_empty(0)
        o, lse, _ = e.triattn_fwd(q, k, v, bias, 32 ** -0.5, True, E, E)
        delta = e.triattn_delta(do, o)
        f = lambda: e.triattn_bwd(q, k, v, bias, do, lse, delta, 32 ** -0.5, E.float(), E, E, E, E.float())
        for _ in range(3): f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): f()
        for _ in range(3): g.replay()
        torch.cuda.synchronize()
        agg = {}
        for rep in range(2):
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(10): g.replay()
                torch.cuda.synchronize()
            for ev in p.events():
                if ev.device_type.name == "CUDA":
                    nm = ev.name.replace("(anonymous namespace)::", "").replace("void ", "")[:22]
                    agg.setdefault(nm, []).append(ev.device_time)
        import statistics as st
        print(f"L{L} H{H} {tag:22s}: " + "  ".join(f"{a} {st.median(t):.1f}" for a, t in sorted(agg.items(), key=lambda x: -st.median(x[1]))), flush=True)

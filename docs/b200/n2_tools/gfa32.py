# gfa32.py S [N] [bf]: GattnFwd modes alone for the F32IO (training / fp32) build (bf=0) or the bf16 build (bf=1): auto / whole / split / cs2 / cs4, graph median (L2 flushed)
import statistics, sys, torch
from foldjet_engine.kernels.swa_dit.cuda import sm100
S = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 1; bf = int(sys.argv[3]) if len(sys.argv) > 3 else 0
H, D, C = 4, 32, 128
dev = torch.device("cuda"); K = sm100.kernels(0); g = K.gattn if bf else K.gattn32
torch.manual_seed(0)
dt = torch.bfloat16 if bf else torch.float32
Qh = torch.randn(N, H, S, D, device=dev, dtype=dt); kvdt = torch.bfloat16 if (bf or sm100.F32IO_MODE == 2) else torch.float32; Kh, Vh = (torch.randn(N, H, S, D, device=dev, dtype=kvdt) for _ in range(2)); print("F32IO_MODE", sm100.F32IO_MODE)
seq = torch.full((N,), S, device=dev, dtype=torch.int32); flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
print("auto: split", g.splits(N, S), "cluster", g.cluster_split(N, S), "items", N * H * ((S // 128 + 1) // 2), "F32IO" if not bf else "bf16")
variants = [("auto", None, None), ("whole", False, 0), ("split", True, 0), ("cs2", False, 2), ("cs4", False, 4), ("cs8", False, 8)]
res, outs = {}, {}
for nm, sp, cs in variants:
    g.split = sp; g.cs = cs
    O = torch.empty(N * S, C, device=dev, dtype=dt if bf else torch.float32); L = torch.empty(N, H, S, device=dev, dtype=torch.float32)
    try:
        run, _, _ = g.bind(Qh, Kh, Vh, seq, O=O, LSE=L, log2=True)
        run(); torch.cuda.synchronize(); outs[nm] = O.clone()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr): run()
        res[nm] = gr
    except Exception as e:
        print(nm, "FAIL", repr(e)[:300])
tm = {k: [] for k in res}
for rep in range(3):
    for k, gr in res.items():
        ts = []
        for _ in range(25):
            flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); gr.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
        tm[k].append(statistics.median(ts))
print(f"S {S} N {N}: " + "  ".join(f"{k} {statistics.median(tm[k]):.1f}" for k in res))

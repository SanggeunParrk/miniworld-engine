# gfy2.py S MODE TAG... : like gfy but MODE: seq (each object 15 reps in turn), il (interleaved 3 rounds x 15), il1 (interleaved, 3 reps per visit)
import statistics, sys, torch
from torch.profiler import profile, ProfilerActivity
from foldjet_engine.kernels.swa_dit.cuda import sm100
S = int(sys.argv[1]); mode = sys.argv[2]; tags = sys.argv[3:]
H, D, C, N = 4, 32, 128, 1
dev = torch.device("cuda"); torch.manual_seed(0)
Qh, Kh, Vh = (torch.randn(N, H, S, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
seq = torch.full((N,), S, device=dev, dtype=torch.int32); flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
objs = {}
for t in tags:
    nm = "gz_" + t
    sm100.CUBINS[nm] = ("gattn_fwd", ("-DSPLITB=1", f"-D{t}=1"))
    g = sm100.GattnFwd(split_cubin=nm); g.split = True; g.cs = 0
    objs[t] = g
res = {}
for t, g in objs.items():
    O = torch.empty(N * S, C, device=dev, dtype=torch.bfloat16); L = torch.empty(N, H, S, device=dev, dtype=torch.float32)
    run, _, _ = g.bind(Qh, Kh, Vh, seq, O=O, LSE=L, log2=True); run(); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): run()
    res[t] = gr
for t, g in objs.items():
    ws, cnt = g.ws[Qh.device]
    print(f"{t}: ws {ws.data_ptr():x} ({ws.data_ptr() % (1 << 24):x} mod16M) cnt {cnt.data_ptr():x}", flush=True)
print(f"Q {Qh.data_ptr():x} K {Kh.data_ptr():x} V {Vh.data_ptr():x} seq {seq.data_ptr():x}")
def meas(gr, n):
    ts = []
    for _ in range(n):
        flush.zero_(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr: gr.replay(); torch.cuda.synchronize()
        ts += [e.time_range.end - e.time_range.start for e in pr.events() if "gattn_fwd" in e.name]
    return statistics.median(ts)
if mode == "seq":
    print(" ".join(f"{t}:{meas(res[t], 15):.1f}" for t in tags))
else:
    n = 15 if mode == "il" else 3
    for rep in range(3): print(" ".join(f"{t}:{meas(res[t], n):.1f}" for t in tags), flush=True)

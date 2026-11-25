# gfz2.py S : split gattn_fwd time vs the placement of CNT (atomic counters): adjacent small block after L, big isolated block, 128B-offset variants
import statistics, sys, torch
from torch.profiler import profile, ProfilerActivity
from foldjet_engine.kernels.swa_dit.cuda import sm100
S = int(sys.argv[1]); H, D, C, N = 4, 32, 128, 1
dev = torch.device("cuda"); torch.manual_seed(0)
Qh, Kh, Vh = (torch.randn(N, H, S, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
seq = torch.full((N,), S, device=dev, dtype=torch.int32); flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
nws = sm100.num_sms() * 4 * 128 * 33
keep = []
def trial(name, mk_cnt):
    g = sm100.GattnFwd(split_cubin="gattn_fwd_s"); g.split = True; g.cs = 0
    O = torch.empty(N * S, C, device=dev, dtype=torch.bfloat16); L = torch.empty(N, H, S, device=dev, dtype=torch.float32)
    ws = torch.empty(nws, device=dev, dtype=torch.float32); cnt = mk_cnt()
    g.ws[dev] = (ws, cnt)
    keep.append((O, L, ws, cnt))
    run, _, _ = g.bind(Qh, Kh, Vh, seq, O=O, LSE=L, log2=True); run(); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): run()
    ts = []
    for _ in range(12):
        flush.zero_(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr: gr.replay(); torch.cuda.synchronize()
        ts += [e.time_range.end - e.time_range.start for e in pr.events() if "gattn_fwd" in e.name]
    print(f"{name:28s} {statistics.median(ts):6.1f} us  cnt-L {(cnt.data_ptr() - L.data_ptr()):#x} (L bytes {L.numel() * 4:#x})  cnt-ws {(cnt.data_ptr() - ws.data_ptr()):#x}", flush=True)
def small(): return torch.zeros(4096, device=dev, dtype=torch.int32)
def big(): return torch.zeros(1 << 19, device=dev, dtype=torch.int32)
def bigoff(k):
    def f():
        b = torch.zeros((1 << 19) + 4096, device=dev, dtype=torch.int32)
        return b[k:k + 4096]
    return f
for i in range(3): trial(f"small#{i}", small)
for i in range(2): trial(f"big#{i}", big)
for k in (32, 1024, 2048, 8192):
    trial(f"big+{k * 4}B", bigoff(k))
for i in range(2): trial(f"small#{3 + i}", small)

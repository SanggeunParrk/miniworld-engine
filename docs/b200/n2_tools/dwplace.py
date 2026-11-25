# dwplace.py : DwModTf32 kernel time (CUPTI, L2 flushed) against the placement of its workspace gws (and out) -- random MB pads before the allocations
import random, statistics, sys, torch
from torch.profiler import profile, ProfilerActivity
from foldjet_engine.kernels.swa_dit.cuda import sm100
from foldjet_engine.kernels.swa_dit.cuda.sm100 import tf32_bwd
from foldjet_engine.kernels.swa_dit.cuda.sm100 import tf32_fwd as _f
dev = torch.device("cuda"); torch.manual_seed(0); random.seed(5)
M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096; C = 128
K = tf32_bwd.kernels_bwd(0)
dmod = torch.randn(M, 768, device=dev) * 0.05; a = torch.randn(M, 128, device=dev)
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
k = K.dwmod; k.run(dmod, a)  # load
kern = k.k
ma = _f._f32map(dmod, [32, M, 24], [768 * 4, 128], [32, 32, 4], swizzle="128a32")
mb = _f._f32map(a, [32, M, 4], [128 * 4, 128], [32, 32, 4], swizzle="128a32")
keep = []
for t in range(14):
    keep.append(torch.empty(random.randint(0, 40) << 20, dtype=torch.uint8, device=dev))
    out = torch.empty(6 * C, C, device=dev); gws = torch.empty(6 * 8 * 128 * 256, device=dev); keep.append((out, gws))
    def run(): kern((48, 1, 1), (256, 1, 1), ma, mb, ma, mb, ma, mb, ma, mb, gws, int(M), 0, 0, 0, 0, 0, ma, mb, out, out, out, out, None)
    run(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): run()
    ts = []
    for _ in range(15):
        flush.zero_(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr: g.replay(); torch.cuda.synchronize()
        ts += [e.time_range.end - e.time_range.start for e in pr.events() if "dw_tf32" in e.name]
    print(f"{statistics.median(ts):6.1f} us  gws {gws.data_ptr() >> 20}MB (mod16M {(gws.data_ptr() % (1 << 24)) >> 10}KB) out {out.data_ptr() % (1 << 24) >> 10}KB", flush=True)

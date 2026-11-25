# sqprobe.py FLAG... -- n,M ... : kernel durations (profiler, one flushed graph replay) of the D 512 chain with extra -D flags on the squeeze GEMM
import sys, statistics, torch
from foldjet_engine.kernels.transition.cuda import fused_wide_sm100a as fw
a = sys.argv[1:]; i = a.index("--"); fw.SQ_FLAGS = tuple(a[:i]); fw.FOLD_LN = False; cases = a[i + 1:]
from foldjet_engine.modules import Transition
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for c in cases:
    n, M = (int(v) for v in c.split(","))
    torch.manual_seed(0)
    m = Transition(512, n=n, backend="cuda").cuda().eval()
    for p in m.parameters():
        if p.dim() > 1: torch.nn.init.normal_(p, 0, 512 ** -0.5)
    x = torch.randn(1, M, 512, device="cuda", dtype=torch.bfloat16)
    def fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): return m(x)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            flush.zero_(); g.replay()
        torch.cuda.synchronize()
    ev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA and "gemm_nd" in e.name]
    d = [e.time_range.end - e.time_range.start for e in ev]
    print(f"{' '.join(a[:i]) or 'base':16s} n {n} M {M}: squeeze {statistics.median(d):8.1f} us (min {min(d):.1f})", flush=True)

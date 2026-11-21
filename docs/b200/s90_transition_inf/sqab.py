# sqab.py n,M ... : D 512 inference chain: graph median of the step and squeeze kernel duration for the squeeze variants (orig / PARTS / RX), alternated in one process (cubins built per variant on first use)
import statistics, sys, torch
from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as fw
from miniworld_engine.modules import Transition
fw.FOLD_LN = False
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
CFG = {"orig": (False, False), "parts": (False, True), "rx": (True, False)}
for c in sys.argv[1:]:
    n, M = (int(v) for v in c.split(","))
    torch.manual_seed(0)
    m = Transition(512, n=n, backend="cuda").cuda().eval()
    for p in m.parameters():
        if p.dim() > 1: torch.nn.init.normal_(p, 0, 512 ** -0.5)
    x = torch.randn(1, M, 512, device="cuda", dtype=torch.bfloat16)
    gs = {}; outs = {}
    for name, (rx, parts) in CFG.items():
        fw.SQ_RX, fw.SQ_PARTS = rx, parts
        def fn():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): return m(x)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): outs[name] = fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): outs[name] = fn()
        gs[name] = g
    res = {k: [] for k in CFG}; sq = {k: [] for k in CFG}
    for rep in range(3):
        for name, g in gs.items():
            ts = []
            for _ in range(30):
                flush.zero_(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
            res[name].append(statistics.median(ts))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
                for _ in range(3):
                    flush.zero_(); g.replay()
                torch.cuda.synchronize()
            d = [e.time_range.end - e.time_range.start for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA and "gemm_nd" in e.name]
            sq[name].append(statistics.median(d))
    print(f"n {n} M {M}: " + "  ".join(f"{k}: step {statistics.median(res[k]):.1f} squeeze {statistics.median(sq[k]):.1f}" for k in CFG) + f"   max|rx - orig| {(outs['rx'].float() - outs['orig'].float()).abs().max().item():.3e} max|parts - orig| {(outs['parts'].float() - outs['orig'].float()).abs().max().item():.3e}", flush=True)

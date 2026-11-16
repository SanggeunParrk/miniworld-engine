"""Per-kernel breakdown (torch.profiler, CUDA time) of one OPM case per backend: python kprof.py ROW L S BACKEND... [--iters 5]
Calls are spaced by 0.3 s idle (cool-ish clocks); prints kernel name, calls per step, mean us per call."""
import argparse, copy, dataclasses, time
from collections import defaultdict
import torch
from miniworld_engine.measure import config, speed
from miniworld_engine.modules.primitives import unpadded_masks

ap = argparse.ArgumentParser()
ap.add_argument("row", type=int); ap.add_argument("L", type=int); ap.add_argument("S", type=int); ap.add_argument("backends", nargs="+")
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--sustain", type=float, default=0.0)   # > 0: back-to-back for this many seconds before and no idle between the profiled calls
a = ap.parse_args()
dev = torch.device("cuda")
cfg = config.Config(modules=("OuterProductMean",), rows=(a.row,), lengths=(a.L,), depths="min", modes=("eval",), samples="min")
case = dataclasses.replace(next(iter(config.cases(cfg))), depth=a.S)
base = config.build(case, device=dev)
for b in a.backends:
    module = copy.deepcopy(base)
    with unpadded_masks():
        inputs = config.inputs(case, module, device=dev)
    step = speed.make_step(module, inputs, backend=b, train=False, autocast=config.autocast(case), seed=config.seed(case), compile_mode=None)
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    if a.sustain > 0:
        t_end = time.time() + a.sustain
        while time.time() < t_end:
            step()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(a.iters):
            if a.sustain <= 0:
                time.sleep(0.3)
            step()
            torch.cuda.synchronize()
    agg = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name][0] += 1
            agg[e.name][1] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    tot = sum(v[1] for v in agg.values()) / a.iters
    print(f"== row {a.row} L {a.L} S {a.S} {b}: {tot:.1f} us per step (kernels)", flush=True)
    for k, (n, t) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"   {t / a.iters:10.1f} us  x{n / a.iters:.0f}  {k[:150]}", flush=True)
    del module, inputs, step
    torch.cuda.empty_cache()

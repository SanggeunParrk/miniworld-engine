"""Per-kernel time of the backward (CUDA-graph replay, profiler kernel split): python time_parts.py 384 [--extra=...]"""
import argparse
import collections
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
from bench_common import graph_ms  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("length", type=int, nargs="+")
ap.add_argument("--two", type=int, default=-1)
ap.add_argument("--pwx", type=int, default=0)
ap.add_argument("--extra", nargs="*", default=[])
a = ap.parse_args()
ext = TB.build(extra=" ".join(a.extra).split())
for L in a.length:
    mod, x = TB.TA.fixture(L)
    dy = torch.randn_like(x)
    pk = TB.pack(mod)
    bufs = {}
    fn = lambda: TB.backward(ext, x, dy, pk, bufs)  # noqa: E731
    if a.pwx:
        fn = lambda: TB.backward_pwx(ext, x, dy, pk, bufs, nrep=a.pwx)  # noqa: E731
    if a.two >= 0:
        fn = lambda: TB.backward_2k(ext, x, dy, pk, bufs, gpx=a.two)  # noqa: E731
    ms = graph_ms(fn)[0]
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    kt = collections.defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kt[e.name[:40]] += e.device_time_total / 10
    sol = TB.sol_us(x.shape[0])
    print(f"L{L}: {ms*1e3:.1f} us (SoL {sol:.0f} -> {sol/ms/10:.1f}%)  " + "  ".join(f"{k} {v:.1f}" for k, v in sorted(kt.items(), key=lambda kv: -kv[1])), flush=True)

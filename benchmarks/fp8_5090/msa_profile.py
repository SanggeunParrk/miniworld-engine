"""Kernel breakdown of the default RTX 5090 OPM / PWA inference (where the time goes before the e4m3 routes).

    python msa_profile.py opm:cm:cz:L:S[:a] ... pwa:D:C:cz:S:N ...

Eager calls under torch.profiler (3 calls after 3 warm-up calls), device time per kernel name; plus the module's graph time (``graph_time``)."""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _msa_common import graph_time, opm_case, pwa_case  # noqa: E402

for c in sys.argv[1:]:
    kind, spec = c.split(":", 1)
    m, inp = (opm_case if kind == "opm" else pwa_case)(spec)

    def f():
        with torch.no_grad():
            return m(*inp)

    for _ in range(3):
        f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            f()
        torch.cuda.synchronize()
    evs = [e for e in p.key_averages() if e.device_time_total > 0]
    tot = sum(e.device_time_total for e in evs) / 3
    print(f"== {c}: kernels {tot:.0f} us / call, graph {graph_time(f):.0f} us", flush=True)
    for e in sorted(evs, key=lambda e: -e.device_time_total)[:8]:
        print(f"  {e.device_time_total / 3:9.0f} us {100 * e.device_time_total / 3 / tot:5.1f} %  {e.count // 3}x  {e.key[:110]}", flush=True)
    del m, inp
    torch.cuda.empty_cache()

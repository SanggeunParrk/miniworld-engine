"""Per-call time distribution of the sequential backward (events, no graph) + graph time, for a build: python seq_diag.py L [flags]"""
import statistics, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
from bench_common import graph_ms  # noqa: E402

L = int(sys.argv[1]); flags = sys.argv[2:]
ext = TB.build(extra=flags)
mod, x = TB.TA.fixture(L)
dy = torch.randn_like(x)
pk = TB.pack(mod)
nsm = torch.cuda.get_device_properties(0).multi_processor_count
prof = torch.zeros(nsm, 4, dtype=torch.int64, device="cuda") if "-DSEQ_PROF" in flags else None
bufs = {}
for name, fn in [("seq", lambda: TB.backward_seq(ext, x, dy, pk, bufs, prof=prof)), ("two", lambda: TB.backward_2k(ext, x, dy, pk, bufs))]:
    for _ in range(20):
        fn()
    ts = []
    for _ in range(40):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
        if prof is not None and name == "seq" and len(ts) % 10 == 0:
            p = prof.cpu(); t0 = p[:, 0].min()
            print(f"  run {len(ts)}: last CTA start {(p[:,0].max()-t0)/1e3:.1f}  PX end max {(p[:,1].max()-t0)/1e3:.1f}  W end max {(p[:,2].max()-t0)/1e3:.1f} us")
    ts.sort()
    print(f"L{L} {name} {' '.join(flags)}: events min {ts[0]:.1f} med {statistics.median(ts):.1f} max {ts[-1]:.1f} us | graph {graph_ms(fn)[0]*1e3:.1f} us", flush=True)

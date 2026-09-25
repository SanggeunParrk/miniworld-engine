"""Sustained (NVML) and graph-replay training-step time for chosen forward / backward cubins (L384, R = 9):
  FWD=build/tfwd2.cubin BWD=build/tbwd.cubin python step_sim.py"""
import os, sys, torch
sys.argv = sys.argv[:1]
from energy import measure
from common import make_inputs, graph_time
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
M, D, H = 384 * 384, 128, 512
x, wa, wb, ws, g, b = make_inputs(384)
dy = torch.randn_like(x) * 0.1
for pair in os.environ.get("PAIRS", "tfwd2:tbwd").split(","):
    fn, bn = pair.split(":")
    f = FusedFwd2(f"build/{fn}.cubin"); f.set_weights(wa, wb, ws)
    run_i, *_ = f.bind(x, g, b, save=False)
    tr = FusedTrain(f, repl=9, cubin=f"build/{bn}.cubin"); tr.b.xch = "xch" in bn
    st = tr.bind(x, g, b, dy); st(); torch.cuda.synchronize()
    r = measure(st, 28 * M * D * H)
    print(f"{fn:10s} + {bn:12s}: step sustained {r['us_per_call']:6.1f} us {r['W']:4.0f} W  graph20 {graph_time(st):6.1f} us | fwd graph20 {graph_time(run_i):5.1f} us", flush=True)

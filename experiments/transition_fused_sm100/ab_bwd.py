"""Alternating A/B of bf16 backward cubins (v8 forward): bwd cold / sustained / energy and step sustained / cold, L384, R = 9."""
import sys
cubs = sys.argv[1:]; sys.argv = sys.argv[:1]
import torch
from energy import measure
from common import make_inputs, graph_time
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
M = 384 * 384; x, wa, wb, ws, g, b = make_inputs(384); dy = torch.randn_like(x) * 0.1
f = FusedFwd2(); f.set_weights(wa, wb, ws)
import os
REPLS = [int(v) for v in os.environ.get("REPLS", "9").split(",")]
for rep in range(int(os.environ.get("REPS", "2"))):
    for c, R in [(c, R) for c in cubs for R in REPLS]:
        st = FusedTrain(f, repl=R, cubin=f"build/{c}.cubin").bind(x, g, b, dy); st()
        r = measure(st.keep[1], 22 * M * 128 * 512); rs = measure(st, 28 * M * 128 * 512)
        print(f"{c:12s} R{R} bwd cold {graph_time(st.keep[1]):6.1f} sust {r['us_per_call']:6.1f} {r['J_per_call'] * 1e3:6.1f} mJ | step sust {rs['us_per_call']:6.1f} cold {graph_time(st):6.1f}", flush=True)

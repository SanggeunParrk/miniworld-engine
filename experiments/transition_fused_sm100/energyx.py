"""Sustained energy / time of the bf16 backward variants (after the bf16 forward), L384, plus the training step."""
import os, sys, torch
sys.argv = sys.argv[:1]
from energy import measure
from common import make_inputs, graph_time
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
M, D, H = 384 * 384, 128, 512
x, wa, wb, ws, g, b = make_inputs(384); dy = torch.randn_like(x) * 0.1
f = FusedFwd2(); f.set_weights(wa, wb, ws)
for tag, kw in (("v9 R9", dict(repl=9)), ("bwdx R12", dict(repl=12, x=True)), ("bwdx R13", dict(repl=13, x=True))):
    st = FusedTrain(f, **kw).bind(x, g, b, dy); st()
    rb = measure(st.keep[1], 22 * M * D * H); rs = measure(st, 28 * M * D * H)
    print(f"{tag:9s}: bwd sustained {rb['us_per_call']:6.1f} us {rb['J_per_call']*1e3:6.1f} mJ | step sustained {rs['us_per_call']:6.1f} us {rs['W']:4.0f} W | step cold {graph_time(st):6.1f}", flush=True)

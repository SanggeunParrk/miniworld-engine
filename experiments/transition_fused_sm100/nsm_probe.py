"""Does the power-capped card need all 148 SMs? Sustained time of the v8 forward and v9 backward on fewer SMs."""
import sys, torch
sys.argv = sys.argv[:1]
from energy import measure
from common import make_inputs
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
x, wa, wb, ws, g, b = make_inputs(384)
M, D, H = 384 * 384, 128, 512
for nsm in (148, 132, 120):
    f = FusedFwd2(); f.nsm = nsm; f.set_weights(wa, wb, ws)
    run, *_ = f.bind(x, g, b, save=False)
    r = measure(run, 6 * M * D * H)
    tr = FusedTrain(f, repl=9); tr.b.nsm = nsm
    st = tr.bind(x, g, b, torch.randn_like(x) * 0.1); st(); torch.cuda.synchronize()
    rb = measure(st.keep[1], 22 * M * D * H)
    print(f"{nsm} SMs: fwd infer {r['us_per_call']:6.1f} us {r['W']:4.0f} W | bwd {rb['us_per_call']:6.1f} us {rb['W']:4.0f} W", flush=True)

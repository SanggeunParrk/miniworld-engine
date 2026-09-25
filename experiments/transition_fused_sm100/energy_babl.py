"""Energy / time of backward ablation builds (training step = fused forward + backward, L384, sustained) and backward alone."""
import sys, torch
sys.argv = sys.argv[:1]
from energy import measure
from common import make_inputs, graph_time
from fwd_op import FusedFwd
from bwd_op import FusedTrain
x, wa, wb, ws, g, b = make_inputs(384)
M, D, H = 384 * 384, 128, 512
f = FusedFwd(); f.set_weights(wa, wb, ws)
dy = torch.randn_like(x) * 0.1
import os
NAMES = os.environ.get("NAMES", "tbwd,babl_gate,babl_epi,babl_both").split(",")
for name in NAMES:
    for R in (9,):
        st = FusedTrain(f, repl=R, cubin=f"build/{name}.cubin").bind(x, g, b, dy)
        st(); torch.cuda.synchronize()
        rb = st.keep[1]
        r = measure(rb, 22 * M * D * H)
        print(f"{name:10s} R{R}: bwd sustained {r['us_per_call']:6.1f} us  {r['W']:5.0f} W  {r['pJ_per_flop']:.3f} pJ/FLOP   graph20 {graph_time(rb):6.1f} us", flush=True)

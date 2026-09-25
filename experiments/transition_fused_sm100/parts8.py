"""Cold graph time of each launch of the relaxed-precision step (L384): quantize, forward, backward main, reduction."""
import os, sys, torch
from common import make_inputs, graph_time
from fwd8_op import Train8
R = int(os.environ.get("R", "14"))
x, wa, wb, ws, g, b = make_inputs(384)
dy = torch.randn_like(x) * 0.1
tr = Train8(R, bcubin="build/tbwd8x.cubin"); st = tr.bind(x, wa, wb, ws, g, b, dy); st()
run_q, run_f, run_b = st.keep
bb = tr.b
print(f"quant {graph_time(run_q):.1f}  fwd {graph_time(run_f):.1f}  bwd(main+reduce) {graph_time(run_b):.1f}  step {graph_time(st):.1f} us")

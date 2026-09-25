import sys, torch
from common import make_inputs, graph_time
from fwd8_op import Train8
L = int(sys.argv[1])
x, wa, wb, ws, g, b = make_inputs(L); dy = torch.randn_like(x) * 0.1
for rep in range(2):
    for f in sys.argv[2:]:
        st = Train8(14, fcubin=f"build/{f}.cubin").bind(x, wa, wb, ws, g, b, dy); st(); st.infer()
        print(f, rep, f"infer {graph_time(st.infer):.1f}  fwd-train {graph_time(st.keep[1]):.1f}", flush=True)

"""Sustained energy / power of the e4m3 forward (training mode, saves) and inference, L384."""
import sys, torch
sys.argv = sys.argv[:1]
from energy import measure
from common import make_inputs, graph_time
from fwd8_op import Train8
M, D, H = 384 * 384, 128, 512
x, wa, wb, ws, g, b = make_inputs(384)
dy = torch.randn_like(x) * 0.1
import os
names = os.environ.get("FNAMES", "tfwd8").split(",")
todo = []
for n in names:
    st = Train8(14, fcubin=f"build/{n}.cubin", bcubin="build/tbwd8x.cubin").bind(x, wa, wb, ws, g, b, dy); st()
    todo.append((n, st.keep[1]))
if len(names) == 1:
    todo += [("fwd infer", st.infer), ("quant", st.keep[0]), ("bwd", st.keep[2])]
for name, fn in todo:
    r = measure(fn, 6 * M * D * H)
    print(f"{name:10s}: sustained {r['us_per_call']:6.1f} us {r['W']:5.0f} W {r['J_per_call']*1e3:6.1f} mJ  cold {graph_time(fn):6.1f}", flush=True)

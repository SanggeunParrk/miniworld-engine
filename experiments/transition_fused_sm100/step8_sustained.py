"""Training step (L384) sustained (~3 s NVML) and cold graph timing: relaxed-precision step vs the bf16 step, plus the cuBLAS ceiling
in both regimes, measured back to back so the SoL ratios use one thermal state."""
import os, sys, torch
R = int(os.environ.get("R", "14")); B = os.environ.get("BCUBIN", "build/tbwd8x.cubin"); F = os.environ.get("FCUBIN", "build/tfwd8.cubin")
sys.argv = sys.argv[:1]
from energy import measure
from common import make_inputs, graph_time
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
from fwd8_op import Train8
L = int(os.environ.get("L", "384"))
M, D, H = L * L, 128, 512
FL = 28 * M * D * H
x, wa, wb, ws, g, b = make_inputs(L)
dy = torch.randn_like(x) * 0.1
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); bb = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
cb = lambda: torch.mm(a, bb.t())
rc = measure(cb, 2 * 8192 ** 3); cc = graph_time(cb)
print(f"cuBLAS 8192^3 bf16: sustained {rc['TFLOPS']:.0f} TFLOPS, cold graph {2 * 8192**3 / cc / 1e6:.0f} TFLOPS", flush=True)
st8 = Train8(R, fcubin=F, bcubin=B).bind(x, wa, wb, ws, g, b, dy); st8()
f = FusedFwd2(); f.set_weights(wa, wb, ws); st16 = FusedTrain(f).bind(x, g, b, dy); st16()
for name, st in (("e4m3 step", st8), ("bf16 step", st16)):
    r = measure(st, FL); c = graph_time(st)
    print(f"{name}: sustained {r['us_per_call']:6.1f} us {r['W']:4.0f} W {r['J_per_call']*1e3:6.1f} mJ | cold {c:6.1f} us | "
          f"SoL (design 28 MDH, bf16 ceiling): sustained {FL / r['us_per_call'] / 1e6 / rc['TFLOPS'] * 100:.1f} %, cold {FL / c / 1e6 / (2 * 8192**3 / cc / 1e6) * 100:.1f} %", flush=True)

"""bf16 training step (v8 + v9 defaults) sustained and cold, with the cuBLAS bf16 ceiling measured back to back; L from env."""
import os, sys
sys.argv = sys.argv[:1]
import torch
from energy import measure
from common import make_inputs, graph_time
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
L = int(os.environ.get("L", "384")); M = L * L; FL = 28 * M * 128 * 512
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); bb = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
rc = measure(lambda: torch.mm(a, bb.t()), 2 * 8192 ** 3)
x, wa, wb, ws, g, b = make_inputs(L); dy = torch.randn_like(x) * 0.1
f = FusedFwd2(); f.set_weights(wa, wb, ws); st = FusedTrain(f).bind(x, g, b, dy); st()
r = measure(st, FL)
print(f"L{L} bf16 step: sustained {r['us_per_call']:.1f} us ({r['W']:.0f} W), cold {graph_time(st):.1f} us | cuBLAS {rc['TFLOPS']:.0f} TFLOPS | SoL {FL / r['us_per_call'] / 1e6 / rc['TFLOPS'] * 100:.1f} %")

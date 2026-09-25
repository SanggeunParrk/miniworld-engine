"""Sustained (~3 s, NVML energy) training-step time at L384 / L768 for several DW replica counts, v8 2-CTA forward + v9 backward."""
import sys, torch
Ls = [int(a) for a in sys.argv[1:]] or [384]
sys.argv = sys.argv[:1]
from energy import measure
from common import make_inputs
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
D, H = 128, 512
for L in Ls:
    M = L * L
    x, wa, wb, ws, g, b = make_inputs(L)
    f = FusedFwd2(); f.set_weights(wa, wb, ws)
    dy = torch.randn_like(x) * 0.1
    for R in (8, 9, 10):
        st = FusedTrain(f, repl=R).bind(x, g, b, dy); st(); torch.cuda.synchronize()
        r = measure(st, 28 * M * D * H)
        print(f"L{L} R{R}: step {r['us_per_call']:7.1f} us  {r['W']:5.0f} W  {r['pJ_per_flop']:.3f} pJ/FLOP  {r['TFLOPS']:.0f} TFLOPS", flush=True)

"""bf16 forward A/B: bitwise output comparison and alternating timing (inference, training-save, sustained), L384 / L768."""
import sys
cubs = sys.argv[1:]; sys.argv = sys.argv[:1]
import torch
from energy import measure
from common import make_inputs, graph_time
from fwd_op import FusedFwd2
for L in (384, 768):
    x, wa, wb, ws, g, b = make_inputs(L); outs = []
    for rep in range(2):
        for c in cubs:
            f = FusedFwd2(f"build/{c}.cubin"); f.set_weights(wa, wb, ws)
            ri, out, *_ = f.bind(x, g, b, save=False); rt, out2, xn, *_ = f.bind(x, g, b, save=True); ri(); rt(); torch.cuda.synchronize()
            if rep == 0: outs.append((out.clone(), xn.clone()))
            r = measure(ri, 6 * L * L * 128 * 512)
            print(f"L{L} {c:10s} infer cold {graph_time(ri):6.1f} sust {r['us_per_call']:6.1f} | train cold {graph_time(rt):6.1f}", flush=True)
    print(f"L{L} outputs bit-identical: {all(torch.equal(outs[0][k], o[k]) for o in outs[1:] for k in range(2))}")

"""Bitwise comparison of the bf16 training-step outputs between two backward cubins (same forward, same inputs)."""
import sys, torch
from common import make_inputs
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
x, wa, wb, ws, g, b = make_inputs(384); dy = (torch.randn(x.shape, generator=torch.Generator().manual_seed(7)) * 0.1).to("cuda", torch.bfloat16)
f = FusedFwd2(); f.set_weights(wa, wb, ws)
outs = []
for c in sys.argv[1:3]:
    st = FusedTrain(f, repl=9, cubin=c).bind(x, g, b, dy); st(); torch.cuda.synchronize()
    outs.append({k: v.clone() for k, v in st.grads.items()})
for k in outs[0]:
    a, bb = outs[0][k].float(), outs[1][k].float()
    print(k, "bit-identical" if torch.equal(outs[0][k], outs[1][k]) else f"differs: max |d| {(a - bb).abs().max().item():.3e} rel {((a - bb).norm() / a.norm()).item():.2e}")

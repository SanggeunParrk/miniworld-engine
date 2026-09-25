import torch
from common import make_inputs
from fwd_op import FusedFwd
from bwd_op import FusedTrain
x, wa, wb, ws, g, b = make_inputs(384)
f = FusedFwd(); f.set_weights(wa, wb, ws)
st = FusedTrain(f).bind(x, g, b, torch.randn_like(x) * 0.1); st(); torch.cuda.synchronize()
rb = st.keep[1]
for _ in range(10): rb()
torch.cuda.synchronize()

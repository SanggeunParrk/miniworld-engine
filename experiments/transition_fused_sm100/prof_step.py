"""One bf16 training step (v8 forward + v9 backward, L384) for ncu: warm-up, then one profiled step."""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import torch
from common import make_inputs
from fwd_op import FusedFwd2
from bwd_op import FusedTrain
x, wa, wb, ws, g, b = make_inputs(384); dy = torch.randn_like(x) * 0.1
f = FusedFwd2(); f.set_weights(wa, wb, ws)
st = FusedTrain(f, repl=int(os.environ.get("R", "9"))).bind(x, g, b, dy)
for _ in range(3): st()
torch.cuda.synchronize()
st(); torch.cuda.synchronize()

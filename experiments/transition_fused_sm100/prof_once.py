"""Run the fused forward (training build) and backward once each at one length -- the target for ncu / nsys."""
import argparse, torch
from common import make_inputs
from fwd_op import FusedFwd
from bwd_op import FusedTrain
p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=384); p.add_argument("--iters", type=int, default=3)
a = p.parse_args()
x, wa, wb, ws, gamma, beta = make_inputs(a.length)
dy = torch.randn_like(x) * 0.1
f = FusedFwd(); f.set_weights(wa, wb, ws)
run_i, *_ = f.bind(x, gamma, beta, save=False)
step = FusedTrain(f).bind(x, gamma, beta, dy)
for _ in range(a.iters):
    run_i(); step()
torch.cuda.synchronize()

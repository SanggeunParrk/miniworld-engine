import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import torch
from common import make, hoist_mod, C
from ops import FfnFwd
A, S = 48, 3072
q, cb, cos, sin, su, w = make(A, S); mod = hoist_mod(cb, w["wmod"]) * 0.5
M = A * S
g = torch.randn(M, C, device="cuda").to(torch.bfloat16); o = torch.randn(M, C, device="cuda").to(torch.bfloat16)
run, _ = FfnFwd().bind(q.reshape(M, C), g, o, mod, w["wo"], w["wu"], w["wd"], A, 1, save=False)
for _ in range(3): run()
torch.cuda.synchronize()

import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import torch
from common import make
from ops import Fwd2
L = int(os.environ.get("L", "768"))
q, k, v, bias = make(48, L)
run, O, LSE = Fwd2(os.environ.get("CUBIN", "build/fwd2_c.cubin")).bind(q, k, v, bias)
for _ in range(3): run()
torch.cuda.synchronize(); run(); torch.cuda.synchronize()

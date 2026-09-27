import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import torch
from common import make, hoist_mod
from ops import QkvgFwd
A, S = 48, 3072
q, cb, cos, sin, su, w = make(A, S); mod = hoist_mod(cb, w["wmod"])
run, _ = QkvgFwd(os.environ.get("CUBIN", "build/qkvg_fwd.cubin")).bind(q, mod, cos, sin, w["wqkv"], w["wg"], A, 1, save=bool(int(os.environ.get("SAVE", "0"))))
for _ in range(3): run()
torch.cuda.synchronize()

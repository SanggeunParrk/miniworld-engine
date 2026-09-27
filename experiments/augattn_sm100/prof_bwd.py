import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import torch
from common import make, H, D
from ops import Fwd2, Dqb, Dkv
L = int(os.environ.get("L", "384")); A = 48
q, k, v, bias = make(A, L)
do = torch.randn_like(q, dtype=torch.float32).to(torch.bfloat16)
frun, O, LSE = Fwd2("build/attn_fwd2.cubin").bind(q, k, v, bias); frun()
Dd = torch.randn(A, H, L, device="cuda") * 0.1
r1, *_ = Dqb(os.environ.get("DQB", "build/attn_dqb.cubin")).bind(q, k, v, do, bias, LSE, Dd)
r2, *_ = Dkv(os.environ.get("DKV", "build/attn_dkv.cubin")).bind(q, k, v, do, bias.transpose(1, 2).contiguous(), LSE, Dd)
for _ in range(2): r1(); r2()
torch.cuda.synchronize(); r1(); r2(); torch.cuda.synchronize()

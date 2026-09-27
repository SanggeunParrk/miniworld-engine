"""(ncu) the DX role alone: dx, dgamma, dbeta against fp32 autograd, and its time.  python test_dx.py 384 768"""
import copy
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402

ext = TB.build(extra=os.environ.get("DX_EXTRA", "").split())
L = int(sys.argv[1])
mod, x = TB.TA.fixture(L)
dy = torch.randn_like(x)
pk = TB.pack(mod)
xf = x.float()
mean, var = xf.mean(1), xf.var(1, unbiased=False)
rstd = torch.rsqrt(var + mod.ln_in.eps)
stats = torch.stack([mean, rstd], 1).contiguous()
xn = ((xf - mean[:, None]) * rstd[:, None] * mod.ln_in.weight.float() + mod.ln_in.bias.float()).bfloat16()
dx = torch.empty_like(x)
dgb = torch.empty(108, 256, device="cuda")
for _ in range(4):
    ext.bwd_dx(x, xn, dy, stats, pk["wdx"], pk["gamma"], dx, dgb)
torch.cuda.synchronize()

"""Gradient magnitude robustness: dy scaled by 1, 1e-3, 1e-5, 1e-7 -- rel-RMS of every gradient, 3-kernel vs one-launch ring.
    python test_scale.py 384"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
from bench_common import rel_rms  # noqa: E402

ext = TB.build()
L = int(sys.argv[1])
mod, x = TB.TA.fixture(L)
pk = TB.pack(mod)
names = ["dx", "dgamma", "dbeta", "dWa", "dWb", "dWs"]
for sc in (1.0, 1e-3, 1e-5, 1e-7):
    torch.manual_seed(7)
    dy = (torch.randn_like(x.float()) * sc).bfloat16()
    m32 = copy.deepcopy(mod).float()
    x32 = x.float().requires_grad_(True)
    p32 = [m32.ln_in.weight, m32.ln_in.bias, m32.expand_a.weight, m32.expand_b.weight, m32.squeeze.weight]
    ref = torch.autograd.grad(m32(x32), [x32] + p32, dy.float())
    for tag, fn in (("3-kernel", lambda: TB.backward(ext, x, dy, pk, {})), ("ring", lambda: TB.backward_ring(ext, x, dy, pk, {}, ndxp=64))):
        got = fn()
        torch.cuda.synchronize()
        print(f"dy x {sc:.0e} {tag:9s}: " + " ".join(f"{n} {rel_rms(g, r):.2e}" for n, g, r in zip(names, got, ref)), flush=True)
